import copy
import csv
import itertools
import logging
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from enum import Enum
from typing import Any, Callable, NamedTuple, Optional, TYPE_CHECKING, Union

import torch
import torch.distributed as dist
from torch._dynamo import OptimizedModule
from torch.distributed.fsdp import FSDPModule, UnshardHandle
from torch.nn.modules.loss import _Loss
from torch.profiler import record_function

from torch.distributed.pipelining._utils import generate_stage_to_rank_mapping
from torch.distributed.pipelining.stage import _PipelineStageBase

from .mbp_stage import _MbpStageBase

def g_str(s):
    return "\033[32m" + s + "\033[0m"
def r_str(s):
    return "\033[31m" + s + "\033[0m"
def b_str(s):
    return "\033[34m" + s + "\033[0m"
def y_str(s):
    return "\033[33m" + s + "\033[0m"

class _MbpSchedule(ABC):
    def __init__(
        self,
        microbatch_idx: int,
        microbatch_size: int,
        loss_fn: Optional[Callable[..., torch.Tensor]] = None,
        args_chunk_spec: Optional[tuple[Any, ...]] = None,
        kwargs_chunk_spec: Optional[dict[str, Any]] = None,
        output_merge_spec: Optional[Union[dict[str, Any], tuple[Any]]] = None,
        global_rank: Optional[int] = None,
    ):
        # From arguments
        self._microbatch_idx = microbatch_idx
        self._microbatch_size = microbatch_size
        self._loss_fn = loss_fn
        self._global_rank = global_rank

        # Chunking specification for positional inputs. (default: `None`)
        self._args_chunk_spec = args_chunk_spec
        # Chunking specification for keyword inputs. (default: `None`)
        self._kwargs_chunk_spec = kwargs_chunk_spec
        self._output_merge_spec = output_merge_spec
        """
        # args_chunk_spec and kwargs_chunk_spec specify how to chunk inputs.
        # They are used to convert batch to microbatches in `step(x)`.  See
        # `TensorChunkSpec` for helper methods for creating them.
        """

        # Derived
        self._has_backward = self._loss_fn is not None

        # Holds the losses for each microbatch.
        self._internal_loss: torch.Tensor = None

    def _maybe_compute_loss(self, stage, output, target_mb):
        if stage.is_last and self._has_backward:
            # For single microbatch, target_mbs is just the target
            loss = self._compute_loss(output, target_mb)
            self._internal_loss = loss

    def _maybe_get_loss(self, stage):
        if stage.is_last and self._has_backward:
            if self._internal_loss is None:
                raise RuntimeError(
                    f"Loss is not available. "
                    f"Available loss: {self._internal_loss} for {self._microbatch_idx}"
                )
            return self._internal_loss
        else:
            return None

    def _update_losses(self, stages, losses):
        """
        Update the losses to those in the internal state
        """
        # if stages not a list turn into a list
        if not isinstance(stages, list):
            stages = [stages]
        contains_last_stage = any(stage.is_last for stage in stages)

        # Return losses if there is a container passed in
        if contains_last_stage and losses is not None:
            if len(self._internal_loss) != 1:
                raise RuntimeError(
                    f"Expecting 1 loss but got {len(self._internal_loss)}"
                )

            # Clean external container first
            losses.clear()
            # Copy internal losses to external container
            losses.extend(self._internal_loss)

        self._internal_loss = None

    @abstractmethod
    def _step_microbatches(
        self,
        args: Optional[tuple[Any, ...]] = None,
        kwargs: Optional[dict[str, Any]] = None,
        target: Optional[torch.Tensor] = None,
        losses: Optional[list] = None,
        mbp_ctrl=None,
    ):
        """
        Run one iteration of the pipeline schedule with single microbatch.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, *args, target=None, losses: Optional[list] = None, **kwargs):
        """
        Run one iteration of the pipeline schedule with *whole-batch* input.
        No chunking needed for single microbatch.

        args: positional arguments to the model (as in non-pipeline case).
        kwargs: keyword arguments to the model (as in non-pipeline case).
        target: target for the loss function.
        losses: a list to store the loss.
        """
        raise NotImplementedError

    def _compute_loss(self, output, target):
        return self._loss_fn(output, target)  # type: ignore[misc]

def _batch_p2p(p2p_ops: list[dist.P2POp], desc: Optional[str] = None):
    """
    Simple wrapper over batch_isend_irecv from torch.distributed, which just adds a descriptive logger on top.
    """
    if len(p2p_ops) == 0:
        return None
    desc_str = f"{desc}, " if desc else ""
    return dist.batch_isend_irecv(p2p_ops).pop()


def _sorted_batch_p2p(
    p2p_ops: list[dist.P2POp], desc: Optional[str] = None
) -> dict[int, dist.Work]:
    """
    Sorts the list of P2P ops by the peer rank, and then calls
    batch_isend_irecv. Return a dictionary of works by peer rank. This function
    helps us avoid hangs in case of skip connections.
    """
    # Arrange p2p_ops by peer rank:
    #   int is the peer rank;
    #   List is the list of ops towards the peer
    ops_by_peer: dict[int, list[dist.P2POp]] = defaultdict(list)
    work_by_peer: dict[int, dist.Work] = {}
    if len(p2p_ops) == 0:
        return work_by_peer

    # Classify the ops by peer rank
    for op in p2p_ops:
        ops_by_peer[op.peer].append(op)

    # Call batch_isend_irecv per peer, in sorted order of the peers (to avoid hangs)
    for peer, ops in sorted(ops_by_peer.items()):
        work_by_peer[peer] = _batch_p2p(ops, desc=desc)

    return work_by_peer


class MbpScheduleSingle(_MbpSchedule):
    """
    Base class for single-stage schedules.
    Implements the `step` method.
    Derived classes should implement `_step_microbatches`.

    Gradients are not scaled since we only have one microbatch.
    """

    def __init__(
        self,
        stage: _MbpStageBase,
        microbatch_idx: int,
        microbatch_size: int,
        loss_fn: Optional[Callable] = None,
        args_chunk_spec: Optional[tuple[Any, ...]] = None,
        kwargs_chunk_spec: Optional[dict[str, Any]] = None,
        output_merge_spec: Optional[Union[dict[str, Any], tuple[Any]]] = None,
        global_rank: Optional[int] = None,
    ):
        # Init parent
        super().__init__(
            microbatch_idx=microbatch_idx,
            microbatch_size=microbatch_size,
            loss_fn=loss_fn,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
            global_rank=global_rank,
        )
        # Self attributes
        self._stage = stage
        self._num_stages = stage.num_stages
        # Set the same has_backward flag for stage object
        self._stage.has_backward = self._has_backward
        self._stage_initialized = False

    def _initialize_stage(self, args, kwargs):
        # For single microbatch, we only need to prepare for 1 chunk
        self._stage._prepare_forward_infra(args, kwargs)
        if self._has_backward:
            self._stage._prepare_backward_infra()
        self._stage_initialized = True

    def step(self, *args, target=None, losses: Optional[list] = None,
            mbp_ctrl=None,
            **kwargs):
        """
        Run one iteration with single microbatch input.
        No chunking needed - inputs are used directly.
        """
        # Clean per iteration
        self._stage.clear_runtime_states()
        # Run single microbatch
        self._step_microbatches(args, kwargs, target, losses,
                                mbp_ctrl)

        # Return outputs directly (no merging needed)
        if self._stage.is_last:
            return self._stage.output_chunk
        else:
            return None


class ScheduleMbp(MbpScheduleSingle):
    """
    The GPipe schedule for single microbatch.
    Processes one microbatch with immediate communication completion.
    """

    def _step_microbatches(
        self,
        args: Optional[tuple[Any, ...]] = None,
        kwargs: Optional[dict[str, Any]] = None,
        target_mb: Optional[torch.Tensor] = None,
        losses: Optional[list] = None,
        mbp_ctrl=None,
    ):
        """
        Run single microbatch - simplified version of GPipe schedule.
        """
        # For single microbatch, we can simplify input validation
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        if target_mb is None:
            target_mb = None

        if not self._stage_initialized:
            self._initialize_stage(args, kwargs)

        works = {}

        # Forward pass
        with record_function(f"Forward {self._microbatch_idx}"):
            ops = self._stage.get_fwd_recv_ops()      
            work_sync = _sorted_batch_p2p(ops, desc="fwd_recv")
            for work in work_sync.values():
                work.wait()
            logging.info(g_str(f"Rank {self._global_rank}: ") 
                         + b_str(f"Forwarding {self._microbatch_idx}") 
                         + f", receiving {ops}")

            output = self._stage.forward_one_chunk(args, kwargs)
 
            ops = self._stage.get_fwd_send_ops()
            logging.info(g_str(f"Rank {self._global_rank}: ") 
                         + b_str(f"Forwarded {self._microbatch_idx}") + f", sending {ops}")
            works.update(_sorted_batch_p2p(ops, desc="fwd_send"))

            if mbp_ctrl is not None:
                mbp_ctrl.add(1)

        # Compute loss if this is the last stage
        if self._stage.is_last and self._has_backward and target_mb is not None:
            loss = self._compute_loss(output, target_mb)
            if losses is not None:
                losses.clear()
                losses.append(loss)

        # No loss function, no need to run backward
        if not self._has_backward:
            return   

        # Backward pass
        with record_function(f"Backward {self._microbatch_idx}"):
            ops = self._stage.get_bwd_recv_ops()
            work_sync = _sorted_batch_p2p(ops, desc="bwd_recv")
            for work in work_sync.values():
                work.wait()
            logging.info(g_str(f"Rank {self._global_rank}: ")+ r_str(f"Backwarding {self._microbatch_idx}") + f", receiving {ops}")

            # For single microbatch, loss is directly available
            loss = loss if self._stage.is_last else None
            self._stage.backward_one_chunk(loss=loss)

            ops = self._stage.get_bwd_send_ops()
            logging.info(g_str(f"Rank {self._global_rank}: ")+ r_str(f"Backwarded {self._microbatch_idx}") + f", sending {ops}")
            works.update(_sorted_batch_p2p(ops, desc="bwd_send"))

        # Wait immediately for single microbatch
        for work in works.values():
            work.wait()
        
        if self._microbatch_idx == self._microbatch_size - 1:
            print(g_str(f"Rank {self._global_rank}: ") + b_str(f"Adding shared gradients back to model"))
            # Add shared gradients back to model here
            # self._stage.scale_grads(1)
            self._stage.run_fsdp_post_backward()
        else:
            print(g_str(f"Rank {self._global_rank}: ") + b_str(f"Doing gradient reduction for non-last microbatch"))
            # Do gradient reduction for non-last microbatch
            pass
        

        # Return losses if there is a container passed in
        self._update_losses(self._stage, losses)

