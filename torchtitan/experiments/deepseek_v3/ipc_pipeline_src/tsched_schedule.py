import copy
import csv
import itertools
import logging
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from enum import Enum
from typing import Any, Callable, NamedTuple, Optional, TYPE_CHECKING, Union
import threading
import torch
import torch.distributed as dist
from ipc_pipeline_src.thread_scheduler import ContextScheduler
from torch._dynamo import OptimizedModule
from torch.distributed.fsdp import FSDPModule, UnshardHandle
from torch.nn.modules.loss import _Loss
from torch.profiler import record_function

from torch.distributed.pipelining._utils import generate_stage_to_rank_mapping
from torch.distributed.pipelining.stage import _PipelineStageBase

from .tsched_stage import _TschedStageBase

def g_str(s):
    return "\033[32m" + s + "\033[0m"
def r_str(s):
    return "\033[31m" + s + "\033[0m"
def b_str(s):
    return "\033[34m" + s + "\033[0m"
def y_str(s):
    return "\033[33m" + s + "\033[0m"
def c_str(s):
    return "\033[36m" + s + "\033[0m"
def m_str(s):
    return "\033[35m" + s + "\033[0m"

class _TschedSchedule(ABC):
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

    def _maybe_compute_loss(self, stage, output, target):
        if stage.is_last and self._has_backward:
            loss = self._compute_loss(output, target)
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
            # Clean external container first
            losses.clear()
            # Copy internal losses to external container
            losses.append(self._internal_loss)

        self._internal_loss = None

    @abstractmethod
    def _step_microbatches(
        self,
        args: Optional[tuple[Any, ...]] = None,
        kwargs: Optional[dict[str, Any]] = None,
        target: Optional[torch.Tensor] = None,
        losses: Optional[list] = None,
        scheduler: Optional[ContextScheduler] = None,
        exec_id: Optional[int] = None,
        step_idx=0,
    ):
        """
        Run one iteration of the pipeline schedule with single microbatch.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, *args, target=None, losses: Optional[list] = None, 
             scheduler: Optional[ContextScheduler] = None, exec_id: Optional[int] = None, **kwargs):
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
    out = dist.batch_isend_irecv(p2p_ops)
    print(f"Rank {dist.get_rank()} Returning from batch_p2p {desc}: {out}")
    return out

def _batch_p2p_non_coalescing(p2p_ops: list[dist.P2POp], desc: Optional[str] = None, microbatch_idx: int = 0):
    """
    Process P2P operations sorted by peer rank to avoid hangs in case of skip connections.
    Force individual operations instead of coalescing.
    """
    if len(p2p_ops) == 0:
        return []
    
    current_stream = torch.cuda.current_stream()
    
    # Ensure all operations use the current stream
    with torch.cuda.stream(current_stream):
        
        # Sort operations by peer rank to avoid hangs
        sorted_ops = sorted(p2p_ops, key=lambda op: op.peer)
        
        work_objects = []
        for p2p_op in sorted_ops:
            # Call individual isend/irecv directly
            if p2p_op.op == dist.isend:
                print(c_str(f"[COMMS R{dist.get_rank()}]") + b_str(f" {desc}: ") + 
                      f"Sending microbatch {microbatch_idx} to rank {p2p_op.peer}: {p2p_op}")
                work = dist.isend(
                    p2p_op.tensor,
                    dst=p2p_op.peer,
                    group=p2p_op.group,
                )
                print(c_str(f"[COMMS R{dist.get_rank()}]") + b_str(f" {desc}: ") + 
                      f"Sent microbatch {microbatch_idx} to rank {p2p_op.peer}: {p2p_op}")
            elif p2p_op.op == dist.irecv:
                print(m_str(f"[COMMS R{dist.get_rank()}]") + r_str(f" {desc}: ") + 
                      f"Receiving microbatch {microbatch_idx} from rank {p2p_op.peer}: {p2p_op}")
                work = dist.irecv(
                    p2p_op.tensor,
                    src=p2p_op.peer,
                    group=p2p_op.group,
                )
            
            if work:
                # IMMEDIATE SYNCHRONIZATION FIX - wait for operation to complete
                # work.wait()
                work_objects.append(work)  # Still return for compatibility
    
    print(y_str(f"[COMMS R{dist.get_rank()}]") + f" Returning from non-coalescing batch_p2p {desc}: {work_objects}")
    return work_objects


class TschedScheduleSingle(_TschedSchedule):
    """
    Base class for single-stage schedules.
    Implements the `step` method.
    Derived classes should implement `_step_microbatches`.

    Gradients are not scaled since we only have one microbatch.
    """

    def __init__(
        self,
        stage: _TschedStageBase,
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
            scheduler: Optional[ContextScheduler] = None,
            exec_id: Optional[int] = None,
            step_idx=0,
            **kwargs):
        """
        Run one iteration with single microbatch input.
        No chunking needed - inputs are used directly.
        """
        # Clean per iteration
        self._stage.clear_runtime_states()
        # Run single microbatch
        self._step_microbatches(args, kwargs, target, losses, scheduler, exec_id, step_idx)

        # Return outputs directly (no merging needed)
        if self._stage.is_last:
            return self._stage.output_chunk
        else:
            return None


class ScheduleTsched(TschedScheduleSingle):
    """
    The GPipe schedule for single microbatch.
    Processes one microbatch with immediate communication completion.
    """
    
    def initialize_stage(
        self,
        *args,
        target: Optional[torch.Tensor] = None,
        losses: Optional[list] = None,
        scheduler: Optional[ContextScheduler] = None,
        exec_id: Optional[int] = None,
        step_idx=0,
        **kwargs,
    ):
        """
        Initialize the stage.
        """
        if not self._stage_initialized:
            self._initialize_stage(args, kwargs)

    def _step_microbatches(
        self,
        args: Optional[tuple[Any, ...]] = None,
        kwargs: Optional[dict[str, Any]] = None,
        target: Optional[torch.Tensor] = None,
        losses: Optional[list] = None,
        scheduler: Optional[ContextScheduler] = None,
        exec_id: Optional[int] = None,
        step_idx=0,
    ):
        """
        Run single microbatch - simplified version of GPipe schedule.
        """
        # For single microbatch, we can simplify input validation
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        if target is None:
            target = None
            
        if not self._stage_initialized:
            self._initialize_stage(args, kwargs)

        works = []
        ident = threading.current_thread().ident
        # Wait for the current microbatch to be scheduled
        # Forward pass

        scheduler.attach_exec_to_context(exec_id) 

        with record_function(f"[T{ident} R{self._global_rank} M{self._microbatch_idx}] Forward"):
            ops, comm_key = self._stage.get_fwd_recv_ops()   
            if comm_key != -1:
                scheduler.schedule_comm(exec_id, 
                                        lambda: _batch_p2p_non_coalescing(ops, 
                                        desc="fwd_recv", microbatch_idx=self._microbatch_idx),
                                        comm_key, True)
                print(g_str(f"[T{ident} R{self._global_rank} E{self._microbatch_idx}] ") + 
                      b_str(f"Forwarding {self._microbatch_idx}") + 
                      f", received {ops}, comm_key: {comm_key}")
            
        
            with torch.profiler.record_function(f"Forward {step_idx}"):
                output = self._stage.forward_one_chunk(args, kwargs)
        

            ops, comm_key = self._stage.get_fwd_send_ops()
            if comm_key != -1:
                print(g_str(f"[T{ident} R{self._global_rank} M{self._microbatch_idx}] ") + 
                            b_str(f"Forwarded {self._microbatch_idx}") + 
                            f", sending {ops}, comm_key: {comm_key}")
                scheduler.schedule_comm(exec_id, 
                                        lambda: _batch_p2p_non_coalescing(ops, 
                                        desc="fwd_send", microbatch_idx=self._microbatch_idx),
                                        comm_key, False)

            # Compute loss if this is the last stage
            self._maybe_compute_loss(self._stage, output, target)
            
        # No loss function, no need to run backward
        
        scheduler.enter_serialized_region(exec_id, region_id=2, 
                                              region_name="Backward")
        # Backward pass
        with record_function(f"[T{ident} R{self._global_rank} M{self._microbatch_idx}] Backward"):
            ops, comm_key = self._stage.get_bwd_recv_ops()
            if comm_key != -1:
                scheduler.schedule_comm(exec_id, 
                                        lambda: _batch_p2p_non_coalescing(ops, 
                                        desc="bwd_recv", microbatch_idx=self._microbatch_idx),
                                        comm_key, True)
                print(g_str(f"[T{ident} R{self._global_rank} M{self._microbatch_idx}] ") + 
                            r_str(f"Backwarding {self._microbatch_idx}") + 
                            f", received {ops}, comm_key: {comm_key}")

            # For single microbatch, loss is directly available
            loss = self._maybe_get_loss(self._stage)
            with torch.profiler.record_function(
                f"[T{ident} R{self._global_rank} M{self._microbatch_idx}] Backward pass"):
                self._stage.backward_one_chunk(loss=loss)
                
            ops, comm_key = self._stage.get_bwd_send_ops()
            if comm_key != -1:
                print(g_str(f"[T{ident} R{self._global_rank} M{self._microbatch_idx}] ") + 
                            r_str(f"Backwarded {self._microbatch_idx}") + 
                            f", sending {ops}, comm_key: {comm_key}")
                scheduler.schedule_comm(exec_id, 
                                        lambda: _batch_p2p_non_coalescing(ops, 
                                        desc="bwd_send", microbatch_idx=self._microbatch_idx),
                                        comm_key, False)
            self._update_losses(self._stage, losses)
            
        scheduler.exit_serialized_region(exec_id, region_id=2, 
                                            region_name="Backward")
        scheduler.detach_exec_from_context(exec_id) 
        # Wait immediately for single microbatch
        for work in works:
            work.wait()

        # Return losses if there is a container passed in
        

