# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import operator
from abc import ABC, abstractmethod
from typing import Any, Callable, cast, Optional, Union

import torch
import torch.distributed as dist
import torch.fx as fx
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensor
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.fx.node import Argument, map_aggregate
from torch.nn.parallel import DistributedDataParallel
from torch.utils._pytree import tree_map_only

from torch.distributed.pipelining._backward import stage_backward, stage_backward_input, stage_backward_weight
from torch.distributed.pipelining._debug import map_debug_info
from torch.distributed.pipelining._utils import flatten_args, PipeInfo, validate_tensors_metadata

def g_str(s):
    return "\033[32m" + s + "\033[0m"
def r_str(s):
    return "\033[31m" + s + "\033[0m"
def b_str(s):
    return "\033[34m" + s + "\033[0m"
def y_str(s):
    return "\033[33m" + s + "\033[0m"

__all__ = [
    "MbpStage",
    "build_stage",
]

logger = logging.getLogger(__name__)


def _normalize_model_output_as_tuple(output: Any) -> tuple[Any]:
    """[Note: pipeline model output type]

    The output of the model passed to pipelining can be any type, controlled by the user.

    However, there are 2 API surfaces that complicate this.
    (1) the outputs of intermediate stages are passed via Send/Recv ops to subsequent stages. The implicit assumption
    is that each element of the outputs is a tensor.  Otherwise, Send/Recv would not be supported.  The exception
    is the last layer of the model, which can output anything any which won't be communicated via Send/Recv.
    (2) the outputs of the last layer of the model are returned to the user, or, passed to the loss function.
    The loss function can be written in any way, such that its inputs match the outputs of the model.

    It would be convenient if we could strictly type the output signature of the pipeline stage wrapping the model,
    but we do not want to impose an unnecessary constraint on user provided models.

    Currently, we let user provided models return either a Tensor or a tuple of Tensors from each stage. Due to
    torch.export tracing, compiled models may also return a list instead of a Tuple, which we will normalize back to a
    tuple for consistency.

    TODO: should we be stricter about asserting that stage modules (intermediate and output) all return only Tensor
    values?
    """
    if type(output) is list:
        # HACK: this is a hacky workaround for the fact that export creates
        # output in list format
        output = tuple(output)

    # Unify output form to tuple for easy correspondance with
    # `act_send_info`
    output_tuple = output if type(output) is tuple else (output,)
    return output_tuple


class _RootArgPlaceholder:
    """
    Placeholder for model-level inputs.
    """

    def __init__(self, tensor):
        self.meta = tensor.to("meta")


class _RecvInfo:
    """
    Represents a stage input.
    """

    def __init__(
        self,
        input_name: str,
        source: int,
        buffer: torch.Tensor,
    ):
        # Name of this input
        self.input_name = input_name
        # Stage index of the source of this input
        self.source = source
        # Buffer to receive the input into.
        self.buffer = buffer

    def __repr__(self):
        return f"_RecvInfo(input={self.input_name}, source={self.source}, shape={self.buffer.size()})"


# An input can be either a received activation or a model input
InputInfo = Union[_RecvInfo, _RootArgPlaceholder]


def _make_tensor_from_meta(
    example: Union[torch.Tensor, FakeTensor],
    device: torch.device,
) -> torch.Tensor:
    """
    Create a real tensor from a tensor.
    """
    return torch.empty(
        example.size(),
        dtype=example.dtype,
        layout=example.layout,
        device=device,
    )


class _MbpStageBase(ABC):
    """
    Base class for pipeline stages.
    Defines or implements common methods used by the `_MbpStage` used by
    the tracing frontend and `MbpStage` used by manual frontend.
    """

    def __init__(
        self,
        submodule: torch.nn.Module,
        microbatch_idx: int,
        stage_index: int,
        num_stages: int,
        device: torch.device,
        group: Optional[dist.ProcessGroup] = None,
        dw_builder: Optional[Callable[[], Callable[..., None]]] = None,
    ):
        """
        Args:
            submodule (torch.nn.Module): The module to be executed in this stage.
            microbatch_idx (int): The index of the microbatch.
            stage_index (int): The microbatch index of this stage.
            num_stages (int): The total number of stages in this pipeline.
            device (torch.device): The device to run this stage on.
            group (Optional[dist.ProcessGroup]): The process group to use for communication.
                If `None`, the default process group will be used.
                Default: `None`.
            dw_builder (Optional[Callable[[], Callable[..., None]]): If provided, dw_builder is a builder function
                that will build a new dw_runner function that will run parts of module backward that were intentionally
                skipped during the module's actual backward pass. The builder must be invoked by stage after stage runs
                model backwards, and stage should save the latest dw_runner to run during weight pas (W).
                If not provided, a dw_runner will be generated automatically by traversing the autograd graph.
                When used with schedules that only have F and B steps, the fresh dw_runner function will be called as
                part of I (input backwards). When used with F,I,W schedules, the dw_runner function implements 'W'.
        """
        super().__init__()
        if stage_index >= num_stages:
            raise ValueError(
                f"Stage index {stage_index} is out of range of {num_stages}"
            )

        self.submod = submodule
        self.microbatch_idx = microbatch_idx
        self.stage_index = stage_index
        self.num_stages = num_stages
        self.device = device
        self.group = group

        self.dw_builder = dw_builder

        # backward state
        self.backward_state: tuple[Any, ...] = ()

        # store dw_runner per microbatch_id
        self.dw_runner: Callable[..., None] = lambda: None

        # `group_rank` is rank in process group `group`.
        self.group_rank = dist.get_rank(self.group)
        self.group_size = dist.get_world_size(self.group)
        if self.group_size > self.num_stages:
            raise RuntimeError(
                f"Pipeline group size {self.group_size} cannot be larger than number of stages {self.num_stages}"
            )

        # Run time states
        self._outputs_meta: Optional[tuple[torch.Tensor, ...]] = None
        # map microbatch ID to list of forward tensor args
        self.fwd_cache: tuple[Any, list[torch.Tensor]] = ()
        # map microbatch ID to list of backward grad tensor args
        self.bwd_cache: tuple[Optional[torch.Tensor], ...] = ()
        # Caching chunk outputs for final output merge or reduction
        self.output_chunk: Any = ()

        # Initialize has_backward to false; this will be set to true if loss
        # function is passed to pipeline schedule
        self.has_backward = False
        # Log prefix
        self.log_prefix = f"[Stage {self.stage_index}]"

        # Forward infra
        self.args_recv_info: tuple[InputInfo, ...] = ()
        self.act_send_info: list = []

        # Backward infra will created lazily
        self.grad_recv_info: tuple[_RecvInfo, ...] = ()
        self.grad_send_info: Optional[list[Optional[int]]] = None

        # To be populated later by the Schedule
        self.stage_index_to_group_rank: dict[int, int] = {
            i: i % self.group_size for i in range(self.num_stages)
        }

    @property
    def has_backward(self) -> bool:
        """
        Returns true if this stage has a backward pass.
        """
        return self._has_backward

    @has_backward.setter
    def has_backward(self, has_backward: bool):
        self._has_backward = has_backward

    @property
    def is_first(self):
        """
        Returns true if this stage is the first stage in the pipeline.
        """
        return self.stage_index == 0

    @property
    def is_last(self):
        """
        Returns true if this stage is the last stage in the pipeline.
        """
        return self.stage_index == self.num_stages - 1

    def _configure_outputs_meta(self, outputs_meta: tuple[torch.Tensor, ...]):
        """
        Track the output shapes/dtype of this stage since they determine the send operation(s) which must match
        recv operations of the next stage.  The next stage _will_ be freezing its recv buffers based on its initial
        configuration, so it's important to also freeze/validate the output side to avoid any send/recv mismatches
        which could show up as hangs, silent corruption, or other errors.
        """
        assert self._outputs_meta is None, (
            "Attempting to reconfigure output_meta, which is not supported"
        )
        self._outputs_meta = tuple(outputs_meta)  # type: ignore[assignment]

    def get_outputs_meta(self) -> tuple[torch.Tensor, ...]:
        """Get the output metadata (meta tensors) reprensenting the outputs of this stage"""
        assert self._outputs_meta is not None, (
            "Attempted to get_outputs_meta() without configuring output meta"
        )
        return self._outputs_meta

    def _create_grad_send_info(
        self,
        args_recv_info: tuple,
    ) -> list[Optional[int]]:
        """
        Create a list of stage indices to send gradients to.
        """
        grad_send_info: list[Optional[int]] = []

        def map_recv_to_send(a):
            # Note: we send gradients back to previous stage as long as in
            # forward it is a received input, regardless of whether it requires
            # grad. It is up to the previous stage to disgard this gradient.
            if isinstance(a, _RecvInfo):
                grad_send_info.append(a.source)
                return a.source
            else:
                grad_send_info.append(None)
                return None

        map_aggregate(args_recv_info, map_recv_to_send)

        return grad_send_info

    @abstractmethod
    def _prepare_forward_infra(
        self,
        args: tuple[Any, ...],
        kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, ...]:
        raise NotImplementedError

    def _prepare_backward_infra(self):
        # `grad_recv_info` is a mirror of `act_send_info`
        self.grad_recv_info = self._create_grad_recv_info(
            self.act_send_info
        )

    @abstractmethod
    def _create_grad_recv_info(
        self,
        act_send_info: dict,
    ) -> tuple[_RecvInfo, ...]:
        raise NotImplementedError

    def _get_recv_ops(
        self,
        recv_infos: tuple[InputInfo, ...],
    ) -> list[dist.P2POp]:
        """
        Helper function shared by `get_fwd_recv_ops` and `get_bwd_recv_ops`.
        Returns a list of ops that correspond to the recv infos.
        """
        ops: list[dist.P2POp] = []
        for info in recv_infos:
            if not isinstance(info, _RecvInfo):
                continue

            peer_rank = self.stage_index_to_group_rank[info.source]
            peer_global_rank = (
                peer_rank
                if self.group is None
                else dist.get_global_rank(self.group, peer_rank)
            )
            ops.append(
                dist.P2POp(dist.irecv, info.buffer, peer_global_rank, self.group)
            )

        return ops

    """[Note: V-schedule special case]

    V-Schedules have a special case where 2 stages with adjacent stage_id are on the same rank.

    ex: 2 ranks, 4 stages forms a simple V:
    rank0:  stage 0                   stage 3
    rank1:          stage 1  stage 2

    stage 0,1 and 2,3 communicate activations using send/recv as usual, but stage 1,2 do not need to
    use communication ops.  Instead, they should pass tensor data directly via function call.

    set_local_fwd_input and (get_local_bwd_output + set_local_bwd_input) facilitate this optimization, and
    should be called at the appropriate time during the pipeline schedule (after forward or backward execution).
    """

    def set_local_fwd_input(self, prev_stage_outputs: Any) -> None:
        """
        Moves 'prev_stage_outputs' from another stage on the same rank into place as inputs for this stage. Avoids
        copying tensor data or using send/recv op.  Detaches original tensor and sets requires_grad so the
        tensor can serve as a leaf for autograd and gradients can be collected from it during backward.
        """
        recv_infos: tuple[InputInfo, ...] = self.args_recv_info

        # See [Note: pipeline model output type]
        prev_stage_outputs = _normalize_model_output_as_tuple(prev_stage_outputs)

        for info, tensor in zip(recv_infos, prev_stage_outputs):
            assert isinstance(tensor, torch.Tensor), (
                f"expected tensor values as outputs from prev stage, got {type(tensor)}"
            )
            assert isinstance(info, _RecvInfo), (
                "set_local_Fwd_input should only be called on non-first stage, which should always have RecvInfo"
            )

            # We don't need to do a data copy here, since we can directly pass the activation tensor reference from
            # one stage to the next.  However, we do need to mark the activation as a leaf tensor since it will serve
            # as the input tensor for a fresh autograd graph, not part of the previous stage's autograd graph.
            # TODO: confirm, do we use this activation as the root of the backward call for the previous stage? does
            # detach have any affect on that?
            info.buffer = tensor.detach().requires_grad_(True)

    def get_local_bwd_output(self):
        """
        Returns the input grad tensors for this stage, which correspond to the stage inputs during forward.
        """
        assert self.has_backward, (
            "can't steal_bwd_input if this stage doesn't have backward"
        )
        assert not self.is_first, "can't get bwd output if this stage is first"

        return self.bwd_cache

    def set_local_bwd_input(
        self, next_stage_bwd_outputs: tuple[Optional[torch.Tensor], ...]
    ) -> None:
        """
        Moves 'grad input' tensors from the next stage to 'grad_output' on this stage, avoiding a copy or send/recv.
        Does not detach or set '_requires_grad'.
        """
        assert isinstance(next_stage_bwd_outputs, tuple), (
            f"Expected tuple, got {type(next_stage_bwd_outputs)}"
        )

        assert self.has_backward, (
            "can't set bwd input if this stage doesn't have backward"
        )
        assert not self.is_last, "can't set bwd input if this stage is last"
        recv_infos = self.grad_recv_info
        for info, tensor in zip(recv_infos, next_stage_bwd_outputs):
            assert isinstance(tensor, torch.Tensor), (
                f"expected tensor values as outputs from prev stage, got {type(tensor)}"
            )
            assert isinstance(info, _RecvInfo), (
                f"Expected a recv info, got {type(info)}"
            )
            info.buffer = tensor

    def get_fwd_recv_ops(self) -> list[dist.P2POp]:
        """
        Returns a list of ops that are needed to receive the input arguments
        for this stage.
        """
        recv_infos: tuple[InputInfo, ...] = self.args_recv_info

        return self._get_recv_ops(recv_infos)

    def get_bwd_recv_ops(self) -> list[dist.P2POp]:
        """
        Returns a list of ops that are needed to receive the gradients
        for this stage.
        """
        if not self.has_backward or self.is_last:
            return []

        recv_infos = self.grad_recv_info    
        return self._get_recv_ops(recv_infos)

    def get_fwd_send_ops(self) -> list[dist.P2POp]:
        """
        Get the activation send ops for current stage's forward.
        """
        output = self.output_chunk
        # Unify output form to tuple for easy correspondance with
        # `act_send_info`
        output_tuple = output if type(output) is tuple else (output,)

        ops: list[dist.P2POp] = []

        for idx, out in enumerate(output_tuple):
            dst_stages = self.act_send_info[idx]
            for dst in dst_stages:
                if dst is None:
                    continue
                peer_rank = self.stage_index_to_group_rank[dst]
                peer_global_rank = (
                    peer_rank
                    if self.group is None
                    else dist.get_global_rank(self.group, peer_rank)
                )
                ops.append(dist.P2POp(dist.isend, out, peer_global_rank, self.group))

        return ops

    def get_bwd_send_ops(self) -> list[dist.P2POp]:
        """
        Get the gradient send ops for current stage's backward.
        """
        if not self.has_backward or self.is_first:
            return []

        # Create bwd send infra lazily
        if self.grad_send_info is None:
            # Send info for input grads during backward:
            # List of destinations corresponding to input grads
            # Can be None if an input has no grad
            # `grad_send_info` is a mirror of `args_recv_info`
            self.grad_send_info = self._create_grad_send_info(self.args_recv_info)

        ops: list[dist.P2POp] = []
        grads_input = self.bwd_cache
        for grad, grad_recv_stage in zip(grads_input, self.grad_send_info):
            if isinstance(grad, torch.Tensor) and grad_recv_stage is not None:
                peer_rank = self.stage_index_to_group_rank[grad_recv_stage]
                peer_global_rank = (
                    peer_rank
                    if self.group is None
                    else dist.get_global_rank(self.group, peer_rank)
                )
                ops.append(dist.P2POp(dist.isend, grad, peer_global_rank, self.group))
            else:
                if not (grad is None and grad_recv_stage is None):
                    raise RuntimeError(
                        f"[{self.stage_index}] for chunk {self.microbatch_idx} has gradients {grad} "
                        f"and is expecting to send gradients to stage {grad_recv_stage}"
                    )
        return ops

    def clear_runtime_states(self) -> None:
        """
        Clear runtime states of the stage.
        """
        # map microbatch ID to list of forward tensor args
        self.fwd_cache = ()
        # Caching chunk outputs for final output merge or reduction
        self.output_chunk = ()

        # Clear grad of input buffers in between schedule steps. This is because
        # `torch.autograd.backward()` will accumulate gradients into leaf
        # tensors by default. For gradients to pass back to previous stages, we
        # don't want such accumulation.
        for a in self.args_recv_info:  # iterate over all input args
            if isinstance(a, _RecvInfo):
                # Set to None is the newer and recommended way to clear grads, compared to `zero_()`.
                # See https://github.com/pytorch/pytorch/pull/92731
                a.buffer.grad = None

    def _map_tensor_from_recv_info(
        self,
        recv_infos: tuple[InputInfo, ...],
    ):
        """
        Map tensors from recv infos to a list.
        """

        def get_recv_tensor(info):
            if isinstance(info, _RecvInfo):
                return info.buffer
            else:
                raise AssertionError(f"Expected _RecvInfo but got {type(info)}")

        return map_aggregate(cast(Argument, recv_infos), get_recv_tensor)

    def _retrieve_recv_activations(self):
        """
        Retrieve the activations received for the current stage during forward.
        """
        recv_infos = self.args_recv_info
        activations = self._map_tensor_from_recv_info(recv_infos)
        return activations

    def _retrieve_recv_grads(
        self,
    ):
        """
        Retrieve the gradients received for the current stage during backward.
        """
        recv_infos = self.grad_recv_info
        grads = self._map_tensor_from_recv_info(recv_infos)
        return grads

    def forward_maybe_with_nosync(self, *args, **kwargs):
        # If submod is wrapped with DDP, we use the `no_sync` context manager to
        # avoid gradient all-reduce per microbatch
        if isinstance(self.submod, DistributedDataParallel):
            with self.submod.no_sync():  # type: ignore[operator]
                out_val = self.submod(*args, **kwargs)
        else:
            out_val = self.submod(*args, **kwargs)
        return out_val

    def backward_maybe_with_nosync(
        self,
        backward_type,
        bwd_kwargs: dict,
    ) -> tuple[tuple[Optional[torch.Tensor], ...], Optional[list[dict[str, Any]]]]:
        """
        Whether using PP with FSDP or DDP, there are some runtime differences between the last backward step and the
        other steps.  Namely, we need to accumulate gradients on previous steps and reduce them on the last step, but
        there are additional state-variables and performance considerations depending on the data parallelism used.
        This helper should adapt any pipeline parallel schedule to work with common/supported data parallel libraries.
        """

        def perform_backward(
            backward_type,
        ) -> Callable[
            [],
            tuple[tuple[Optional[torch.Tensor], ...], Optional[list[dict[str, Any]]]],
        ]:
            if backward_type == "full":
                return lambda: (
                    stage_backward(
                        bwd_kwargs["stage_output"],
                        bwd_kwargs["output_grads"],
                        bwd_kwargs["input_values"],
                    ),
                    None,
                )
            elif backward_type == "input":
                return lambda: stage_backward_input(
                    bwd_kwargs["stage_output"],
                    bwd_kwargs["output_grads"],
                    bwd_kwargs["input_values"],
                    self.submod.parameters(),
                )
            elif backward_type == "weight":
                return lambda: (
                    stage_backward_weight(
                        self.submod.parameters(), bwd_kwargs["param_groups"]
                    ),
                    None,
                )
            else:
                raise RuntimeError(f"Unknown backward type: {backward_type}")

        # If submod is wrapped by DDP
        if isinstance(self.submod, DistributedDataParallel):
            raise NotImplementedError("DDP is not supported")
        # If submod is a FSDP module
        elif isinstance(self.submod, FSDPModule):
            self.submod.set_is_last_backward(False)
            self.submod.set_reshard_after_backward(False)
            self.submod.set_requires_gradient_sync(False)
            result = perform_backward(backward_type)()
        else:
            # Non-DP submodule, regular backward
            result = perform_backward(backward_type)()

        grads, param_groups = result
        return grads, param_groups
    
    def run_fsdp_post_backward(self) -> None:
        self.submod.set_is_last_backward(True)
        self.submod.set_reshard_after_backward(True)
        self.submod.set_requires_gradient_sync(True)
        fsdp_state = fully_shard.state(self.submod)  # type: ignore[attr-defined]
        for state in fsdp_state._state_ctx.all_states:
            if state._fsdp_param_group:
                state._fsdp_param_group.post_backward()

        # it would be much better if pipelining backward invoked .backward so autograd hooks
        # worked and modules like DDP/FSDP behaved as expected.  Working around this for the time being,
        # we need to call this too to ensure FSDP syncs its grad reduction ops back to the default stream.
        fsdp_state._root_post_backward_final_callback()

    def forward_one_chunk(
        self,
        args: tuple[Any, ...],
        kwargs: Optional[dict[str, Any]] = None,
    ):
        """
        Perform forward pass on the stage with one microbatch.
        `args` and `kwargs` are the inputs from *external* to this stage.
        As of Sept 2024:
        - `args` applies to the first stage only, other stages receives args
          through activation transmission.
        - `kwargs` can be passed to all stages via respective `step` calls.
        """

        if self.is_first:
            # First stage doesn't need to receive anything
            composite_args = args
        else:
            # Receive activations for this chunk
            # Activations only come in args form
            composite_args = self._retrieve_recv_activations()

        composite_kwargs = kwargs or {}

        self._validate_fwd_input(args, kwargs)

        # Compute forward
        try:
            output = self.forward_maybe_with_nosync(*composite_args, **composite_kwargs)

        except Exception as e:
            exc_msg = f"""
            {self.log_prefix} failed to run forward:
            args: {map_debug_info(composite_args)}
            kwargs: {map_debug_info(composite_kwargs)}
            """
            raise RuntimeError(exc_msg) from e

        # See [Note: pipeline model output type]
        output_tuple = _normalize_model_output_as_tuple(output)

        # Prepare for final output merge or reduction
        self.output_chunk = output

        # Save activations and inputs for backward
        flat_args = flatten_args(composite_args)
        flat_kwargs = flatten_args(composite_kwargs)
        flatten_input_tensors = flat_args + flat_kwargs
        self.fwd_cache = (
            output_tuple,  # stage_output
            flatten_input_tensors,  # input_values
        )

        logger.debug(
            "%s Forwarded chunk %s, outputs: %s",
            self.log_prefix,
            self.microbatch_idx,
            map_debug_info(output),
        )
        self._validate_fwd_outputs(output_tuple)

        # We return the original user-provied output, not normalized to tuple.
        # See [Note: pipeline model output type]
        return output

    def backward_one_chunk(
        self,
        loss=None,
        full_backward: bool = True,
    ):
        """
        Perform backward pass on the module.
        This should only be called once per microbatch.

        If full_backward is True (the default), the full backward pass including weight and input gradients will be run,
        and it is an error to call `backward_weight_one_chunk` for this bwd_chunk_id.

        If full_backward is False, it is optional that `dw_runner` was provided to the MbpStage at __init__ time,
        and a subsequent call to `backward_weight_one_chunk` is required to invoke dw_runner and complete the backward.
        """

        (stage_output, input_values) = self.fwd_cache

        # Compute backward
        if self.is_last:
            # Last stage computes gradients from loss and has no gradients from
            # next stage
            bwd_kwargs = {
                "stage_output": loss,
                "output_grads": None,
                "input_values": input_values,
            }
        else:
            # Otherwise, receive gradients from next stage
            grads_output = self._retrieve_recv_grads()
            # If an input to the pipeline requires gradient,
            # `torch.autograd.backward` will accumulate the gradient into the
            # `.grad` field of such input
            bwd_kwargs = {
                "stage_output": stage_output,
                "output_grads": grads_output,
                "input_values": input_values,
            }

        grads_input: tuple[Optional[torch.Tensor], ...] = ()

        # Custom backward function
        if self.dw_builder:
            # TODO: We may want to change our semantics so we are allowed to ignore
            # the 'dw_builder' and call full_backward directly when it is a full_backward op.
            grads_input, _ = self.backward_maybe_with_nosync(
                "full",
                bwd_kwargs,
            )
            if full_backward:
                self.dw_builder()()
            else:
                self.dw_runner = self.dw_builder()
        else:
            if full_backward:
                grads_input, _ = self.backward_maybe_with_nosync(
                    "full", bwd_kwargs
                )
            else:
                param_groups: list[dict[str, Any]] | None = None
                # Skip the backward for the first stage since we will perform the weight update with
                # autograd.backward in backward_weight_one_chunk
                if not self.is_first:
                    if isinstance(bwd_kwargs["stage_output"], torch.Tensor):
                        bwd_kwargs["stage_output"] = (bwd_kwargs["stage_output"],)

                    # perform the partial backwards for the inputs with a custom backward function
                    # when the "stage_ouput" is a loss, then it is a tensor, otherwise it is a tuple of tensors
                    grads_input, param_groups = self.backward_maybe_with_nosync(
                        "input", bwd_kwargs
                    )

                # TODO: we dont need to save this, add to dw_runner?
                self.backward_state = (
                    bwd_kwargs["input_values"],
                    param_groups,
                    bwd_kwargs["stage_output"],
                    bwd_kwargs["output_grads"],
                )
                # Save a placeholder for the dw_runner
                self.dw_runner = lambda: None

        self.bwd_cache = grads_input

        if self.is_last and not self.is_first:
            # Autograd dependencies:
            #    rest_of_autograd_graph -> stage_output -> loss
            # stage_output is no longer used in the last stage for backward and only needed
            # to return to the user in merge_output_chunk, therefore
            # this should be detached to release autograd graph context and free memory earlier
            for t in stage_output:
                if not t._is_view():  # views are not detachable in-place
                    t.detach_()

        logger.debug("%s Backwarded chunk %s", 
                     self.log_prefix, self.microbatch_idx)
    
    def backward_weight_one_chunk(self):
        assert self.dw_runner is not None, (
            f"{self.log_prefix} Attempted to run backward_weight_one_chunk for chunk {self._microbatch_id}"
            " without first calling `backward_one_chunk(full_backward=False)`"
        )

        if self.dw_builder is not None:
            self.dw_runner()
        else:
            (
                input_values,
                param_groups,
                stage_output,
                output_grads,
            ) = self.backward_state

            if self.stage_index != 0:
                bwd_kwargs = {
                    "stage_output": stage_output,
                    "param_groups": param_groups,
                }
                self.backward_maybe_with_nosync(
                    "weight", bwd_kwargs
                )
            else:
                # TODO: figure out a better way to do this:
                # if inputs does not require gradient,
                # then the parameter group will not be fully captured during stage_backward_input
                # in this case, we need call grad directly on the parameters
                # To solve: make input fn do the intersect compute and then finish it off during W
                bwd_kwargs = {
                    "stage_output": stage_output,
                    "output_grads": output_grads,
                    "input_values": input_values,
                }
                self.backward_maybe_with_nosync(
                    "full", bwd_kwargs
                )

    def _validate_fwd_input(self, args, kwargs):
        """Raises a RuntimeError if shapes of input args/kwargs do not match the shapes configured for this stage."""

        if self.is_first:
            # TODO why is there a separate recv_info for each pipeline chunk?
            # kwen2501: to avoid passing a `fwd_chunk_id` to this function, we
            # check all chunks against args_recv_info[0]
            expected_args = self.args_recv_info
        else:
            # We don't check inputs for non-0 stages assuming they don't accept
            # user inputs in canonical pipeline scenarios
            return

        if len(kwargs):
            # TODO- need a mapping of kwarg to position in self.args_recv_info
            # Without it, we are not 100% sure how to match the args and
            # expected_args.
            return

        # TODO- need a mapping of kwarg to position in self.args_recv_info
        # maybe it's impossible to tell whether the len mismatches because
        # (a) the user passed an extra arg or missed an arg
        # (b) the user did not pass a kwarg, which has a default value baked into expected_args
        expected_tensors_meta = [
            e.meta if isinstance(e, _RootArgPlaceholder) else e.buffer
            for e in expected_args
        ]
        validate_tensors_metadata(
            f"Stage {self.stage_index} forward inputs", expected_tensors_meta, args
        )

    def _validate_fwd_outputs(self, outputs: tuple[torch.Tensor, ...]):
        """Raises a RuntimeError if this stage produces an output of unexpected shape/dtype.
        Most likely, this could be cause either by incorrect user specification of output shapes, or becuase
        shape inference was done on the original model but then at runtime the model is wrapped with something like
        mixed precision which changes output dtype.
        """
        expected_tensors_meta = self.get_outputs_meta()
        validate_tensors_metadata(
            f"Stage {self.stage_index} forward outputs", expected_tensors_meta, outputs
        )

class MbpStage(_MbpStageBase):
    """
    A class representing a pipeline stage in a pipeline parallelism setup.

    MbpStage assumes sequential partitioning of the model, i.e. the model is split into chunks where outputs from
    one chunk feed into inputs of the next chunk, with no skip connections.

    MbpStage performs runtime shape/dtype inference automatically by propagating the outputs from stage0 to
    stage1 and so forth, in linear order.  To bypass shape inference, pass the `input_args` and `output_args` to each
    MbpStage instance.

    Args:
        submodule (nn.Module): The PyTorch module wrapped by this stage.
        stage_index (int): The ID of this stage.
        num_stages (int): The total number of stages.
        device (torch.device): The device where this stage is located.
        input_args (Union[torch.Tensor, Tuple[torch.tensor]], optional): The input arguments for the submodule.
        output_args (Union[torch.Tensor, Tuple[torch.tensor]], optional): The output arguments for the submodule.
        group (dist.ProcessGroup, optional): The process group for distributed training. If None, default group.
        dw_builder (Optional[Callable[[], Callable[..., None]]): If provided, dw_builder will build a new dw_runner function
            that will the W action (input weights) for F, I, W (Fwd, Input, Weight) zero bubble schedules.
    """

    def __init__(
        self,
        submodule: nn.Module,
        microbatch_idx: int,
        stage_index: int,
        num_stages: int,
        device: torch.device,
        input_args: Optional[Union[torch.Tensor, tuple[torch.Tensor, ...]]] = None,
        output_args: Optional[Union[torch.Tensor, tuple[torch.Tensor, ...]]] = None,
        group: Optional[dist.ProcessGroup] = None,
        dw_builder: Optional[Callable[[], Callable[..., None]]] = None,
    ):
        super().__init__(submodule, microbatch_idx, stage_index, num_stages, 
                         device, group, dw_builder)
        self.inputs: Optional[list[torch.Tensor]] = None
        self.inputs_meta: Optional[tuple[torch.Tensor, ...]] = None
        # Note: inputs and submod should ideally be on meta device. We decided not to assert this (yet) becuase it
        # might be breaking for existing users.
        if input_args is None:
            assert output_args is None, (
                "If specifying output_args, input_args must also be specified. "
                "Otherwise, shape inference will be performed at runtime"
            )
        else:
            self.inputs_meta = (
                (input_args,) if isinstance(input_args, torch.Tensor) else input_args
            )
            if output_args is None:
                logger.warning(
                    "Deprecation warning: passing input_args and performing init-time shape inference is deprecated. "
                    "MbpStage now supports runtime shape inference using the real inputs provided to schedule step(). "
                    "Either delete `input_args` arg to `MbpStage` to opt-into runtime shape inference, "
                    "or additionally pass `output_args` to `MbpStage` to fully override shape inference. "
                )
                try:
                    with torch.no_grad():
                        output_args = submodule(*self.inputs_meta)
                    output_args = tree_map_only(
                        torch.Tensor, lambda x: x.to("meta"), output_args
                    )
                except Exception as e:
                    raise RuntimeError(
                        "Failed to perform pipeline shape inference- are your inputs on the same device as your module?"
                    ) from e
            assert output_args is not None, (
                "If passing input_args, also pass output_args to override shape inference"
            )
            self._configure_outputs_meta(
                (output_args,) if isinstance(output_args, torch.Tensor) else output_args
            )

        # these are the buffers used in backwards send/recv, they are allocated later
        self.outputs_grad: list[torch.Tensor] = []

        dbg_str = (
            f"Finished pipeline stage init, {self.stage_index=}, {self.is_first=}, "  # noqa: G004
            f"{self.is_last=}, {self.num_stages=}, "
        )
        if self.inputs_meta is not None:
            dbg_str += (
                f"inputs: {[inp.shape for inp in self.inputs_meta]}, "
                f"output: {[output.shape for output in self.get_outputs_meta()]}"
            )
        else:
            dbg_str += " running shape-inference at runtime"

        print(dbg_str)

    def _shape_inference(
        self,
        args: tuple[Any, ...],
        kwargs: Optional[dict[str, Any]] = None,
    ):
        if kwargs is None:
            kwargs = {}
        assert args is not None, "Args may be an empty tuple but not None"

        # We skip recv communication if we're the first stage, but also if the previous stage is on the same rank
        # and can pass its output shapes in as args instead of using send/recv.
        if (
            self.is_first
            # if not first stage, then check if prev stage is on the same rank
            or self.stage_index_to_group_rank[self.stage_index - 1] == self.group_rank
        ):
            logger.debug(
                "Shape inference: stage %s skipping recv, because shape info passed in via `args`",
                self.stage_index,
            )
            args = tree_map_only(torch.Tensor, lambda x: x.to("meta"), args)
        else:
            assert len(args) == 0, (
                "Can't supply input args for shape inference on non-first stage"
            )
            objects = [None]
            logger.debug(
                "Shape inference: stage %s receiving from stage %s",
                self.stage_index,
                self.stage_index - 1,
            )
            dist.recv_object_list(
                objects,
                src=dist.get_global_rank(
                    self.group or dist.distributed_c10d._get_default_group(),
                    self.stage_index_to_group_rank[self.stage_index - 1],
                ),
                group=self.group,
                device=self.device,
            )
            recv_args = objects[0]
            assert isinstance(recv_args, tuple), type(recv_args)
            args = recv_args

        # cache input shapes for use during recv buffer allocation
        self.inputs_meta = args
        args = tree_map_only(
            torch.Tensor, lambda x: torch.zeros_like(x, device=self.device), args
        )

        # set attributes needed for forward
        with torch.no_grad():
            outputs = self.submod(*args, **kwargs)

        # if single tensor, convert so it is always a list
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]

        # communicate meta outputs not real outputs for two reasons
        # 1 - its faster (esp. since obj coll pickles tensor data!)
        # 2 - avoid activating a cuda context for the src rank when unpickling on the recv end!
        outputs_meta = tuple(
            tree_map_only(torch.Tensor, lambda x: x.to("meta"), outputs)
        )
        logger.debug(
            "Shape inference: stage %s inputs %s, outputs %s",
            self.stage_index,
            self.inputs_meta,
            outputs_meta,
        )
        self._configure_outputs_meta(outputs_meta)

        # Passing outputs to the next stage:
        # two cases-
        # 1. Usually: use send/recv communication to pass the output
        # 2. Special case: for V-schedules, 2 'adjacent' stages (e.g. stage 3, 4 in an 8-stage 4-rank V)
        #    pass their shape info via return value and function args rather than send/recv.
        if (
            self.is_last
            # if not last stage, then check if next stage is on the same rank
            or self.stage_index_to_group_rank[self.stage_index + 1] == self.group_rank
        ):
            # Case (2) above: pass shape info via return value and caller passes it as args to next stage's
            # _shape_inference call
            logger.debug(
                "Shape inference: stage %s skipping send to next stage",
                self.stage_index,
            )

        else:
            # Case (1): send shapes via send operation, and ensure not to return it to the caller
            logger.debug(
                "Shape inference: stage %s sending to stage %s",
                self.stage_index,
                self.stage_index + 1,
            )
            dist.send_object_list(
                [outputs_meta],
                dst=dist.get_global_rank(
                    self.group or dist.distributed_c10d._get_default_group(),
                    self.stage_index_to_group_rank[self.stage_index + 1],
                ),
                group=self.group,
                device=self.device,
            )
            outputs_meta = tuple()

        return outputs_meta

    def _prepare_forward_infra(
        self,
        args: tuple[Any, ...],
        kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, ...]:
        # TODO move self.device to an argument from step API (from its input tensors)?
        outputs: tuple[Any, ...] = tuple()
        if self.inputs_meta is None:
            outputs = self._shape_inference(args, kwargs)

        assert self.inputs_meta is not None
        # Receive info during forward
        # TODO: create args_recv_info lazily? (same needed for MbpStage)
        # For single microbatch, we only need one chunk
        if not self.is_first:
            # We assume that we always receive from stage - 1
            recv_infos = tuple(
                [
                    _RecvInfo(
                        f"recv_for_{self.stage_index}_from_{self.stage_index - 1}",
                        self.stage_index - 1,
                        _make_tensor_from_meta(inp, self.device),
                    )
                    for inp in self.inputs_meta
                ]
            )
            # In case there is backward pass, set requires_grad for receive buffers
            if self.has_backward:
                for r in recv_infos:
                    r.buffer.requires_grad_(True)

            self.args_recv_info = recv_infos
        else:
            self.args_recv_info = tuple(
                [_RootArgPlaceholder(i) for i in self.inputs_meta]
            )

        # Send info during forward for each activation
        # only need the rank that is being sent to
        self.act_send_info: dict[int, list] = {}

        for idx in range(len(self.get_outputs_meta())):
            # We assume we always send to stage + 1
            if not self.is_last:
                self.act_send_info[idx] = [self.stage_index + 1]
            else:
                self.act_send_info[idx] = []

        return outputs
    
    def _create_grad_recv_info(
        self,
        act_send_info: dict,
    ) -> tuple[_RecvInfo, ...]:
        grad_recv_info: tuple[_RecvInfo, ...] = ()
        if not self.is_last:
            # Receiving gradients from multiple sources is not supported
            # hence we only take the first destination
            grad_recv_info = tuple(
                [
                    _RecvInfo(
                        f"recv_grad_for_{self.stage_index}_from_{dst_list[0]}",
                        dst_list[0],
                        _make_tensor_from_meta(
                            self.get_outputs_meta()[idx], self.device
                        ),
                    )
                    for idx, dst_list in act_send_info.items()
                ]
            )
        return grad_recv_info
