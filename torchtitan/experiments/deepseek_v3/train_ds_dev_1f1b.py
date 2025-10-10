# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# torchrun --standalone --nproc-per-node 8 train.py
# bash run_training.sh

# this file runs a simple training loop with synthetic data
# and is intended to be used for debugging and development

import os
from random import Random
from typing import Optional

import torch
import torch.distributed as dist
import sys
import zipfile

# from checkpoint import load_weights_from_hf
from model import DeepseekForCausalLM
from model_config import deepseek_config_registry

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (PipelineScheduleSingle, 
                                                    _Action, _ComputationType,
                                                    _wait_batch_p2p, _batch_p2p)

from datetime import datetime

from ipc_pipeline_src.sync_traces import merge_chrome_traces_with_barriers

# Use DeepSeek-V2-Lite as a proxy
model_id = "deepseek-ai/DeepSeek-V2-Lite"

def g_str(s):
    return "\033[32m" + s + "\033[0m"
def r_str(s):
    return "\033[31m" + s + "\033[0m"
def b_str(s):
    return "\033[34m" + s + "\033[0m"
def y_str(s):
    return "\033[33m" + s + "\033[0m"

class Schedule1F1B(PipelineScheduleSingle):
    """
    The 1F1B schedule.
    Will perform one forward and one backward on the microbatches in steady state.
    """

    def _step_microbatches(
        self,
        arg_mbs: Optional[list] = None,
        kwarg_mbs: Optional[list] = None,
        target_mbs: Optional[list] = None,
        losses: Optional[list] = None,
    ):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the 1F1B schedule.

        Args:
            microbatches: list of microbatch args.
        """
        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)

        if not self._stage_initialized:
            self._initialize_stage(arg_mbs[0], kwarg_mbs[0])

        # Last stage has 1 warmup, second-to-last 2 warmups, ...
        # first stage `num_stages` warmups
        warmup_chunks = min(
            self._n_microbatches,
            self._num_stages - self._stage.stage_index,
        )

        # Chunk counters
        fwd_mb_index = 0
        bwd_mb_index = 0

        # Warmup phase
        send_work: list[dist.Work] = []
        fwd_sends = []
        global_rank = dist.get_rank()
        print(g_str(f"Rank {global_rank}: ") + f"Running {self._n_microbatches} microbatches")
        for _ in range(warmup_chunks):
            # Receive activations
            fwd_recvs = self._stage.get_fwd_recv_ops(fwd_mb_index)
            print(g_str(f"Rank {global_rank}: ") + 
                  b_str(f"Warmup Forward {fwd_mb_index}") + f", receiving {fwd_recvs}")
            _wait_batch_p2p(_batch_p2p(fwd_recvs, desc="fwd_recv"))
            

            # Compute
            with torch.profiler.record_function(f"[Rank {fwd_mb_index}-{global_rank}] Warmup Forward {fwd_mb_index}"):
                output = self._stage.forward_one_chunk(
                    fwd_mb_index, arg_mbs[fwd_mb_index], kwarg_mbs[fwd_mb_index]
                )  # type: ignore[index]

            # Clear previous chunk's forward sends (hopefully they have well
            # finished, otherwise, we are heavily communication bound, in which
            # case it doesn't create a lot of benefit to compute next chunk
            # eagerly either)
            _wait_batch_p2p(send_work)

            # Send activations
            fwd_sends = self._stage.get_fwd_send_ops(fwd_mb_index)
            print(g_str(f"Rank {global_rank}: ") + 
                  b_str(f"Warmup Forwarded {fwd_mb_index}") + f", sending {fwd_sends}")
            if fwd_mb_index != warmup_chunks - 1:
                # Safe to fire
                send_work = _batch_p2p(fwd_sends, desc="fwd_send")
            # otherwise:
            #   The last forward send is left for fuse with first 1B in 1B1F below

            # Compute loss
            self._maybe_compute_loss(self._stage, output, target_mbs, fwd_mb_index)
            fwd_mb_index += 1

        # Now we should have send ops left over, to be fused with first 1B of 1B1F phase below.

        # 1B1F phase
        while True:  # Don't worry, we have a break inside
            # We actually do 1B first as the `1B1F` name indicates, so prepare its recv ops
            bwd_recvs = self._stage.get_bwd_recv_ops(bwd_mb_index)
            print(g_str(f"Rank {global_rank}: ") + 
                  r_str(f"1B1F Backward {bwd_mb_index}") + f", receiving {bwd_recvs}")
            # Now, we need to fire the fwd_sends and bwd_recvs together
            _wait_batch_p2p(_batch_p2p(fwd_sends + bwd_recvs, desc="fwd_send_bwd_recv"))

            # Backward one chunk
            loss = self._maybe_get_loss(self._stage, bwd_mb_index)
            self._stage.backward_one_chunk(
                bwd_mb_index,
                loss=loss,
                last_backward=bwd_mb_index == self._n_microbatches - 1,
            )

            # Get the bwd send ops, but don't fire, to be fused with the 1F below
            with torch.profiler.record_function(f"[Rank {bwd_mb_index}-{global_rank}] 1B1F Backward {bwd_mb_index}"):
                bwd_sends = self._stage.get_bwd_send_ops(bwd_mb_index)
            print(g_str(f"Rank {global_rank}: ") + 
                  r_str(f"1B1F Backwarded {bwd_mb_index}") + f", sending {bwd_sends}")
            bwd_mb_index += 1

            if fwd_mb_index == self._n_microbatches:
                # We are done with 1B1F, so break with some left-over bwd_sends
                break

            # We prepare 1F of the `1B1F`
            fwd_recvs = self._stage.get_fwd_recv_ops(fwd_mb_index)
            print(g_str(f"Rank {global_rank}: ") + 
                  b_str(f"1B1F Forward {fwd_mb_index}") + f", receiving {fwd_recvs}")
            # Fuse it with bwd_sends above
            _wait_batch_p2p(_batch_p2p(bwd_sends + fwd_recvs, desc="bwd_send_fwd_recv"))

            # Now do the fwd
            with torch.profiler.record_function(f"[Rank {fwd_mb_index}-{global_rank}] 1B1F Forward {fwd_mb_index}"):
                output = self._stage.forward_one_chunk(
                    fwd_mb_index, arg_mbs[fwd_mb_index], kwarg_mbs[fwd_mb_index]
                )  # type: ignore[index]

            # Compute loss
            self._maybe_compute_loss(self._stage, output, target_mbs, fwd_mb_index)

            # Get the fwd send ops, but don't fire, leave it for the next iter (wrap-around)
            fwd_sends = self._stage.get_fwd_send_ops(fwd_mb_index)
            print(g_str(f"Rank {global_rank}: ") + 
                  b_str(f"1B1F Forwarded {fwd_mb_index}") + f", sending {fwd_sends}")
            fwd_mb_index += 1

        # Remember we still have some bwd_sends left over after the break? Now it is time to fire it
        send_work = _batch_p2p(bwd_sends, desc="bwd_send")
        print(g_str(f"Rank {global_rank}: ") + 
              r_str(f"Cooldown Backward {bwd_mb_index}") + f", sending {bwd_sends}")
        # Cooldown
        while bwd_mb_index < self._n_microbatches:
            # prepare bwd recv ops
            bwd_recvs = self._stage.get_bwd_recv_ops(bwd_mb_index)
            _wait_batch_p2p(_batch_p2p(bwd_recvs, desc="bwd_recv"))
            print(g_str(f"Rank {global_rank}: ") + 
                  r_str(f"Cooldown Backward {bwd_mb_index}") + f", receiving {bwd_recvs}")
            # Backward one chunk
            loss = self._maybe_get_loss(self._stage, bwd_mb_index)
            with torch.profiler.record_function(f"[Rank {bwd_mb_index}-{global_rank}] Cooldown Backward {bwd_mb_index}"):
                self._stage.backward_one_chunk(
                    bwd_mb_index,
                    loss=loss,
                    last_backward=bwd_mb_index == self._n_microbatches - 1,
                )
            # Clear previous chunk's backward sends (hopefully they have well finished)
            _wait_batch_p2p(send_work)

            # Get the bwd send ops, fire it
            bwd_sends = self._stage.get_bwd_send_ops(bwd_mb_index)
            send_work = _batch_p2p(bwd_sends, desc="bwd_send")
            print(g_str(f"Rank {global_rank}: ") + 
                  r_str(f"Cooldown Backwarded {bwd_mb_index}") + f", sending {bwd_sends}")
            bwd_mb_index += 1

        self._stage.scale_grads(
            grad_scale_factor=self._n_microbatches if self.scale_grads else 1
        )

        # Wait for the last backward send to finish
        _wait_batch_p2p(send_work)

        # Return losses if there is a container passed in
        self._update_losses(self._stage, losses)

    def _get_pipeline_order(self) -> Optional[dict[int, list[Optional[_Action]]]]:
        """
        Returns the pipeline order for 1F1B schedule.

        See base method in PipelineScheduleSingle for details on the schedule IR format.
        """
        pipeline_order = {}
        pp_group_size = self._num_stages

        for rank in range(pp_group_size):
            actions: list[Optional[_Action]] = []

            # 1. Warmup phase: initial delay based on rank
            actions.extend([None] * rank)

            # 2. Initial forward passes before 1F1B phase
            num_forward = (pp_group_size - 1) - rank
            forward_mb = 0
            for i in range(num_forward):
                actions.append(_Action(rank, _ComputationType.FORWARD, i))
                forward_mb = i

            # 3. Wait for backward to be ready
            wait_for_1f1b = max(0, 2 * (pp_group_size - 1 - rank))
            actions.extend([None] * wait_for_1f1b)

            # 4. 1F1B steady state phase
            backward_mb = 0
            remaining_forward = self._n_microbatches - num_forward

            while remaining_forward > 0:
                # One forward
                forward_mb += 1
                actions.append(_Action(rank, _ComputationType.FORWARD, forward_mb))
                remaining_forward -= 1

                # One backward
                actions.append(
                    _Action(rank, _ComputationType.FULL_BACKWARD, backward_mb)
                )
                backward_mb += 1

            # 5. Cooldown phase: remaining backward passes
            remaining_backward = self._n_microbatches - backward_mb

            while remaining_backward > 0:
                # Add None and backward actions in alternating pattern
                # based on distance from the last stage
                if (pp_group_size - rank) > 0:
                    actions.append(None)
                    # Decrement the wait counter only if we still have backward passes to do
                    if remaining_backward > 0:
                        actions.append(
                            _Action(rank, _ComputationType.FULL_BACKWARD, backward_mb)
                        )
                        backward_mb += 1
                        remaining_backward -= 1
                else:
                    # If we're at the last stage, just add backward actions without None
                    actions.append(
                        _Action(rank, _ComputationType.FULL_BACKWARD, backward_mb)
                    )
                    backward_mb += 1
                    remaining_backward -= 1

            pipeline_order[rank] = actions
        return pipeline_order


# Run full model
def run_full_model(
    mesh: DeviceMesh,
    mbp_size: int,
    num_hidden_layers: int,
    batch_size: int,
    seq_len: int,
    num_steps: int,
):
    rank = dist.get_rank()
    device_count = torch.cuda.device_count()
    device = torch.device("cuda", rank % device_count)
    pp_mesh = mesh["pp"]
    ep_mesh = mesh["ep"]
    pp_rank = pp_mesh.get_local_rank()
    ep_rank = ep_mesh.get_local_rank()
    pp_size = pp_mesh.size()
    ep_size = ep_mesh.size()

    # Get model configs
    model_args = deepseek_config_registry[model_id]
    # [Note]: I am making the model smaller for testing / avoiding OOM. If you
    # have sufficient GPUs for model parallelism, you can remove this line.
    model_args.num_hidden_layers = num_hidden_layers

    # Apply model parallelism
    model_args.ep_size = ep_size
    model_args.num_stages = pp_size
    model_args.stage_idx = pp_rank
    print(
        f"Parallelism: {rank=}, {ep_size=}, {pp_size=}, {model_args.ep_size=}, {model_args.num_stages=}, {model_args.stage_idx=}"
    )
    # print(model_args)

    # Instantiate model
    with device, mesh:
        model = DeepseekForCausalLM(model_args)

    # Load weights
    # load_weights_from_hf(model, model_id, device)
    model.train()

    # Apply data parallelism
    fsdp_mesh = mesh["fsdp"]
    hsdp_mesh = mesh["ep", "fsdp"]
    print(f"{rank=}, fsdp_mesh: {fsdp_mesh}")
    print(f"{rank=}, hsdp_mesh: {hsdp_mesh}")
    # Using `reshard_after_forward=False` to implement Zero-2, i.e. sharding the
    # optimizer (Zero-1) and gradients (Zero-2), but not the model weights.
    # Reason: the MoE is "sparsely activated" compared to the dense model, thus
    # it will be ineconomical re-gather the weights.
    # for layer in model.model.layers.values():
    #     # Apply FSDP to experts
    #     if hasattr(layer.mlp, "experts"):
    #         for expert in layer.mlp.experts.values():
    #             fully_shard(expert, mesh=fsdp_mesh, reshard_after_forward=False)
    #     # Apply HSDP to other parts such as attention, layernorm, because they
    #     # are doing DDP on EP dimension
    #     fully_shard(layer, mesh=hsdp_mesh, reshard_after_forward=False)

    # # Apply HSDP on root model (lm_head, embeddings, etc)
    # fully_shard(model, mesh=hsdp_mesh, reshard_after_forward=False)

    # Synthetic setting
    microbatches = mbp_size

    # Use Symmetric Memory for MoE token shuffle.
    # TODO: we are rewriting `moe_on_device` function. `setup_symm_mem` is
    # currently supported for forward only. See `generate.py`.
    # model.setup_symm_mem(torch.bfloat16, device)

    # Example inputs
    torch.manual_seed(ep_rank)
    bs = batch_size
    seqlen = seq_len
    x = torch.randint(model_args.vocab_size, (microbatches * bs, seqlen), device=device)
    label = torch.rand(microbatches * bs, seqlen, model_args.vocab_size, device=device)

    # Create loss function
    loss_fn = torch.nn.functional.cross_entropy
    
    with torch.profiler.record_function("BARRIER:EXEC_START"):
        dist.barrier()

    print(f"[Rank {rank}] Running {num_steps} 1f1b steps, {num_hidden_layers=}, {microbatches=}, {bs=}, {seqlen=}")
    # Run forward and backward
    for _ in range(num_steps):
        if pp_size > 1:
            # Create pipeline stage
            stage = PipelineStage(
                model,
                pp_rank,
                pp_size,
                device,
                group=pp_mesh.get_group(),
            )

            # Create pipeline schedule
            losses = []
            pp_schedule = Schedule1F1B(stage, microbatches, loss_fn=loss_fn)

            if pp_rank == 0:
                y = pp_schedule.step(x)
            elif pp_rank == pp_size - 1:
                y = pp_schedule.step(target=label, losses=losses)
                loss = torch.mean(torch.stack(losses))
            else:
                pp_schedule.step()
        else:
            y = model(x)
            loss = loss_fn(y, label)
            loss.backward()

        if pp_rank == pp_size - 1:
            print(f"logits: {y.shape}")
            print(f"{loss=}")

        if pp_rank == 0:
            param = model.get_parameter("model.layers.0.self_attn.q_proj.weight")
            print(f"{torch.linalg.norm(param.grad)=}")

        model.zero_grad()

    print("Backward done")
    with torch.profiler.record_function("BARRIER:EXEC_END"):
        dist.barrier()


if __name__ == "__main__":
    # set device before init_device mesh, otherwise ep will have duplicate device mapping
    pp_size = int(sys.argv[1])
    ep_size = int(sys.argv[2])
    fsdp_size = int(sys.argv[3])
    mbp_size = int(sys.argv[4])
    mbp_rank = int(sys.argv[5])
    batch_size = int(sys.argv[6])
    seq_len = int(sys.argv[7])
    num_hidden_layers = int(sys.argv[8])
    num_steps = int(sys.argv[9])
    run_profiler = sys.argv[10] == "True"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    mesh = dist.init_device_mesh("cuda", (pp_size, ep_size, fsdp_size),
                                 mesh_dim_names=("pp", "ep", "fsdp"))

    # Setup profiler
    run_id = os.getenv("RUN_ID", "0")
    log_dir = f"./tensorboard_traces/run_1f1b_{run_id}_mbp_{mbp_size}_pp_{pp_size}_ep_{ep_size}_fsdp_{fsdp_size}_layers_{num_hidden_layers}_bs_{batch_size}_seqlen_{seq_len}_steps_{num_steps}"
    os.makedirs(log_dir, exist_ok=True)

    # Profile the execution
    if run_profiler:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=0,
                warmup=0,
                active=1,
                repeat=1
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            time_start = datetime.now()
            run_full_model(mesh, mbp_size, num_hidden_layers, batch_size, seq_len, num_steps)
            prof.step()
            time_end = datetime.now()
            print(f"Rank {dist.get_rank()} Time elapsed: {time_end - time_start}\n", end="")
            
            # Export the trace
            trace_path = f"{log_dir}/trace_{mbp_rank}_{dist.get_rank()}.json"
            print(f"Rank {dist.get_rank()} Exporting trace to {trace_path}")
            prof.export_chrome_trace(trace_path)

            torch.cuda.empty_cache()
            if dist.get_rank() == 0:
                merge_chrome_traces_with_barriers(
                    trace_dir=log_dir,
                    output_file=f"{log_dir}/merged_trace.json",
                    barrier_events=["BARRIER:EXEC_START", "BARRIER:EXEC_END"],
                    trace_names=[f"trace_0_0.json", f"trace_0_{ep_size*fsdp_size}.json"],
                    whole_trace=False,
                )
                # compress the merged trace
                zip_name = f"{log_dir}/{run_id}_1f1b_mbp_{mbp_size}_pp_{pp_size}_ep_{ep_size}_fsdp_{fsdp_size}_layers_{num_hidden_layers}_bs_{batch_size}_seqlen_{seq_len}_steps_{num_steps}_merged_trace.zip" 
                with zipfile.ZipFile(zip_name, "w",
                                    compression=zipfile.ZIP_DEFLATED, 
                                    compresslevel=9) as zipf:
                    print(f"Rank {dist.get_rank()} Compressing trace to {zip_name}")
                    zipf.write(f"{log_dir}/merged_trace.json", f"{run_id}_merged_trace.json")
                print(f"Rank {dist.get_rank()} Compressed trace to {zip_name}")
    else:
        time_start = datetime.now()
        run_full_model(mesh, mbp_size, num_hidden_layers, batch_size, seq_len, num_steps)
        time_end = datetime.now()
        print(f"Rank {dist.get_rank()} Time elapsed: {time_end - time_start}\n", end="")
        

    dist.destroy_process_group()
