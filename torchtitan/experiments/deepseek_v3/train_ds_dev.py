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

import torch
import torch.distributed as dist

# from checkpoint import load_weights_from_hf
from model import DeepseekForCausalLM
from model_config import deepseek_config_registry

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.pipelining import PipelineStage, Schedule1F1B
from accelerate import init_empty_weights
from ipc_pipeline_src.ipc_share import (share_model_parameters_ipc, 
                                        create_and_share_tensor_ipc,
                                        copy_and_share_tensor_ipc,
                                        SharedGradientCache)
from ipc_pipeline_src.mbp_schedule import ScheduleMbp
from ipc_pipeline_src.mbp_stage import MbpStage

import torch_ipc_extension # Your compiled extension



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

# Run full model
def run_full_model(
    mesh: DeviceMesh,
):
    pp_mesh = mesh["pp"]
    ep_mesh = mesh["ep"]
    mbp_mesh = mesh["mbp"]
    pp_rank = pp_mesh.get_local_rank()
    ep_rank = ep_mesh.get_local_rank()
    mbp_rank = mbp_mesh.get_local_rank()
    pp_size = pp_mesh.size()
    ep_size = ep_mesh.size()
    mbp_size = mbp_mesh.size()

    rank = dist.get_rank()
    device_count = torch.cuda.device_count()
    device = torch.device("cuda", rank % device_count)

    # Get model configs
    model_args = deepseek_config_registry[model_id]
    # [Note]: I am making the model smaller for testing / avoiding OOM. If you
    # have sufficient GPUs for model parallelism, you can remove this line.
    model_args.num_hidden_layers = 16

    # Apply model parallelism
    model_args.ep_size = ep_size
    model_args.num_stages = pp_size
    model_args.stage_idx = pp_rank
    print(
        b_str(f"Parallelism Setting: ") + 
        f"Global_{rank=}, {ep_size=}, {pp_size=}, {mbp_size=}, "
        f"Ranks: {ep_rank=}, {pp_rank=}, {mbp_rank=}, {device=}, "
        f"{model_args.num_stages=}, {model_args.stage_idx=}"
    )
    # print(model_args)
    
    # Setup MBP groups
    mbp_grp = None
    mbp_group_idx = None
    global_cpu_grp = dist.new_group(
        backend="gloo",
        group_desc=f"global_cpu_group"
    )
    if mbp_size > 1:
        # Get all unique FB groups first
        all_mbp_groups = []
        num_mbp_groups = dist.get_world_size() // mbp_size
        for fb_grp_idx in range(num_mbp_groups ):
            # Calculate ranks for this FB group
            mbp_group_ranks = [(fb_grp_idx + i * num_mbp_groups) for i in range(mbp_size)]
            all_mbp_groups.append(mbp_group_ranks)

        created_mbp_groups = []
        for group_idx, mbp_ranks in enumerate(all_mbp_groups):
            group = dist.new_group(
                ranks=mbp_ranks,
                backend="gloo",
                group_desc=f"mbp_group_{group_idx}_{mbp_ranks}"
            )
            created_mbp_groups.append((group_idx, group, mbp_ranks))

        dist.barrier(group=global_cpu_grp)
        # Find the group this rank belongs to
        current_rank = dist.get_rank()
        for group_idx, group, fb_ranks in created_mbp_groups:

            if current_rank in fb_ranks:
                mbp_group_idx = group_idx
                mbp_grp = group
                break
        dist.barrier(group=global_cpu_grp)

    fsdp_mesh = mesh["fsdp"]
    fsdp_rank = fsdp_mesh.get_local_rank()
    fsdp_size = fsdp_mesh.size()
    hsdp_mesh = mesh["ep", "fsdp"]
    mbp_orthogonal_mesh = mesh["pp", "ep", "fsdp"]
    print(g_str(f"Rank {rank} Mesh: ") + y_str("mbp_mesh ") + f"{mbp_group_idx}: {mbp_mesh}, " +
                y_str("pp_mesh: ") + f"{pp_mesh}, " + y_str("ep_mesh: ") + f"{ep_mesh}, " +
                y_str("sdp_mesh: ") + f"{fsdp_mesh}, " + y_str("hsdp_mesh: ") + f"{hsdp_mesh}, " +
                y_str("mbp_orthogonal_mesh: ") + f"{mbp_orthogonal_mesh}\n", end="")
    
    dist.barrier(group=global_cpu_grp)

    # Instantiate model
    with device, mesh:
        if mbp_rank == 0:    
            # Only MBP rank 0 creates the model with weights
            model = DeepseekForCausalLM(model_args)
        else:
            # Other ranks create the model without weights
            with init_empty_weights():
                model = DeepseekForCausalLM(model_args)

    if mbp_size > 1:
        # Share model parameters across MBP group
        share_model_parameters_ipc(model, mbp_grp)

    # Load weights
    # load_weights_from_hf(model, model_id, device)
    model.train()

    # Apply data parallelism
    # Using `reshard_after_forward=False` to implement Zero-2, i.e. sharding the
    # optimizer (Zero-1) and gradients (Zero-2), but not the model weights.
    # Reason: the MoE is "sparsely activated" compared to the dense model, thus
    # it will be ineconomical re-gather the weights.
    for layer in model.model.layers.values():
        # Apply FSDP to experts
        if hasattr(layer.mlp, "experts"):
            for expert in layer.mlp.experts.values():
                fully_shard(expert, mesh=fsdp_mesh, reshard_after_forward=False)
        # Apply HSDP to other parts such as attention, layernorm, because they
        # are doing DDP on EP dimension
        fully_shard(layer, mesh=hsdp_mesh, reshard_after_forward=False)

    # Apply HSDP on root model (lm_head, embeddings, etc)
    fully_shard(model, mesh=hsdp_mesh, reshard_after_forward=False)

    # Synthetic setting
    microbatches = mbp_size
    max_concurrent_mb = 2

    # Use Symmetric Memory for MoE token shuffle.
    # TODO: we are rewriting `moe_on_device` function. `setup_symm_mem` is
    # currently supported for forward only. See `generate.py`.
    # model.setup_symm_mem(torch.bfloat16, device)

    # Example inputs
    torch.manual_seed(ep_rank)
    bs = 4
    seqlen = 128
    x = torch.randint(model_args.vocab_size, (microbatches * bs, seqlen), device=device)
    label = torch.rand(microbatches * bs, seqlen, model_args.vocab_size, device=device)

    # Create loss function
    loss_fn = torch.nn.functional.cross_entropy 

    # Setup MBP control variables
    # MBP control variables are shared across the first process in each MBP rank.
    mbp_ctrl = None
    if mbp_size > 1 and mbp_group_idx == 0:
        print(b_str(f"Rank {rank} ") + f"Creating MBP control variable mbp_ctrl\n", end="")
        mbp_ctrl_name = f"mbp_ctrl"
        if rank == 0:
            mbp_ctrl = torch_ipc_extension.SharedData(
                name=mbp_ctrl_name, is_creator=True, initial_value=0,
                semaphore_count = max_concurrent_mb
            )
        else:
            mbp_ctrl = torch_ipc_extension.SharedData(
                name=mbp_ctrl_name, is_creator=False
            )
    dist.barrier(group=global_cpu_grp)

    # Setup MBP shared gradient cache
    shared_grad_cache = None
    if mbp_size > 1:
        # Semaphore to limit memory usage
        if mbp_rank == 0:
            mbp_ctrl.sem_wait()
            print(b_str(f"Rank {rank} (MBP rank 0)") + f"Capturing gradient cache, microbatch {mbp_rank}\n", end="")
        else:
            print(g_str(f"Rank {rank} ") + f"Capturing gradient cache, microbatch {mbp_rank}\n", end="")
        # Capture forward cache
        if pp_size > 1:
            # Create pipeline stage
            stage = MbpStage(
                model,
                pp_rank,
                pp_size,
                device,
                group=pp_mesh.get_group(),
            )

            # Wait for semaphore'd first MBP ranks to arrive
            dist.barrier(group=mbp_orthogonal_mesh.get_group())

            pp_schedule = ScheduleMbp(stage, mbp_rank, microbatches, loss_fn=loss_fn)

            if pp_rank == 0:
                y = pp_schedule.step(x)
            else:
                pp_schedule.step()

            dist.barrier(group=global_cpu_grp)

            print(b_str(f"Rank {rank} ") + f"Finished capturing forward cache.")  
        if mbp_rank == 0:  
            mbp_ctrl.sem_post()

    return 0
            
    print(b_str(f"Rank {rank} ") + f"Starting training loop with {microbatches=}, {bs=}, {seqlen=}")
    dist.barrier(group=global_cpu_grp)

    # Run forward and backward
    steps = 2
    for _ in range(steps):
        if pp_size > 1:
            # Create pipeline stage
            stage = PipelineStage(
                model,
                pp_rank,
                pp_size,
                device,
                group=pp_mesh.get_group(),
                do_mbp=(mbp_size > 1),
                mbp_group=mbp_grp,
                global_rank=rank,
                shared_fwd_cache=shared_forward_cache,
            )

            # Create pipeline schedule
            losses = []
            if mbp_size <= 1:
                pp_schedule = Schedule1F1B(stage, microbatches, loss_fn=loss_fn)
                if pp_rank == 0:
                    y = pp_schedule.step(x)
                elif pp_rank == pp_size - 1:
                    y = pp_schedule.step(target=label, losses=losses)
                    loss = torch.mean(torch.stack(losses))
                else:
                    pp_schedule.step()
            else:
                if mbp_rank == 0:
                    pp_schedule = ScheduleForwardOnly(stage, microbatches, loss_fn=loss_fn)
                else:
                    pp_schedule = ScheduleBackwardOnly(stage, microbatches, loss_fn=loss_fn)
                
                if pp_rank == 0:
                    y = pp_schedule.step(x)
                elif pp_rank == pp_size - 1 and mbp_rank == 0:
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

        if pp_rank == 0 and ((mbp_size > 1 and mbp_rank > 1) or (mbp_size <= 1 and mbp_rank == 0)):
            param = model.get_parameter("model.layers.0.self_attn.q_proj.weight")
            print(f"{torch.linalg.norm(param.grad)=}")

        model.zero_grad()

    print(g_str(f"Rank {rank} ") + f"Finished training loop")
    dist.barrier(group=global_cpu_grp)
    print(b_str(f"Rank {rank} ") + f"All processes finished training loop")


if __name__ == "__main__":
    # set device before init_device mesh, otherwise ep will have duplicate device mapping
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]) % num_gpus)

    mesh = dist.init_device_mesh("cuda", (3, 2, 2, 1), 
                                 mesh_dim_names=("mbp", "pp", "ep", "fsdp"))
    
    assert num_gpus >= mesh.size() // mesh["mbp"].size()

    run_full_model(mesh)

    dist.destroy_process_group()
