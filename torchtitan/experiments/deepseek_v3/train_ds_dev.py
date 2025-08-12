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
from re import T

from accelerate.utils.megatron_lm import F
import torch
import torch.distributed as dist

# from checkpoint import load_weights_from_hf
from model import DeepseekForCausalLM
from model_config import deepseek_config_registry

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.pipelining import PipelineStage, Schedule1F1B
from accelerate import init_empty_weights

from ipc_pipeline_src.ipc_share import share_model_parameters_ipc
from ipc_pipeline_src.mbp_schedule import ScheduleMbp
from ipc_pipeline_src.mbp_stage import MbpStage
from ipc_pipeline_src.ipc_gradient import SharedGradientManager
import torch_ipc_extension 

from datetime import datetime

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

def print_memory_usage():
    allocated_mem = torch.cuda.memory_allocated()
    cached_mem = torch.cuda.memory_reserved()
    free_mem, total_mem = torch.cuda.mem_get_info()
    total_used_mem = total_mem - free_mem
    other_process_mem = total_used_mem - cached_mem
    str = (f"Total memory: {total_mem/1024**2:.2f} MB, "
          f"Free memory: {free_mem/1024**2:.2f} MB, "
          f"Total used memory: {total_used_mem/1024**2:.2f} MB, "
          f"Allocated memory: {allocated_mem/1024**2:.2f} MB, "
          f"Cached memory: {cached_mem/1024**2:.2f} MB, "
          f"Other process memory: {other_process_mem/1024**2:.2f} MB")
    return str

# Run full model
def run_full_model(
    mesh: DeviceMesh,
):
    pp_mesh = mesh["pp"]
    ep_mesh = mesh["ep"]
    mbp_mesh = mesh["mbp"]
    fsdp_mesh = mesh["fsdp"]
    pp_rank = pp_mesh.get_local_rank()
    ep_rank = ep_mesh.get_local_rank()
    mbp_rank = mbp_mesh.get_local_rank()
    fsdp_rank = fsdp_mesh.get_local_rank()
    pp_size = pp_mesh.size()
    ep_size = ep_mesh.size()
    mbp_size = mbp_mesh.size()
    fsdp_size = fsdp_mesh.size()

    rank = dist.get_rank()
    device_count = min(torch.cuda.device_count(), pp_size * ep_size * fsdp_size)
    device = torch.device("cuda", rank % device_count)

    # Get model configs
    model_args = deepseek_config_registry[model_id]
    # [Note]: I am making the model smaller for testing / avoiding OOM. If you
    # have sufficient GPUs for model parallelism, you can remove this line.
    model_args.num_hidden_layers = 4

    # Apply model parallelism
    model_args.ep_size = ep_size
    model_args.num_stages = pp_size
    model_args.stage_idx = pp_rank
    if rank == 0:
        print(
            b_str(f"Parallelism Setting: ") + 
            f"Global_{rank=}, {ep_size=}, {pp_size=}, {mbp_size=}, "
            f"Ranks: {mbp_rank=}, {ep_rank=}, {pp_rank=}, {fsdp_rank=}, {device=}, "
            f"{model_args.num_stages=}, {model_args.stage_idx=}"
        )
    
    # Setup MBP groups
    global_cpu_grp = dist.new_group(
        backend="gloo",
        group_desc=f"global_cpu_group"
    )
    # Microbatch parallelism group (MBP) = Process group on the SAME GPU
    # Get all unique MBP groups first
    all_mbp_groups = []
    num_mbp_groups = dist.get_world_size() // mbp_size
    for mbp_grp_idx in range(num_mbp_groups):
        # Calculate ranks for this MBP group
        mbp_group_ranks = [(mbp_grp_idx + i * num_mbp_groups) for i in range(mbp_size)]
        all_mbp_groups.append(mbp_group_ranks)

    mbp_groups = []
    for group_idx, mbp_ranks in enumerate(all_mbp_groups):
        group = dist.new_group(
            ranks=mbp_ranks,
            backend="gloo",
            group_desc=f"mbp_group_{group_idx}_{mbp_ranks}"
        )
        mbp_groups.append((group_idx, group, mbp_ranks))

    dist.barrier(group=global_cpu_grp)
    # Find the group this rank belongs to
    current_rank = dist.get_rank()
    for group_idx, group, mbp_ranks in mbp_groups:
        if current_rank in mbp_ranks:
            mbp_group_idx = group_idx
            mbp_grp = group
            print(b_str(f"Rank {rank} ") + f"Found my MBP group: " +
                    f"{mbp_group_idx} with ranks: {mbp_ranks} on device {device}\n", end="")
            break
    dist.barrier(group=global_cpu_grp)

    fsdp_mesh = mesh["fsdp"]
    fsdp_rank = fsdp_mesh.get_local_rank()
    hsdp_mesh = mesh["ep", "fsdp"]
    print(g_str(f"Rank {rank} Mesh: ") + 
          b_str("mbp_mesh ") + f"{mbp_group_idx}: {mbp_mesh} - " + y_str("mbp rank: ") + f"{mbp_rank}, " +
          b_str("pp_mesh: ") + f"{pp_mesh} - " + y_str("pp rank: ") + f"{pp_rank}, " +
          b_str("ep_mesh: ") + f"{ep_mesh} - " + y_str("ep rank: ") + f"{ep_rank}, " +
          b_str("fsdp_mesh: ") + f"{fsdp_mesh} - " + y_str("fsdp rank: ") + f"{fsdp_rank}, " + 
          b_str("hsdp_mesh: ") + f"{hsdp_mesh} \n", end="")
    
    dist.barrier(group=global_cpu_grp)

    # Instantiate model
    print(g_str(f"Rank {rank} ") + "Before model instantiation " + print_memory_usage() + "\n", end="")
    dist.barrier(group=global_cpu_grp)
    with device, mesh:
        if mbp_rank == 0:    
            # Only MBP rank 0 creates the model with weights
            model = DeepseekForCausalLM(model_args)
        else:
            # Other ranks create the model without weights
            with init_empty_weights():
                model = DeepseekForCausalLM(model_args)
    dist.barrier(group=global_cpu_grp)
    print(g_str(f"Rank {rank} ") + "After model instantiation " + print_memory_usage() + "\n", end="")
    dist.barrier(group=global_cpu_grp)
    if mbp_size > 1:
        # Share model parameters across MBP group
        share_model_parameters_ipc(model, mbp_grp)
        
    print(g_str(f"Rank {rank} ") + "After sharing model parameters " + print_memory_usage() + "\n", end="")
    dist.barrier(group=global_cpu_grp)
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
                fully_shard(expert, 
                    mesh=fsdp_mesh, 
                    reshard_after_forward=False)
        # Apply HSDP to other parts such as attention, layernorm, because they
        # are doing DDP on EP dimension
        fully_shard(layer, 
                    mesh=hsdp_mesh, 
                    reshard_after_forward=False)

    # Apply HSDP on root model (lm_head, embeddings, etc)
    fully_shard(model, 
                mesh=hsdp_mesh, 
                reshard_after_forward=False)
    # dist.barrier(group=global_cpu_grp)
    # print(g_str(f"Rank {rank} ") + f"Model after FSDP {model} \n", end="")
    dist.barrier(group=global_cpu_grp)
    print(g_str(f"Rank {rank} ") + "After applying FSDP " + print_memory_usage() + "\n", end="")
    dist.barrier(group=global_cpu_grp)
    # Synthetic setting
    microbatches = mbp_size 
    max_concurrent_process_groups = 1

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

    if rank == 0:
        print(g_str(f"Rank {rank} ") + f"Runtime settings: {microbatches=}, {max_concurrent_process_groups=}, {bs=}, {seqlen=}\n", end="")
    dist.barrier(group=global_cpu_grp)
    print(g_str(f"Rank {rank} ") + "After synthetic data generation " + print_memory_usage() + "\n", end="")
    dist.barrier(group=global_cpu_grp)
    # Create loss function
    loss_fn = torch.nn.functional.cross_entropy 

    # Setup MBP control synchronization primitives for each MBP group
    mbp_ctrl_name = f"mbp_ctrl_{mbp_group_idx}"
    if mbp_rank == 0:
        mbp_ctrl = torch_ipc_extension.SharedData(
            name=mbp_ctrl_name, is_creator=True, initial_value=0,
            semaphore_count = max_concurrent_process_groups
        )
    else:
        mbp_ctrl = torch_ipc_extension.SharedData(
            name=mbp_ctrl_name, is_creator=False
        )
    dist.barrier(group=global_cpu_grp)
    print(g_str(f"Rank {rank} ") + "After MBP control setup " + print_memory_usage() + "\n", end="")
    dist.barrier(group=global_cpu_grp)
    # Setup MBP shared gradient manage
    # Only the first rank in each MBP group captures the weight gradients
    mbp_grad_manager = SharedGradientManager(model, mesh=fsdp_mesh, group=mbp_grp)
    pp_schedule = None
    stage = None
    if mbp_rank == 0:
        print(g_str(f"Rank {rank}: ") + "Starting gradient discovery run...\n", end="")
        # Create pipeline stage
        stage = MbpStage(
            model,
            pp_rank,
            pp_size,
            device,
            group=pp_mesh.get_group(),
        )

        pp_schedule = ScheduleMbp(stage, mbp_rank, microbatches, 
                                  loss_fn=loss_fn, global_rank=rank)

        if pp_rank == 0:
            pp_schedule.step(x)
        elif pp_rank == pp_size - 1:
            y_dict = pp_schedule.step(target=label)
        else:
            pp_schedule.step()
    dist.barrier(group=global_cpu_grp)
    print(g_str(f"Rank {rank} ") + "After gradient discovery run " + print_memory_usage() + "\n", end="")
    dist.barrier(group=global_cpu_grp)
    # Now, all ranks in the MBP group participate in creating the shared cache.
    # Rank 0 will broadcast the metadata it discovered.
    mbp_grad_manager.setup_shared_cache()
    if mbp_rank == 0:
        del pp_schedule, stage
    torch.cuda.empty_cache()
    print(b_str(f"Rank {rank} ") + "Shared gradient cache setup complete.\n", end="")
    dist.barrier(group=global_cpu_grp)
    print(g_str(f"Rank {rank} ") + "After shared gradient cache setup " + print_memory_usage() + "\n", end="")
    dist.barrier(group=global_cpu_grp)
    print(b_str(f"Rank {rank} ") + f"Starting training loop with {microbatches=}, {bs=}, {seqlen=}\n", end="")
    dist.barrier(group=global_cpu_grp)

    # Run forward and backward
    steps = 2
    for _ in range(steps):
        # Only the first process in each SMB group captures the weight gradients
        if pp_size > 1:
            # Create pipeline stage
            stage = MbpStage(
                model,
                pp_rank,
                pp_size,
                device,
                group=pp_mesh.get_group(),
            )
            pp_schedule = ScheduleMbp(stage, mbp_rank, microbatches, 
                                      loss_fn=loss_fn, global_rank=rank)

            mbp_ctrl.wait_for_value(mbp_rank)
            mbp_ctrl.sem_wait()
            print(g_str(f"Rank {rank} ") + b_str(f"Executing ") + f"F/B pass on microbatch {mbp_rank}\n", end="")

            loss = None
            y_dict = None
            losses = []
            if pp_rank == 0:
                pp_schedule.step(x,
                                 mbp_ctrl=mbp_ctrl,
                                 mbp_group_idx=mbp_group_idx,
                                 mbp_grp=mbp_grp)
            elif pp_rank == pp_size - 1:
                y_dict = pp_schedule.step(target=label, losses=losses,
                                          mbp_ctrl=mbp_ctrl,
                                          mbp_group_idx=mbp_group_idx,
                                          mbp_grp=mbp_grp)
                loss = torch.mean(torch.stack(losses))
            else:
                pp_schedule.step(mbp_ctrl=mbp_ctrl,
                                 mbp_group_idx=mbp_group_idx,
                                 mbp_grp=mbp_grp)
            
            print(g_str(f"Rank {rank} ") + "After F/B pass" + print_memory_usage() + "\n", end="")

            # Release the semaphore
            print(g_str(f"Rank {rank} ") + r_str(f"Finished ") + f"F/B pass on microbatch {mbp_rank}\n", end="")  

            if pp_rank == pp_size - 1:
                first_key = next(iter(y_dict))
                print(y_str(f"Rank {rank} ") + f"{loss=}, logits: {y_dict[first_key].shape}")

            mbp_grad_manager.accumulate_grad()
            model.zero_grad()
            del pp_schedule, stage
            del losses
            if loss is not None:
                del loss
            if y_dict is not None:
                del y_dict
            torch.cuda.empty_cache()
            print(g_str(f"Rank {rank} ") + "After gradient accumulation" + print_memory_usage() + "\n", end="")

            mbp_ctrl.sem_post()

        # Wait for all MBP members to finish gradient accumulation before running optimizer
        dist.barrier(group=mbp_grp)

        if mbp_rank == 0:
            # Run optimizer here
            if rank == 0:
                grad = mbp_grad_manager.get_grad("model.layers.0.self_attn.q_proj.weight")
                print(y_str(f"Rank {rank} ") + f"{torch.linalg.norm(grad)=}")

        dist.barrier(group=mbp_grp)
        mbp_grad_manager.zero_grad()

        # Wait for all processes to finish before the next iteration
        dist.barrier(group=global_cpu_grp)
        if mbp_rank == 0:
            # Reset microbatch counter
            mbp_ctrl.set(0)
            if rank == 0:
                print(g_str(f"Rank {rank} ") + f"/////// Finished iteration {_} ///////\n", end="")

    print(b_str(f"Rank {rank} ") + f"All processes finished training loop\n", end="")
    mbp_grad_manager.destroy()
    if mbp_rank == 0:
        torch_ipc_extension.destroy_shared_data(f"mbp_ctrl")


if __name__ == "__main__":
    mbp_size = 2
    pp_size = 2
    ep_size = 2
    fsdp_size = 2
    
    assert torch.cuda.device_count() >= pp_size * ep_size * fsdp_size
    num_gpus = min(torch.cuda.device_count(), pp_size * ep_size * fsdp_size)
    # set device before init_device mesh, otherwise ep will have duplicate device mapping
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]) % num_gpus)
    mesh = dist.init_device_mesh("cuda", (mbp_size, pp_size, ep_size, fsdp_size), 
                                 mesh_dim_names=("mbp", "pp", "ep", "fsdp"))
    
    time_start = datetime.now()
    run_full_model(mesh)
    time_end = datetime.now()
    print(f"Time elapsed: {time_end - time_start}\n", end="")

    dist.destroy_process_group()
