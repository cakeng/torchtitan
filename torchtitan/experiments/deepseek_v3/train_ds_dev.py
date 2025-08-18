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
import sys
from re import T

from accelerate.utils.megatron_lm import F
import torch
import torch.distributed as dist

# from checkpoint import load_weights_from_hf
from model import DeepseekForCausalLM
from model_config import deepseek_config_registry

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed._tensor import DTensor
from torch.distributed.pipelining import PipelineStage, Schedule1F1B
from accelerate import init_empty_weights

from ipc_pipeline_src.ipc_share import (get_shared_data, destroy_shared_data,
                                        share_model_parameters_ipc,
                                        share_model_gradients_ipc,
                                        create_and_share_tensor_ipc,
                                        copy_and_share_tensor_ipc)
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

def get_memory_usage():
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

def print_memory_usage(rank, mbp_ctrl, str):
    mbp_ctrl.barrier()
    if rank == 0:
        print(g_str(f"Rank {rank} ") + str + ": " + get_memory_usage() + "\n", end="")
    mbp_ctrl.barrier()

# Run full model
def run_full_model(
    mesh: DeviceMesh,
    mbp_rank: int = 0,
    mbp_size: int = 1,
):
    pp_mesh = mesh["pp"]
    ep_mesh = mesh["ep"]
    fsdp_mesh = mesh["fsdp"]
    pp_rank = pp_mesh.get_local_rank()
    ep_rank = ep_mesh.get_local_rank()
    fsdp_rank = fsdp_mesh.get_local_rank()
    pp_size = pp_mesh.size()
    ep_size = ep_mesh.size()
    fsdp_size = fsdp_mesh.size()
    
    rank = dist.get_rank() # PP x EP x FSDP Rank, exists per GPU
    device_count = torch.cuda.device_count()
    device = torch.device("cuda", rank % device_count)

    # Get model configs
    model_args = deepseek_config_registry[model_id]
    # [Note]: I am making the model smaller for testing / avoiding OOM. If you
    # have sufficient GPUs for model parallelism, you can remove this line.
    model_args.num_hidden_layers = 6

    # Apply model parallelism
    model_args.ep_size = ep_size
    model_args.num_stages = pp_size
    model_args.stage_idx = pp_rank
    if rank == 0 and mbp_rank == 0:
        print(
            b_str(f"Parallelism Setting: ") + 
            f"Parallelism: {mbp_size=}, {ep_size=}, {pp_size=}, {fsdp_size=}, "
            f"Ranks: {mbp_rank=}, {rank=}, {ep_rank=}, {pp_rank=}, {fsdp_rank=}, "
            f"{model_args.num_stages=}, {model_args.stage_idx=}\n"
        )
    
    # Setup MBP groups
    max_concurrent_process_groups = 6
    mbp_ctrl_name = f"mbp_ctrl_gpu_{rank}"
    if mbp_rank == 0:
        mbp_ctrl = get_shared_data(mbp_ctrl_name, is_creator=True, 
                                   group_size=mbp_size, initial_value=0, 
                                   semaphore_count=max_concurrent_process_groups)
    else:
        mbp_ctrl = get_shared_data(mbp_ctrl_name, is_creator=False, 
                                   group_size=mbp_size, initial_value=0, 
                                   semaphore_count=max_concurrent_process_groups)

    fsdp_mesh = mesh["fsdp"]
    fsdp_rank = fsdp_mesh.get_local_rank()
    hsdp_mesh = mesh["ep", "fsdp"]
    print(g_str(f"Rank {rank} Mesh: ") + 
          y_str("\n\tmbp rank: ") + f"{mbp_rank}" +
          b_str("\n\tpp_mesh: ") + f"{pp_mesh} - " + y_str("pp rank: ") + f"{pp_rank}" +
          b_str("\n\tep_mesh: ") + f"{ep_mesh} - " + y_str("ep rank: ") + f"{ep_rank}" +
          b_str("\n\tfsdp_mesh: ") + f"{fsdp_mesh} - " + y_str("fsdp rank: ") + f"{fsdp_rank}" + 
          b_str("\n\thsdp_mesh: ") + f"{hsdp_mesh} \n", end="")
    
    mbp_ctrl.barrier()
    print(g_str(f"Rank {rank} ") + "Synced with all ranks in the MBP group\n", end="")
    
      
    # Instantiate model
    print_memory_usage(rank, mbp_ctrl, "Before model instantiation")
    with device, mesh:
        # model = DeepseekForCausalLM(model_args)
        if mbp_rank == 0:
            model = DeepseekForCausalLM(model_args)
        else:
            with init_empty_weights():
                model = DeepseekForCausalLM(model_args)
    print_memory_usage(rank, mbp_ctrl, "After model instantiation")
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
    print_memory_usage(rank, mbp_ctrl, "After applying FSDP")
    
    # Use Symmetric Memory for MoE token shuffle.
    # TODO: we are rewriting `moe_on_device` function. `setup_symm_mem` is
    # currently supported for forward only. See `generate.py`.
    # model.setup_symm_mem(torch.bfloat16, device)

    # Example inputs
    # Synthetic setting
    microbatches = mbp_size
    torch.manual_seed(ep_rank)
    bs = 4
    seqlen = 128
    if mbp_rank == 0:
        x = torch.randint(model_args.vocab_size, (microbatches * bs, seqlen), device=device)
        label = torch.rand(microbatches * bs, seqlen, model_args.vocab_size, device=device)
        x = copy_and_share_tensor_ipc(x, is_creator=True, group_size=mbp_size, 
                                      shm_name=f"mbp_share_input_{rank}")
        label = copy_and_share_tensor_ipc(label, is_creator=True, group_size=mbp_size, 
                                          shm_name=f"mbp_share_label_{rank}")
    else:
        x = copy_and_share_tensor_ipc(None, is_creator=False, group_size=mbp_size, 
                                      shm_name=f"mbp_share_input_{rank}")
        label = copy_and_share_tensor_ipc(None, is_creator=False, group_size=mbp_size, 
                                          shm_name=f"mbp_share_label_{rank}")
    print_memory_usage(rank, mbp_ctrl, "After sharing inputs")
    
    if mbp_rank == 0:
        share_model_parameters_ipc(model, is_creator=True, group_size=mbp_size, 
                                   shm_name=f"mbp_share_model_params_{rank}")
    else:
        share_model_parameters_ipc(model, is_creator=False, group_size=mbp_size, 
                                   shm_name=f"mbp_share_model_params_{rank}")
    print_memory_usage(rank, mbp_ctrl, "After sharing model parameters")

    if rank == 0:
        print(g_str(f"Rank {rank} ") + f"Runtime settings: {microbatches=}, {max_concurrent_process_groups=}, {bs=}, {seqlen=}\n", end="")
    
    # Create loss function
    loss_fn = torch.nn.functional.cross_entropy 
    
    stage = MbpStage(
                model,
                mbp_rank,
                pp_rank,
                pp_size,
                device,
                group=pp_mesh.get_group(),
            )
    pp_schedule = ScheduleMbp(stage, mbp_rank, microbatches, 
                              loss_fn=loss_fn, global_rank=rank)

    print(g_str(f"Rank {rank}: ") + "Starting initialization run...\n", end="")

    if pp_rank == 0:
        pp_schedule.step(x, init_stage_only=(mbp_rank != 0))
    elif pp_rank == pp_size - 1:
        y_dict = pp_schedule.step(target=label, init_stage_only=(mbp_rank != 0))
    else:
        pp_schedule.step(init_stage_only=(mbp_rank != 0))
        
    print_memory_usage(rank, mbp_ctrl, "After initialization run")
    mbp_ctrl.barrier()
    
    # Now, all ranks in the MBP group participate sharing model parameters and gradients.
    # We share model params again as FSDP may change the model parameters.
    if mbp_rank == 0:
        share_model_parameters_ipc(model, is_creator=True, group_size=mbp_size, 
                                   shm_name=f"mbp_share_model_params_{rank}")
        share_model_gradients_ipc(model, is_creator=True, group_size=mbp_size, 
                                  shm_name=f"mbp_share_model_grads_{rank}")
    else:
        share_model_parameters_ipc(model, is_creator=False, group_size=mbp_size, 
                                   shm_name=f"mbp_share_model_params_{rank}")
        share_model_gradients_ipc(model, is_creator=False, group_size=mbp_size, 
                                  shm_name=f"mbp_share_model_grads_{rank}")

    torch.cuda.empty_cache()
    print_memory_usage(rank, mbp_ctrl, "After sharing model and gradients")

    print(b_str(f"Rank {rank} ") + f"Starting training loop with {microbatches=}, {bs=}, {seqlen=}\n", end="")
    
    mbp_ctrl.barrier()

    # Run forward and backward
    steps = 2
    for _ in range(steps):
        # Only the first process in each SMB group captures the weight gradients
        if pp_size > 1:
            # Create pipeline stage
            loss = None
            y = None
            losses = []
            if pp_rank == 0:
                pp_schedule.step(x, mbp_ctrl=mbp_ctrl)
            elif pp_rank == pp_size - 1:
                y = pp_schedule.step(target=label, losses=losses,
                                          mbp_ctrl=mbp_ctrl)
                loss = torch.mean(torch.stack(losses))
            else:
                pp_schedule.step(mbp_ctrl=mbp_ctrl)

            print(g_str(f"Rank {rank} ") + r_str(f"Finished ") + 
                  f"F/B pass on microbatch {mbp_rank}\n", end="")  
            print_memory_usage(rank, mbp_ctrl, "After F/B pass")

            if pp_rank == pp_size - 1:
                print(y_str(f"Rank {rank} ") + f"{loss=}, " +
                      f"logits: {y.shape}, " +
                      f"label: {label.shape}\n", end="")

        # Wait for all MBP members to finish gradient accumulation before running optimizer
        mbp_ctrl.barrier()

        if mbp_rank == 0:
            # Reset microbatch counter
            mbp_ctrl.set(0)
            if rank == 0:
                print(g_str(f"Rank {rank} ") + 
                      f"/////// Finished iteration {_} ///////\n", end="")

    print(b_str(f"Rank {rank} ") + f"All processes finished training loop\n", end="")
    if mbp_rank == 0:
        destroy_shared_data(mbp_ctrl)


if __name__ == "__main__":
    pp_size = int(sys.argv[1])
    ep_size = int(sys.argv[2])
    fsdp_size = int(sys.argv[3])
    mbp_size = int(sys.argv[4])
    mbp_rank = int(sys.argv[5])
    
    # set device before init_device mesh, otherwise ep will have duplicate device mapping
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    mesh = dist.init_device_mesh("cuda", (pp_size, ep_size, fsdp_size), 
                                 mesh_dim_names=("pp", "ep", "fsdp"))
    
    time_start = datetime.now()
    run_full_model(mesh, mbp_rank, mbp_size)
    time_end = datetime.now()
    print(f"Rank {dist.get_rank()} Time elapsed: {time_end - time_start}\n", end="")

    dist.destroy_process_group()
