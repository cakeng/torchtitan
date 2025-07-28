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
from torch.distributed.pipelining import PipelineStage, Schedule1F1B, ScheduleBackwardOnly, ScheduleForwardOnly
from accelerate import init_empty_weights
from fb_pipeline_src.share_model_ipc import share_model_parameters_ipc, create_and_share_tensor_ipc


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
    fbbp_mesh = mesh["fbbp"]
    pp_rank = pp_mesh.get_local_rank()
    ep_rank = ep_mesh.get_local_rank()
    fbbp_rank = fbbp_mesh.get_local_rank()
    pp_size = pp_mesh.size()
    ep_size = ep_mesh.size()
    fbbp_size = fbbp_mesh.size()

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
        f"Global_{rank=}, {ep_size=}, {pp_size=}, {fbbp_size=}, "
        f"Ranks: {ep_rank=}, {pp_rank=}, {fbbp_rank=}, {device=}, "
        f"{model_args.num_stages=}, {model_args.stage_idx=}"
    )
    # print(model_args)
    
    fbbp_grp = None
    global_cpu_grp = dist.new_group(
        backend="gloo",
        group_desc=f"global_cpu_group"
    )
    if fbbp_size > 1:
        # Global synchronization to ensure all processes reach this point before proceeding
        # Get all unique FB groups first
        all_fbbp_groups = []
        num_fbbp_groups = dist.get_world_size() // fbbp_size
        for fb_grp_idx in range(num_fbbp_groups ):
            # Calculate ranks for this FB group
            fbbp_group_ranks = [(fb_grp_idx + i * num_fbbp_groups) for i in range(fbbp_size)]
            all_fbbp_groups.append(fbbp_group_ranks)

        created_fbbp_groups = []
        for group_idx, fbbp_ranks in enumerate(all_fbbp_groups):
            group = dist.new_group(
                ranks=fbbp_ranks,
                backend="gloo",
                group_desc=f"fbbp_group_{group_idx}_{fbbp_ranks}"
            )
            created_fbbp_groups.append((group, fbbp_ranks))

        dist.barrier(group=global_cpu_grp)
        # Find the group this rank belongs to
        current_rank = dist.get_rank()
        for group, fb_ranks in created_fbbp_groups:
            if current_rank in fb_ranks:
                fbbp_grp = group
                break
        dist.barrier(group=global_cpu_grp)


    
    fsdp_mesh = mesh["fsdp"]
    hsdp_mesh = mesh["ep", "fsdp"]
    print(g_str(f"{rank=} ") + f"fbbp_mesh: {fbbp_mesh}, pp_mesh: {pp_mesh}, ep_mesh: {ep_mesh}, "
          f"fsdp_mesh: {fsdp_mesh}, hsdp_mesh: {hsdp_mesh}")
    dist.barrier(group=global_cpu_grp)

    # Instantiate model
    with device, mesh:
        if fbbp_rank == 0:    
            model = DeepseekForCausalLM(model_args)
        else:
            with init_empty_weights():
                model = DeepseekForCausalLM(model_args)

    if fbbp_size > 1:
        share_model_parameters_ipc(model, fbbp_grp)

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
    microbatches = 8

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
                do_fbbp=(fbbp_size > 1),
                fbbp_group=fbbp_grp,
                global_rank=rank,
            )

            # Create pipeline schedule
            losses = []
            if fbbp_size <= 1:
                pp_schedule = Schedule1F1B(stage, microbatches, loss_fn=loss_fn)
                if pp_rank == 0:
                    y = pp_schedule.step(x)
                elif pp_rank == pp_size - 1:
                    y = pp_schedule.step(target=label, losses=losses)
                    loss = torch.mean(torch.stack(losses))
                else:
                    pp_schedule.step()
            else:
                if fbbp_rank == 0:
                    pp_schedule = ScheduleForwardOnly(stage, microbatches, loss_fn=loss_fn)
                else:
                    pp_schedule = ScheduleBackwardOnly(stage, microbatches, loss_fn=loss_fn)
                
                if pp_rank == 0:
                    y = pp_schedule.step(x)
                elif pp_rank == pp_size - 1 and fbbp_rank == 0:
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

        if pp_rank == 0 and ((fbbp_size > 1 and fbbp_rank > 1) or (fbbp_size <= 1 and fbbp_rank == 0)):
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
                                 mesh_dim_names=("fbbp", "pp", "ep", "fsdp"))
    
    assert num_gpus >= mesh.size() // mesh["fbbp"].size()

    run_full_model(mesh)

    dist.destroy_process_group()
