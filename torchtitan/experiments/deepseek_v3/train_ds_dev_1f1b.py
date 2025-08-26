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
import sys
import zipfile

# from checkpoint import load_weights_from_hf
from model import DeepseekForCausalLM
from model_config import deepseek_config_registry

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.pipelining import PipelineStage, Schedule1F1B

from datetime import datetime

from ipc_pipeline_src.sync_traces import merge_chrome_traces_with_barriers

# Use DeepSeek-V2-Lite as a proxy
model_id = "deepseek-ai/DeepSeek-V2-Lite"

# Run full model
def run_full_model(
    mesh: DeviceMesh,
    mbp_size: int,
    num_hidden_layers: int,
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
    model_args.num_hidden_layers = 8

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
    print(f"**** {rank=}, {ep_rank=}")
    torch.manual_seed(ep_rank)
    bs = 4
    seqlen = 128
    x = torch.randint(model_args.vocab_size, (microbatches * bs, seqlen), device=device)
    label = torch.rand(microbatches * bs, seqlen, model_args.vocab_size, device=device)

    # Create loss function
    loss_fn = torch.nn.functional.cross_entropy
    
    with torch.profiler.record_function("BARRIER:EXEC_START"):
        dist.barrier()

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
    num_hidden_layers = int(sys.argv[6])
    num_steps = int(sys.argv[7])
    run_profiler = sys.argv[8] == "True"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    mesh = dist.init_device_mesh("cuda", (pp_size, ep_size, fsdp_size),
                                 mesh_dim_names=("pp", "ep", "fsdp"))

    # Setup profiler
    run_id = os.getenv("RUN_ID", "0")
    log_dir = f"./tensorboard_traces/run_1f1b_{run_id}_mbp_{mbp_size}_pp_{pp_size}_ep_{ep_size}_fsdp_{fsdp_size}"
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
            run_full_model(mesh, mbp_size, num_hidden_layers, num_steps)
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
                zip_name = f"{log_dir}/{run_id}_mbp_{mbp_size}_pp_{pp_size}_ep_{ep_size}_fsdp_{fsdp_size}_merged_trace.zip" 
                with zipfile.ZipFile(zip_name, "w",
                                    compression=zipfile.ZIP_DEFLATED, 
                                    compresslevel=9) as zipf:
                    print(f"Rank {dist.get_rank()} Compressing trace to {zip_name}")
                    zipf.write(f"{log_dir}/merged_trace.json", f"{run_id}_merged_trace.json")
                print(f"Rank {dist.get_rank()} Compressed trace to {zip_name}")
    else:
        time_start = datetime.now()
        run_full_model(mesh, mbp_size, num_hidden_layers, num_steps)
        time_end = datetime.now()
        print(f"Rank {dist.get_rank()} Time elapsed: {time_end - time_start}\n", end="")
        

    dist.destroy_process_group()
