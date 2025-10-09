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

from regex import R
import torch
import torch.distributed as dist
import sys
import zipfile

from torch.nn import init
from accelerate import init_empty_weights
# from checkpoint import load_weights_from_hf
from model import DeepseekForCausalLM
from model_config import deepseek_config_registry

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard

from datetime import datetime

from ipc_pipeline_src.tsched_schedule import ScheduleTsched
from ipc_pipeline_src.tsched_stage import TschedStage
from ipc_pipeline_src.sync_traces import merge_chrome_traces_with_barriers
from ipc_pipeline_src.thread_scheduler import ContextScheduler, ExecutionEngine, ModelProfiler, materialize_meta_model
from ipc_pipeline_src.tsched_device_mesh import init_independent_device_mesh, compare_device_mesh_structures

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

class TorchTitanExecutionEngine(ExecutionEngine):
    def __init__(self, model, x, label, loss_fn, microbatch_size, microbatch_index,
                 pp_rank, pp_size, device, pp_mesh, context_scheduler, 
                 profiler=None, is_dist=False, debug=False, main_thread=False):
        super().__init__(model, x, label, loss_fn, context_scheduler, 
                         profiler=profiler, start_exec=False, is_dist=is_dist, 
                         debug=debug, main_thread=main_thread)
        
        self.microbatch_size = microbatch_size
        self.microbatch_index = microbatch_index
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.device = device
        self.pp_mesh = pp_mesh
        
        self.stage = TschedStage(
            self.model,
            self.microbatch_size,
            self.microbatch_index,
            self.pp_rank,
            self.pp_size,
            self.device,
            group=self.pp_mesh.get_group(),
        )
        # Create pipeline schedule
        self.losses = []
        global_rank = dist.get_rank()
        self.pp_schedule = ScheduleTsched(self.stage, 
                                          self.microbatch_index,
                                          self.microbatch_size, 
                                          loss_fn=self.loss_fn,
                                          global_rank=global_rank)
        
        print(g_str(f"[T{self.tid}]") + " ExecutionEngine initialized, "
              f"exec id " + y_str(f"{self.exec_id}") + ", microbatch id " + 
              y_str(f"{self.microbatch_index}") + ", global rank " + 
              y_str(f"{global_rank}" + ", PP mesh " + y_str(f"{self.pp_mesh.get_group()}")))
        
        if self.pp_rank == 0:
            y = self.pp_schedule.initialize_stage(self.x, scheduler=self.scheduler, exec_id=self.exec_id)
        elif self.pp_rank == self.pp_size - 1:
            y = self.pp_schedule.initialize_stage(target=self.label, losses=self.losses, 
                                                  scheduler=self.scheduler, exec_id=self.exec_id)
        else:
            self.pp_schedule.initialize_stage(scheduler=self.scheduler, exec_id=self.exec_id)
        
        self.init_exec(model=self.stage.submod)
        if not self.main_thread:
            self.start()

    def step(self):
        if self.pp_rank == 0:
            y = self.pp_schedule.step(self.x, scheduler=self.scheduler, exec_id=self.exec_id)
        elif self.pp_rank == self.pp_size - 1:
            y = self.pp_schedule.step(target=self.label, losses=self.losses, 
                                    scheduler=self.scheduler, exec_id=self.exec_id)
            loss = torch.mean(torch.stack(self.losses))
        else:
            self.pp_schedule.step(scheduler=self.scheduler, exec_id=self.exec_id)

        if self.pp_rank == self.pp_size - 1:
            print(f"logits: {y.shape}")
            print(f"{loss=}")

        if self.pp_rank == 0:
            param = self.model.get_parameter("model.layers.0.self_attn.q_proj.weight")
            print(f"{torch.linalg.norm(param.grad)=}")

        print("Backward done")

# Run full model
def run_full_model(
    meshes: list[DeviceMesh],
    mbp_size: int,
    num_hidden_layers: int,
    num_steps: int,
):
    rank = dist.get_rank()
    device_count = torch.cuda.device_count()
    device = torch.device("cuda", rank % device_count)
    microbatches = mbp_size
    debug = False

    mesh = meshes[0]
    pp_mesh = mesh["pp"]
    ep_mesh = mesh["ep"]
    pp_rank = pp_mesh.get_local_rank()
    ep_rank = ep_mesh.get_local_rank()
    pp_size = pp_mesh.size()
    ep_size = ep_mesh.size()
    # warmup_nccl()

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
        base_model = DeepseekForCausalLM(model_args)
        print(y_str(f"[Rank {rank}]") + " Base model instantiated")

    # Load weights
    # load_weights_from_hf(model, model_id, device)
    base_model.train()

    # Example inputs
    torch.manual_seed(ep_rank)
    bs = 16
    seqlen = 128
    x = torch.randint(model_args.vocab_size, (microbatches, bs, seqlen), device=device)
    label = torch.rand(microbatches, bs, seqlen, model_args.vocab_size, device=device)

    # Create loss function
    loss_fn = torch.nn.functional.cross_entropy
    
    context_scheduler = ContextScheduler(microbatches, 
                                        debug=debug, is_dist=True)
    profiler = None
    # profiler = ModelProfiler(base_model, x[0], label[0], loss_fn)

    with device, mesh, init_empty_weights():
        model = DeepseekForCausalLM(model_args)
    model.train()
    materialize_meta_model(model, base_model)
    dist.barrier()
    main_engine = TorchTitanExecutionEngine(
                        model, x[0], label[0], loss_fn,
                        microbatches, 0, pp_rank, pp_size, device, pp_mesh, 
                        context_scheduler, profiler=profiler, is_dist=True, 
                        debug=debug, main_thread= True)


    for t in range(1, microbatches):
        mesh = meshes[t]
        pp_mesh = mesh["pp"]
        ep_mesh = mesh["ep"]
        pp_rank = pp_mesh.get_local_rank()
        ep_rank = ep_mesh.get_local_rank()

        with device, mesh, init_empty_weights():
            model = DeepseekForCausalLM(model_args)
        model.train()
        materialize_meta_model(model, base_model)
        dist.barrier()
        TorchTitanExecutionEngine(
                        model, x[t], label[t], loss_fn,
                        microbatches, t, pp_rank, pp_size, device, pp_mesh, 
                        context_scheduler, profiler=profiler, is_dist=True, 
                        debug=debug)
    

        # Apply data parallelism
        # fsdp_mesh = mesh["fsdp"]
        # hsdp_mesh = mesh["ep", "fsdp"]
        # print(y_str(f"[Rank {rank}]") + f" fsdp_mesh: {fsdp_mesh}")
        # print(y_str(f"[Rank {rank}]") + f" hsdp_mesh: {hsdp_mesh}")
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
        
        # Use Symmetric Memory for MoE token shuffle.
        # TODO: we are rewriting `moe_on_device` function. `setup_symm_mem` is
        # currently supported for forward only. See `generate.py`.
        # model.setup_symm_mem(torch.bfloat16, device)

    # dist.barrier()
    # context_scheduler.check_gloo_comms()
    # dist.barrier()
    # assert 0

    with torch.profiler.record_function("BARRIER:EXEC_START"):
        dist.barrier()

    print(y_str(f"[Rank {rank}]") + " Running " + f"{num_steps} thread-parallel steps, "
          f"{num_hidden_layers=}, {microbatches=}, {bs=}, {seqlen=}")
    # Run forward and backward
    for step_idx in range(num_steps):
        print(y_str(f"[Rank {rank}]") + " Starting step " + f"{step_idx}")
        context_scheduler.start()
        main_engine.step()
        main_engine.sync_step_completion()
        context_scheduler.wait_completion()
        print(y_str(f"[Rank {rank}]") + " Step " + f"{step_idx} completed.")
        dist.barrier()
        
    print(y_str(f"[Rank {rank}]") + " All steps completed.")
        
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
    
    original_mesh = dist.init_device_mesh("cuda", (pp_size, ep_size, fsdp_size),
                                          mesh_dim_names=("pp", "ep", "fsdp"))

    meshes = []
    for i in range(mbp_size):
        # mesh = init_independent_device_mesh("cuda", (pp_size, ep_size, fsdp_size),
        #                                     mesh_dim_names=("pp", "ep", "fsdp"),
        #                                     mesh_id=f"mb_{i}")
        # assert compare_device_mesh_structures(original_mesh, mesh, verbose=False)
        mesh = dist.init_device_mesh("cuda", (pp_size, ep_size, fsdp_size),
                                     mesh_dim_names=("pp", "ep", "fsdp"))
        # if i == 0:
        #     pp_group_orig = original_mesh.get_group("pp")
        #     pp_group = mesh.get_group("pp")
        #     print(f"Rank {dist.get_rank()} pp_group_orig: {pp_group_orig}")
        #     print(f"Rank {dist.get_rank()} pp_group: {pp_group}")
        #     assert pp_group_orig != pp_group
        print(f"Rank {dist.get_rank()} Mesh {i} created")
        meshes.append(mesh)

    # Setup profiler
    run_id = os.getenv("RUN_ID", "0")
    log_dir = f"./tensorboard_traces/run_tsched_{run_id}_mbp_{mbp_size}_pp_{pp_size}_ep_{ep_size}_fsdp_{fsdp_size}_layers_{num_hidden_layers}_steps_{num_steps}"
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
            run_full_model(meshes, mbp_size, num_hidden_layers, num_steps)
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
                zip_name = f"{log_dir}/{run_id}_tsched_mbp_{mbp_size}_pp_{pp_size}_ep_{ep_size}_fsdp_{fsdp_size}_layers_{num_hidden_layers}_steps_{num_steps}_merged_trace.zip" 
                with zipfile.ZipFile(zip_name, "w",
                                    compression=zipfile.ZIP_DEFLATED, 
                                    compresslevel=9) as zipf:
                    print(f"Rank {dist.get_rank()} Compressing trace to {zip_name}")
                    zipf.write(f"{log_dir}/merged_trace.json", f"{run_id}_merged_trace.json")
                print(f"Rank {dist.get_rank()} Compressed trace to {zip_name}")
    else:
        time_start = datetime.now()
        run_full_model(meshes, mbp_size, num_hidden_layers, num_steps)
        time_end = datetime.now()
        print(f"Rank {dist.get_rank()} Time elapsed: {time_end - time_start}\n", end="")
        

    dist.destroy_process_group()
