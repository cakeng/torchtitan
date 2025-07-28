# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# torchrun --standalone --nproc-per-node 8 train.py
# bash run_training.sh

# this file runs a real training loop with real data, optimizer, metrics, etc.

import os
import time

from typing import Iterable

import torch
import torch.profiler
import torch.distributed as dist

import torchtitan.components.ft as ft

from torchtitan.components.lr_scheduler import build_lr_schedulers

from torchtitan.components.metrics import build_metrics_processor
from torchtitan.components.optimizer import build_optimizers

from torchtitan.config_manager import ConfigManager, JobConfig

from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.distributed import ParallelDims

from torchtitan.experiments.deepseek_v3.infra.parallelize_deepseek import (
    parallelize_deepseek,
)

# from checkpoint import load_weights_from_hf

from torchtitan.experiments.deepseek_v3.model_config import deepseek_config_registry

from torchtitan.experiments.deepseek_v3.tokenizers.hf_tokenizer import (
    get_hf_tokenizer,
    remove_notset_root_handlers,
)

# from torchtitan.experiments.deepseek_v3.train_configs.custom_args import JobConfig
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.utils import get_device_info

from torch.distributed.pipelining import (PipelineStage, 
                                          Schedule1F1B,
                                          ScheduleGPipe,
                                          ScheduleForwardOnly,
                                          ScheduleBackwardOnly,)

import torch.multiprocessing as mp
from multiprocessing import Manager
from multiprocessing.managers import BaseManager

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


def cross_entropy_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Common cross-entropy loss function for Transformer models training."""
    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(), labels.flatten(0, 1)
    )


# temp global
_device_type, _device_info = get_device_info()


class Trainer:
    job_config: JobConfig
    device: torch.device

    # states
    step: int


def next_batch(
    data_iterator: Iterable, metrics_processor
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:

    data_load_start = time.perf_counter()
    batch = next(data_iterator)
    input_dict, labels = batch
    metrics_processor.ntokens_since_last_log += labels.numel()
    metrics_processor.data_loading_times.append(time.perf_counter() - data_load_start)

    for k, _ in input_dict.items():
        input_dict[k] = input_dict[k].to(_device_type)
    labels = labels.to(_device_type)
    return input_dict, labels


# Run full model
def run_full_model(
    config: JobConfig,
):
    # setup mesh
    pp_dim = config.parallelism.pipeline_parallel_degree
    ep_dim = config.parallelism.expert_parallel_degree
    fb_dim = config.parallelism.forward_backward_parallel_degree
    assert fb_dim <= 3, "Forward-backward parallelism cannot have a degree more than 2."
    if not ep_dim:
        ep_dim = 3
        logger.info("Using default EP degree 3")

    fsdp_dim = config.parallelism.data_parallel_shard_degree

    # set device before init_device mesh, otherwise ep will have duplicate device mapping
    num_device = torch.cuda.device_count()
    logger.info(g_str("Num GPUs: ") + f"{num_device}")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]) % num_device)

    world_mesh = dist.init_device_mesh(
        "cuda", (fb_dim, pp_dim, ep_dim, fsdp_dim), mesh_dim_names=("fb", "pp", "ep", "fsdp")
    )
    logger.info(g_str("Mesh: ") + f"{world_mesh}")

    rank = dist.get_rank()
    os_rank = os.environ["LOCAL_RANK"]
    logger.info(r_str(f"OS Rank ") + f"{os_rank}, " + r_str(f"Dist Rank ") + f"{rank}")

    cpu_gloo_grp = dist.new_group(
        backend="gloo",
        group_desc="cpu_gloo_group"
    )
    # Global synchronization to ensure all processes reach this point before proceeding
    dist.barrier(group=cpu_gloo_grp)
    logger.info(g_str(f"Rank {rank}: All processes synchronized."))
    
    fb_grp = None
    if fb_dim > 1:
        # Get all unique FB groups first
        all_fb_groups = []
        for fb_grp_idx in range(dist.get_world_size() // fb_dim):
            # Calculate ranks for this FB group
            fb_group_ranks = [fb_grp_idx, fb_grp_idx +  (dist.get_world_size() // fb_dim)]
            all_fb_groups.append(fb_group_ranks)

        created_groups = []
        for group_idx, fb_ranks in enumerate(all_fb_groups):
            group = dist.new_group(
                ranks=fb_ranks,
                backend="gloo",
                group_desc=f"fb_gloo_group_{group_idx}_{fb_ranks}"
            )
            created_groups.append((group, fb_ranks))
        
        # Find the group this rank belongs to
        current_rank = dist.get_rank()
        for group, fb_ranks in created_groups:
            if current_rank in fb_ranks:
                fb_grp = group
                logger.info(y_str(f"Rank {rank} assigned to FB GLOO group: ") + f"{fb_ranks}")
                break

    device_count = torch.cuda.device_count()
    device = torch.device("cuda", rank % device_count)

    logger.info(g_str(f"Rank {rank}:") + f" {device=}")

    model_args = deepseek_config_registry.get(model_id, None)
    if model_args is None:
        raise ValueError(f"Model {model_id} not found in registry.")

    # TODO - remove this for full model\

    # model_args.num_hidden_layers = 16
    (
        model,
        model_parts,
        fb_size,
        fb_rank,
        pp_size,
        pp_rank,
        pp_mesh,
        ep_size,
        ep_rank,
    ) = parallelize_deepseek(world_mesh, device, model_args, rank, fb_grp, cpu_gloo_grp)

    # build tokenizer
    tokenizer = get_hf_tokenizer(model_id)

    # TODO - ep is not the same as dp really...just a temp shim atm.
    dataloader = build_hf_dataloader(
        dp_world_size=ep_size, dp_rank=ep_rank, tokenizer=tokenizer, job_config=config
    )

    # Synthetic setting
    microbatches = 4
    # microbatches = pp_size * 2

    # Use Symmetric Memory for MoE token shuffle.
    # TODO: we are rewriting `moe_on_device` function. `setup_symm_mem` is
    # currently supported for forward only. See `generate.py`.
    # model.setup_symm_mem(torch.bfloat16, device)

    torch.manual_seed(ep_rank)
    bs = config.training.local_batch_size  # * microbatches  # 4
    seqlen = config.training.seq_len  # 128

    # metrics manager
    proxy_parallel_dims = ParallelDims(
        dp_replicate=ep_size,
        dp_shard=fsdp_dim,
        pp=pp_size,
        fb=fb_size,
        cp=1,
        tp=1,
        world_size=world_mesh.size(),
        enable_loss_parallel=False,
    )

    metrics_processor = build_metrics_processor(
        config, proxy_parallel_dims, model_args=None
    )
    metrics_processor.num_flops_per_token = 100

    color = metrics_processor.color
    device_memory_monitor = metrics_processor.device_memory_monitor

    # logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
    device_module, device_type = utils.device_module, utils.device_type
    device_mem_stats = device_memory_monitor.get_peak_stats()
    logger.info(
        f"{color.yellow}{device_type.upper()} memory usage for model:  {color.reset}"
        f"{color.blue}{device_mem_stats.max_reserved_gib:.2f}GiB {color.reset}"
        f"{color.green}({device_mem_stats.max_reserved_pct:.2f}%){color.reset}"
    )

    # if pp_rank == 0 and ep_rank == 0:
    #     print(r_str(f"Rank {rank}: ") + f"{pp_size=}_{pp_rank=}_{ep_size=}_{ep_rank=}_{fb_size=} - Model: \n{model}")

    # Create loss function
    loss_fn = cross_entropy_loss  # torch.nn.functional.cross_entropy

    ft_manager = ft.init_ft_manager(config)
    optimizer = build_optimizers(model_parts, config, ft_manager)

    lr_scheduler = build_lr_schedulers(optimizer, config)

    # Run forward and backward
    steps = config.training.steps

    loss = float("inf")
    data_iterator = iter(dataloader)

    datetime_str = time.strftime("%Y%m%d-%H%M%S")
    trace_name = f"ds_real_{fb_rank=}_{pp_rank=}_{ep_rank=}_{datetime_str}"

    dist.barrier(group=cpu_gloo_grp)
    logger.info(g_str(f"Rank {rank}: All processes synchronized."))

    shared_fwd_cache = None
    if fb_dim > 1:
        # --- Centralized Manager Creation ---
        # Only one rank (global rank 0) will create the manager server.
        # This manager will host one dictionary per pipeline stage.
        if rank == 0:
            manager = Manager()
            # Create a list of dictionaries, one for each pipeline stage.
            all_caches = [manager.dict() for _ in range(pp_size)]
            authkey_bytes = bytes(manager._authkey)
            # Prepare the connection info and the list of all cache proxies to be broadcast.
            data_to_broadcast = [manager.address, authkey_bytes, all_caches]
        else:
            # All other ranks prepare a placeholder to receive the broadcasted data.
            data_to_broadcast = [None, None, None]

        # Broadcast the connection info and the list of caches from rank 0 to all other ranks.
        dist.broadcast_object_list(data_to_broadcast, src=0, group=cpu_gloo_grp)

        # All ranks now extract the connection info and the list of caches.
        address, authkey_bytes, all_caches = data_to_broadcast

        # If this rank is not the manager host, it must connect as a client.
        if rank != 0:
            class SharedCacheManager(BaseManager):
                pass
            # Register the 'dict' type so the client knows how to create a proxy for it.
            SharedCacheManager.register('dict')
            manager = SharedCacheManager(address=address, authkey=authkey_bytes)
            manager.connect()

        # Each rank selects the specific dictionary that corresponds to its pipeline stage rank (pp_rank).
        # This ensures the forward and backward processes of the same stage share the same dictionary.
        shared_fwd_cache = all_caches[pp_rank]

    # --- Test shared cache functionality ---
    if fb_dim > 1:
        logger.info(y_str(f"Rank {rank}: Testing shared forward cache..."))
        
        # Use a predictable key based on the pipeline rank, which is shared
        # by the forward and backward processes of a single stage.
        test_key = f"pp_stage_{pp_rank}"
        
        # The forward process (fb_rank 0) writes a test value.
        if fb_rank == 0:
            test_value = f"hello from rank {rank}"
            shared_fwd_cache[test_key] = test_value
            logger.info(g_str(f"Rank {rank} (forward):") + f" Wrote key '{test_key}' to shared cache.")

        # Synchronize within the forward-backward pair to ensure the write completes before the read.
        dist.barrier(group=fb_grp)

        # The backward process (fb_rank 1) reads and verifies the value.
        if fb_rank != 0:
            try:
                retrieved_value = shared_fwd_cache[test_key]
                logger.info(g_str(f"Rank {rank} (backward):") + f" Successfully read key '{test_key}' with value: '{retrieved_value}'. Cache is working.")
                # A more robust test could check the value, but existence is a good first step.
                assert "hello from" in retrieved_value
            except (KeyError, AssertionError) as e:
                logger.error(r_str(f"Rank {rank} (backward):") + f" FAILED cache test for key '{test_key}'. Error: {e}. Cache keys: {list(shared_fwd_cache.keys())}")
                assert False, "Cache test failed."

        # Synchronize again to ensure the backward process has finished reading.
        dist.barrier(group=fb_grp)
        
        # Cleanup: only the forward process should clear the key.
        if fb_rank == 0:
            shared_fwd_cache.pop(test_key, None)
            logger.info(g_str(f"Rank {rank} (forward):") + f" Cleaned up test key '{test_key}'.")

        # Global barrier to ensure all tests are done before proceeding.
        logger.info(y_str(f"Rank {rank}: Finished testing shared cache. Synchronizing all processes."))
        dist.barrier(group=cpu_gloo_grp)    

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=0, active=1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs', trace_name),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for step in range(1):
            optimizer.zero_grad()

            inputs, label = next_batch(data_iterator, metrics_processor)
            x = inputs["input"]

            if pp_size > 1:
                # Create pipeline stage
                stage = PipelineStage(
                    model,
                    pp_rank,
                    pp_size,
                    device,
                    group=pp_mesh.get_group(),
                    do_fb_pipeline=(fb_dim > 1),
                    fb_group=fb_grp,
                    global_rank=rank, 
                    shared_fwd_cache=shared_fwd_cache
                )

                # Create pipeline schedule
                if fb_dim > 1:
                    if fb_rank == 0:
                        pp_schedule = ScheduleForwardOnly(stage, microbatches, loss_fn=loss_fn)
                    else:
                        pp_schedule = ScheduleBackwardOnly(stage, microbatches, loss_fn=loss_fn)
                    
                    losses = []
                    if pp_rank == 0:
                        y = pp_schedule.step(x)
                    elif pp_rank == pp_size - 1:
                        # last rank...run loss function
                        y = pp_schedule.step(target=label, losses=losses)
                        loss = torch.mean(torch.stack(losses))
                    else:
                        pp_schedule.step()

                else:
                    losses = []
                    pp_schedule = ScheduleGPipe(stage, microbatches, loss_fn=loss_fn)
                
                    if pp_rank == 0:
                        y = pp_schedule.step(x)
                    elif pp_rank == pp_size - 1:
                        # last rank...run loss function
                        y = pp_schedule.step(target=label, losses=losses)
                        loss = torch.mean(torch.stack(losses))
                    else:
                        pp_schedule.step()
            else:
                y = model(x)
                loss = loss_fn(y, label)
                loss.backward()

            if pp_rank == pp_size - 1:

                global_avg_loss = global_max_loss = loss  # .detach().item()

                metrics_processor.log(step, global_avg_loss, global_max_loss,
                                    -1.0)
                
            if fb_dim > 1:
                # Use a barrier to ensure all processes have finished their backward passes
                # before we attempt to clear the cache.
                dist.barrier(group=fb_grp)
                # Only one rank (the one that created it) should perform the clear operation.
                if fb_rank == 0:
                    shared_fwd_cache.clear()
                # A second barrier ensures no process starts the next step until the
                # cache is confirmed to be empty.
                
            # optimizer.step()
            # lr_scheduler.step()
            if fb_dim > 1:
                logger.info(g_str(f"Rank {rank}: Step {step} finished - Synchronizing..."))
                dist.barrier(group=cpu_gloo_grp)

            prof.step()
        logger.info("Profiler trace exported to ./profiler_logs/" + trace_name)

    metrics_processor.close()
    logger.info("Training completed")


if __name__ == "__main__":

    try:
        # Use the 'forkserver' start method if available, as it can be faster
        # than 'spawn' for repeated process creation. Fall back to 'spawn'.
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn")
    except RuntimeError:
        # The start method can only be set once.
        pass


    init_logger()
    # we do this to remove a root logger that is added by HF
    # otherwise we get duplicate logs
    remove_notset_root_handlers()
    
    config_manager = ConfigManager()
    config = config_manager.parse_args()

    run_full_model(config)

    dist.destroy_process_group()
