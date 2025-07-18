# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional

import torch
import torch.distributed as dist

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard

# from checkpoint import load_weights_from_hf
from torchtitan.experiments.deepseek_v3.model import DeepseekForCausalLM

from torchtitan.tools.logging import logger


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

# from ..model.moe import MoE


# Get model parallel subgroup by name:
# e.g. "pp", "ep", None
def get_group(dim_name: Optional[str] = None) -> dist.ProcessGroup:
    glob = torch.distributed.device_mesh._mesh_resources.get_current_mesh()
    return glob.get_group(dim_name)


def parallelize_deepseek(
    # model: nn.Module,
    world_mesh: DeviceMesh,
    device: torch.device,
    model_args,
    rank: int,
    fb_gloo_grp: dist.ProcessGroup,
    cpu_gloo_grp: dist.ProcessGroup,
    # parallel_dims: ParallelDims,
    # job_config: JobConfig,
):
    """
    Apply parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    logger.info("Applying parallelism to the model...")
    world_size = int(os.environ["WORLD_SIZE"])

    fb_mesh = world_mesh["fb"]
    fb_rank = fb_mesh.get_local_rank()
    fb_size = fb_mesh.size()

    pp_mesh = world_mesh["pp"]
    ep_mesh = world_mesh["ep"]
    pp_rank = pp_mesh.get_local_rank()
    ep_rank = ep_mesh.get_local_rank()
    pp_size = pp_mesh.size()
    ep_size = ep_mesh.size()

    # Apply data parallelism
    fsdp_mesh = world_mesh["fsdp"]
    hsdp_mesh = world_mesh["ep", "fsdp"]

    hsdp_size = hsdp_mesh.size()

    # Apply model parallelism
    model_args.ep_size = ep_size
    model_args.num_stages = pp_size
    model_args.stage_idx = pp_rank
    logger.info(
        y_str(f"Rank {rank} Parallelism: ") + 
        f"{fb_size=}, {ep_size=}, {pp_size=}, {model_args.ep_size=}, {model_args.num_stages=}, {model_args.stage_idx=}"
    )
    # print(model_args)
    # verify world size matches parallelized total
    parallelized_world_size = pp_size * hsdp_size * fb_size
    logger.info(g_str(f"Total Parallelized World size: ") + f"{parallelized_world_size}")
    assert (
        world_size == parallelized_world_size
    ), f"mismatch between total world size {world_size=} and parallelized total {parallelized_world_size}"

    model=None

    # Instantiate model
    logger.info("")
    with device, world_mesh:
        if fb_rank == 0:    
            model = DeepseekForCausalLM(model_args)
        else:
            with torch.device("meta"):
                model = DeepseekForCausalLM(model_args)
    # Load weights
    # load_weights_from_hf(model, model_id, device)
    model.train()

    dist.barrier(group=cpu_gloo_grp)
    logger.info(g_str(f"Rank {rank}: All processes synchronized."))
    # Share weights within fb ranks
    if fb_size > 1:
        if fb_rank == 0:
            model.share_memory() # Make model's storage shareable via IPC
            # Get state dict with shared tensors
            shared_state_dict = {}
            for name, param in model.named_parameters():
                shared_state_dict[name] = param.data
            
            print(r_str(f"Rank {rank}: ") + f"Sharing model state dict on GPU {device}")
            dist.broadcast_object_list([shared_state_dict], group=fb_gloo_grp, group_src=0)
        else:
            received_data = [None]
            print(r_str(f"Rank {rank}: ") + f"Receiving shared model state dict on GPU {device}")
            dist.broadcast_object_list(received_data, group=fb_gloo_grp, group_src=0)
            shared_state_dict = received_data[0]
            
            if shared_state_dict is not None:
                # Load shared tensors into model (this will replace meta tensors)
                model.load_state_dict(shared_state_dict, strict=False, assign=True)
            
            # 안전하게 meta buffers만 처리
            def materialize_meta_buffers(module, target_device):
                # 현재 모듈의 직접적인 버퍼만 처리 (점이 없는 이름들)
                for name, buffer in list(module.named_buffers(recurse=False)):
                    if buffer.is_meta:
                        # logger.info(y_str(f"Rank {rank}: ") + f"Materializing meta buffer {name} on GPU {target_device}")
                        # meta buffer를 실제 device에서 초기화
                        new_buffer = torch.empty_like(buffer, device=target_device)
                        # RoPE 관련 버퍼인 경우 적절히 초기화
                        if 'inv_freq' in name:
                            # inv_freq는 특별히 계산해서 초기화해야 함
                            pass  # 모델 내부에서 lazy 초기화됨
                        module.register_buffer(name, new_buffer, persistent=False)
                
                # 재귀적으로 자식 모듈들 처리
                for child in module.children():
                    materialize_meta_buffers(child, target_device)
            
            materialize_meta_buffers(model, device)
            
            print(r_str(f"Rank {rank}: ") + f"Loaded shared model state dict and materialized buffers on GPU {device}")
        
        dist.barrier(group=fb_gloo_grp)

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

    # 모델 파트 분리: 파이프라인 병렬화 단계별로 파트 분리
    # pp_size가 1이면 전체 모델, 아니면 각 파이프라인 stage별로 파트 분리
    model_parts = []
    if pp_size > 1 and hasattr(model.model, "layers"):
        # 각 파이프라인 stage에 해당하는 레이어만 파트로 분리
        # 예시: model.model.layers는 dict이므로, 각 stage에 해당하는 레이어만 추출
        # 실제 분할 방식은 모델 구조에 따라 다를 수 있음
        # 여기서는 단순히 전체 레이어를 하나의 파트로 반환 (실제 분할 필요시 수정)
        model_parts.append(model.model)
    else:
        model_parts.append(model)

    return (
        model,
        model_parts,
        fb_size,
        fb_rank,
        pp_size,
        pp_rank,
        pp_mesh,
        ep_size,
        ep_rank,
    )
