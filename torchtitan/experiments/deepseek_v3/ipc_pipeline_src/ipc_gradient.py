import torch
import torch.distributed as dist
from typing import Dict, Tuple, List
from torch.distributed.tensor import DTensor, DeviceMesh, distribute_tensor
from torch.distributed.tensor.placement_types import Replicate

import torch_ipc_extension
from ipc_pipeline_src.ipc_share import create_and_share_tensor_ipc

def g_str(s):
    return "\033[32m" + s + "\033[0m"
def r_str(s):
    return "\033[31m" + s + "\033[0m"
def b_str(s):
    return "\033[34m" + s + "\033[0m"
def y_str(s):
    return "\033[33m" + s + "\033[0m"

def get_dtensor_info(name: str, grad: DTensor):
    """Prints sharding information for a DTensor."""
    out = ""
    if isinstance(grad, DTensor):
        out = (
            f"DTensor '{name}':\n"
            f"  Global Shape: {grad.shape}\n"
            f"  Local Shape:  {grad.to_local().shape}\n"
            f"  Device Mesh:  {grad.device_mesh}\n"
            f"  Placements:   {grad.placements}\n"
        )
    return out

def get_gradient_weight_info(model: torch.nn.Module):
    """
    Returns a dictionary of gradient weight information for the model.
    """
    grad_weight_info = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad = param.grad
            weight = param.data
            grad_mesh = grad.device_mesh if isinstance(grad, DTensor) else None
            weight_mesh = weight.device_mesh if isinstance(weight, DTensor) else None
            grad_placements = grad.placements if isinstance(grad, DTensor) else None
            weight_placements = weight.placements if isinstance(weight, DTensor) else None
            grad_weight_info[name] = {
                "Gradient": (grad.shape, grad.dtype, grad_mesh, grad_placements),
                "Weight": (weight.shape, weight.dtype, weight_mesh, weight_placements)
            }

    return grad_weight_info

class SharedGradientManager:
    """
    Manages the discovery, creation, and accumulation of gradients into
    shared IPC tensors for multi-process training. The cache holds local
    torch.Tensors that are backed by shared memory.
    """
    def __init__(self, model: torch.nn.Module, group: dist.ProcessGroup):
        """
        Args:
            model: The model whose gradients will be managed.
            mesh: The DeviceMesh that defines the process topology for DTensors.
            group: The process group over which gradients will be shared.
        """
        self.model = model
        self.group = group
        self.rank = dist.get_rank(group=self.group)
        self.world_size = dist.get_world_size(group=self.group)
        self.global_rank = dist.get_global_rank(group=self.group, group_rank=self.rank)
        # The cache now correctly holds torch.Tensor objects
        self.grad_cache: Dict[str, torch.Tensor] = {}

    def setup_shared_cache(self, src_rank: int = 0):
        """
        Discovers gradient metadata from the model after a backward pass,
        and creates the shared IPC tensors for accumulation.
        """
        grad_metadata = {}
        if self.rank == src_rank:
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad = param.grad
                    if isinstance(grad, DTensor):
                        grad = grad.to_local()
                    else:
                        print(r_str(f"Rank {self.global_rank} ") + f"Gradient for {name} is not a DTensor\n", end="")
                    grad_metadata[name] = (grad.shape, grad.dtype)
                    param.grad = None
                    grad = None
                    
        metadata_list = [grad_metadata]
        src_global_rank = dist.get_global_rank(group=self.group, group_rank=src_rank)
        dist.broadcast_object_list(metadata_list, src=src_global_rank, group=self.group)
        
        if self.rank != src_rank:
            grad_metadata = metadata_list[0]

        for name, (shape, dtype) in grad_metadata.items():
            # This creates a plain torch.Tensor backed by shared memory
            device = torch.device("cuda", torch.cuda.current_device())
            shared_local_tensor, _ = create_and_share_tensor_ipc(
                shape, dtype, device, group=self.group, src_rank=src_rank
            )
            
            # The cache correctly stores the plain tensor
            self.grad_cache[name] = shared_local_tensor
            
        torch.cuda.empty_cache()
        self.zero_grad()

    def accumulate_grad(self):
        """
        Accumulates gradients from the model into the shared tensor cache.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name in self.grad_cache:
                    cached_grad = self.grad_cache[name]
                    if isinstance(param.grad, DTensor):
                        print(b_str(f"Rank {self.global_rank} ") + 
                              get_dtensor_info(name, param.grad) + "\n", end="")
                    else:
                        print(r_str(f"Rank {self.global_rank} ") + 
                              f"Gradient for {name} is not a DTensor\n", end="")
                    
                    # Convert DTensor grad to a local tensor before adding
                    local_grad_shard = (
                        param.grad.to_local()
                        if isinstance(param.grad, DTensor)
                        else param.grad
                    )
                    if local_grad_shard.shape == cached_grad.shape:
                        torch_ipc_extension.acquire(cached_grad)
                        cached_grad.add_(local_grad_shard)
                        torch_ipc_extension.release(cached_grad)
                    else:
                        print(r_str(f"Rank {self.global_rank} ") + f"Gradient shape mismatch for "
                              f"{name} of shape {local_grad_shard.shape} and cached "
                              f"shape {cached_grad.shape}\n", end="")

                    param.grad = None

    def get_grad(self, name: str) -> torch.Tensor:
        """Retrieves a gradient tensor from the cache."""
        if name not in self.grad_cache:
            raise ValueError(f"Parameter {name} not found in the gradient cache, entries: {self.grad_cache.keys()}")
        return self.grad_cache[name]

    def zero_grad(self):
        """
        Resets all cached gradient tensors to zero.
        """
        for cached_grad in self.grad_cache.values():
            # FIX: cached_grad is already a local tensor, no .to_local() needed.
            torch_ipc_extension.acquire(cached_grad)
            cached_grad.zero_()
            torch_ipc_extension.release(cached_grad)
        dist.barrier(group=self.group)

    def destroy(self):
        """
        Clears the cache, allowing Python's GC to destroy the tensors
        and trigger the C++ deleters to clean up OS resources.
        """
        dist.barrier(group=self.group)
        self.grad_cache.clear()
        dist.barrier(group=self.group)
