import torch
import torch.distributed as dist
from typing import Dict, Tuple, List
from torch.distributed.tensor import DTensor, DeviceMesh
from torch.distributed.tensor.placement_types import Replicate

# Assume your compiled extension is named 'torch_ipc_extension'
import torch_ipc_extension

def g_str(s):
    return "\033[32m" + s + "\033[0m"
def r_str(s):
    return "\033[31m" + s + "\033[0m"
def b_str(s):
    return "\033[34m" + s + "\033[0m"
def y_str(s):
    return "\033[33m" + s + "\033[0m"

class SharedGradientManager:
    """
    Manages the discovery, creation, and accumulation of gradients into
    shared, distributed IPC tensors (DTensors) for multi-process training.
    """
    def __init__(self, model: torch.nn.Module, mesh: DeviceMesh, group: dist.ProcessGroup):
        """
        Args:
            model: The model whose gradients will be managed.
            mesh: The DeviceMesh that defines the process topology for DTensors.
            group: The process group over which gradients will be shared.
        """
        self.model = model
        self.mesh = mesh
        self.group = group
        self.rank = dist.get_rank(group=self.group)
        self.world_size = dist.get_world_size(group=self.group)
        self.global_rank = dist.get_global_rank(group=self.group, group_rank=self.rank)
        self.grad_cache: Dict[str, DTensor] = {}

    def setup_shared_cache(self, src_rank: int = 0):
        """
        Discovers gradient metadata from the model after a backward pass,
        and creates the shared, distributed IPC tensors for accumulation.
        """
        grad_metadata = {}
        if self.rank == src_rank:
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad = param.grad
                    if isinstance(grad, DTensor):
                        grad_metadata[name] = (grad.shape, grad.dtype, grad.placements)
                    else:
                        grad_metadata[name] = (grad.shape, grad.dtype, [Replicate()])

        metadata_list = [grad_metadata]
        src_global_rank = dist.get_global_rank(group=self.group, group_rank=src_rank)
        dist.broadcast_object_list(metadata_list, src=src_global_rank, group=self.group)
        
        if self.rank != src_rank:
            grad_metadata = metadata_list[0]

        for name, (global_shape, dtype, placements) in grad_metadata.items():
            local_shard_shape = DTensor.from_local(
                torch.empty(0, dtype=dtype, device=self.mesh.device_type),
                self.mesh,
                placements,
                run_check=False
            ).to_local().shape

            # FIX: If the local shard for this rank has zero elements,
            # there's no need to create a shared tensor for it.
            if torch.Size(local_shard_shape).numel() == 0:
                continue

            shared_local_tensor, _ = torch_ipc_extension.create_tensor_and_get_ipc(
                local_shard_shape, dtype, self.mesh.device_type
            )
            
            shared_dtensor = DTensor.from_local(
                shared_local_tensor, self.mesh, placements, run_check=True
            )
            
            self.grad_cache[name] = shared_dtensor
        
        self.zero_grad()

    def accumulate_grad(self):
        """
        Accumulates gradients from the model into the shared DTensor cache.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name in self.grad_cache:
                    cached_grad_dtensor = self.grad_cache[name]
                    
                    # Acquire the lock on the local shard for atomic update.
                    local_shard = cached_grad_dtensor.to_local()
                    torch_ipc_extension.acquire(local_shard)
                    
                    # Perform the DTensor to DTensor addition.
                    cached_grad_dtensor.add_(param.grad)
                    
                    torch_ipc_extension.release(local_shard)

                    param.grad = None
                    if self.rank == 0:
                        param.grad = cached_grad_dtensor

    def get_grad(self, name: str) -> DTensor:
        """Retrieves a distributed gradient tensor from the cache."""
        if name not in self.grad_cache:
            raise ValueError(f"Parameter {name} not found in the gradient cache")
        return self.grad_cache[name]

    def zero_grad(self):
        """
        Resets all cached distributed gradient tensors to zero.
        This is a collective operation.
        """
        for cached_grad in self.grad_cache.values():
            # DTensor.zero_() correctly zeros all local shards.
            # We still use a lock to make the operation atomic.
            local_shard = cached_grad.to_local()
            torch_ipc_extension.acquire(local_shard)
            cached_grad.zero_()
            torch_ipc_extension.release(local_shard)
        dist.barrier(group=self.group)

    def destroy(self):
        """
        Clears the cache, allowing Python's GC to destroy the tensors
        and trigger the C++ deleters to clean up OS resources.
        """
        dist.barrier(group=self.group)
        self.grad_cache.clear()
        dist.barrier(group=self.group)
