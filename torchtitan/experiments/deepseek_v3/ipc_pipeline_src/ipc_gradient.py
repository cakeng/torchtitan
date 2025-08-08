import torch
import torch.distributed as dist
from typing import Dict, Tuple, List
from torch.distributed.tensor import DTensor, DeviceMesh, distribute_tensor
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
    shared IPC tensors for multi-process training. The cache holds local
    torch.Tensors that are backed by shared memory.
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
                        grad_metadata[name] = (grad.shape, grad.dtype, grad.placements)
                    else:
                        grad_metadata[name] = (grad.shape, grad.dtype, [Replicate()])

        metadata_list = [grad_metadata]
        src_global_rank = dist.get_global_rank(group=self.group, group_rank=src_rank)
        dist.broadcast_object_list(metadata_list, src=src_global_rank, group=self.group)
        
        if self.rank != src_rank:
            grad_metadata = metadata_list[0]

        for name, (global_shape, dtype, placements) in grad_metadata.items():
            meta_tensor = torch.empty(global_shape, dtype=dtype, device="meta")
            conceptual_dtensor = distribute_tensor(meta_tensor, self.mesh, placements)
            local_shard_shape = conceptual_dtensor.to_local().shape

            if torch.Size(local_shard_shape).numel() == 0:
                continue

            # This creates a plain torch.Tensor backed by shared memory
            shared_local_tensor, _ = torch_ipc_extension.create_tensor_and_get_ipc(
                local_shard_shape, dtype, torch.device(self.mesh.device_type)
            )
            
            # The cache correctly stores the plain tensor
            self.grad_cache[name] = shared_local_tensor
        
        self.zero_grad()

    def accumulate_grad(self):
        """
        Accumulates gradients from the model into the shared tensor cache.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name in self.grad_cache:
                    cached_grad = self.grad_cache[name]
                    
                    # Convert DTensor grad to a local tensor before adding
                    local_grad_shard = param.grad.to_local() if isinstance(param.grad, DTensor) else param.grad
                    
                    torch_ipc_extension.acquire(cached_grad)
                    cached_grad.add_(local_grad_shard)
                    torch_ipc_extension.release(cached_grad)

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
