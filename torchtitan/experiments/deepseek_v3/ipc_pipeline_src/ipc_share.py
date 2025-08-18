"""
IPC (Inter-Process Communication) Tensor Sharing Module

This module provides high-level functions for sharing PyTorch tensors and model parameters
across multiple processes using shared memory and IPC mechanisms.

The underlying C++ extension (torch_ipc_extension) provides:
- SharedData: A robust, process-shared data class using pthreads
- IPC tensor creation, copying, and sharing functions
- Process synchronization primitives (barriers, mutexes, semaphores)
"""

import subprocess
import os
import json
import shutil
import math
import time
import random
import gc
import pickle
from sympy import O
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights
from typing import Tuple, List, Optional, Union, Any, Dict

# Import the C++ extension
import torch_ipc_extension

# Type hints for the C++ extension functions and classes
# This makes Pylance aware of the available functions and their signatures

# SharedData class type hints
class SharedData:
    """
    A robust, process-shared data class using pthreads for inter-process synchronization.
    
    This class provides:
    - Process barriers for synchronization
    - Atomic integer operations with condition variables
    - Generic data buffer for arbitrary data exchange
    - Mutex-based exclusive access control
    - Semaphore-based resource management
    
    Attributes:
        name (str): The unique name of the shared memory segment
        is_creator (bool): Whether this process created the shared memory
        barrier_size (int): Number of processes that must reach the barrier
        initial_value (int): Initial value for the shared integer
        semaphore_count (int): Initial semaphore count
    """
    
    def __init__(self, name: str, is_creator: bool, barrier_size: int, 
                 initial_value: int = 0, semaphore_count: int = 1) -> None:
        """
        Initialize or attach to a shared memory segment.
        
        Args:
            name: Unique name for the shared memory segment
            is_creator: Whether this process should create the shared memory
            barrier_size: Number of processes that must reach the barrier
            initial_value: Initial value for the shared integer (creator only)
            semaphore_count: Initial semaphore count (creator only)
        """
        pass
    
    def barrier(self) -> None:
        """Wait for all processes to reach this barrier."""
        pass
    
    def get(self) -> int:
        """Get the current value of the shared integer."""
        pass
    
    def set(self, value: int) -> None:
        """Set the value of the shared integer."""
        pass
    
    def add(self, delta: int) -> None:
        """Atomically add to the shared integer and return the new value."""
        pass
    
    def wait_for_value(self, target_value: int) -> None:
        """Wait until the shared integer reaches a target value."""
        pass
    
    def write_data(self, data: bytes) -> None:
        """Write bytes to the shared data buffer."""
        pass
    
    def read_data(self) -> bytes:
        """Read bytes from the shared data buffer."""
        pass
    
    def sem_wait(self) -> None:
        """Wait on (decrement) the semaphore."""
        pass
    
    def sem_post(self) -> None:
        """Post to (increment) the semaphore."""
        pass

def acquire(tensor: torch.Tensor) -> None:
    """Acquire the lock for a given IPC tensor."""
    pass

def release(tensor: torch.Tensor) -> None:
    """Release the lock for a given IPC tensor with device synchronization."""
    pass

def release_async(tensor: torch.Tensor) -> None:
    """Release the lock for a given IPC tensor without device synchronization."""
    pass

def destroy_shared_data(shared_data: torch_ipc_extension.SharedData) -> None:
    """Destroy and unlink shared memory for a SharedData object."""
    pass

# Re-export the actual C++ extension classes and functions
# This ensures the type hints are available while maintaining the real functionality
__all__ = [
    'SharedData',
    'acquire', 'release', 'release_async',
    'destroy_shared_data',
    'get_shared_data',
    'share_model_parameters_ipc',
    'create_and_share_tensor_ipc',
    'copy_and_share_tensor_ipc'
]

def g_str(s):
    return "\033[32m" + s + "\033[0m"
def r_str(s):
    return "\033[31m" + s + "\033[0m"
def b_str(s):
    return "\033[34m" + s + "\033[0m"
def y_str(s):
    return "\033[33m" + s + "\033[0m"

def _check_tensor_ipc_comptability(tensor: torch.Tensor):
    "IPC does not support tensors that are not contiguous, has offset > 0, or number of elements less than 1."
    assert tensor is not None, "Tensors must not be None."
    assert tensor.is_contiguous(), "Non-contiguous tensors are not supported by IPC."
    tensor_size = torch.Size(tensor.shape)
    assert tensor_size.numel() >= 1, "Tensors with fewer than 1 elements are not supported by IPC."
    assert tensor.storage_offset() == 0, "Tensors with storage offset > 0 are not supported by IPC."

def _analyze_referrers(obj, name):
    """A helper function to print detailed information about an object's referrers."""
    print(f"--- Analyzing referrers for: {name} ---")
    referrers = gc.get_referrers(obj)
    print(f"Found {len(referrers)} referrers.")
    for i, ref in enumerate(referrers):
        print(f"  Referrer {i+1}:")
        print(f"    Type: {type(ref)}")
        if isinstance(ref, dict):
            for k, v in ref.items():
                if v is obj:
                    print(f"    Found in dict with key: '{k}'")
        elif isinstance(ref, (list, tuple, set)):
            print(f"    Found in a container of length {len(ref)}")
        elif isinstance(ref, types.FrameType):
            print(f"    It's a frame object (temporary reference from code execution)")
            print(f"      Code: {ref.f_code.co_name} in {ref.f_code.co_filename}:{ref.f_lineno}")
        else:
            try:
                print(f"    Content (snippet): {str(ref)[:200]}")
            except Exception:
                print("    Content could not be displayed.")
    print("-" * (20 + len(name)))

def _get_gradient_weight_info(model: torch.nn.Module):
    """
    Returns a dictionary of gradient weight information for the model.
    """
    grad_weight_info = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            # The parameter itself is the DTensor after fully_shard.
            # Do NOT use .data, as it returns the underlying torch.Tensor.
            weight = param
            grad = param.grad

            # Default to None for non-DTensor params or missing grads
            weight_mesh, weight_placements = None, None
            grad_mesh, grad_placements = None, None
            grad_shape, grad_dtype = None, None

            # Check the weight tensor
            if isinstance(weight, DTensor):
                weight_mesh = weight.device_mesh
                weight_placements = weight.placements

            # Check the gradient tensor
            if grad is not None:
                grad_shape = grad.shape
                grad_dtype = grad.dtype
                if isinstance(grad, DTensor):
                    grad_mesh = grad.device_mesh
                    grad_placements = grad.placements
            
            # If the weight is a DTensor, its shape attribute will still
            # correctly report the global shape.
            grad_weight_info[name] = {
                "Gradient": (
                    grad_shape, grad_dtype, grad_mesh, grad_placements
                ),
                "Weight": (
                    weight.shape, weight.dtype, weight_mesh, weight_placements
                )
            }

    return grad_weight_info

def _set_param_or_buffer(model, name, new_tensor):
    """Recursively finds and replaces a parameter or buffer in a model."""
    module_path, _, attr_name = name.rpartition('.')
    target_module = model.get_submodule(module_path)
    
    # Store old tensor reference before replacement
    old_tensor = None
    
    if attr_name in target_module._parameters:
        old_tensor = target_module._parameters[attr_name]
        # CRITICAL: Use nn.Parameter wrapper for parameters
        target_module._parameters[attr_name] = \
            nn.Parameter(new_tensor, requires_grad=old_tensor.requires_grad \
                             if old_tensor is not None else True)
        new_tensor.grad = old_tensor.grad
        del old_tensor.grad
    elif attr_name in target_module._buffers:
        old_tensor = target_module._buffers[attr_name]
        target_module._buffers[attr_name] = new_tensor
    else:
        raise AttributeError(f"{name} is not a parameter or buffer in the model.")

    # Explicitly delete old tensor reference to help with memory cleanup
    if old_tensor is not None:
        del old_tensor
    torch.cuda.empty_cache()
        
def _set_grad_in_param_or_buffer(model, name, new_tensor):
    """Recursively finds and replaces a parameter or buffer in a model."""
    module_path, _, attr_name = name.rpartition('.')
    target_module = model.get_submodule(module_path)
    
    if attr_name in target_module._parameters:
        target_module._parameters[attr_name].grad = new_tensor
    elif attr_name in target_module._buffers:
        target_module._buffers[attr_name].grad = new_tensor
    else:
        raise AttributeError(f"{name} is not a parameter or buffer in the model.")
    torch.cuda.empty_cache()
    
_shm_name_history = set()
def get_shared_data(shm_name: str, is_creator: bool, group_size: int,
                    initial_value: int = 0, semaphore_count: int = 1,
                    data_buffer_size: int = 4*1024) -> torch_ipc_extension.SharedData:
    """Wraps the SharedData class constructor and checks if the shared memory name is valid."""
    assert shm_name not in _shm_name_history, \
        r_str(f"Shared memory name {shm_name} already exists.")
    run_id = os.getenv('RUN_ID', "0")
    if run_id == "0":
        raise Warning(r_str(f"Environment variable RUN_ID is not set. " + \
            "Please set it to a unique value to avoid shared memory name collision."))
    shm_name = f"torch_ipc_extension_{shm_name}_{run_id}"
    _shm_name_history.add(shm_name)
    return torch_ipc_extension.SharedData(
        name=shm_name, is_creator=is_creator, barrier_size=group_size,
        initial_value=initial_value, semaphore_count=semaphore_count,
        data_buffer_size=data_buffer_size
    )
    
def destroy_shared_data(shared_data: torch_ipc_extension.SharedData):
    """Destroy and unlink shared memory for a SharedData object."""
    if shared_data.get_name() not in _shm_name_history:
        raise Warning(r_str(f"Shared memory name {shared_data.get_name()} does not exist."))
    _shm_name_history.remove(shared_data.get_name())
    shared_data.destroy()
    
def share_model_gradients_ipc(
    model: nn.Module,
    is_creator: bool,
    group_size: int,
    shm_name: str,
):
    """
    Shares model gradients from a source rank using C++-level IPC,
    independent of torch.distributed.
    """
    shm = get_shared_data(shm_name, is_creator, group_size, 
                          data_buffer_size=128*1024*1024)

    if is_creator:
        handles_and_meta = []
        param_names = [(True, name) for name, param in model.named_parameters()
                       if param.requires_grad]
        buffer_names = [(False, name) for name, buffer in model.named_buffers()
                        if buffer.requires_grad]
        
        for is_param, name in param_names + buffer_names:
            module_path, _, attr_name = name.rpartition('.')
            parent_module = model.get_submodule(module_path)
            original_tensor = getattr(parent_module, attr_name)
            original_grad = original_tensor.grad
            is_dtensor = isinstance(original_grad, DTensor)
            device_mesh = None
            placements = None
            # print(f"Original grad {name} is_dtensor: {is_dtensor}, is_param: {is_param}, "
            #       f"requires_grad: {original_tensor.requires_grad}, "
            #       f"grad is None: {original_grad is None}\n", end="")
            if is_dtensor:
                device_mesh = original_grad.device_mesh
                placements = original_grad.placements
                original_grad = original_grad.to_local()

            _check_tensor_ipc_comptability(original_grad)
            
            shared_grad, handle = torch_ipc_extension.copy_tensor_and_get_ipc(
                original_grad
            )
            shared_grad.zero_()
            
            device = original_grad.device
            device_uuid = str(torch.cuda.get_device_properties(device.index).uuid)
            meta = (
                name, handle, original_grad.shape, original_grad.dtype,
                original_grad.stride(), original_grad.storage_offset(),
                device_uuid, is_dtensor, device_mesh, placements
            )
            handles_and_meta.append(meta)
            
            if is_dtensor:
                shared_grad = DTensor.from_local(shared_grad,
                                                 device_mesh=device_mesh,
                                                 placements=placements)
            # setattr(original_tensor, "grad", shared_grad)
            _set_grad_in_param_or_buffer(model, name, shared_grad)
        
        # Serialize and write all metadata at once
        payload = pickle.dumps(handles_and_meta)
        shm.write_data(payload)
        

    # Barrier 1: Wait for src_rank to write data
    shm.barrier()

    if not is_creator:
        payload = shm.read_data()
        handles_and_meta = pickle.loads(payload)
        
        device = torch.device("cuda", torch.cuda.current_device())
        device_uuid = str(torch.cuda.get_device_properties(device.index).uuid)
        for name, handle, shape, dtype, stride, offset, source_uuid, \
            is_dtensor, device_mesh, placements \
                in handles_and_meta:
            assert device_uuid == source_uuid, \
                r_str("Rank ") + f"{dist.get_rank()} " + \
                r_str("Source and destination tensor not on the same device.") + \
                f" Source UUID: {source_uuid}, destination UUID: {device_uuid}"
                
            shared_grad = torch_ipc_extension.open_ipc_and_get_tensor(
                handle, device, dtype, shape, stride, offset
            )
            if is_dtensor:
                shared_grad = DTensor.from_local(shared_grad,
                                                 device_mesh=device_mesh,
                                                 placements=placements)
                
            module_path, _, attr_name = name.rpartition('.')
            parent_module = model.get_submodule(module_path)
            original_tensor = getattr(parent_module, attr_name)
            # setattr(original_tensor, "grad", shared_grad)
            _set_grad_in_param_or_buffer(model, name, shared_grad)
            
    # Barrier 2: Wait for all ranks to finish setting parameters
    gc.collect()
    torch.cuda.empty_cache()
    shm.barrier()
    if is_creator:
        destroy_shared_data(shm)

def share_model_parameters_ipc(
    model: nn.Module,
    is_creator: bool,
    group_size: int,
    shm_name: str,
):
    """
    Shares model parameters from a source rank using C++-level IPC,
    independent of torch.distributed.
    """
    shm = get_shared_data(shm_name, is_creator, group_size, 
                          data_buffer_size=128*1024*1024)

    if is_creator:
        handles_and_meta = []
        param_names = [(True, name) for name, _ in model.named_parameters()]
        buffer_names = [(False, name) for name, _ in model.named_buffers()]
        
        for is_param, name in param_names + buffer_names:
            module_path, _, attr_name = name.rpartition('.')
            parent_module = model.get_submodule(module_path)
            original_tensor = getattr(parent_module, attr_name)
            if is_param:
                original_tensor = original_tensor.data
            requires_grad = original_tensor.requires_grad
            is_dtensor = isinstance(original_tensor, DTensor)
            device_mesh = None
            placements = None
            if is_dtensor:
                device_mesh = original_tensor.device_mesh
                placements = original_tensor.placements
                original_tensor = original_tensor.to_local()
                print(f"Original tensor {name} is_dtensor: {is_dtensor}, "
                      f"is_param: {is_param}, device_mesh: {device_mesh}, "
                      f"placements: {placements}\n", end="")

            _check_tensor_ipc_comptability(original_tensor)
            
            shared_tensor, handle = torch_ipc_extension.copy_tensor_and_get_ipc(
                original_tensor
            )
            
            device = original_tensor.device
            device_uuid = str(torch.cuda.get_device_properties(device.index).uuid)
            meta = (
                name, handle, original_tensor.shape, original_tensor.dtype,
                original_tensor.stride(), original_tensor.storage_offset(),
                device_uuid, is_param, requires_grad, is_dtensor, 
                device_mesh, placements
            )
            handles_and_meta.append(meta)
            
            if is_dtensor:
                shared_tensor = DTensor.from_local(shared_tensor,
                                                   device_mesh=device_mesh,
                                                   placements=placements)
            _set_param_or_buffer(model, name, shared_tensor)

        # Serialize and write all metadata at once
        payload = pickle.dumps(handles_and_meta)
        shm.write_data(payload)
        

    # Barrier 1: Wait for src_rank to write data
    shm.barrier()

    if not is_creator:
        payload = shm.read_data()
        handles_and_meta = pickle.loads(payload)
        
        device = torch.device("cuda", torch.cuda.current_device())
        device_uuid = str(torch.cuda.get_device_properties(device.index).uuid)
        for name, handle, shape, dtype, stride, offset, source_uuid, \
            is_param, requires_grad, is_dtensor, device_mesh, placements \
                in handles_and_meta:
            assert device_uuid == source_uuid, \
                r_str("Rank ") + f"{dist.get_rank()} " + \
                r_str("Source and destination tensor not on the same device.") + \
                f" Source UUID: {source_uuid}, destination UUID: {device_uuid}"
                
            shared_tensor = torch_ipc_extension.open_ipc_and_get_tensor(
                handle, device, dtype, shape, stride, offset
            )
            if is_dtensor:
                shared_tensor = DTensor.from_local(shared_tensor,
                                                   device_mesh=device_mesh,
                                                   placements=placements)
            _set_param_or_buffer(model, name, shared_tensor)
            
    # Barrier 2: Wait for all ranks to finish setting parameters
    gc.collect()
    torch.cuda.empty_cache()
    shm.barrier()
    if is_creator:
        destroy_shared_data(shm)

def create_and_share_tensor_ipc(
    shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    is_creator: bool,
    group_size: int,
    shm_name: str
):
    """Creates a shared tensor on src_rank and shares it via C++ IPC."""
    shm = get_shared_data(shm_name, is_creator, group_size)
    
    shared_tensor = None
    if is_creator:
        assert torch.Size(shape).numel() >= 1, "Tensor must not be empty."
        shared_tensor, handle = torch_ipc_extension.create_tensor_and_get_ipc(
            shape, dtype, device
        )
        device_uuid = str(torch.cuda.get_device_properties(device.index).uuid)
        meta = (handle, shape, dtype, shared_tensor.stride(), 0, device_uuid)
        payload = pickle.dumps(meta)
        shm.write_data(payload)

    # Barrier 1: Wait for src_rank to create tensor and write metadata
    shm.barrier()
    
    if not is_creator:
        payload = shm.read_data()
        handle, shape, dtype, stride, offset, source_uuid = pickle.loads(payload)
        device = torch.device("cuda", torch.cuda.current_device())
        device_uuid = str(torch.cuda.get_device_properties(device.index).uuid)
        assert device_uuid == source_uuid, \
            r_str("Rank ") + f"{dist.get_rank()} " + \
            r_str("Source and destination tensor not on the same device.") + \
            f" Source UUID: {source_uuid}, destination UUID: {device_uuid}"
        shared_tensor = torch_ipc_extension.open_ipc_and_get_tensor(
            handle, device, dtype, shape, stride, offset
        )

    # Barrier 2: Wait for all ranks to receive the tensor
    shm.barrier()
    if is_creator:
        destroy_shared_data(shm)
    return shared_tensor


def copy_and_share_tensor_ipc(
    tensor: torch.Tensor,
    is_creator: bool,
    group_size: int,
    shm_name: str,
):
    """Copies a tensor to shared memory on src_rank and shares it via C++ IPC."""
    shm = get_shared_data(shm_name, is_creator, group_size)

    shared_tensor = None
    if is_creator:
        assert tensor is not None, "Tensor must not be None for creator rank."
        _check_tensor_ipc_comptability(tensor)
        shared_tensor, handle = torch_ipc_extension.copy_tensor_and_get_ipc(
            tensor
        )
        device_uuid = str(torch.cuda.get_device_properties(tensor.device.index).uuid)
        meta = (
            handle, tensor.shape, tensor.dtype,
            tensor.stride(), tensor.storage_offset(),
            device_uuid
        )
        payload = pickle.dumps(meta)
        shm.write_data(payload)
    
    # Barrier 1: Wait for src_rank to copy tensor and write metadata
    shm.barrier()

    if not is_creator:
        assert tensor is None, "Tensor must be None for non-creator rank."
        payload = shm.read_data()
        handle, shape, dtype, stride, offset, source_uuid = pickle.loads(payload)
        device = torch.device("cuda", torch.cuda.current_device())
        device_uuid = str(torch.cuda.get_device_properties(device.index).uuid)
        assert device_uuid == source_uuid, \
            r_str("Rank ") + f"{dist.get_rank()} " + \
            r_str("Source and destination tensor not on the same device.") + \
            f" Source UUID: {source_uuid}, destination UUID: {device_uuid}"
        shared_tensor = torch_ipc_extension.open_ipc_and_get_tensor(
            handle, device, dtype, shape, stride, offset
        )

    # Barrier 2: Wait for all ranks to receive the tensor
    shm.barrier()
    if is_creator:
        destroy_shared_data(shm)
    return shared_tensor

def _get_driver_memory_usage(device_index: int):
    """
    Gets the total used GPU memory and total GPU memory directly from
    the driver for a specific device.

    This function is useful for verifying memory allocated outside of PyTorch's
    default caching allocator, such as memory allocated via a custom C++
    extension like the one we built.

    Args:
        device_index: The logical index of the GPU device (e.g., 0).
                     This accounts for CUDA_VISIBLE_DEVICES.

    Returns:
        A tuple of (total_used_memory_mb, total_memory_mb).
        Returns (None, None) if the driver tool is not found or parsing fails.
    """
    used_mb = None
    total_mb = None

    # Case 1: NVIDIA GPUs using nvidia-smi
    nvidia_smi_path = shutil.which("nvidia-smi")
    if nvidia_smi_path:
        try:
            # Get the actual physical GPU index by querying all GPUs and mapping
            # the logical device index to the physical one
            list_cmd = [
                nvidia_smi_path,
                "--query-gpu=index",
                "--format=csv,noheader,nounits"
            ]
            result = subprocess.run(
                list_cmd, capture_output=True, text=True, check=True
            )
            
            # Parse the list of available GPU indices
            available_gpus = [int(idx.strip()) for idx in result.stdout.strip().split('\n') if idx.strip()]
            
            # Get CUDA_VISIBLE_DEVICES if set
            cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            if cuda_visible_devices:
                # Parse CUDA_VISIBLE_DEVICES to get the mapping
                visible_indices = [int(idx.strip()) for idx in cuda_visible_devices.split(',') if idx.strip()]
                if device_index < len(visible_indices):
                    physical_device_index = visible_indices[device_index]
                else:
                    print(f"Device index {device_index} out of range for CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
                    return None, None
            else:
                # No CUDA_VISIBLE_DEVICES, use device_index directly
                physical_device_index = device_index
            
            # Verify the physical device index is valid
            if physical_device_index not in available_gpus:
                print(f"Physical device index {physical_device_index} not found in available GPUs: {available_gpus}")
                return None, None
            
            # Get total memory for the specified device
            total_cmd = [
                nvidia_smi_path,
                f"--query-gpu=memory.total",
                f"--format=csv,noheader,nounits",
                f"--id={physical_device_index}"
            ]
            result = subprocess.run(
                total_cmd, capture_output=True, text=True, check=True
            )
            total_mb = int(result.stdout.strip()) * 1024 * 1024
            # print(f"nvidia-smi output 1: {result.stdout}, total_mb: {total_mb}")

            # Get total used memory for the specified device
            used_cmd = [
                nvidia_smi_path,
                f"--query-gpu=memory.used",
                f"--format=csv,noheader,nounits",
                f"--id={physical_device_index}"
            ]
            result = subprocess.run(
                used_cmd, capture_output=True, text=True, check=True
            )
            used_mb = int(result.stdout.strip()) * 1024 * 1024
            # print(f"nvidia-smi output 2: {result.stdout}, used_mb: {used_mb}")
            
            return used_mb, total_mb
        except Exception as e:
            print(f"Error processing nvidia-smi output: {e}")
            return None, None

    # Case 2: AMD GPUs using rocm-smi
    rocm_smi_path = shutil.which("rocm-smi")
    if rocm_smi_path:
        try:
            # Query the specific device and get output in JSON format
            cmd = [
                rocm_smi_path, "-d", str(device_index), 
                "--showmeminfo", "vram", "--showuse", "--json"
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )
            data = json.loads(result.stdout)
            
            card_key = f"card{device_index}"
            card_data = data[card_key]
            
            # Get total memory
            total_vram_str = card_data.get("VRAM Total Memory (B)")
            
            total_mb = -1  # Default to -1 if not found
            total_mb = int(total_vram_str)

            # Get memory used by the current process (fallback to total used if per-process not available)
            used_mb = -1  # Default to -1 if not found
            used_vram_str = card_data.get("VRAM Total Used Memory (B)")
            if used_vram_str is not None:
                used_mb = int(used_vram_str)
            
            return used_mb, total_mb
        except Exception as e:
            print(f"Error processing rocm-smi output: {e}")
            return None, None

    print("Could not find 'nvidia-smi' or 'rocm-smi' in PATH.")
    return None, None
              
def test_barrier_synchronization(rank, world_size):
    """
    Tests the pthread barrier for correct synchronization. This must be the
    first test as others rely on a working barrier.
    """
    test_name = "barrier_sync_test"
    if rank == 0:
        print(y_str(f"\n--- Running Test: Barrier Synchronization ---"))

    shm = None
    try:
        # All ranks create/open the object. The barrier_size is critical.
        shm = get_shared_data(test_name, rank == 0, world_size)
        
        # Phase 1: Ensure all processes wait
        if rank == 0:
            # Rank 0 will set a value *after* the barrier.
            # If the barrier works, other ranks should see the new value.
            pass
        else:
            # Other ranks sleep for a variable amount of time.
            time.sleep(random.uniform(0.01, 0.05))

        starting_val = shm.get()
        assert starting_val == 0, \
            r_str(f"[Rank {rank}] Barrier test failed! Expected 0, got {starting_val}")
        print(b_str(f"[Rank {rank}] Arrived at barrier 1" + 
                    f" value: {shm.get()}\n"), end="")
        shm.barrier()
        print(y_str(f"[Rank {rank}] Passed barrier 1" + 
                    f" value: {shm.get()}\n"), end="")

        for _ in range(10):
            shm.add(1)

        print(b_str(f"[Rank {rank}] Arrived at barrier 2") + 
              f" value: {shm.get()}\n", end="")
        shm.barrier()
        print(y_str(f"[Rank {rank}] Passed barrier 2") + 
              f" value: {shm.get()}\n", end="")

        final_val = shm.get()
        assert final_val == world_size * 10, \
            r_str(f"[Rank {rank}] Barrier test failed! Expected {world_size * 10}, "
                  f"got {final_val}")

        # Final barrier to ensure clean exit before destruction.
        shm.barrier()
        if rank == 0:
            print(g_str("[Rank 0] ✓ Test 'barrier' passed.\n"), end="")

    finally:
        # Final barrier and cleanup
        if shm:
            shm.barrier()
        if rank == 0:
            destroy_shared_data(shm)

def test_data_exchange(rank, world_size):
    """Tests writing and reading arbitrary data from the shared buffer."""
    test_name = "data_exchange_test"
    if rank == 0:
        print(y_str(f"\n--- Running Test: Data Exchange ---"))

    shm = None
    try:
        shm = get_shared_data(test_name, rank == 0, world_size)

        original_data = {
            'message': f'hello_world_from_rank_{rank}',
            'rank': rank,
            'data': [1, 2, 3, {'nested': True}]
        }

        if rank == 0:
            # Rank 0 serializes and writes the data
            payload = pickle.dumps(original_data)
            shm.write_data(payload)
            print(g_str(f"[Rank 0] Wrote {len(payload)} bytes to shared buffer\n\t") + 
                  f"Payload: {original_data}\n", end="")

        print(b_str(f"[Rank {rank}] Arrived at barrier\n"), end="")
        shm.barrier()
        print(y_str(f"[Rank {rank}] Passed barrier 1\n"), end="")

        # All other ranks read, deserialize, and verify
        if rank != 0:
            payload = shm.read_data()
            received_data = pickle.loads(payload)
            
            # We check against rank 0's original data
            original_rank_0_data = {
                'message': f'hello_world_from_rank_{0}',
                'rank': 0, # The data was from rank 0
                'data': [1, 2, 3, {'nested': True}]
            }
            print(b_str(f"[Rank {rank}] Received data: ") + 
                  f"{received_data}\n\t" + 
                  f"Original data: {original_rank_0_data}\n", end="")
            assert received_data == original_rank_0_data, \
                r_str(f"[Rank {rank}] Data exchange failed!")

        print(b_str(f"[Rank {rank}] Arrived at barrier 2\n"), end="")
        shm.barrier()
        print(y_str(f"[Rank {rank}] Passed barrier 2\n"), end="")
        
        # Test 2: Data exchange with multiple ranks
        if world_size > 1:
            if rank == 1:
                payload = pickle.dumps(original_data)
                shm.write_data(payload)
                print(g_str(f"[Rank {rank}] Wrote {len(payload)} bytes to shared buffer\n\t") + 
                      f"Payload: {original_data}\n", end="")
                
            shm.barrier()
            
            if rank != 1:
                payload = shm.read_data()
                received_data = pickle.loads(payload)
                original_rank_1_data = {
                    'message': f'hello_world_from_rank_{1}',
                    'rank': 1, # The data was from rank 1
                    'data': [1, 2, 3, {'nested': True}]
                }
                print(b_str(f"[Rank {rank}] Received data: ") + 
                      f"{received_data}\n\t" + 
                      f"Original data: {original_rank_1_data}\n", end="")
                assert received_data == original_rank_1_data, \
                    r_str(f"[Rank {rank}] Data exchange failed!")
        
        shm.barrier()
        if rank == 0:
            print(g_str("[Rank 0] ✓ Test 'data_exchange' passed.\n"), end="")

    finally:
        if shm:
            shm.barrier()
        if rank == 0:
            destroy_shared_data(shm)
    
def test_atomic_operations(rank, world_size):
    """Tests the atomic integer operations (get, set, add, etc.)."""
    test_name = "atomic_operations_test"
    if rank == 0:
        print(y_str(f"\n--- Running Test: Atomic Operations ---"))

    shm = None
    try:
        shm = get_shared_data(test_name, rank == 0, world_size,
                              initial_value=100, semaphore_count=1)
        
        shm.barrier()

        # Test 1: get/set
        if rank == 0:
            shm.set(42)
        
        shm.barrier()
        
        val = shm.get()
        assert val == 42, \
            r_str(f"[Rank {rank}] failed get/set test! Expected 42, got {val}")
        if rank == 0:
            print(g_str("[Rank 0] ✓ Test 'get/set' passed."))
        
        shm.barrier()

        # Test 2: add (replaces fetch_add logic)
        for _ in range(rank):
            shm.add(1)
        
        shm.barrier()
        
        if rank == 0:
            expected_sum = 42 + sum(range(world_size))
            final_val = shm.get()
            assert final_val == expected_sum, \
                r_str(f"[Rank {rank}] Atomic 'add' failed! Expected {expected_sum}, got {final_val}")
            print(g_str("[Rank 0] ✓ Test 'add' passed."))

        shm.barrier()

        # Test 3: wait_for_value
        if rank == 0:
            shm.set(world_size - 1)
        shm.barrier()

        # Ranks wait for their turn in reverse order
        target_val = world_size - 1 - rank
        shm.wait_for_value(target_val)
        
        val = shm.get()
        print(b_str(f"[Rank {rank}] wait_for_value section, value: {val}."))
        assert val == target_val, \
            f"Atomic 'wait_for_value' failed! Expected {target_val}, got {val}"
        
        # Signal the next rank
        if val > 0:
            shm.set(val - 1)
        
        shm.barrier()
        if rank == 0:
            print(g_str(f"[Rank {rank}] ✓ Test 'wait_for_value' passed."))

    finally:
        if shm:
            shm.barrier()
        if rank == 0:
            destroy_shared_data(shm)

def test_mutex_synchronization(rank, world_size):
    """Tests the mutex for exclusive access to a critical section."""
    test_name = "mutex_sync_test"
    num_adds = 100
    if rank == 0:
        print(y_str(f"\n--- Running Test: Mutex Synchronization ({num_adds} adds/rank) ---"))

    shm = None
    try:
        shm = get_shared_data(test_name, rank == 0, world_size)
        shm.barrier()

        # --- Part 1: Demonstrate the race condition WITHOUT a mutex ---
        # This read-modify-write pattern is a classic race condition.
        # Do not use add, it will not work as add is atomic.
        for _ in range(num_adds):
            current_val = shm.get()
            shm.set(current_val + 1)
        
        shm.barrier()

        if rank == 0:
            final_val = shm.get()
            expected_val = num_adds * world_size
            print(f"[Rank 0] Without mutex, expected {expected_val}, got {final_val}.")
            # This will likely fail, which is the point of the test.
            if final_val != expected_val:
                print(g_str("[Rank 0] ✓ Race condition correctly demonstrated (as expected)."))
            else:
                print(r_str("[Rank 0] ✗ Race condition did not occur (unlikely, but possible)."))
            # Reset for the real test
            shm.set(0)

        shm.barrier()

        # --- Part 2: Test the mutex for correctness ---
        for _ in range(num_adds):
            shm.mutex_acquire()
            # Critical Section
            current_val = shm.get()
            shm.set(current_val + 1)
            # End Critical Section
            shm.mutex_release()
        
        shm.barrier()

        # Verification
        if rank == 0:
            final_val = shm.get()
            expected_val = num_adds * world_size
            assert final_val == expected_val, \
                r_str(f"[Rank 0] Mutex test failed! Expected {expected_val}, got {final_val}")
            print(g_str(f"[Rank 0] ✓ Test 'mutex' passed. Final value: {final_val}"))

    finally:
        if shm:
            shm.barrier()
        if rank == 0:
            destroy_shared_data(shm)

def test_semaphore_synchronization(rank, world_size):
    """Tests the semaphore for controlling concurrent access."""
    semaphore_slots = max(1, world_size // 2)
    test_name = "semaphore_sync_test"
    if rank == 0:
        print(y_str(f"\n--- Running Test: Semaphore ({semaphore_slots} slots) ---"))

    shm = None
    try:
        shm = get_shared_data(test_name, rank == 0, world_size,
                              initial_value=0, semaphore_count=semaphore_slots)
        shm.barrier()

        shm.sem_wait()

        print(b_str(f"[Rank {rank}] Entering") + " critical section.")
        
        active_procs = shm.add(1)
        print (f"[Rank {rank}] Active processes inside: {active_procs}")
        assert active_procs <= semaphore_slots, \
            r_str(f"[Rank {rank}] Semaphore failed! {active_procs} procs inside, limit {semaphore_slots}")
        
        time.sleep(random.uniform(0.05, 0.15))
        
        shm.add(-1)
        
        print(y_str(f"[Rank {rank}] Exiting") + " critical section.")
        shm.sem_post()
        
        shm.barrier()

        if rank == 0:
            final_active_procs = shm.get()
            assert final_active_procs == 0, \
                r_str(f"[Rank 0] Semaphore cleanup failed! Expected 0, got {final_active_procs}")
            print(g_str(f"[Rank 0] ✓ Test 'semaphore' passed."))

    finally:
        if shm:
            shm.barrier()
        if rank == 0:
            destroy_shared_data(shm)

def test_memory_management(rank, world_size, device):
    """Test to verify no memory leaks in IPC tensor operations."""
    if rank == 0:
        print(f"\n[Rank {rank}] Testing memory management...")
    
    # Test 1: Create and share multiple tensors
    configs = [
        # Small tensors
        (torch.Size([8, 8]), torch.float32, 2),
        (torch.Size([16, 16]), torch.float16, 2),
        (torch.Size([32,]), torch.int32, 4),
        (torch.Size([32,]), torch.int8, 4),
        # Medium tensors
        (torch.Size([128, 64]), torch.float32, 4),
        (torch.Size([256, 128]), torch.float16, 4),
        (torch.Size([128, 128]), torch.int64, 2),
        (torch.Size([128, 128]), torch.uint8, 2),
        # Large tensors
        (torch.Size([1024, 512]), torch.float32, 8),
        (torch.Size([2048, 256]), torch.float16, 8),
        (torch.Size([512, 512]), torch.int32, 4),
        # Large number of allocs
        (torch.Size([128, 64]), torch.float32, 512),
        (torch.Size([256, 128]), torch.float16, 256),
        (torch.Size([128, 128]), torch.int64, 100),
        (torch.Size([128, 128]), torch.uint8, 2000),
        # Very large tensor (stress test, reduce num_alloc for safety)
        (torch.Size([1024, 1024, 1024]), torch.float32, 1),
        (torch.Size([512, 1024, 1024]), torch.float16, 3),
    ]
    
    shm_name = f"test_memory_management"
    shm = get_shared_data(shm_name, rank == 0, world_size)

    for i, (shape, dtype, num_alloc) in enumerate(configs):
        initial_allocated, _ = _get_driver_memory_usage(device.index)
        shm.barrier()

        tensors = []
        for j in range(num_alloc):
            shared_tensor = create_and_share_tensor_ipc(
                shape, dtype, device, is_creator=(rank == 0),
                group_size=world_size, shm_name=f"{shm_name}_{i}_{j}")
            tensors.append(shared_tensor)
        if rank == 0:
            size = shared_tensor.element_size() * shared_tensor.numel()
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                if size < 2 * 1024 * 1024:
                    # The AMD driver allocates in 2MiB chunks, so we need to round up
                    size = 2 * 1024 * 1024
            else:
                if size < 512 * 1024:
                    # The NVIDIA driver allocates in 512KiB chunks, so we need to round up
                    size = 512 * 1024
            size *= num_alloc
        shm.barrier()
        
        mid_allocated, _ = _get_driver_memory_usage(device.index)

        shm.barrier()
        
        # Test 2: Delete references and check memory cleanup
        shared_tensor = None
        del tensors
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()  # Ensure CUDA operations complete

        shm.barrier()
        
        final_allocated, _ = _get_driver_memory_usage(device.index)
        
        if rank == 0:
            print (f"--- Test {i+1}, config: shape={shape}, dtype={dtype}, num_alloc={num_alloc}, size={size/1024**2:.3f}MB ---")
            print(f"Memory usage - Initial: {initial_allocated/1024**2:.2f}MB, "
                f"Peak: {mid_allocated/1024**2:.2f}MB, "
                f"Final: {final_allocated/1024**2:.2f}MB")
            
            # Allow some tolerance for memory management overhead
            fail = False
            d_alloc = (mid_allocated - initial_allocated) - size 
            if abs(d_alloc)  > 4 * 1024 * 1024:  # 4MB tolerance
                fail = True
                print(r_str(f"Warning: Potential memory allocation problem detected. "
                            f"Allocation Delta: {d_alloc/1024**2:.2f}MB, expected {size/1024**2:.2f}MB"))

            d_alloc = final_allocated - initial_allocated
            # There seems to be a constant overhead for managing the
            # closed IPC handles, but this is not a leak
            # as it does not scale with the number or the size of the tensors.
            if abs(d_alloc) > 4 * 1024 * 1024:  # 4MB tolerance
                fail = True
                print(r_str(f"Warning: Potential memory leak detected. Final Delta: {d_alloc/1024**2:.2f}MB"))
            
            if not fail:
                print(g_str("✓ Memory management test passed"))
        shm.barrier()
            
    shm.barrier()
    if rank == 0:
        destroy_shared_data(shm)
    
def _test_single_tensor_sharing(rank, world_size, device, shape, dtype, 
                               custom_stride, storage_offset):
    original_tensor = None
    shared_tensor_created = None
    shared_tensor_copied = None
    received_tensor = None

    def config_to_str():
        return f"shape {shape}, dtype {dtype}, stride {custom_stride}, offset {storage_offset}"

    shm_name = f"test_single_tensor_sharing_{config_to_str()}"
    shm = None

    try:
        # Create shared object
        shm = get_shared_data(shm_name, rank == 0, world_size)
        # Create base tensor with enough storage for offset
        if rank == 0:
            if storage_offset > 0:
                # Create larger storage to accommodate offset
                total_elements = torch.Size(shape).numel() + storage_offset
                base_storage = torch.randn(total_elements, dtype=dtype, device=device)
                if custom_stride is not None:
                    original_tensor = base_storage.as_strided(shape, custom_stride, storage_offset)
                else:
                    original_tensor = base_storage[storage_offset:storage_offset + torch.Size(shape).numel()].view(shape)
            else:
                if custom_stride is not None:
                    # Calculate required storage size for custom stride
                    storage_size = storage_offset
                    for dim_size, dim_stride in zip(shape, custom_stride):
                        if dim_size > 1:
                            storage_size += (dim_size - 1) * dim_stride
                    storage_size += 1  # Add one for the last element
                    base_storage = torch.randn(storage_size, dtype=dtype, device=device)
                    original_tensor = base_storage.as_strided(shape, custom_stride, storage_offset)
                else:
                    if dtype in [torch.float16, torch.float32, torch.float64]:
                        original_tensor = torch.randn(shape, dtype=dtype, device=device)
                    elif dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]:
                        # Use randint for integer types
                        info = torch.iinfo(dtype)
                        original_tensor = torch.randint(
                            low=info.min, high=info.max, size=shape, dtype=dtype, device=device
                        )
                    elif dtype == torch.bool:
                        # Generate random bools
                        original_tensor = (torch.randint(0, 2, size=shape, device=device) < 0)
                    else:
                        raise ValueError(f"Unsupported dtype: {dtype}")
            
        # Create shared tensor from original tensor
        shared_tensor_created = create_and_share_tensor_ipc(
            shape, dtype, device, is_creator=(rank == 0), group_size=world_size,
            shm_name=f"{shm_name}_created")
        shared_tensor_copied = copy_and_share_tensor_ipc(
            original_tensor, is_creator=(rank == 0), group_size=world_size,
            shm_name=f"{shm_name}_copied")
        shm.barrier()
        
        # Test 1: Bit-level comparison
        if rank == 0:
            # Send original data for comparison
            dist.broadcast(original_tensor, src=0)
            shm.barrier()
            dist.broadcast(shared_tensor_created, src=0)
        else:
            # Receive and compare
            received_tensor = torch.empty_like(shared_tensor_copied)
            dist.broadcast(received_tensor, src=0)
            
            # Bit-level comparison using int8 view
            shared_bytes = shared_tensor_copied.contiguous().view(torch.int8)
            received_bytes = received_tensor.contiguous().view(torch.int8)
            
            assert torch.equal(shared_bytes, received_bytes), \
                r_str(f"Bit-level mismatch for config " + config_to_str() + "!") + \
                f"\nShared copied tensor:\n{shared_tensor_copied}" + \
                f"\nReceived tensor:\n{received_tensor}"
            print(g_str(f"[Rank {rank}] ✓ Bit-level comparison passed for created IPC tensor, ") +
                  "config " + config_to_str())
            shm.barrier()
            # Receive and compare
            received_tensor = torch.empty_like(shared_tensor_created)
            dist.broadcast(received_tensor, src=0)
            
            # Bit-level comparison using int8 view
            shared_bytes = shared_tensor_created.contiguous().view(torch.int8)
            received_bytes = received_tensor.contiguous().view(torch.int8)
            
            assert torch.equal(shared_bytes, received_bytes), \
                r_str(f"Bit-level mismatch for config " + config_to_str() + "!") + \
                f"\nShared copeid tensor:\n{shared_tensor_created}" + \
                f"\nReceived tensor:\n{received_tensor}"
            print(g_str(f"[Rank {rank}] ✓ Bit-level comparison passed for copied IPC tensor, ") +
                  "config " + config_to_str())
            
        shm.barrier()
        
        # Test 2: Modify from rank 0 and verify on rank 1
        if rank == 0:
            # Modify the tensor
            if dtype in [torch.float16, torch.float32, torch.float64]:
                shared_tensor_created.fill_(3.14159)
            elif dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
                shared_tensor_created.fill_(42)
            elif dtype == torch.uint8:
                shared_tensor_created.fill_(255)
            elif dtype == torch.bool:
                shared_tensor_created.fill_(True)
            
            modified_data = shared_tensor_created.clone()
            dist.broadcast(modified_data, src=0)
            shm.barrier()

            if dtype in [torch.float16, torch.float32, torch.float64]:
                shared_tensor_copied.fill_(3.14159)
            elif dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
                shared_tensor_copied.fill_(42)
            elif dtype == torch.uint8:
                shared_tensor_copied.fill_(255)
            elif dtype == torch.bool:
                shared_tensor_copied.fill_(True)
            
            modified_data = shared_tensor_copied.clone()
            dist.broadcast(modified_data, src=0)

        else:
            # Verify changes are visible
            received_tensor = torch.empty_like(shared_tensor_created)
            dist.broadcast(received_tensor, src=0)
            
            shared_bytes = shared_tensor_created.contiguous().view(torch.int8)
            received_bytes = received_tensor.contiguous().view(torch.int8)
            
            assert torch.equal(shared_bytes, received_bytes), \
                r_str(f"Modification not visible for config " + config_to_str() + "!") + \
                f"\nShared created tensor:\n{shared_tensor_created}" + \
                f"\nReceived tensor:\n{received_tensor}"
            print(g_str(f"[Rank {rank}] ✓ Rank 0 modification visible for created IPC tensor, ") +
                  "config " + config_to_str())

            shm.barrier()
            # Verify changes are visible
            received_tensor = torch.empty_like(shared_tensor_copied)
            dist.broadcast(received_tensor, src=0)
            
            shared_bytes = shared_tensor_copied.contiguous().view(torch.int8)
            received_bytes = received_tensor.contiguous().view(torch.int8)
            
            assert torch.equal(shared_bytes, received_bytes), \
                r_str(f"Modification not visible for config " + config_to_str() + "!") + \
                f"\nShared copied tensor:\n{shared_tensor_copied}" + \
                f"\nReceived tensor:\n{received_tensor}"
            print(g_str(f"[Rank {rank}] ✓ Rank 0 modification visible for copied IPC tensor, ") +
                  "config " + config_to_str())
        
        shm.barrier()
        
        # Test 3: Modify from rank 1 and verify on rank 0
        if rank == 1:
            if shared_tensor_created.numel() > 0:
                # Modify the tensor from rank 1
                if dtype in [torch.float16, torch.float32, torch.float64]:
                    shared_tensor_created[0] = float('inf')
                elif dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
                    shared_tensor_created[0] = -17
                elif dtype == torch.uint8:
                    shared_tensor_created[0] = 11
                elif dtype == torch.bool:
                    shared_tensor_created[0] = False
                
                modified_data = shared_tensor_created.clone()
                dist.broadcast(modified_data, src=1)

                shm.barrier()
                # Modify the tensor from rank 1
                if dtype in [torch.float16, torch.float32, torch.float64]:
                    shared_tensor_copied[0] = float('inf')
                elif dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
                    shared_tensor_copied[0] = -17
                elif dtype == torch.uint8:
                    shared_tensor_copied[0] = 11
                elif dtype == torch.bool:
                    shared_tensor_copied[0] = False
                
                modified_data = shared_tensor_copied.clone()
                dist.broadcast(modified_data, src=1)
        else:
            # Verify changes are visible on rank 0
            received_tensor = torch.empty_like(shared_tensor_created)
            dist.broadcast(received_tensor, src=1)
            
            original_bytes = shared_tensor_created.contiguous().view(torch.int8)
            received_bytes = received_tensor.contiguous().view(torch.int8)
            
            assert torch.equal(original_bytes, received_bytes), \
                r_str(f"Rank 1 modification not visible on rank 0 for config " + config_to_str() + "!") + \
                f"\nShared created tensor:\n{shared_tensor_created}" + \
                f"\nReceived tensor:\n{received_tensor}"
            print(g_str(f"[Rank {rank}] ✓ Rank 1 modification visible for created IPC tensor, ") +
                  "config " + config_to_str())
            
            shm.barrier()
            # Verify changes are visible on rank 0
            received_tensor = torch.empty_like(shared_tensor_copied)
            dist.broadcast(received_tensor, src=1)
            
            original_bytes = shared_tensor_copied.contiguous().view(torch.int8)
            received_bytes = received_tensor.contiguous().view(torch.int8)
            
            assert torch.equal(original_bytes, received_bytes), \
                r_str(f"Rank 1 modification not visible on rank 0 for config " + config_to_str() + "!") + \
                f"\nShared copied tensor:\n{shared_tensor_copied}" + \
                f"\nReceived tensor:\n{received_tensor}"
            print(g_str(f"[Rank {rank}] ✓ Rank 1 modification visible for copied IPC tensor, ") +
                  "config " + config_to_str())
        
        shm.barrier()

        # Test 4: Mutex-locked modification on all ranks
        num_adds = 10
        if dtype != torch.bool and dtype != torch.int8 and dtype != torch.uint8 and shared_tensor_created.numel() > 0:
            shared_tensor_created.flatten()[0] = 0
            shm.barrier()
            init_val = shared_tensor_created.flatten()[0].item()
            print(b_str(f"[Rank {rank}] Entering")+ " critical section without mutex, " +
                        f"Shared tensor value: {init_val} \n", end="")
            for i in range(num_adds):
                shared_tensor_created.flatten()[0] += (rank + 1)
            final_val = shared_tensor_created.flatten()[0].item()
            expected_val = init_val + (num_adds * (rank + 1))
            print(y_str(f"[Rank {rank}] Exiting") + " critical section without mutex, " +
                        f"Shared tensor value: {final_val}, expected: {expected_val} \n", end="")
            shm.barrier()

            if rank == 0:
                print(y_str(f"[Rank {rank}] Concurrent tensor modification without mutex result: ") + 
                      f"{shared_tensor_created.flatten()[0]}\n", end="")
            shm.barrier()
                
            shared_tensor_created.flatten()[0] = 0
            shm.barrier()
            torch_ipc_extension.acquire(shared_tensor_created)
            init_val = shared_tensor_created.flatten()[0].item()
            print(b_str(f"[Rank {rank}] Entering") + " critical section with mutex, " +
                        f"Shared tensor value: {init_val} \n", end="")      
            for i in range(num_adds):
                shared_tensor_created.flatten()[0] += (rank + 1)
            final_val = shared_tensor_created.flatten()[0].item()
            expected_val = init_val + (num_adds * (rank + 1))
            print(y_str(f"[Rank {rank}] Exiting") + " critical section with mutex, " +
                        f"Shared tensor value: {final_val}, expected: {expected_val} \n", end="")
            torch_ipc_extension.release(shared_tensor_created)
            shm.barrier()

            if rank == 0:
                # Verify changes are visible on rank 0
                expected_result = (sum(range(world_size)) + world_size) * num_adds
                result = shared_tensor_created.flatten()[0].item()
                
                print(y_str(f"[Rank {rank}] Concurrent tensor modification with mutex result: ") + 
                      f"{result}, " + y_str("expected: ") + f"{expected_result}\n", end="")
                
                if isinstance(result, float) or isinstance(expected_result, float):
                    assert math.isclose(result, expected_result, rel_tol=1e-5), \
                        r_str(f"[Rank {rank}] Concurrent Tensor modification with mutex incorrect for config " + config_to_str() + "!") + \
                        f"\nExpected Result:\n{expected_result}" + \
                        f"\nActual Result:\n{result}"
                else:
                    assert int(expected_result) == int(result), \
                        r_str(f"[Rank {rank}] Concurrent Tensor modification with mutex incorrect for config " + config_to_str() + "!") + \
                        f"\nExpected Result:\n{expected_result}" + \
                        f"\nActual Result:\n{result}"
                print(g_str(f"[Rank {rank}] ✓ Concurrent Tensor modification with mutex correct, ") +
                    "config " + config_to_str())

            
        shm.barrier()
        
        if rank == 0:
            print(g_str(f"[Rank {rank}] ✓ All tests passed for config " + config_to_str()))
            
    except Exception as e:
        print(r_str(f"[Rank {rank}] ✗ Test failed for config " + config_to_str()))
        raise
    finally:
        if shm is not None:
            shm.barrier()
            if rank == 0:
                destroy_shared_data(shm)

def test_single_tensor_sharing(rank, world_size, device):
    
    """Comprehensive test for single tensor IPC sharing across various configurations."""
    print(f"\n[Rank {rank}] Starting single tensor sharing tests...")
    
    # Test configurations: (shape, dtype, stride_multiplier, storage_offset)
    test_configs = [
        # Basic contiguous tensors
        ((10,), torch.float32, None, 0),
        ((5, 4), torch.float16, None, 0),
        ((2, 3, 4), torch.int32, None, 0),
        ((2, 2, 2, 2), torch.float64, None, 0),
        
        # Different data types
        ((100,), torch.int8, None, 0),
        ((100,), torch.uint8, None, 0),
        # ((50,), torch.int16, None, 0), torch.broadcast_to not supported for int16
        ((50,), torch.int64, None, 0),
        ((50,), torch.bool, None, 0),
        
        # Large tensors
        ((4096, 4096), torch.float16, None, 0),
        ((100, 100, 100), torch.float32, None, 0),
        
        # Edge cases
        ((1,), torch.float32, None, 0),  # Single element
        ((1, 1, 1), torch.float32, None, 0),  # Multi-dim single element
    ]
    
    for i, (shape, dtype, custom_stride, storage_offset) in enumerate(test_configs):
        if rank == 0:
            print(f"\n--- Test {i+1}: shape={shape}, dtype={dtype}, "
                  f"stride={custom_stride}, offset={storage_offset} ---")
        _test_single_tensor_sharing(rank, world_size, device, shape, dtype, custom_stride, storage_offset)
    if rank == 0:
        print(g_str(f"\n[Rank {rank}] ✓ All single tensor sharing tests completed successfully!"))

class _SimpleModel(nn.Module):
    """A basic model for initial testing."""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(20, 5)
        self.register_buffer('my_buffer', torch.randn(1, 5))

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x))) + self.my_buffer

def main():
    """Main execution function with robust unit tests."""
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA/HIP enabled GPU.")

    dist.init_process_group("gloo")
    world_size = dist.get_world_size()
    rank = int(os.environ["RANK"])
    target_device_id = 0
    torch.cuda.set_device(target_device_id)
    device = torch.device("cuda", target_device_id)

    # --- Test 0: Shared CPU Data Test ---
    if rank == 0:
        print("="*50)
        print("RUNNING TEST 0: Shared CPU Data")
        print("="*50)
            
    test_barrier_synchronization(rank, world_size)
    dist.barrier()
    test_data_exchange(rank, world_size)
    dist.barrier()
    test_atomic_operations(rank, world_size)
    dist.barrier()
    test_mutex_synchronization(rank, world_size)
    dist.barrier()
    test_semaphore_synchronization(rank, world_size)
    dist.barrier()

    dist.barrier()
    if rank == 0:
        print("--- Shared CPU Data Test PASSED ---")

    # --- Test 1: Memory Management Test ---
    if rank == 0:
        print("="*50)
        print("RUNNING TEST 1: Memory Management")
        print("="*50)
    
    test_memory_management(rank, world_size, device)
    
    dist.barrier()
    if rank == 0:
        print("--- Memory Management Test PASSED ---")
    
    # --- Test 2: Single Tensor Sharing ---
    if rank == 0:
        print("="*50)
        print("RUNNING TEST 2: Single Tensor Sharing")
        print("="*50)
    
    test_single_tensor_sharing(rank, world_size, device)
    
    dist.barrier()
    if rank == 0:
        print("--- Single Tensor Sharing Test PASSED ---")
    
    # --- Test 3: Simple Model ---
    if rank == 0:
        print("\n" + "="*40)
        print("RUNNING TEST 3: SimpleModel")
        print("="*40)
        
    dist.barrier()
    
    if rank == 0:
        simple_model = _SimpleModel().to(device)
    else:
        with init_empty_weights():
            simple_model = _SimpleModel()
    simple_model.eval()
            
    share_model_parameters_ipc(simple_model, is_creator=(rank == 0),
                               group_size=world_size,
                               shm_name=f"simple_model_test")
    
    simple_model.train()
    
    if rank == 0:
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.1)
        test_input = torch.randn(4, 10, device=device)
        output = simple_model(test_input)
        loss = output.sum()
        weight_before = simple_model.layer1.weight.data.clone()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        weight_after = simple_model.layer1.weight.data
        assert not torch.equal(weight_before, weight_after), (
            r_str(f"[Rank {rank}] Weight mismatch after sharing! \n") +
            f"Expected:\n{weight_before}\nGot:\n{weight_after}"
        )
        print(y_str("[Rank 0] SimpleModel weights updated on source rank."))
        dist.broadcast(weight_after, src=0)
    else:
        expected_weight = torch.empty_like(simple_model.layer1.weight.data)
        dist.broadcast(expected_weight, src=0)
        assert torch.equal(simple_model.layer1.weight.data, expected_weight), (
            r_str(f"[Rank {rank}] Weight mismatch after sharing! \n") +
            f"Expected:\n{expected_weight}\nGot:\n{simple_model.layer1.weight.data}"
        )
        print(g_str(f"[Rank {rank}] ✓ Verified SimpleModel weight update is visible."))
        
    dist.barrier()
    if rank == 0:
        print("--- SimpleModel Test PASSED ---")

    # --- Test 4: DeepSeek-V2-Lite Model ---
    if rank == 0:
        print("\n" + "="*40)
        print("RUNNING TEST 4: DeepSeek-V2-Lite")
        print("="*40)
    
    # This barrier prevents a race condition where rank 0 is busy loading
    # the model while other ranks proceed.
    dist.barrier()

    model_id = "deepseek-ai/DeepSeek-V2-lite-chat"
    # model_id = "moonshotai/Moonlight-16B-A3B"
    model_dtype = torch.float16

    if rank == 0:
        print(f"[Rank 0] Loading {model_id}...\n", end="")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=model_dtype, trust_remote_code=True
        ).to(device)
    else:
        print(f"[Rank {rank}] Initializing empty meta model...\n", end="")
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config, torch_dtype=model_dtype, trust_remote_code=True
            )
            
    model.train()
            
    dist.barrier()
    
    mem_before_used, mem_before_total = _get_driver_memory_usage(torch.cuda.current_device())
    print(g_str(f"Rank {rank} ") + "mem_before_used: " + f"{mem_before_used/1024**2:.2f} MB, "
          f"mem_before_total: {mem_before_total/1024**2:.2f} MB\n", end="")
    
    dist.barrier()

    share_model_parameters_ipc(model, is_creator=(rank == 0),
                               group_size=world_size,
                               shm_name=f"deepseek_v2_lite_test")
    
    dist.barrier()

    mem_after_used, mem_after_total = _get_driver_memory_usage(torch.cuda.current_device())
    print(g_str(f"Rank {rank} ") + "mem_after_used: " + f"{mem_after_used/1024**2:.2f} MB, "
          f"mem_after_total: {mem_after_total/1024**2:.2f} MB\n", end="")

    assert mem_after_used < mem_before_used + 100*1024**2, (
        r_str(f"[Rank {rank}] Memory leak detected after sharing model parameters!") + \
        f"mem_before_used: {mem_before_used/1024**2:.2f} MB, " + \
        f"mem_after_used: {mem_after_used/1024**2:.2f} MB, " + \
        f"mem_after_total: {mem_after_total/1024**2:.2f} MB"
    )

    dist.barrier()
    
    # --- STAGE 1: Direct Parameter Verification ---
    target_param_name = "model.layers.1.mlp.experts.0.up_proj.weight"
    if rank == 0:
        print(f"\n[Rank 0] Verifying parameter '{target_param_name}'...")
        target_param = model.get_parameter(target_param_name).data
        dist.broadcast(target_param, src=0)
    else:
        target_param_on_worker = model.get_parameter(target_param_name).data
        expected_param = torch.empty_like(target_param_on_worker)
        dist.broadcast(expected_param, src=0)
        assert torch.equal(target_param_on_worker, expected_param), (
            r_str(f"[Rank {rank}] Parameter mismatch after sharing! \n") +
            f"Expected:\n{expected_param}\nGot:\n{target_param_on_worker}"
        )
        print(g_str(f"[Rank {rank}] ✓ Verified parameter '{target_param_name}' matches rank 0.\n"), end="")
    
    dist.barrier()
    if rank == 0:
        print("--- Parameter sharing verification PASSED ---")

    # --- STAGE 2: Forward Pass (Logits) Verification ---
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    prompt = "Hello, this is a test of the shared model."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    if rank == 0:
        torch.manual_seed(42)
        with torch.no_grad():
            output_before = model(**inputs)
            logits_before = output_before.logits
            print(y_str(f"\n[Rank 0] Logits sum BEFORE update: {logits_before.sum()}\n"), end="")
        dist.broadcast(logits_before, src=0)
    else:
        torch.manual_seed(42)
        with torch.no_grad():
            output_before = model(**inputs)
            logits_before = output_before.logits
            print(f"\n[Rank {rank}] Logits sum BEFORE update: {logits_before.sum()}\n", end="")

        expected_logits_before = torch.empty_like(logits_before)
        dist.broadcast(expected_logits_before, src=0)

        local_bytes = logits_before.view(torch.int8)
        expected_bytes = expected_logits_before.view(torch.int8)
        assert torch.equal(local_bytes, expected_bytes), \
            r_str(f"[Rank {rank}] Initial logits do not match between ranks at the byte level!")
        print(g_str(f"[Rank {rank}] ✓ Verified initial logits match rank 0.\n"), end="")

    dist.barrier()

    # --- STAGE 3: Verification after weight update ---
    if rank == 0:
        print(f"\n[Rank 0] Updating weights...\n", end="")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        output = model(**inputs, labels=inputs["input_ids"])
        loss = output.loss
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        print("[Rank 0] Weights updated successfully.\n", end="")
        dist.barrier()
        
        torch.manual_seed(42)
        with torch.no_grad():
            output_after = model(**inputs)
            logits_after = output_after.logits
        assert not torch.allclose(logits_before, logits_after)
        dist.broadcast(logits_after, src=0)
    else:
        dist.barrier() # Wait for rank 0 to finish updating weights
        torch.manual_seed(42)
        with torch.no_grad():
            output_after = model(**inputs)
            logits_after = output_after.logits

        expected_logits_after = torch.empty_like(logits_after)
        dist.broadcast(expected_logits_after, src=0)

        local_bytes_after = logits_after.view(torch.int8)
        expected_bytes_after = expected_logits_after.view(torch.int8)
        assert torch.equal(local_bytes_after, expected_bytes_after), \
            r_str(f"[Rank {rank}] Final logits do not match between ranks at the byte level!")
        print(g_str(f"[Rank {rank}] ✓ Verified final logits match rank 0.\n"), end="")

    dist.barrier()

    # --- STAGE 4: Shared gradient verification ---
    mem_before_grad, mem_before_grad_total = \
        _get_driver_memory_usage(torch.cuda.current_device())
    print(g_str(f"Rank {rank} ") + "mem_before_grad: " + f"{mem_before_grad/1024**2:.2f} MB, "
          f"mem_before_grad_total: {mem_before_grad_total/1024**2:.2f} MB\n", end="")
    
    if rank == 0:
        print(f"\n[Rank 0] Sharing gradient...\n", end="")
        share_model_gradients_ipc(model, is_creator=(rank == 0),
                                  group_size=world_size,
                                  shm_name=f"deepseek_v2_lite_grad_test")
        dist.barrier()
        mem_after_grad, mem_after_grad_total = \
            _get_driver_memory_usage(torch.cuda.current_device())
        print(g_str(f"Rank {rank} ") + "mem_after_grad: " + f"{mem_after_grad/1024**2:.2f} MB, "
              f"mem_after_grad_total: {mem_after_grad_total/1024**2:.2f} MB\n", end="")
        assert mem_after_grad < mem_before_grad + 100*1024**2, (
            r_str(f"[Rank {rank}] Memory leak detected after sharing gradient!") + \
            f"mem_before_grad: {mem_before_grad/1024**2:.2f} MB, " + \
            f"mem_after_grad: {mem_after_grad/1024**2:.2f} MB, " + \
            f"mem_after_grad_total: {mem_after_grad_total/1024**2:.2f} MB"
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        dist.barrier() #1
        # Wait rank 1 to update the gradient
        dist.barrier() #2
        print(f"[Rank {rank}] Verifying shared gradient...\n", end="")
        target_grad = model.get_parameter(target_param_name).grad
        expected_grad = torch.empty_like(target_grad)
        dist.broadcast(expected_grad, src=1)
        assert torch.equal(target_grad, expected_grad), (
            r_str(f"[Rank {rank}] Gradient mismatch after sharing! \n") +
            f"Expected:\n{expected_param}\nGot:\n{target_param}"
        )
        print(g_str(f"[Rank {rank}] ✓ Verified shared gradient matches rank 1.\n"), end="")
        # Run the optimizer step to update the weights
        optimizer.step()
        dist.barrier() #3
        print(f"[Rank {rank}] Broadcasting updated parameter '{target_param_name}'...\n", end="")
        target_param = model.get_parameter(target_param_name).data
        dist.broadcast(target_param, src=0)
    elif rank == 1:
        share_model_gradients_ipc(model, is_creator=(rank == 0),
                                  group_size=world_size,
                                  shm_name=f"deepseek_v2_lite_grad_test")
        dist.barrier()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        dist.barrier() #1
        # Update the gradient
        output = model(**inputs, labels=inputs["input_ids"])
        loss = output.loss
        # loss.backward() will accumulate new gradients to the shared gradients
        loss.backward()
        dist.barrier() #2
        target_grad = model.get_parameter(target_param_name).grad
        print(f"[Rank {rank}] Broadcasting shared gradient...\n", end="")
        dist.broadcast(target_grad, src=1)
        # Wait rank 0 to run the optimizer step and update the parameter
        dist.barrier() #3
        print(f"[Rank {rank}] Verifying updated parameter '{target_param_name}'...\n", end="")
        target_param = model.get_parameter(target_param_name).data
        expected_param = torch.empty_like(target_param)
        dist.broadcast(expected_param, src=0)
        assert torch.equal(target_param, expected_param), (
            r_str(f"[Rank {rank}] Parameter mismatch after sharing! \n") +
            f"Expected:\n{expected_param}\nGot:\n{target_param}"
        )
        print(g_str(f"[Rank {rank}] ✓ Verified updated parameter '{target_param_name}' matches rank 0.\n"), end="")
    else:
        share_model_gradients_ipc(model, is_creator=(rank == 0),
                                  group_size=world_size,
                                  shm_name=f"deepseek_v2_lite_grad_test")
        dist.barrier()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        dist.barrier() #1
        # Wait rank 1 to update the gradient
        dist.barrier() #2
        print(f"[Rank {rank}] Verifying shared gradient...\n", end="")
        target_grad = model.get_parameter(target_param_name).grad
        expected_grad = torch.empty_like(target_grad)
        dist.broadcast(expected_grad, src=1)
        assert torch.equal(target_grad, expected_grad), (
            r_str(f"[Rank {rank}] Gradient mismatch after sharing! \n") +
            f"Expected:\n{expected_param}\nGot:\n{target_param}"
        )
        print(g_str(f"[Rank {rank}] ✓ Verified shared gradient matches rank 1.\n"), end="")
        dist.barrier() #3
        print(f"[Rank {rank}] Verifying updated parameter '{target_param_name}'...\n", end="")
        target_param = model.get_parameter(target_param_name).data
        expected_param = torch.empty_like(target_param)
        dist.broadcast(expected_param, src=0)
        assert torch.equal(target_param, expected_param), (
            r_str(f"[Rank {rank}] Parameter mismatch after sharing! \n") +
            f"Expected:\n{expected_param}\nGot:\n{target_param}"
        )
        print(g_str(f"[Rank {rank}] ✓ Verified updated parameter '{target_param_name}' matches rank 0.\n"), end="")

    dist.barrier()
    if rank == 0:
        print("\n--- DeepSeek-V2-Lite Test PASSED ---")
        print(g_str("\n//// All verification successful on all ranks! ////"))

if __name__ == "__main__":
    if "torch_ipc_extension" in globals():
         main()
