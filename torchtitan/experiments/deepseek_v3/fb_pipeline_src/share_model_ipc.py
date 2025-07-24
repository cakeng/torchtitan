import os
import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights

# Import the C++ extension after it has been built.
import torch_ipc_extension

def g_str(s):
    return "\033[32m" + s + "\033[0m"
def r_str(s):
    return "\033[31m" + s + "\033[0m"
def b_str(s):
    return "\033[34m" + s + "\033[0m"
def y_str(s):
    return "\033[33m" + s + "\033[0m"

def share_model_parameters_ipc(
    model: nn.Module,
    group: dist.ProcessGroup = None,
    src_rank: int = 0
):
    """Shares model parameters from a source rank using the C++ IPC extension."""
    # Add a barrier to ensure all processes enter the function together.
    rank = dist.get_rank(group=group)

    if rank == src_rank:
        handles_and_meta = []
        tensors_to_share = list(model.named_parameters()) + list(
            model.named_buffers()
        )
        
        # Store original tensors to prevent premature deallocation
        original_tensors = {}
        
        for name, tensor in tensors_to_share:
            check_tensor_ipc_comptability(tensor)
            
            # Store original tensor reference to prevent deallocation
            original_tensors[name] = tensor
            
            handle, shared_tensor = torch_ipc_extension.copy_tensor_and_get_ipc(tensor)
            device = torch.device("cuda", torch.cuda.current_device())
            device_index = device.index if hasattr(device, "index") else device
            device_uuid = str(torch.cuda.get_device_properties(device_index).uuid)
            meta = (
                name, handle, tensor.shape, tensor.dtype,
                tensor.stride(), tensor.storage_offset(),
                device_uuid
            )
            handles_and_meta.append(meta)
            
            # CRITICAL: Replace the parameter/buffer immediately to avoid duplicates
            _set_param_or_buffer(model, name, shared_tensor)
            
            # Clear original tensor reference after replacement
            del original_tensors[name]
        
        object_to_broadcast = [handles_and_meta]
        dist.broadcast_object_list(object_to_broadcast, src=src_rank, group=group)
    else:
        received_objects = [None]
        dist.broadcast_object_list(received_objects, src=src_rank, group=group)
        handles_and_meta = received_objects[0]
        
        for name, handle, shape, dtype, stride, storage_offset, source_uuid in handles_and_meta:
            device = torch.device("cuda", torch.cuda.current_device())
            device_index = device.index if hasattr(device, "index") else device
            device_uuid = str(torch.cuda.get_device_properties(device_index).uuid)
            assert device_uuid == source_uuid, \
                r_str("Source tensor and destination tensor must be on the same device.") + \
                f" Source device UUID: {source_uuid}, destination device UUID: {device_uuid}"
            shared_tensor = torch_ipc_extension.open_ipc_and_get_tensor(
                handle, device, dtype, shape, stride, storage_offset)
            _set_param_or_buffer(model, name, shared_tensor)
            
    # Add a barrier to ensure all processes have completed sharing
    # before exiting the function.
    dist.barrier(group=group)

def check_tensor_ipc_comptability(tensor: torch.Tensor):
    "IPC does not support tensors that are not contiguous, has offset > 0, or number of elements less than 1."
    assert tensor is not None, "Tensors must not be None."
    assert tensor.is_contiguous(), "Non-contiguous tensors are not supported by IPC."
    tensor_size = torch.Size(tensor.shape)
    assert tensor_size.numel() >= 1, "Tensors with fewer than 1 elements are not supported by IPC."
    assert tensor.storage_offset() == 0, "Tensors with storage offset > 0 are not supported by IPC."

def create_and_share_tensor_ipc(
    shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    group: dist.ProcessGroup = None,
    src_rank: int = 0
):
    """Creates a shared tensor from a source rank using the C++ IPC extension."""
    rank = dist.get_rank(group=group)
    shared_tensor = None
    
    if rank == src_rank:
        assert torch.Size(shape).numel() >= 1, "Tensors with fewer than 1 elements are not supported by IPC."
        handle, shared_tensor = torch_ipc_extension.create_tensor_and_get_ipc(
            shape, dtype, device
        )
        # Get the device UUID for more robust device identification
        device_index = device.index if hasattr(device, "index") else device
        device_uuid = str(torch.cuda.get_device_properties(device_index).uuid)
        meta = (handle, shape, dtype, shared_tensor.stride(), 0, device_uuid)
        object_to_broadcast = [meta]
        dist.broadcast_object_list(object_to_broadcast, src=src_rank, group=group)
    else:
        received_objects = [None]
        dist.broadcast_object_list(received_objects, src=src_rank, group=group)
        handle, shape, dtype, stride, storage_offset, source_uuid = received_objects[0]
        device_index = device.index if hasattr(device, "index") else device
        device_uuid = str(torch.cuda.get_device_properties(device.index).uuid)
        assert device_uuid == source_uuid, \
            r_str("Source tensor and destination tensor must be on the same device.") + \
            f" Source device UUID: {source_uuid}, destination device UUID: {device_uuid}"
        shared_tensor = torch_ipc_extension.open_ipc_and_get_tensor(
            handle, device, dtype, shape, stride, storage_offset)

    dist.barrier(group=group)
    return shared_tensor

def copy_and_share_tensor_ipc(
    tensor: torch.Tensor = None,
    group: dist.ProcessGroup = None,
    src_rank: int = 0
):
    """Creates a shared tensor from an existing tensor using the C++ IPC extension."""
    rank = dist.get_rank(group=group)
    shared_tensor = None
    
    if rank == src_rank:
        check_tensor_ipc_comptability(tensor)
        
        # Keep reference to original tensor during IPC creation
        original_tensor_ref = tensor
        
        handle, shared_tensor = torch_ipc_extension.copy_tensor_and_get_ipc(tensor)
        device = torch.device("cuda", torch.cuda.current_device())
        device_index = device.index if hasattr(device, "index") else device
        device_uuid = str(torch.cuda.get_device_properties(device_index).uuid)
        meta = (
            handle, tensor.shape, tensor.dtype,
            tensor.stride(), tensor.storage_offset(),
            device_uuid
        )
        object_to_broadcast = [meta]
        dist.broadcast_object_list(object_to_broadcast, src=src_rank, group=group)
        
        # Clear reference after successful sharing
        del original_tensor_ref
    else:
        received_objects = [None]
        dist.broadcast_object_list(received_objects, src=src_rank, group=group)
        handle, shape, dtype, stride, storage_offset, source_uuid = received_objects[0]
        device = torch.device("cuda", torch.cuda.current_device())
        device_index = device.index if hasattr(device, "index") else device
        device_uuid = str(torch.cuda.get_device_properties(device_index).uuid)
        assert device_uuid == source_uuid, \
            r_str("Source tensor and destination tensor must be on the same device.") + \
            f" Source device UUID: {source_uuid}, destination device UUID: {device_uuid}"
        shared_tensor = torch_ipc_extension.open_ipc_and_get_tensor(
            handle, device, dtype, shape, stride, storage_offset)
        
    dist.barrier(group=group)      
    return shared_tensor

def _set_param_or_buffer(model, name, new_tensor):
    """Recursively finds and replaces a parameter or buffer in a model."""
    module_path, _, attr_name = name.rpartition('.')
    target_module = model.get_submodule(module_path)
    
    # Store old tensor reference before replacement
    old_tensor = None
    
    if attr_name in target_module._parameters:
        old_tensor = target_module._parameters[attr_name]
        # CRITICAL: Use nn.Parameter wrapper for parameters
        target_module._parameters[attr_name] = nn.Parameter(new_tensor, requires_grad=old_tensor.requires_grad if old_tensor is not None else True)
    elif attr_name in target_module._buffers:
        old_tensor = target_module._buffers[attr_name]
        target_module._buffers[attr_name] = new_tensor
    else:
        raise AttributeError(f"{name} is not a parameter or buffer in the model.")
    
    # Explicitly delete old tensor reference to help with memory cleanup
    if old_tensor is not None:
        del old_tensor

def verify_no_memory_leaks():
    """Helper function to verify memory usage."""
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()

def test_memory_management(rank, device):
    """Test to verify no memory leaks in IPC tensor operations."""
    if rank == 0:
        print(f"\n[Rank {rank}] Testing memory management...")
    
    initial_allocated, initial_reserved = verify_no_memory_leaks()
    
    # Test 1: Create and share multiple tensors
    tensors = []
    for i in range(5):
        shape = (1000, 1000)
        dtype = torch.float32
        shared_tensor = create_and_share_tensor_ipc(shape, dtype, device, src_rank=0)
        tensors.append(shared_tensor)
    
    mid_allocated, mid_reserved = verify_no_memory_leaks()
    
    # Test 2: Delete references and check memory cleanup
    del tensors
    torch.cuda.synchronize()  # Ensure CUDA operations complete
    
    final_allocated, final_reserved = verify_no_memory_leaks()
    
    if rank == 0:
        print(f"Memory usage - Initial: {initial_allocated/1024**2:.2f}MB, "
              f"Peak: {mid_allocated/1024**2:.2f}MB, "
              f"Final: {final_allocated/1024**2:.2f}MB")
        
        # Allow some tolerance for memory management overhead
        memory_increase = final_allocated - initial_allocated
        if memory_increase > 100 * 1024 * 1024:  # 100MB tolerance
            print(y_str(f"Warning: Potential memory leak detected. Increase: {memory_increase/1024**2:.2f}MB"))
        else:
            
            print(g_str("✓ Memory management test passed"))

class SimpleModel(nn.Module):
    """A basic model for initial testing."""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(20, 5)
        self.register_buffer('my_buffer', torch.randn(1, 5))

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x))) + self.my_buffer
    
def single_tensor_sharing_test(rank, device, shape, dtype, custom_stride, storage_offset):
    original_tensor = None
    shared_tensor_created = None
    shared_tensor_copied = None
    received_tensor = None

    def config_to_str():
        return f"shape {shape}, dtype {dtype}, stride {custom_stride}, offset {storage_offset}"

    try:
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
            shape, dtype, device, src_rank=0
        )
        shared_tensor_copied = copy_and_share_tensor_ipc(
            original_tensor, src_rank=0
        )
        dist.barrier()
        
        # Test 1: Bit-level comparison
        if rank == 0:
            # Send original data for comparison
            dist.broadcast(original_tensor, src=0)
            dist.barrier()
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
            dist.barrier()
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
            
        dist.barrier()
        
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
            dist.barrier()

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

            dist.barrier()
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
        
        dist.barrier()
        
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

                dist.barrier()
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
            
            dist.barrier()
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
        
        dist.barrier()
        
        if rank == 0:
            print(g_str(f"[Rank {rank}] ✓ All tests passed for config " + config_to_str()))
            
    except Exception as e:
        print(r_str(f"[Rank {rank}] ✗ Test failed for config " + config_to_str()))
        raise

def test_single_tensor_sharing(rank, device):
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
        single_tensor_sharing_test(rank, device, shape, dtype, custom_stride, storage_offset)
        dist.barrier()
    if rank == 0:
        print(g_str(f"\n[Rank {rank}] All single tensor sharing tests completed successfully!"))

def main():
    """Main execution function with robust unit tests."""
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA/HIP enabled GPU.")

    dist.init_process_group("gloo")
    
    rank = int(os.environ["RANK"])
    target_device_id = 0
    torch.cuda.set_device(target_device_id)
    device = torch.device("cuda", target_device_id)

    # --- Test 1: Memory Management Test ---
    if rank == 0:
        print("="*50)
        print("RUNNING TEST 1: Memory Management")
        print("="*50)
    
    test_memory_management(rank, device)
    
    dist.barrier()
    if rank == 0:
        print("--- Memory Management Test COMPLETED ---")
    
    # --- Test 2: Single Tensor Sharing ---
    if rank == 0:
        print("="*50)
        print("RUNNING TEST 2: Single Tensor Sharing")
        print("="*50)
    
    test_single_tensor_sharing(rank, device)
    
    dist.barrier()
    if rank == 0:
        print("--- Single Tensor Sharing Test PASSED ---")
    
    # --- Test 3: Simple Model ---
    if rank == 0:
        print("\n" + "="*40)
        print("RUNNING TEST 3: SimpleModel")
        print("="*40)
    
    if rank == 0:
        simple_model = SimpleModel().to(device)
    else:
        with init_empty_weights():
            simple_model = SimpleModel()

    share_model_parameters_ipc(simple_model)
    
    if rank == 0:
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.1)
        test_input = torch.randn(4, 10, device=device)
        output = simple_model(test_input)
        loss = output.sum()
        weight_before = simple_model.layer1.weight.data.clone()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        weight_after = simple_model.layer1.weight.data
        assert not torch.equal(weight_before, weight_after)
        print(y_str("[Rank 0] SimpleModel weights updated on source rank."))
        dist.broadcast(weight_after, src=0)
    else:
        expected_weight = torch.empty_like(simple_model.layer1.weight.data)
        dist.broadcast(expected_weight, src=0)
        assert torch.equal(simple_model.layer1.weight.data, expected_weight)
        print(g_str(f"[Rank {rank}] Verified SimpleModel weight update is visible."))
        
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

    model_id = "deepseek-ai/deepseek-v2-lite"
    model_dtype = torch.float16

    if rank == 0:
        print(f"[Rank 0] Loading {model_id}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=model_dtype, trust_remote_code=True
        ).to(device)
    else:
        print(f"[Rank {rank}] Initializing empty meta model...")
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config, torch_dtype=model_dtype, trust_remote_code=True
            )

    share_model_parameters_ipc(model)
    
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
            r_str(f"Parameter mismatch after sharing! \n") +
            f"Expected:\n{expected_param}\nGot:\n{target_param_on_worker}"
        )
        print(g_str(f"[Rank {rank}] Verified parameter '{target_param_name}' matches rank 0."))
    
    dist.barrier()
    if rank == 0:
        print("--- Parameter sharing verification PASSED ---")

    # --- STAGE 2: Forward Pass (Logits) Verification ---
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    prompt = "Hello, this is a test of the shared model."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    if rank == 0:
        torch.manual_seed(42)
        with torch.no_grad():
            output_before = model(**inputs)
            logits_before = output_before.logits
            print(y_str(f"\n[Rank 0] Logits sum BEFORE update: {logits_before.sum()}"))
        dist.broadcast(logits_before, src=0)
    else:
        torch.manual_seed(42)
        with torch.no_grad():
            output_before = model(**inputs)
            logits_before = output_before.logits
            print(f"\n[Rank {rank}] Logits sum BEFORE update: {logits_before.sum()}")

        expected_logits_before = torch.empty_like(logits_before)
        dist.broadcast(expected_logits_before, src=0)

        local_bytes = logits_before.view(torch.int8)
        expected_bytes = expected_logits_before.view(torch.int8)
        assert torch.equal(local_bytes, expected_bytes), \
            r_str("Initial logits do not match between ranks at the byte level!")
        print(g_str(f"[Rank {rank}] Verified initial logits match rank 0."))

    dist.barrier()

    # --- STAGE 3: Verification after weight update ---
    if rank == 0:
        print(f"\n[Rank 0] Updating weights...")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        output = model(**inputs, labels=inputs["input_ids"])
        loss = output.loss
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        print("[Rank 0] Weights updated successfully.")
        dist.barrier()
        
        torch.manual_seed(42)
        with torch.no_grad():
            output_after = model(**inputs)
            logits_after = output_after.logits
            print(y_str(f"[Rank 0] Logits sum AFTER update:  {logits_after.sum()}"))
        assert not torch.allclose(logits_before, logits_after)
        dist.broadcast(logits_after, src=0)
    else:
        dist.barrier() # Wait for rank 0 to finish updating weights
        torch.manual_seed(42)
        with torch.no_grad():
            output_after = model(**inputs)
            logits_after = output_after.logits
            print(f"\n[Rank {rank}] Logits sum AFTER update:  {logits_after.sum()}")

        expected_logits_after = torch.empty_like(logits_after)
        dist.broadcast(expected_logits_after, src=0)

        local_bytes_after = logits_after.view(torch.int8)
        expected_bytes_after = expected_logits_after.view(torch.int8)
        assert torch.equal(local_bytes_after, expected_bytes_after), \
            r_str("Final logits do not match between ranks at the byte level!")
        print(g_str(f"[Rank {rank}] Verified final logits match rank 0."))

    dist.barrier()
    if rank == 0:
        print("\n--- DeepSeek-V2-Lite Test PASSED ---")
        print("\nAll verification successful on all ranks!")

if __name__ == "__main__":
    if "torch_ipc_extension" in globals():
         main()
