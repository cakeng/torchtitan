import os
import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights

# Import the C++ extension after it has been built.
import torch_ipc_extension

def share_model_parameters_ipc(
    model: nn.Module,
    group: dist.ProcessGroup = None,
    src_rank: int = 0
):
    """Shares model parameters from a source rank using the C++ IPC extension."""
    # Add a barrier to ensure all processes enter the function together.
    dist.barrier(group=group)
    
    rank = dist.get_rank(group=group)

    if rank == src_rank:
        handles_and_meta = []
        tensors_to_share = list(model.named_parameters()) + list(
            model.named_buffers()
        )
        for name, tensor in tensors_to_share:
            handle = torch_ipc_extension.get_ipc_handle(tensor)
            meta = (
                name, handle, tensor.shape, tensor.dtype,
                tensor.stride(), tensor.storage_offset(),
                # CRITICAL FIX: Use .storage() to be consistent with the C++
                # extension, which also uses the typed storage.
                tensor.untyped_storage().nbytes() 
            )
            handles_and_meta.append(meta)
        
        object_to_broadcast = [handles_and_meta]
        dist.broadcast_object_list(object_to_broadcast, src=src_rank, group=group)
    else:
        received_objects = [None]
        dist.broadcast_object_list(received_objects, src=src_rank, group=group)
        handles_and_meta = received_objects[0]
        
        for name, handle, shape, dtype, stride, storage_offset, storage_nbytes in handles_and_meta:
            device = torch.device("cuda", torch.cuda.current_device())
            shared_tensor = torch_ipc_extension.open_ipc_handle(
                handle, device, dtype, shape, stride, storage_offset,
                storage_nbytes
            )
            _set_param_or_buffer(model, name, shared_tensor)
            
    # Add a barrier to ensure all processes have completed sharing
    # before exiting the function.
    dist.barrier(group=group)


def _set_param_or_buffer(model, name, new_tensor):
    """Recursively finds and replaces a parameter or buffer in a model."""
    module_path, _, attr_name = name.rpartition('.')
    target_module = model.get_submodule(module_path)
    
    if attr_name in target_module._parameters:
        target_module._parameters[attr_name] = nn.Parameter(new_tensor)
    elif attr_name in target_module._buffers:
        target_module._buffers[attr_name] = new_tensor
    else:
        raise AttributeError(f"{name} is not a parameter or buffer in the model.")

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

def main():
    """Main execution function with robust unit tests."""
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA/HIP enabled GPU.")

    dist.init_process_group("gloo")
    
    rank = int(os.environ["RANK"])
    target_device_id = 0
    torch.cuda.set_device(target_device_id)
    device = torch.device("cuda", target_device_id)
    
    # --- Test 1: Simple Model ---
    if rank == 0:
        print("="*40)
        print("RUNNING TEST 1: SimpleModel")
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
        print("[Rank 0] SimpleModel weights updated on source rank.")
        dist.broadcast(weight_after, src=0)
    else:
        expected_weight = torch.empty_like(simple_model.layer1.weight.data)
        dist.broadcast(expected_weight, src=0)
        assert torch.equal(simple_model.layer1.weight.data, expected_weight)
        print(f"[Rank {rank}] Verified SimpleModel weight update is visible.")
        
    dist.barrier()
    if rank == 0:
        print("--- SimpleModel Test PASSED ---")

    # --- Test 2: DeepSeek-V2-Lite Model ---
    if rank == 0:
        print("\n" + "="*40)
        print("RUNNING TEST 2: DeepSeek-V2-Lite")
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
            f"Parameter mismatch after sharing! \n"
            f"Expected:\n{expected_param}\nGot:\n{target_param_on_worker}"
        )
        print(f"[Rank {rank}] Verified parameter '{target_param_name}' matches rank 0.")
    
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
            print(f"\n[Rank 0] Logits sum BEFORE update: {logits_before.sum()}")
        dist.broadcast(logits_before, src=0)
    else:
        config = model.config
        logits_shape = (inputs['input_ids'].shape[0], inputs['input_ids'].shape[1], config.vocab_size)
        expected_logits_before = torch.empty(logits_shape, dtype=model_dtype, device=device)
        dist.broadcast(expected_logits_before, src=0)
        
        torch.manual_seed(42)
        with torch.no_grad():
            output_before = model(**inputs)
            logits_before = output_before.logits
            print(f"\n[Rank {rank}] Logits sum BEFORE update: {logits_before.sum()}")

        # CRITICAL FIX: Use .view(torch.int8) for a robust byte-level comparison.
        local_bytes = logits_before.view(torch.int8)
        expected_bytes = expected_logits_before.view(torch.int8)
        assert torch.equal(local_bytes, expected_bytes), \
            "Initial logits do not match between ranks at the byte level!"
        print(f"[Rank {rank}] Verified initial logits match rank 0.")

    dist.barrier()

    # --- STAGE 3: Verification after weight update ---
    if rank == 0:
        print(f"\n[Rank 0] Updating weights...")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        output = model(**inputs, labels=inputs["input_ids"])
        loss = output.loss
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        print("[Rank 0] Weights updated successfully.")
        
        torch.manual_seed(42)
        with torch.no_grad():
            output_after = model(**inputs)
            logits_after = output_after.logits
            print(f"[Rank 0] Logits sum AFTER update:  {logits_after.sum()}")
        assert not torch.allclose(logits_before, logits_after)
        dist.broadcast(logits_after, src=0)
    else:
        config = model.config
        logits_shape = (inputs['input_ids'].shape[0], inputs['input_ids'].shape[1], config.vocab_size)
        expected_logits_after = torch.empty(logits_shape, dtype=model_dtype, device=device)
        dist.broadcast(expected_logits_after, src=0)
        
        torch.manual_seed(42)
        with torch.no_grad():
            output_after = model(**inputs)
            logits_after = output_after.logits
            print(f"\n[Rank {rank}] Logits sum AFTER update:  {logits_after.sum()}")

        # CRITICAL FIX: Use .view(torch.int8) for a robust byte-level comparison.
        local_bytes_after = logits_after.view(torch.int8)
        expected_bytes_after = expected_logits_after.view(torch.int8)
        assert torch.equal(local_bytes_after, expected_bytes_after), \
            "Final logits do not match between ranks at the byte level!"
        print(f"[Rank {rank}] Verified final logits match rank 0.")

    dist.barrier()
    if rank == 0:
        print("\n--- DeepSeek-V2-Lite Test PASSED ---")
        print("\nAll verification successful on all ranks!")

if __name__ == "__main__":
    if "torch_ipc_extension" in globals():
         main()
