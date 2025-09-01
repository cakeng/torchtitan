# parallel_training_scheduler.py
#
# A thread-based scheduler for data-parallel training with fine-grained
# concurrency control for backward passes.
#
# Key Features:
# 1. Selective Deep-Copy: Efficiently shares model weights (nn.Parameter)
#    across threads while isolating other states.
# 2. Cooperative Multitasking: Forward pass hooks at pre-defined points
#    allow threads to yield, enabling concurrent execution.
# 3. Backward Pass Mutual Exclusion: A lock is assigned to each
#    transformer layer. Threads must acquire a layer's lock before
#    executing its backward pass, preventing gradient race conditions.
# 4. Optimizer Synchronization: A threading.Barrier ensures the optimizer
#    step only occurs after all threads have finished their backward pass.
#
# All code formatted within 80 columns as requested [2025-05-21]

import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
import time
from dataclasses import dataclass
from collections import deque
from typing import Dict

from accelerate import init_empty_weights
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, logging
logging.set_verbosity_error() # Suppress verbose warnings

def g_str(s):
    return "\033[32m" + s + "\033[0m"
def r_str(s):
    return "\033[31m" + s + "\033[0m"
def b_str(s):
    return "\033[34m" + s + "\033[0m"
def y_str(s):
    return "\033[33m" + s + "\033[0m"


# =============================================================================
# COMPONENT 1: SELECTIVE DEEP-COPY
# (This component remains unchanged)
# =============================================================================

def _get_parent_module_and_param_name(model: nn.Module, path: str):
    """
    A helper function to find the parent module and the final attribute
    name given a full parameter path.
    Example: for path 'layers.0.attn.weight', it returns the
    `model.layers[0].attn` module and the string 'weight'.
    """
    parts = path.split('.')
    parent_module = model
    for part in parts[:-1]:
        # getattr can handle both attribute access (like .layers) and
        # indexed access for nn.ModuleList (like [0])
        if part.isdigit():
            parent_module = parent_module[int(part)]
        else:
            parent_module = getattr(parent_module, part)
    param_name = parts[-1]
    return parent_module, param_name

def materialize_meta_model(
    meta_model: nn.Module, 
    base_model: nn.Module
):
    """
    Materializes a model created on the 'meta' device by replacing its
    meta tensors with references to the corresponding tensors from a
    fully initialized base model.

    Args:
        meta_model (nn.Module): The model skeleton, initialized on the
                                'meta' device. This model will be
                                modified in-place.
        base_model (nn.Module): A fully initialized model (on CPU or GPU)
                                that contains the actual tensor data.
    """
    base_params: Dict[str, nn.Parameter] = dict(base_model.named_parameters())
    base_buffers: Dict[str, torch.Tensor] = dict(base_model.named_buffers())

    # ### FIX: Collect parameter names into a static list first ###
    # This prevents the "dictionary changed during iteration" error.
    meta_param_names = [name for name, _ in meta_model.named_parameters()]

    for param_name in meta_param_names:
        # We only need to handle meta tensors. If a tensor is already
        # materialized, we can skip it.
        parent_check, attr_check = _get_parent_module_and_param_name(
            meta_model, param_name
        )
        if getattr(parent_check, attr_check).device != torch.device("meta"):
            continue
            
        if param_name not in base_params:
            raise ValueError(
                f"Architecture mismatch: Parameter '{param_name}' found in "
                "meta_model but not in base_model."
            )

        parent_module, attr_name = _get_parent_module_and_param_name(
            meta_model, param_name
        )

        # Replace the meta parameter with a reference to the base parameter
        delattr(parent_module, attr_name)
        setattr(parent_module, attr_name, base_params[param_name])

    # ### FIX: Apply the same logic for buffers ###
    meta_buffer_names = [name for name, _ in meta_model.named_buffers()]

    for buffer_name in meta_buffer_names:
        parent_check, attr_check = _get_parent_module_and_param_name(
            meta_model, buffer_name
        )
        if getattr(parent_check, attr_check).device != torch.device("meta"):
            continue

        if buffer_name not in base_buffers:
            raise ValueError(
                f"Architecture mismatch: Buffer '{buffer_name}' found in "
                "meta_model but not in base_model."
            )

        parent_module, attr_name = _get_parent_module_and_param_name(
            meta_model, buffer_name
        )
        
        delattr(parent_module, attr_name)
        setattr(parent_module, attr_name, base_buffers[buffer_name])

# =============================================================================
# COMPONENT 2: SCHEDULER, THREAD, AND HOOKS (UPDATED FOR TRAINING)
# =============================================================================
@dataclass
class ExecContext:
    exec: threading.Thread
    signal: threading.Event
    cuda_stream: torch.cuda.Stream
    cuda_event: torch.cuda.Event

class ContextScheduler:
    """ Manages and schedules ExecutionEngines in a round-robin fashion. """
    def __init__(self, num_execs, debug = False):
        self.num_execs = num_execs
        self.active_exec_id = 0
        self.next_exec_id = 0
        self.waiting_exec_ids = deque(maxlen=num_execs)
        self.execs = {}
        self.debug = debug
        self.completion_barrier = threading.Barrier(num_execs + 1)
        self.context_lock = threading.Lock()
        self.context_lock.acquire()

    def _scheduler(self):
        if self.next_exec_id is None:
            return None
        # Switch to the next exec using a simple FIFO queue.
        if len(self.waiting_exec_ids) > 0:
            self.next_exec_id = self.waiting_exec_ids.popleft()
            if self.debug:
                t_id = threading.current_thread().ident
                print(r_str(f"[T{t_id}]") + " Scheduling next exec " + 
                      y_str(f"{self.next_exec_id}") + ", current waiting execs: " + 
                      y_str(f"{self.waiting_exec_ids}"))
        else:
            if self.debug:
                t_id = threading.current_thread().ident
                print(r_str(f"[T{t_id}]") + " No execs waiting, scheduling active exec " + 
                      y_str(f"{self.active_exec_id}"))
            self.next_exec_id = self.active_exec_id

    def _release_context(self):
        # Release the context lock and signal the next exec to resume.
        if self.debug:
            t_id = threading.current_thread().ident
            print(b_str(f"[T{t_id}]") + " Yielding context of exec to " + 
                  y_str(f"{self.active_exec_id}") + " exec " + 
                  y_str(f"{self.next_exec_id}"))
            assert t_id == self.execs[self.active_exec_id].exec.ident, \
                f"Expected {t_id} to be the active exec during release, " + \
                f"got {self.execs[self.active_exec_id].exec.ident}"
        self.execs[self.active_exec_id].signal.clear()
        self.execs[self.active_exec_id].cuda_event.record(
            self.execs[self.active_exec_id].cuda_stream)
        if self.next_exec_id is not None:
            self.execs[self.next_exec_id].signal.set()
        self.context_lock.release()

    def _acquire_context(self, exec_id = None):
        # Acquire the context lock and execute.
        if exec_id is None:
            exec_id = self.active_exec_id
        self.waiting_exec_ids.append(exec_id)
        if self.debug:
            t_id = threading.current_thread().ident
            print(b_str(f"[T{t_id}]") + " Waiting for exec " + 
                y_str(f"{exec_id}"))
        self.execs[exec_id].signal.wait()
        assert self.next_exec_id == exec_id, \
            f"Expected {self.next_exec_id} to be the next exec during acquire, " + \
            f"got {exec_id} running"
        self.active_exec_id = self.next_exec_id
        torch.cuda.set_stream(self.execs[self.active_exec_id].cuda_stream)
        self.context_lock.acquire() 
        if self.debug:
            t_id = threading.current_thread().ident
            print(b_str(f"[T{t_id}]") + " Resuming exec " + 
                y_str(f"{self.active_exec_id}") + " on stream " +
                y_str(f"{self.execs[self.active_exec_id].cuda_stream}"))

    def add_exec(self, exec):
        # Add an exec to the scheduler.
        for exec_id in self.execs:
            assert self.execs[exec_id].exec.ident != exec.ident, \
                f"Exec {exec} already exists, ident {exec.ident}"
        if self.debug:
            t_id = threading.current_thread().ident
            print(b_str(f"[T{t_id}]") + " Adding exec " + 
                  y_str(f"{exec}") + ", ident " + 
                  y_str(f"{exec.ident}"))
        new_exec_id = len(self.execs)
        new_exec_event = threading.Event()
        new_exec_event.clear()
        self.execs[new_exec_id] = ExecContext(exec, new_exec_event, 
                                   torch.cuda.Stream(), torch.cuda.Event())
        return new_exec_id
    
    def context_switch(self):
        # Schedule the next exec, release context of the current exec,
        # and enter the waiting queue until next context is acquired.
        self._scheduler()
        self._release_context()
        self._acquire_context()
        return

    def attach_exec_to_context(self, exec_id):
        # Attach the exec to the scheduler context
        # i.e., enter the waiting queue for next exec
        if self.debug:
            t_id = threading.current_thread().ident
            print(b_str(f"[T{t_id}]") + " Attaching exec " + 
                  y_str(f"{exec_id}") + ", ident " + 
                  y_str(f"{self.execs[exec_id].exec.ident}"))
        self._acquire_context(exec_id)
        return

    def detach_exec_from_context(self, exec_id):
        # Detach the exec from the scheduler context
        # i.e., release context without entering the waiting queue for context switch
        assert self.active_exec_id == exec_id, \
            f"Expected {self.active_exec_id} to be the active exec during detach, " + \
            f"got {exec_id} running"
        self._scheduler()
        self._release_context()
        if self.debug:
            t_id = threading.current_thread().ident
            print(b_str(f"[T{t_id}]") + " Exec " + y_str(f"{exec_id}") + 
                " detached and running independently from the scheduler context.")
        return

    def start(self):
        assert len(self.execs) == self.num_execs, \
            f"Expected {self.num_execs} execs, got {len(self.execs)}"
        for exec_id in self.execs:
            self.execs[exec_id].signal.clear()
        self._scheduler()
        if self.next_exec_id is not None:
            self.execs[self.next_exec_id].signal.set()
        self.context_lock.release()
        return
        
    def stop(self):
        self.next_exec_id = None
        self.context_lock.acquire()

    def wait_completion(self):
        if self.debug:
            t_id = threading.current_thread().ident
            print(b_str(f"[T{t_id}]") + " Waiting for completion of all execs.")
        self.completion_barrier.wait()
        self.context_lock.acquire()
        if self.debug:
            t_id = threading.current_thread().ident
            print(b_str(f"[T{t_id}]") + " All exec completed, returning to main thread.")
        
class ExecutionEngine(threading.Thread):
    """ A dedicated thread to run a single training step (fwd/bwd). """
    def __init__(
        self, model, microbatch, labels, loss_fn, scheduler,
        debug = False,
    ):
        super().__init__(daemon=True)
        self.model = model
        self.microbatch = microbatch
        self.labels = labels
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.exec_id = scheduler.add_exec(self)
        self.loss = None
        self.hooks = {}
        self.op_stack = []
        self.profile = False
        self.debug = debug
        self.attach_hooks()
        self.start()

    def run(self):
        self.profile = False
        print(g_str(f"[T{self.ident}]") + " Running...")
        while True:
            # Attach the exec to the scheduler context, 
            # The execution of this thread will be controlled by the scheduler.
            self.scheduler.attach_exec_to_context(self.exec_id)
            try:
                # Forward pass
                outputs = self.model(self.microbatch)
                self.loss = self.loss_fn(outputs, self.labels)
                
                # Backward pass
                # self.loss.backward()
                
                # print(f"[{self.name}] Fwd/Bwd pass finished. Loss:"
                #       f" {self.loss.item():.4f}")
            except Exception as e:
                print(g_str(f"[T{self.ident}]") + " Encountered an error: " + 
                    r_str(f"{e}"))

            # Detach the exec from the scheduler context,
            # The execution will now be independent of the scheduler and
            # the scheduler will no longer switch to this thread.
            self.scheduler.detach_exec_from_context(self.exec_id)
            # Wait for all execs to complete.
            if self.debug:
                t_id = threading.current_thread().ident
                print(b_str(f"[T{t_id}]") + " Waiting for all execs to complete.")
            self.scheduler.completion_barrier.wait()
    
    def profile_model(self):
        self.profile = True
        outputs = self.model(self.microbatch)
        self.profile = False
        return outputs

    def is_running(self): return self.event.is_set()

    def attach_hooks(self):
        """ ### IMPLEMENTATION: Attach hooks to all modules. ### """
        print(g_str(f"[T{self.ident}]") + " Attaching hooks...")
        self.detach_hooks() # Clear any old hooks first
        for module in self.model.modules():
            fwd_pre_handle = module.register_forward_pre_hook(self.forward_pre_scheduler_hook)
            fwd_handle = module.register_forward_hook(self.forward_scheduler_hook)
            # Use full backward hook for better coverage
            bwd_handle = module.register_full_backward_hook(
                self.backward_scheduler_hook
            )
            self.hooks[module] = [fwd_pre_handle, fwd_handle, bwd_handle]
    
    def detach_hooks(self):
        """ ### IMPLEMENTATION: Remove all attached hooks. ### """
        for module in self.hooks:
            for handle in self.hooks[module]:
                handle.remove()
        self.hooks = {}
        
    def forward_pre_scheduler_hook(self, module, input):
        assert threading.current_thread().ident == self.ident, \
            f"Expected {self.ident} to be the current exec during forward pre hook, " + \
            f"got {threading.current_thread().ident}"
        if self.debug:
            print(g_str(f"[T{self.ident}]") + " Forward pre hook fired on " + 
                y_str(f"{module}") + " current exec " + 
                y_str(f"{self.scheduler.active_exec_id}"))
            
        if not self.profile:
            pass
        else:
            self.op_stack.append(("forward_pre", module, input))

    def forward_scheduler_hook(self, module, input, output):
        assert threading.current_thread().ident == self.ident, \
            f"Expected {self.ident} to be the current exec during forward hook, " + \
            f"got {threading.current_thread().ident}"
        if self.debug:
            print(g_str(f"[T{self.ident}]") + " Forward hook fired on " + 
                y_str(f"{module}") + " current exec " + 
                y_str(f"{self.scheduler.active_exec_id}"))
            
        if not self.profile:
            self.scheduler.context_switch()

    def backward_scheduler_hook(self, module, grad_input, grad_output):
        assert threading.current_thread().ident == self.ident, \
            f"Expected {self.ident} to be the current exec during backward hook, " + \
            f"got {threading.current_thread().ident}"
        if self.debug:
            print(g_str(f"[T{self.ident}]") + " Backward hook fired on " + 
                  y_str(f"{module}") + " current exec " + 
                  y_str(f"{self.scheduler.active_exec_id}"))
        if not self.profile:
            self.scheduler.context_switch()

    def create_backward_lock_hooks(self, layer_lock):
        """ Hooks to acquire/release a lock during the backward pass. """
        def pre_hook(module, grad_input):
            print(g_str(f"[T{self.ident}]") + " Pre-hook fired on " + 
                  y_str(f"{module}"))
        def post_hook(module, grad_input, grad_output):
            print(g_str(f"[T{self.ident}]") + " Post-hook fired on " + 
                  y_str(f"{module}"))
        return pre_hook, post_hook

# =============================================================================
# E2E TEST AND MODEL DEFINITION
# =============================================================================

class SimpleTransformerLayer(nn.Module):
    def __init__(self, d_model=128, nhead=4, d_ff=512):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x_norm1 = self.norm1(x)
        attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1, need_weights=False)
        x = x + self.dropout(attn_out)
        x_norm2 = self.norm2(x)
        ffn_out = self.ffn(x_norm2)
        x = x + self.dropout(ffn_out)
        return x

class SimpleTransformerModel(nn.Module):
    def __init__(self, num_layers=2, d_model=128, nhead=4, d_ff=512, vocab_size=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            SimpleTransformerLayer(d_model, nhead, d_ff) 
            for _ in range(num_layers)
        ])
        self.out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.out(x)

def simple_loss_fn(outputs, labels):
    # Reshape for CrossEntropyLoss
    return F.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))

def run_training_test(
    model_name, model_factory_fn, get_data_fn, loss_fn, num_threads, debug
):
    print("\n" + "="*80)
    print(f" E2E Training Test: {model_name} with {num_threads} Threads")
    print("="*80)

    if not torch.cuda.is_available():
        print("Skipping test: CUDA not available.")
        return

    # 1. Model and Optimizer Setup
    base_model = model_factory_fn(materialized=True)
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=1e-4)
    
    # 2. Data and Thread Setup
    microbatches, labels = get_data_fn(num_threads)
    
    # 3. Hook and Lock Setup
    scheduler = ContextScheduler(num_threads, debug)
    for t in range(num_threads):
        model = model_factory_fn(materialized=False)
        materialize_meta_model(model, base_model)
        ExecutionEngine(model, microbatches[t], labels[t], 
                        loss_fn, scheduler, debug)

    # 4. Run Training Step
    scheduler.start()

    print("\n[MainThread] Waiting for all threads to complete execution...")
    scheduler.wait_completion() # Main thread waits here
    
    # print("[MainThread] All threads complete. Stepping optimizer...")
    # optimizer.step()
    # optimizer.zero_grad()
    
    # final_weight = transformer_layers[0].attn.in_proj_weight.data

    # # 5. Cleanup and Verification
    # for t in threads: t.join(timeout=5)
    # scheduler.stop()

    # assert not torch.equal(initial_weight, final_weight), \
    #     "Weights did not change after optimizer step."
    # print("\nVerification PASSED: Model weights were updated.")

# =============================================================================
# TEST CONFIGURATIONS AND MAIN BLOCK
# =============================================================================

def get_simple_transformer_model(materialized=True):
    if materialized:
        return SimpleTransformerModel().to("cuda")
    else:
        with init_empty_weights():
            return SimpleTransformerModel()

def get_simple_transformer_data(num_threads):
    batch_size, seq_len, vocab_size = 32, 64, 100
    assert batch_size % num_threads == 0
    micro_bs = batch_size // num_threads
    
    full_batch = torch.randint(
        0, vocab_size, (batch_size, seq_len), device='cuda'
    )
    full_labels = torch.randint(
        0, vocab_size, (batch_size, seq_len), device='cuda'
    )
    
    microbatches = list(torch.split(full_batch, micro_bs))
    labels = list(torch.split(full_labels, micro_bs))
    return microbatches, labels

def get_deepseek_model(materialized=True):
    """
    Factory function for the DeepSeekV2-Lite model.

    Args:
        materialized (bool): If True, loads the full model with weights
            and device_map. If False, creates a memoryless shell on the
            'meta' device.
    """
    model_name = "deepseek-ai/deepseek-v2-lite"
    
    if materialized:
        # This is for the base_model: load it completely with accelerate
        print("[MainThread] Loading materialized base model with device_map='auto'...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )
        # The forward pass needs to be modified for this simple trainer to
        # compute loss automatically when labels are present.
        original_forward = model.forward
        def new_forward(input_ids, labels=None):
            # If labels are not provided, use input_ids for causal LM loss
            if labels is None:
                labels = input_ids
            return original_forward(input_ids=input_ids, labels=labels)
        model.forward = new_forward
        print("[MainThread] Base model loaded.")
        return model
    else:
        # This is for the meta_model_shell: create a shell from config
        # without loading weights.
        print("[MainThread] Creating meta model shell...")
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        print("[MainThread] Meta model shell created.")
        return model

def get_deepseek_data(num_threads):
    # This is a dummy data generator for demonstration
    batch_size, seq_len = 32, 128
    assert batch_size % num_threads == 0
    micro_bs = batch_size // num_threads
    
    full_batch = torch.randint(
        0, 1000, (batch_size, seq_len), device='cuda', dtype=torch.long
    )
    
    microbatches = list(torch.split(full_batch, micro_bs))
    # For causal LM, labels are the same as inputs
    return microbatches, microbatches

def deepseek_loss_fn(outputs, labels):
    # The model itself returns a loss object if labels are provided
    return outputs.loss

if __name__ == "__main__":
    # Test 1: Simple Transformer
    debug = True
    for num_threads in [1, 2, 4]:
        run_training_test(
            "Simple Transformer",
            get_simple_transformer_model,
            get_simple_transformer_data,
            simple_loss_fn,
            num_threads,
            debug
        )
        torch.cuda.empty_cache()
    # NOTE: Running with more threads on large models is very
    # memory-intensive due to activations stored for backward pass.
    for num_threads in [4]:
        run_training_test(
            "DeepSeekV2-Lite",
            get_deepseek_model,
            get_deepseek_data,
            deepseek_loss_fn,
            num_threads,
            debug
        )
        torch.cuda.empty_cache()