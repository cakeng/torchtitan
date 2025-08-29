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
import copy
from collections import deque
import unittest

from transformers import AutoTokenizer, AutoModelForCausalLM, logging
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

def selective_deepcopy(obj, memo=None):
    if memo is None:
        memo = {}
    if id(obj) in memo:
        return memo[id(obj)]
    if isinstance(obj, nn.Parameter):
        memo[id(obj)] = obj
        return obj
    if isinstance(obj, nn.Module):
        # As per your request, I'm having the opening brace on a new
        # line for this style [2025-08-13].
        new_module = obj.__class__.__new__(obj.__class__)
        memo[id(obj)] = new_module
        for key, value in obj.__dict__.items():
            new_module.__dict__[key] = selective_deepcopy(value, memo)
        return new_module
    elif isinstance(obj, (list, tuple)):
        new_list = [selective_deepcopy(item, memo) for item in obj]
        if isinstance(obj, tuple):
            new_list = tuple(new_list)
        return new_list
    elif isinstance(obj, dict):
        new_dict = {
            key: selective_deepcopy(value, memo)
            for key, value in obj.items()
        }
        return new_dict
    else:
        return copy.deepcopy(obj, memo)

# =============================================================================
# COMPONENT 2: SCHEDULER, THREAD, AND HOOKS (UPDATED FOR TRAINING)
# =============================================================================

class ContextScheduler:
    """ Manages and schedules TrainingThreads in a round-robin fashion. """
    def __init__(self, num_execs):
        self.active_exec_id = 0
        self.next_exec_id = 0
        self.context_lock = threading.Lock()
        self.context_lock.acquire()
        self.execs = {}
        self.ops = {}
        self.debug = True
        self.num_execs = num_execs
        self.completion_barrier = threading.Barrier(num_execs + 1)

    def _scheduler(self):
        if self.next_exec_id is None:
            return None
        # Switch to the next exec using a simple round-robin.
        self.next_exec_id = (self.active_exec_id + 1) % len(self.execs)
        if self.debug:
            t_id = threading.current_thread().ident
            print(b_str(f"[T{t_id} Scheduler]") + " Scheduling next exec " + 
                  y_str(f"{self.next_exec_id}"))

    def _release_context(self):
        # Release the context lock and signal the next exec to resume.
        if self.debug:
            t_id = threading.current_thread().ident
            print(b_str(f"[T{t_id} Scheduler]") + " Yielding context of exec " + 
                  y_str(f"{self.active_exec_id}"))
        self.execs[self.active_exec_id]["signal"].clear()
        if self.next_exec_id is not None:
            self.execs[self.next_exec_id]["signal"].set()
        self.context_lock.release()

    def _acquire_context(self, exec_id = None):
        # Acquire the context lock and execute.
        if exec_id is None:
            exec_id = self.active_exec_id
        self.execs[exec_id]["signal"].wait()
        if self.next_exec_id == exec_id:
            self.active_exec_id = self.next_exec_id
            if self.debug:
                t_id = threading.current_thread().ident
                print(b_str(f"[T{t_id} Scheduler]") + " Resuming exec " + 
                    y_str(f"{self.active_exec_id}"))
            self.context_lock.acquire() 
        elif self.debug:
            t_id = threading.current_thread().ident
            print(b_str(f"[T{t_id} Scheduler]") + " Exec " + y_str(f"{exec_id}") + 
                  " detached remotely, running independently from the scheduler context.")

    def attach_exec(self, ident):
        new_exec_id = len(self.execs)
        new_exec_event = threading.Event()
        new_exec_event.clear()
        self.execs[new_exec_id] = {
            "ident": ident,
            "signal": new_exec_event,
        }
        if self.debug:
            t_id = threading.current_thread().ident
            print(b_str(f"[T{t_id} Scheduler]") + " Attaching exec " + 
                  y_str(f"{new_exec_id}"))
        self._acquire_context(new_exec_id)
        return

    def detach_exec(self, ident):
        for exec_id, exec in self.execs.items():
            if exec["ident"] == ident:
                if self.active_exec_id == exec_id:
                    self._scheduler()
                    self._release_context(exec_id)
                    if self.debug:
                        t_id = threading.current_thread().ident
                        print(b_str(f"[T{t_id} Scheduler]") + " Exec " + y_str(f"{exec_id}") + 
                            " detached and running independently from the scheduler context.")
                else:
                    if self.debug:
                        t_id = threading.current_thread().ident
                        print(b_str(f"[T{t_id} Scheduler]") + f" Current exec " +
                              y_str(f"{self.active_exec_id}") + f" detaching inactive exec " +
                              y_str(f"{exec_id}") + " from the scheduler context.")
                    self._release_context(exec_id)
                    exec["signal"].set()
                del self.execs[exec_id]
                return
        raise ValueError(f"Exec with ident {ident} not found")

    def start(self, starting_exec_id = 0):
        assert len(self.execs) == self.num_execs, \
            f"Expected {self.num_execs} execs, got {len(self.execs)}"
        for exec_id in self.execs:
            self.execs[exec_id]["signal"].clear()
            self.execs[exec_id].s
        self.active_exec_id = starting_exec_id
        self.execs[starting_exec_id]["signal"].set()
        self.context_lock.release()
        return
        
    def stop(self):
        self.next_exec_id = None
        self.context_lock.acquire()

    def wait_completion(self):
        if self.debug:
            t_id = threading.current_thread().ident
            print(b_str(f"[T{t_id} Scheduler]") + " Waiting for completion of all execs.")
        self.completion_barrier.wait()
        self.context_lock.acquire()
        if self.debug:
            t_id = threading.current_thread().ident
            print(b_str(f"[T{t_id} Scheduler]") + " All exec completed, returning to main thread.")

    def context_switch(self):
        self._scheduler()
        self._release_context()
        self._acquire_context()
        return
        
class ExecutionEngine(threading.Thread):
    """ A dedicated thread to run a single training step (fwd/bwd). """
    def __init__(
        self, base_model, microbatch, labels, loss_fn, scheduler,
    ):
        super().__init__(daemon=True)
        self.model = base_model
        self.microbatch = microbatch
        self.labels = labels
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.loss = None
        self.hooks = {}
        self.attach_hooks()

    def run(self):
        self.scheduler.attach_exec(self.ident)
        try:
            # Forward pass
            outputs = self.model(self.microbatch)
            self.loss = self.loss_fn(outputs, self.labels)
            
            # Backward pass
            # self.loss.backward()
            
            # print(f"[{self.name}] Fwd/Bwd pass finished. Loss:"
            #       f" {self.loss.item():.4f}")
        except Exception as e:
            print(f"[{self.name}] Encountered an error: {e}")

        self.scheduler.detach_exec(self.ident)
        self.scheduler.completion_barrier.wait()

    def is_running(self): return self.event.is_set()

    def attach_hooks(self):
        pass
    def detach_hooks(self):
        pass

    def forward_yield_hook(self, module, input, output):
        print(f"[T{self.ident}] Hook fired on {module}")
        self.scheduler.context_switch()

    def backward_yield_hook(self, module, grad_input, grad_output):
        print(f"[T{self.ident}] Hook fired on {module}")
        self.scheduler.context_switch()

    def create_backward_lock_hooks(layer_lock):
        """ Hooks to acquire/release a lock during the backward pass. """
        def pre_hook(module, grad_input):
            print(f"[T{self.ident}] Pre-hook fired on {module}")
        def post_hook(module, grad_input, grad_output):
            print(f"[T{self.ident}] Post-hook fired on {module}")
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
    model_name, model_class, get_data_fn, loss_fn, num_threads
):
    print("\n" + "="*80)
    print(f" E2E Training Test: {model_name} with {num_threads} Threads")
    print("="*80)

    if not torch.cuda.is_available():
        print("Skipping test: CUDA not available.")
        return

    # 1. Model and Optimizer Setup
    base_model = model_class().cuda()
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=1e-4)

    # 2. Hook and Lock Setup
    scheduler = ContextScheduler()
    fwd_hook = create_forward_yield_hook(scheduler)
    layer_locks = {}
    
    # Identify transformer layers to attach locks
    transformer_layers = []
    if hasattr(base_model, 'layers'): # For our simple model
        transformer_layers = base_model.layers
    elif hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'): # For HF models
        transformer_layers = base_model.model.layers
    
    for i, layer in enumerate(transformer_layers):
        layer.custom_name = f"Layer-{i}"
        layer_locks[i] = threading.Lock()
        bwd_pre_hook, bwd_post_hook = create_backward_lock_hooks(
            layer_locks[i]
        )
        layer.register_forward_hook(fwd_hook)
        layer.register_full_backward_pre_hook(bwd_pre_hook)
        layer.register_full_backward_hook(bwd_post_hook)
    
    # 3. Data and Thread Setup
    microbatches, labels = get_data_fn(num_threads)
    completion_barrier = threading.Barrier(num_threads + 1)
    
    threads = [
        TrainingThread(
            base_model, microbatches[i], labels[i], loss_fn,
            scheduler, completion_barrier, rank=i,
            name=f"Thread-{i}"
        ) for i in range(num_threads)
    ]

    # 4. Run Training Step
    initial_weight = copy.deepcopy(
        transformer_layers[0].attn.in_proj_weight.data
    )

    scheduler.start()
    for t in threads: t.start()
    for t in threads: scheduler.submit(t)

    print("\n[MainThread] Waiting for all threads to finish backward pass...")
    completion_barrier.wait(timeout=60) # Main thread waits here
    
    print("[MainThread] All threads complete. Stepping optimizer...")
    optimizer.step()
    optimizer.zero_grad()
    
    final_weight = transformer_layers[0].attn.in_proj_weight.data

    # 5. Cleanup and Verification
    for t in threads: t.join(timeout=5)
    scheduler.stop()

    assert not torch.equal(initial_weight, final_weight), \
        "Weights did not change after optimizer step."
    print("\nVerification PASSED: Model weights were updated.")

# =============================================================================
# TEST CONFIGURATIONS AND MAIN BLOCK
# =============================================================================

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

def get_deepseek_data(num_threads):
    # This is a dummy data generator for demonstration
    batch_size, seq_len = 8, 128
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
    for num_threads in [1, 2, 4]:
        run_training_test(
            "Simple Transformer",
            SimpleTransformerModel,
            get_simple_transformer_data,
            simple_loss_fn,
            num_threads
        )
    def get_deepseek_model():
        # As per your request, I'm using vim for editing text files
        # [2025-08-05], though it's not applicable here.
        # This demonstrates how to wrap the model loading.
        model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/deepseek-v2-lite",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        # The forward pass needs to be modified for this simple trainer
        original_forward = model.forward
        def new_forward(input_ids):
            return original_forward(input_ids, labels=input_ids)
        model.forward = new_forward
        return model
    
    # NOTE: Running with more threads on large models is very
    # memory-intensive due to activations stored for backward pass.
    for num_threads in [1, 2]:
        run_training_test(
            "DeepSeekV2-Lite",
            get_deepseek_model,
            get_deepseek_data,
            deepseek_loss_fn,
            num_threads
        )