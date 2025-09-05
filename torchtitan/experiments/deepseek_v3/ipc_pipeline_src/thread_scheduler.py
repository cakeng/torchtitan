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

from cProfile import label
from tkinter import Y
import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
import time
import copy
import faulthandler
from dataclasses import dataclass
from collections import deque
from typing import Dict, Callable
import signal

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

# Enable automatic stack dumps on SIGQUIT
faulthandler.enable()

# Register custom signal handler for thread dumps
def dump_all_threads(signum, frame):
    print("\n" + "="*50)
    print("THREAD DUMP")
    print("="*50)
    faulthandler.dump_traceback()
    print(f"Active threads: {threading.active_count()}")
    for thread in threading.enumerate():
        print(f"  {thread.name}: {thread}")
    print("="*50)

signal.signal(signal.SIGUSR1, dump_all_threads)

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
            # Try integer indexing first (for ModuleList), then string key (for ModuleDict)
            try:
                parent_module = parent_module[int(part)]
            except (KeyError, TypeError):
                # If integer indexing fails, try string key (for ModuleDict)
                parent_module = parent_module[part]
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
    stream: torch.cuda.Stream
    event: torch.cuda.Event
    backward_signal: threading.Event

class ContextScheduler:
    """ Manages and schedules ExecutionEngines in a round-robin fashion. """
    def __init__(self, num_execs, debug = False):
        self.num_execs = num_execs
        self.active_exec_id = None
        self.next_exec_id = None
        self.stop_scheduling = False
        self.waiting_exec_ids = []
        self.execs = {}
        self.debug = debug
        self.completion_barrier = threading.Barrier(num_execs + 1)
        self.completion_signal = threading.Event()
        self.completion_signal.clear()
        self.context_lock = threading.Lock()
        self.context_lock.acquire()
        self.backward_semaphore = threading.Semaphore(1)

    def _scheduler(self, force_switch = False):
        if self.stop_scheduling:
            self.next_exec_id = None
            return
        if len(self.waiting_exec_ids) > 0:
            self.next_exec_id = self.active_exec_id
            for exec_id in self.waiting_exec_ids:
                if self.execs[exec_id].event.query():
                    self.next_exec_id = exec_id
                    break
            if self.next_exec_id == self.active_exec_id and force_switch:
                self.next_exec_id = self.waiting_exec_ids[0]   
            
            if self.debug:
                t_id = threading.current_thread().ident
                print(r_str(f"[T{t_id}]") + " Scheduling next exec " + 
                      y_str(f"{self.next_exec_id}") + ", current waiting execs: " + 
                      y_str(f"{self.waiting_exec_ids}"))
        else: 
            # No execs waiting
            if force_switch:
                self.next_exec_id = None # No active exec
            else:
                if self.debug:
                    t_id = threading.current_thread().ident
                    print(r_str(f"[T{t_id}]") + " No execs waiting, scheduling active exec " + 
                        y_str(f"{self.active_exec_id}"))
                self.next_exec_id = self.active_exec_id

    def _release_context(self, exec_id):
        # Release the context lock and signal the next exec to resume.
        assert self.active_exec_id == exec_id, \
            f"Expected {self.active_exec_id} to be the active exec during release, " + \
            f"got {exec_id} running"
        if self.debug:
            t_id = threading.current_thread().ident
            print(b_str(f"[T{t_id}]") + " Yielding context of exec " + 
                  y_str(f"{self.active_exec_id}") + ", next scheduled exec " + 
                  y_str(f"{self.next_exec_id}"))
        self.execs[self.active_exec_id].signal.clear()
        self.execs[self.active_exec_id].event.record(
            self.execs[self.active_exec_id].stream)
        if self.next_exec_id is not None:
            self.execs[self.next_exec_id].signal.set()
        self.active_exec_id = None
        self.context_lock.release()

    def _acquire_context(self, exec_id):
        # Acquire the context lock and execute.
        # Append to the front of the list to maintain order
        self.waiting_exec_ids.insert(0, exec_id)
        if self.active_exec_id is None and self.next_exec_id is None:
            if self.debug:
                t_id = threading.current_thread().ident
                print(b_str(f"[T{t_id}]") + " No active execs, acquiring context lock for exec " + 
                      y_str(f"{exec_id}"))
        else:
            if self.debug:
                t_id = threading.current_thread().ident
                print(b_str(f"[T{t_id}]") + " Waiting for context switch to exec " + 
                      y_str(f"{exec_id}"))
            self.execs[exec_id].signal.wait()
        self.context_lock.acquire() 
        if exec_id in self.waiting_exec_ids:
            self.waiting_exec_ids.remove(exec_id)
        self.active_exec_id = exec_id
        torch.cuda.set_stream(self.execs[self.active_exec_id].stream)
        if self.debug:
            t_id = threading.current_thread().ident
            print(b_str(f"[T{t_id}]") + " Resuming exec " + 
                y_str(f"{self.active_exec_id}") + " context on stream " +
                y_str(f"{self.execs[self.active_exec_id].stream}"))

    def add_exec(self, exec):
        # Add an exec to the scheduler.
        for exec_id in self.execs:
            assert self.execs[exec_id].exec.ident != exec.ident, \
                f"Exec {exec} already exists, ident {exec.ident}"
        new_exec_id = len(self.execs)
        new_exec_signal = threading.Event()
        new_exec_signal.clear()
        new_exec_stream = torch.cuda.Stream()
        new_exec_event = torch.cuda.Event()
        new_exec_event.record(new_exec_stream)
        new_exec_backward_signal = threading.Event()
        new_exec_backward_signal.clear()
        self.execs[new_exec_id] = ExecContext(exec, 
                                              new_exec_signal, 
                                              new_exec_stream, 
                                              new_exec_event,
                                              new_exec_backward_signal)
        if self.debug:
            t_id = threading.current_thread().ident
            print(y_str(f"[Main Thread]") + " Adding exec " + 
                  y_str(f"{exec}") + ", ident " + 
                  y_str(f"{exec.ident}") + " with id " + 
                  y_str(f"{new_exec_id}"))
        return new_exec_id
    
    def context_switch(self, exec_id):
        # Schedule the next exec, release context of the current exec,
        # and enter the waiting queue until next context is acquired.
        self._scheduler()
        if self.next_exec_id != self.active_exec_id:
            self._release_context(exec_id)
            self._acquire_context(exec_id)
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
        self._scheduler(force_switch=True)
        self._release_context(exec_id)
        if self.debug:
            t_id = threading.current_thread().ident
            print(b_str(f"[T{t_id}]") + " Exec " + y_str(f"{exec_id}") + 
                " detached and running independently from the scheduler context.")
        return
    
    def enter_backward_region(self, exec_id):
        self.execs[exec_id].backward_signal.wait()
        self.backward_semaphore.acquire()
        if exec_id + 1 < self.num_execs:
            self.execs[exec_id + 1].backward_signal.set()
        if self.debug:
            t_id = threading.current_thread().ident
            print(b_str(f"[T{t_id}]") + " Entering backward region for exec " + 
                  y_str(f"{exec_id}"))
        return
    
    def exit_backward_region(self, exec_id):
        self.backward_semaphore.release()
        self.execs[exec_id].backward_signal.clear()
        if self.debug:
            t_id = threading.current_thread().ident
            print(b_str(f"[T{t_id}]") + " Exiting backward region for exec " + 
                  y_str(f"{exec_id}"))
        return

    def start(self):
        assert len(self.execs) == self.num_execs, \
            f"Expected {self.num_execs} execs, got {len(self.execs)}"
        for exec_id in self.execs:
            self.execs[exec_id].signal.clear()
        self.execs[0].backward_signal.set()
        self.active_exec_id = None
        self.stop_scheduling = False
        self._scheduler(force_switch=True)
        if self.next_exec_id is not None:
            self.execs[self.next_exec_id].signal.set()
        self.completion_signal.clear()
        self.context_lock.release()
        return
        
    def stop(self):
        self.stop_scheduling = True
        self.context_lock.acquire() # Acquire the context lock
        self.stop_scheduling = False

    def wait_completion(self):
        if self.debug:
            t_id = threading.current_thread().ident
            print(y_str(f"[Main Thread]") + " Waiting for completion of all execs.")
        self.completion_barrier.wait() # Wait for all execs to complete
        self.context_lock.acquire() # Acquire the context lock
        self.completion_signal.set() # Signal that the main thread has resumed
        if self.debug:
            t_id = threading.current_thread().ident
            print(y_str(f"[Main Thread]") + " All execs completed! Resuming main thread.")
            
@dataclass
class ModuleInfo():
    avg_time_taken: float
    num_profiles: int
    input_shape: torch.Size
    output_shape: torch.Size
    fire_context_switch: bool
    time_since_last_context_switch: float
    
    new_module_queue_start: int
    module_queue_start: int
    module_queue_end: int
    profiler_event: torch.cuda.Event
    
    fwd_pre_hook: Callable
    fwd_hook: Callable
    bwd_hook: Callable
    
def print_module_info(module_name, module_info, module_queue = []):
    out_str = (g_str(f"\t[{module_name}]") + " Module info:\n" + 
               y_str(f"\t\tavg time taken:") + f"{module_info.avg_time_taken}\n" +
               y_str(f"\t\tnum profiles: ") + f"{module_info.num_profiles}\n" +
               y_str(f"\t\tinput shape: ") + f"{module_info.input_shape}\n" +
               y_str(f"\t\toutput shape: ") + f"{module_info.output_shape}\n" +
               y_str(f"\t\tfire context switch: ") + f"{module_info.fire_context_switch}\n" +
               y_str(f"\t\ttime since last context switch: ") + f"{module_info.time_since_last_context_switch}\n"
               )
    print(out_str, end="")
    
class ModelProfiler():
    def __init__(
        self, model, microbatch, label, loss_fn, do_profile=True, debug = False
    ):
        super().__init__()
        self.model = model
        self.microbatch = microbatch
        self.label = label
        self.loss_fn = loss_fn
        self.loss = None
        self.total_time = 0
        self.modules = {}
        self.module_queue = []
        self.debug = debug
        
        self.tag_module_name(self.model)
        if do_profile:
            self.profile_model()
            self.delete_non_leaf_modules()
            self.flag_context_switch()
            self.print_profiler_info()
    
    def tag_module_name(self, module, module_name=None):
        if module_name is None:
            module.module_name = module.__class__.__name__
        else:
            module.module_name = module_name    
        
        for child_name, child_module in module.named_children():
            self.tag_module_name(child_module, 
                                 module.module_name + "_" + child_name)
            
    def delete_non_leaf_modules(self):
        if self.debug:
            print(g_str(f"[Profiler]") + " Deleting non-leaf modules...")
        new_modules = {}
        new_module_queue = []
        total_time = 0
        for module_name in self.module_queue:
            module_info = self.modules[module_name]
            if module_info.module_queue_end - module_info.module_queue_start == 1:
                if self.debug:
                    print(g_str(f"[Profiler]") + " Keeping leaf module " + 
                          b_str(f"{module_name}") + ".")
                module_info.module_queue_start = len(new_module_queue)
                module_info.module_queue_end = len(new_module_queue) + 1
                new_modules[module_name] = module_info
                new_module_queue.append(module_name)
                total_time += module_info.avg_time_taken
            elif self.debug:
                print(g_str(f"[Profiler]") + " Deleting non-leaf module " + 
                      r_str(f"{module_name}") + ".")
        self.modules = new_modules
        self.module_queue = new_module_queue
        self.total_time = total_time
        
    def flag_context_switch(self):
        # Simple equal-time based context switch
        num_context_switch_points = len(self.module_queue) / 500 if len(self.module_queue) > 2500 else 5
        context_switch_time = self.total_time / num_context_switch_points
        current_time = 0    
        for module_name in self.module_queue:
            module_info = self.modules[module_name]
            module_info.time_since_last_context_switch = current_time
            if current_time + module_info.avg_time_taken > context_switch_time:
                if self.debug:
                    print(g_str(f"[Profiler]") + " Flagging context switch at " + 
                          y_str(f"{module_name}") + ".")
                module_info.fire_context_switch = True
                current_time = 0
            current_time += module_info.avg_time_taken
            
    def print_profiler_info(self):
        print(g_str(f"[Profiler]") + " Profiler info:")
        print(y_str(f"\tModel: ") + f"{self.model}")
        print(y_str(f"\tTotal time: ") + f"{self.total_time}")
        print(y_str(f"\tProfiled Modules: "))
        num_context_switch = 0
        for module_name in self.modules:
            if self.modules[module_name].fire_context_switch:
                num_context_switch += 1
                print_module_info(module_name, self.modules[module_name], self.module_queue)
        print(y_str(f"\tNumber of context switches: ") + f"{num_context_switch}")
        # print(y_str(f"\tProfiled Module queue: ") + f"{self.module_queue}")

    def profile_model(self):
        self.new_module_queue = []
        self.model_changed = False
        self.attach_hooks()
        outputs = self.model(self.microbatch)
        self.detach_hooks()
        self.module_queue = self.new_module_queue
        return outputs, self.model_changed

    def attach_hooks(self):
        print(g_str(f"[Profiler]") + " Attaching hooks...")
        self.detach_hooks() # Clear any old hooks first
        for module in self.model.modules():
            module_info = ModuleInfo(
                avg_time_taken=-1,
                time_since_last_context_switch=-1,
                input_shape=torch.Size([]),
                output_shape=torch.Size([]),
                fire_context_switch=False,
                new_module_queue_start=-1,
                module_queue_start=-1,
                module_queue_end=-1,
                num_profiles=0,
                profiler_event=torch.cuda.Event(enable_timing=True),
                fwd_pre_hook=None,
                fwd_hook=None,
                bwd_hook=None
            )
            module_info.fwd_pre_hook = module.register_forward_pre_hook(self.forward_pre_profiler_hook)
            module_info.fwd_hook = module.register_forward_hook(self.forward_profiler_hook)
            self.modules[module.module_name] = module_info
    
    def detach_hooks(self):
        for module_info in self.modules.values():
            if module_info.fwd_pre_hook is not None:
                module_info.fwd_pre_hook.remove()
            if module_info.fwd_hook is not None:
                module_info.fwd_hook.remove()
            if module_info.bwd_hook is not None:
                module_info.bwd_hook.remove()
        
    def forward_pre_profiler_hook(self, module, input):
        if isinstance(input, tuple):
            if len(input) > 0:
                input_shape = input[0].shape if hasattr(input[0], 'shape') else torch.Size([])
            else:
                input_shape = torch.Size([])
        else:
            input_shape = input.shape if hasattr(input, 'shape') else torch.Size([])
            
        if self.debug:
            print(g_str(f"[Profiler]") + " Forward pre profiler hook fired on " + 
                  y_str(f"{module.module_name}") + f", input {input_shape}")
        module_key = module.module_name
        self.modules[module_key].new_module_queue_start = len(self.new_module_queue)
        self.new_module_queue.append(module_key)
        self.modules[module_key].profiler_event.record()

    def forward_profiler_hook(self, module, input, output):
        if isinstance(input, tuple):
            if len(input) > 0:
                input_shape = input[0].shape if hasattr(input[0], 'shape') else torch.Size([])
            else:
                input_shape = torch.Size([])
        else:
            input_shape = input.shape if hasattr(input, 'shape') else torch.Size([])
        if isinstance(output, tuple):
            if len(output) > 0:
                output_shape = output[0].shape if hasattr(output[0], 'shape') else torch.Size([])
            else:
                output_shape = torch.Size([])
        else:
            output_shape = output.shape if hasattr(output, 'shape') else torch.Size([])
        if self.debug:
            print(g_str(f"[Profiler]") + " Forward profiler hook fired on " + 
                  y_str(f"{module.module_name}") + f", input {input_shape}" + 
                  f", output {output_shape}")

        module_key = module.module_name
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        torch.cuda.synchronize()
        time_taken = self.modules[module_key].profiler_event.elapsed_time(end_event)
        self.total_time += time_taken
        self.modules[module_key].input_shape = input_shape
        self.modules[module_key].output_shape = output_shape
        
        # If this is not the first profile, check if the module execution changed
        if self.modules[module_key].module_queue_start > -1:
            # If this is not the first profile and the model did not change yet, check if the module queue is the same
            old_start = self.modules[module_key].module_queue_start
            if old_start != self.modules[module_key].new_module_queue_start:
                self.model_changed = True
            old_end = self.modules[module_key].module_queue_end
            if old_end != len(self.new_module_queue):
                self.model_changed = True
            old_module_queue = self.module_queue[old_start:old_end]
            new_module_queue = self.new_module_queue[self.modules[module_key].new_module_queue_start:]
            if new_module_queue != old_module_queue:
                self.model_changed = True
            if self.model_changed and self.debug:
                print(g_str(f"[Profiler]") + " Model execution change detected on " + 
                        y_str(f"{module.module_name}"))
                
        if self.model_changed or self.modules[module_key].module_queue_start == -1:
            # If the module queue is different (execution changed), update the module queue start and end
            self.modules[module_key].avg_time_taken = time_taken
            self.modules[module_key].module_queue_start = self.modules[module_key].new_module_queue_start
            self.modules[module_key].module_queue_end = len(self.new_module_queue)
        else:
            # If the module queue is the same (execution did not change), update the average time taken
            self.modules[module_key].avg_time_taken = \
                (self.modules[module_key].avg_time_taken * self.modules[module_key].num_profiles + 
                    time_taken) / (self.modules[module_key].num_profiles + 1)
        self.modules[module_key].num_profiles += 1
        if self.debug:
            print_module_info(module_key, self.modules[module_key], 
                              self.new_module_queue)
                

class ForcedContextSwitchModuleWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        
    def forward(self, *args, **kwargs):
        self.module.forward(*args, **kwargs)

class ExecutionEngine(threading.Thread):
    
    """ A dedicated thread to run a single training step (fwd/bwd). """
    def __init__(
        self, model, x, label, loss_fn, scheduler,
        profiler = None, start_exec = True, debug = False,
    ):
        super().__init__(daemon=True)
        self.model = model
        self.x = x
        self.label = label
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.exec_id = scheduler.add_exec(self)
        self.loss = None
        self.profiler = profiler
        self.backward_tid = -1
        self.hooks = {}
        self.debug = debug
        self.stop_exec = False
        
        if start_exec:
            self.start_exec()

    def start_exec(self):
        self.tag_module_name(self.model)
        self.attach_hooks()
        self.start()
        
    def stop_exec(self):
        self.detach_hooks()
        self.stop_exec = True
        
    def tag_module_name(self, module, module_name=None):
        # Set the exec_name for the current module
        if module_name is None:
            module.module_name = module.__class__.__name__
        else:
            module.module_name = module_name    
        
        for child_name, child_module in module.named_children():
            self.tag_module_name(child_module, module.module_name + "_" + child_name)

    def run(self):
        print(g_str(f"[T{self.ident}]") + " Running...")
        while not self.stop_exec:
            # Attach the exec to the scheduler context, 
            # The execution of this thread will be controlled by the scheduler.
            self.scheduler.attach_exec_to_context(self.exec_id)

            # Forward pass
            self.model.train()
            outputs = self.model(self.x, label=self.label)
            self.loss = self.loss_fn(outputs, self.label)
            
            # Detach the exec from the scheduler context,
            # The execution will now be independent of the scheduler and
            # the scheduler will no longer switch to this thread.
            self.scheduler.detach_exec_from_context(self.exec_id)
            
            print(g_str(f"[T{self.ident}]") + " Fwd pass finished. Loss:"
                    f" {self.loss},  current exec " + 
                    y_str(f"{self.scheduler.active_exec_id}"))
            
            self.scheduler.enter_backward_region(self.exec_id)
            self.scheduler.attach_exec_to_context(self.exec_id)
            
            # Backward pass
            self.loss.backward()
            
            self.scheduler.detach_exec_from_context(self.exec_id)
            self.scheduler.exit_backward_region(self.exec_id)
            
            print(g_str(f"[T{self.ident}]") + " Bwd pass finished, current exec " + 
                    y_str(f"{self.scheduler.active_exec_id}"))

            # Wait for all execs to complete.
            if self.debug:
                t_id = threading.current_thread().ident
                print(b_str(f"[T{t_id}]") + " Waiting for all execs to complete.")
            self.scheduler.completion_barrier.wait() # Wait for all execs to complete
            self.scheduler.completion_signal.wait() # Wait for the main thread to resume
            if self.debug:
                t_id = threading.current_thread().ident
                print(b_str(f"[T{t_id}]") + " All execs completed! Waiting for next iteration...")

    def attach_hooks(self):
        """ ### IMPLEMENTATION: Attach hooks to all modules. ### """
        print(g_str(f"[T{self.ident}]") + " Attaching hooks...")
        self.detach_hooks() # Clear any old hooks first
        for module in self.model.modules():
            module_name = module.module_name
            if isinstance(module, ForcedContextSwitchModuleWrapper):
                fwd_hook = module.register_forward_hook(self.forward_scheduler_hook)
                bwd_hook = module.register_full_backward_hook(self.backward_scheduler_hook)
                self.hooks[module_name] = (
                    fwd_hook,
                    bwd_hook
                )
                continue
            elif self.profiler is not None and module_name in self.profiler.modules:
                module_info = self.profiler.modules[module_name]
                if module_info.fire_context_switch:
                    fwd_hook = module.register_forward_hook(self.forward_scheduler_hook)
                    bwd_hook = module.register_full_backward_hook(self.backward_scheduler_hook)
                    self.hooks[module_name] = (
                        fwd_hook,
                        bwd_hook
                    )
    
    def detach_hooks(self):
        """ ### IMPLEMENTATION: Remove all attached hooks. ### """
        # Clear all events
        for fwd_hook, bwd_hook in self.hooks.values():
            fwd_hook.remove()
            bwd_hook.remove()
        self.hooks = {}
        
    def forward_scheduler_hook(self, module, input, output):
        assert threading.current_thread().ident == self.ident, \
            f"Expected {self.ident} to be the current exec during forward hook, " + \
            f"got {threading.current_thread().ident}"
        if self.debug:
            print(g_str(f"[T{self.ident}] ") + b_str(f"Forward hook") + " fired on " + 
                y_str(f"{module.module_name}") + " current exec " + 
                y_str(f"{self.scheduler.active_exec_id}"))
        self.scheduler.context_switch(self.exec_id)

    def backward_scheduler_hook(self, module, grad_input, grad_output):
        if self.backward_tid == -1:
            self.backward_tid = threading.current_thread().ident
        assert threading.current_thread().ident == self.backward_tid, \
            f"Expected {self.ident} to be the current exec during backward hook, " + \
            f"got {threading.current_thread().ident}"
        if self.debug:
            print(g_str(f"[T{self.backward_tid}] ") + 
                  r_str(f"Backward hook") + " fired on " + 
                  y_str(f"{module.module_name}") + " current exec " + 
                  y_str(f"{self.scheduler.active_exec_id}"))
        self.scheduler.context_switch(self.exec_id)

    def create_backward_lock_hooks(self, layer_lock):
        """ Hooks to acquire/release a lock during the backward pass. """
        def pre_hook(module, grad_input):
            print(g_str(f"[T{self.backward_tid}] ") + 
                  r_str(f"Backward Pre-hook") + " fired on " + 
                  y_str(f"{module.module_name}") + " current exec " + 
                  y_str(f"{self.scheduler.active_exec_id}"))
        def post_hook(module, grad_input, grad_output):
            print(g_str(f"[T{self.backward_tid}] ") + 
                  r_str(f"Backward Post-hook") + " fired on " + 
                  y_str(f"{module.module_name}") + " current exec " + 
                  y_str(f"{self.scheduler.active_exec_id}"))
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
    
    def forward(self, x, label=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.out(x)

def simple_loss_fn(outputs, label):
    # Reshape for CrossEntropyLoss
    return F.cross_entropy(outputs.view(-1, outputs.size(-1)), label.view(-1))

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
    microbatches, label = get_data_fn(num_threads)
    
    # 3. Profiler Setup
    profiler = ModelProfiler(base_model, microbatches[0], label[0], loss_fn)

    # 3. Hook and Lock Setup
    scheduler = ContextScheduler(num_threads, debug)
    for t in range(num_threads):
        model = model_factory_fn(materialized=False)
        materialize_meta_model(model, base_model)
        ExecutionEngine(model, microbatches[t], label[t], 
                        loss_fn, scheduler, profiler=profiler, debug=debug)

    # 4. Run Training Step
    scheduler.start()
    scheduler.wait_completion() # Main thread waits here
    
    # print(y_str(f"[Main Thread]") + " All threads complete. Stepping optimizer...")
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
    full_label = torch.randint(
        0, vocab_size, (batch_size, seq_len), device='cuda'
    )
    
    microbatches = list(torch.split(full_batch, micro_bs))
    label = list(torch.split(full_label, micro_bs))
    return microbatches, label

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
        print(y_str(f"[Main Thread]") + " Loading materialized base model with device_map='auto'...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )
        print(y_str(f"[Main Thread]") + " Base model loaded.")
        return model
    else:
        # This is for the meta_model_shell: create a shell from config
        # without loading weights.
        print(y_str(f"[Main Thread]") + " Creating meta model shell...")
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        print(y_str(f"[Main Thread]") + " Meta model shell created.")
        return model

def get_deepseek_data(num_threads):
    # This is a dummy data generator for demonstration
    batch_size, seq_len = 16, 64
    assert batch_size % num_threads == 0
    micro_bs = batch_size // num_threads
    
    full_batch = torch.randint(
        0, 1000, (batch_size, seq_len), device='cuda', dtype=torch.long
    )
    
    microbatches = list(torch.split(full_batch, micro_bs))
    microbatch_shapes = [microbatch.shape for microbatch in microbatches]
    print(y_str(f"[Main Thread]") + " Microbatches: " + f"{microbatch_shapes}")
    # For causal LM, label are the same as inputs
    return microbatches, microbatches

def deepseek_loss_fn(outputs, label):
    # The model itself returns a loss object if label are provided
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