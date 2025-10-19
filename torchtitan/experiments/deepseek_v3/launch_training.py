import os
from pdb import run
import subprocess
import threading
import sys
from datetime import datetime
import random
import zipfile
from ipc_pipeline_src.sync_traces import merge_chrome_traces_with_barriers

run_type = sys.argv[1] if len(sys.argv) > 1 else ""
mbp_size = 4
pp_size = 2
ep_size = 2
fsdp_size = 1
batch_size = 16
seq_len = 128
num_hidden_layers = 12
num_steps = 3
run_profiler = "True"

if run_type == "1f1b":
    train_script = "train_ds_dev_1f1b.py"
elif run_type == "sched":
    train_script = "train_ds_dev_sched.py"
elif run_type == "mbp":
    train_script = "train_ds_dev.py"
else:
    raise ValueError(f"Invalid run type: {run_type}")

num_gpus = pp_size * ep_size * fsdp_size

run_id = datetime.now().strftime("%Y%m%d%H%M%S")
#export run_id to env
os.environ["RUN_ID"] = run_id
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(0 + i) for i in range(num_gpus))
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "16"
os.environ["CUDA_SCALE_LAUNCH_QUEUES"] = "4x"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "3"
os.environ["NCCL_LAUNCH_ORDER_IMPLICIT"] = "1"

# # Merlin network env
# os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
# os.environ["GLOO_SOCKET_IFNAME"] = "eth0"
# # NCCL Env
# os.environ["NCCL_CGA_CLUSTER_SIZE"] = "0"
# os.environ["NCCL_IB_PCI_RELAXED_ORDERING"] = "1"
# os.environ["NCCL_LAUNCH_ORDER_IMPLICIT"] = "1"
# # export NCCL_IB_QPS_PER_CONNECTION=2
# # export NCCL_IB_SPLIT_DATA_ON_QPS=0
# os.environ["NCCL_NCHANNELS_PER_NET_PEER"] = "8"
# # export NCCL_MIN_NCHANNELS=16
# # export NCCL_PXN_DISABLE=1
# os.environ["NCCL_P2P_NET_CHUNKSIZE"] = "2097152"

def stream_output(process, rank, stream_type):
    """Stream output from a process in real-time"""
    for line in iter(process.stdout.readline if stream_type == "stdout" else process.stderr.readline, ''):
        if line:
            # Prefix each line with rank info for clarity
            prefix = f""
            if run_type == "mbp":
                prefix = f"[MBP {rank}] "
                if stream_type == "stderr":
                    prefix = f"[MBP {rank}-ERR] "
            else:
                if stream_type == "stderr":
                    prefix = f"[ERR] "
            print(f"{prefix}{line.rstrip()}", flush=True)

# Launch four different training jobs asynchronously
processes = []
num_process_groups = mbp_size if run_type == "mbp" else 1
for i in range(num_process_groups):
    port = 29500 + random.randint(0, 30000)
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        f"--master_port={port}",
        f"--master_addr=127.0.0.1",
        f"--node_rank=0",                
        f"--nnodes=1",
        train_script,
        str(pp_size),
        str(ep_size),
        str(fsdp_size),
        str(mbp_size),
        str(i),
        str(batch_size),
        str(seq_len),
        str(num_hidden_layers),
        str(num_steps),
        run_profiler
    ]
    cmd_str = " ".join(cmd)
    
    env = os.environ.copy()
    env.update({
        "NCCL_SOCKET_IFNAME": "lo",
        "NCCL_P2P_DISABLE": "1", 
        "NCCL_IB_DISABLE": "1",
        "NCCL_NET_GDR_LEVEL": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port)
    })
    
    # Launch process asynchronously with real-time output streaming
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Start output streaming threads only for first 2 instances
    if i < num_process_groups:
        stdout_thread = threading.Thread(target=stream_output, args=(process, i, "stdout"))
        stderr_thread = threading.Thread(target=stream_output, args=(process, i, "stderr"))
        
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        
        stdout_thread.start()
        stderr_thread.start()
        
        processes.append((process, stdout_thread, stderr_thread))
    else:
        # For instances beyond the first 3, just append the process without streaming threads
        processes.append((process, None, None))
    print(f"Launched MBP rank {i} with PID {process.pid}, cmd: {cmd_str}")

# Wait for all processes to complete
print(f"\nWaiting for all {num_process_groups} processes to complete...")
for i, (process, stdout_thread, stderr_thread) in enumerate(processes):
    return_code = process.wait()
    print(f"\n=== MBP Rank {i} (PID {process.pid}) completed with return code {return_code} ===")
    print("=" * 60)

log_name = f"run_{run_type}_{run_id}_mbp_{mbp_size}_pp_{pp_size}_ep_{ep_size}_fsdp_{fsdp_size}_layers_{num_hidden_layers}_bs_{batch_size}_seqlen_{seq_len}_steps_{num_steps}"
log_dir = f"./tensorboard_traces/{log_name}"
os.makedirs(log_dir, exist_ok=True)
    
merge_chrome_traces_with_barriers(
    trace_dir=log_dir,
    output_file=f"{log_dir}/merged_trace.json",
    barrier_events=["BARRIER:EXEC_START", "BARRIER:EXEC_END"],
    trace_names=[f"trace_0_2.json", f"trace_0_3.json"],
    whole_trace=False,
)

# compress the merged trace
zip_name = f"{log_dir}/{log_name}_merged_trace.zip" 
with zipfile.ZipFile(zip_name, "w",
                    compression=zipfile.ZIP_DEFLATED, 
                    compresslevel=9) as zipf:
    print(f"Compressing trace to {zip_name}")
    zipf.write(f"{log_dir}/merged_trace.json", f"{log_name}_merged_trace.json")
print(f"Compressed trace to {zip_name}")

print(f"\nAll processes completed. Run ID: {run_id}")