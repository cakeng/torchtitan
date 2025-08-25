import os
import subprocess
import threading
import sys
from datetime import datetime
import random

mbp_size = 10
pp_size = 2
ep_size = 4
fsdp_size = 1
run_profiler = "False"
num_gpus = pp_size * ep_size * fsdp_size

run_id = datetime.now().strftime("%Y%m%d%H%M%S")
#export run_id to env
os.environ["RUN_ID"] = run_id
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "16"
os.environ["CUDA_SCALE_LAUNCH_QUEUES"] = "4x"

def stream_output(process, rank, stream_type):
    """Stream output from a process in real-time"""
    for line in iter(process.stdout.readline if stream_type == "stdout" else process.stderr.readline, ''):
        if line:
            # Prefix each line with rank info for clarity
            prefix = f"[MBP {rank}] "
            if stream_type == "stderr":
                prefix = f"[MBP {rank}-ERR] "
            print(f"{prefix}{line.rstrip()}", flush=True)

# Launch four different training jobs asynchronously
processes = []
for i in range(mbp_size):
    port = 29500 + random.randint(0, 10000)
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        f"--master_port={port}",
        "train_ds_dev.py",
        str(pp_size),
        str(ep_size),
        str(fsdp_size),
        str(mbp_size),
        str(i),
        run_profiler
    ]
    cmd_str = " ".join(cmd)
    
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
    if i < 12:
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
print(f"\nWaiting for all {len(processes)} processes to complete...")
for i, (process, stdout_thread, stderr_thread) in enumerate(processes):
    return_code = process.wait()
    print(f"\n=== MBP Rank {i} (PID {process.pid}) completed with return code {return_code} ===")
    print("=" * 60)

print(f"\nAll processes completed. Run ID: {run_id}")