import os
import subprocess
import threading
import sys
from datetime import datetime
import random

mbp_size = 2
pp_size = 2
ep_size = 2
fsdp_size = 2

num_gpus = pp_size * ep_size * fsdp_size

run_id = datetime.now().strftime("%Y%m%d%H%M%S")
#export run_id to env
os.environ["RUN_ID"] = run_id

def stream_output(process, rank, stream_type):
    """Stream output from a process in real-time"""
    for line in iter(process.stdout.readline if stream_type == "stdout" else process.stderr.readline, ''):
        if line:
            # Prefix each line with rank info for clarity
            prefix = f"[MBP-{rank}] "
            if stream_type == "stderr":
                prefix = f"[MBP-{rank}-ERR] "
            print(f"{prefix}{line.rstrip()}", flush=True)

# Launch four different training jobs asynchronously
processes = []
for i in range(mbp_size):
    port = 29500 + random.randint(0, 1000)
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        f"--master_port={port}",
        "train_ds_dev.py",
        str(pp_size),
        str(ep_size),
        str(fsdp_size),
        str(mbp_size),
        str(i)
    ]
    
    # Launch process asynchronously with real-time output streaming
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Start output streaming threads
    stdout_thread = threading.Thread(target=stream_output, args=(process, i, "stdout"))
    stderr_thread = threading.Thread(target=stream_output, args=(process, i, "stderr"))
    
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    
    stdout_thread.start()
    stderr_thread.start()
    
    processes.append((process, stdout_thread, stderr_thread))
    print(f"Launched MBP rank {i} with PID {process.pid}, cmd: {cmd}")

# Wait for all processes to complete
print(f"\nWaiting for all {len(processes)} processes to complete...")
for i, (process, stdout_thread, stderr_thread) in enumerate(processes):
    return_code = process.wait()
    print(f"\n=== MBP Rank {i} (PID {process.pid}) completed with return code {return_code} ===")
    print("=" * 60)

print(f"\nAll processes completed. Run ID: {run_id}")