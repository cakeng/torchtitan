import json
import glob
import os
from typing import Dict, List, Tuple
import time

def debug_trace_events(traces: Dict):
    """Debug function to see what events are actually in the traces."""
    print("\n=== DEBUG: All Event Names ===")
    
    for rank, trace_data in traces.items():
        print(f"\nRank {rank} events:")
        if 'traceEvents' not in trace_data:
            print("  No traceEvents found")
            continue
        
        # Collect all unique event names
        event_names = set()
        event_categories = set()
        
        for event in trace_data['traceEvents']:
            if 'name' in event:
                event_names.add(event['name'])
            if 'cat' in event:
                event_categories.add(event['cat'])
        
        print(f"  Event names ({len(event_names)}): {sorted(list(event_names))}")
        print(f"  Event categories ({len(event_categories)}): {sorted(list(event_categories))}")
        
        # Show events with specific patterns
        barrier_like_events = [e for e in event_names if 'barrier' in e.lower() or 'sync' in e.lower() or 'wait' in e.lower()]
        if barrier_like_events:
            print(f"  Barrier-like events: {barrier_like_events}")

def merge_chrome_traces_with_barriers(trace_dir: str, output_file: str, 
                                      barrier_events: List[str] = None, 
                                      trace_names: List[str] = None ):
    """
    Merge multiple Chrome trace files with barrier synchronization.
    
    Args:
        trace_dir: Directory containing trace files
        output_file: Output merged trace file
        barrier_events: List of event names that represent barriers (e.g., ['barrier', 'sync', 'wait'])
    """
    trace_files = []
    if trace_names is None or len(trace_names) == 0:
        trace_files = glob.glob(os.path.join(trace_dir, "*.json"))
    else:
        for trace_name in trace_names:
            if trace_name.endswith(".json"):
                trace_files.append(os.path.join(trace_dir, trace_name))
            else:
                trace_files.append(os.path.join(trace_dir, f"{trace_name}.json"))
    
    if not trace_files:
        print("No trace files found")
        return
    
    # Default barrier event names
    if barrier_events is None:
        barrier_events = []
    
    print(f"Found {len(trace_files)} trace files: {trace_files}")
    
    # Load all traces
    traces = {}
    for trace_file in trace_files:
        # Extract both MBP rank (x) and distributed rank (y) from filename
        filename = os.path.basename(trace_file)
        if filename.startswith('trace_'):
            parts = filename.split('_')
            mbp_rank = parts[1]  # x in trace_x_y.json
            dist_rank = parts[2].split('.')[0]  # y in trace_x_y.json
            rank = f"{mbp_rank}_{dist_rank}"  # Create composite key like "1_0"
        else:
            # Fallback for other filename formats
            rank = os.path.basename(trace_file).split('_')[1].split('.')[0]
        
        with open(trace_file, 'r') as f:
            traces[rank] = json.load(f)
        print(f"Loaded trace for rank {rank} (MBP {mbp_rank}, Dist {dist_rank})")
    
    # Find barrier synchronization points
    barrier_times = find_barrier_synchronization_points(traces, barrier_events)
    
    if not barrier_times:
        print("No barrier events found, merging without synchronization")
        merge_simple(traces, output_file)
        return
    
    # Synchronize traces based on barriers
    synchronized_traces = synchronize_traces_at_barriers(traces, barrier_times)
    
    # Merge synchronized traces
    merge_synchronized_traces(synchronized_traces, output_file)
    
    print(f"Merged {len(trace_files)} synchronized traces into {output_file}")

def find_barrier_synchronization_points(traces: Dict, barrier_events: List[str]) -> List[Tuple[str, float]]:
    """Find barrier events across all traces and their timestamps."""
    barrier_times = []
    
    print(f"Looking for barrier events: {barrier_events}")
    
    for rank, trace_data in traces.items():
        if 'traceEvents' not in trace_data:
            print(f"Rank {rank}: No traceEvents found")
            continue
        
        print(f"Rank {rank}: Found {len(trace_data['traceEvents'])} events")
        
        # Look for barrier events - ONLY CPU events to avoid duplicates
        for event in trace_data['traceEvents']:
            event_name = event.get('name', '').lower()
            event_cat = event.get('cat', '').lower()
            
            # Only consider CPU user_annotation events to avoid GPU duplicates
            if event_cat != 'user_annotation':
                continue
                
            # Debug: Print events that might be barriers
            if any(barrier.lower() in event_name for barrier in barrier_events):
                print(f"Rank {rank}: Found potential barrier event: {event}")
            
            is_barrier = any(barrier.lower() in event_name for barrier in barrier_events)
            
            if is_barrier and 'ts' in event:
                barrier_times.append((rank, event['ts'], event.get('name', 'Unknown')))
                print(f"Rank {rank}: Confirmed barrier event: {event.get('name', 'Unknown')}")
    
    print(f"Total barrier events found: {len(barrier_times)}")
    for rank, ts, name in barrier_times:
        print(f"  Rank {rank}: {name} at {ts}")
    
    # Group barriers by approximate time (within a small window)
    grouped_barriers = group_barriers_by_time(barrier_times)
    
    print(f"Found {len(grouped_barriers)} barrier synchronization points")
    for i, barriers in enumerate(grouped_barriers):
        print(f"  Barrier {i}: {[b[2] for b in barriers]}")
    
    return grouped_barriers

def group_barriers_by_time(barrier_times: List[Tuple], time_window: float = 1000.0):
    """Group barriers that occur within a time window (microseconds)."""
    if not barrier_times:
        return []
    
    # Sort by timestamp
    barrier_times.sort(key=lambda x: x[1])
    
    grouped = []
    current_group = [barrier_times[0]]
    
    for barrier in barrier_times[1:]:
        if barrier[1] - current_group[-1][1] <= time_window:
            current_group.append(barrier)
        else:
            grouped.append(current_group)
            current_group = [barrier]
    
    grouped.append(current_group)
    return grouped

def synchronize_traces_at_barriers(traces: Dict, barrier_groups: List) -> Dict:
    """Synchronize traces by aligning them at barrier points."""
    synchronized_traces = {}
    
    for rank, trace_data in traces.items():
        synchronized_traces[rank] = {
            'traceEvents': [],
            'metadata': trace_data.get('metadata', {})
        }
        
        if 'traceEvents' not in trace_data:
            continue
        
        # Find the first barrier for this rank
        first_barrier_time = None
        for barrier_group in barrier_groups:
            for barrier_rank, barrier_time, _ in barrier_group:
                if barrier_rank == rank:
                    first_barrier_time = barrier_time
                    break
            if first_barrier_time is not None:
                break
        
        if first_barrier_time is None:
            # No barriers found, use original trace
            synchronized_traces[rank]['traceEvents'] = trace_data['traceEvents']
            continue
        
        # Adjust timestamps relative to first barrier
        for event in trace_data['traceEvents']:
            if 'ts' in event:
                # Adjust timestamp so first barrier occurs at time 0
                adjusted_event = event.copy()
                adjusted_event['ts'] = event['ts'] - first_barrier_time
                synchronized_traces[rank]['traceEvents'].append(adjusted_event)
            else:
                synchronized_traces[rank]['traceEvents'].append(event)
    
    return synchronized_traces

def merge_synchronized_traces(synchronized_traces: Dict, output_file: str):
    """Merge synchronized traces with Perfetto-compatible process separation."""
    merged_trace = {
        'traceEvents': [],
        'metadata': {
            'merged_from': list(synchronized_traces.keys()),
            'synchronization': 'barrier-based',
            'timestamp': time.time()
        }
    }
    
    def get_ranks(rank_str):
        try:
            if '_' in rank_str:
                parts = rank_str.split('_')
                mbp_rank = int(parts[0])
                dist_rank = int(parts[1])
                return mbp_rank, dist_rank
            else:
                mbp_rank = int(rank_str)
                dist_rank = 0
                return mbp_rank, dist_rank
        except:
            return 0, 0
    
    # Sort by MBP rank first, then by distributed rank
    def sort_key(rank_str):
        mbp_rank, dist_rank = get_ranks(rank_str)
        return (mbp_rank, dist_rank)
    
    sorted_ranks = sorted(synchronized_traces.keys(), key=sort_key, reverse=True)
    print(f"Sorted ranks by (MBP, dist) order: {sorted_ranks}")
    
    # Assign process IDs with smaller, Perfetto-compatible ranges
    for rank in sorted_ranks:
        trace_data = synchronized_traces[rank]
        mbp_rank, dist_rank = get_ranks(rank)

        # Use smaller PID ranges: CPU starts at 1, GPU starts at 100
        pid = 10000 + (99 - mbp_rank) * 100 + (99 - dist_rank)
                
        print(f"MBP{mbp_rank}-D{dist_rank}: PID {pid}")
        
        if 'traceEvents' not in trace_data:
            continue
        
        for event in trace_data['traceEvents']:
            # event['pid'] = pid
            event['name'] = f"[Rank {mbp_rank}-{dist_rank}] {event['name']}"
            merged_trace['traceEvents'].append(event)
    
    with open(output_file, 'w') as f:
        json.dump(merged_trace, f, indent=2)
    
def merge_synchronized_traces_old(synchronized_traces: Dict, output_file: str):
    """Merge synchronized traces into a single file."""
    merged_trace = {
        'traceEvents': [],
        'metadata': {
            'merged_from': list(synchronized_traces.keys()),
            'synchronization': 'barrier-based',
            'timestamp': time.time()
        }
    }
    
    # Add rank information to event names for clarity
    for rank, trace_data in synchronized_traces.items():
        for event in trace_data['traceEvents']:
            # Add rank prefix to event names for identification
            if 'name' in event:
                event['name'] = f"[Rank {rank}] {event['name']}"
            else:
                event['name'] = f"[Rank {rank}] Event"
            
            merged_trace['traceEvents'].append(event)
    
    # Save merged trace
    with open(output_file, 'w') as f:
        json.dump(merged_trace, f, indent=2)

def merge_simple(traces: Dict, output_file: str):
    """Simple merge without synchronization (fallback)."""
    merged_trace = {
        'traceEvents': [],
        'metadata': {
            'merged_from': list(traces.keys()),
            'synchronization': 'none',
            'timestamp': time.time()
        }
    }
    
    for rank, trace_data in traces.items():
        if 'traceEvents' in trace_data:
            for event in trace_data['traceEvents']:
                if 'name' in event:
                    event['name'] = f"[Rank {rank}] {event['name']}"
                else:
                    event['name'] = f"[Rank {rank}] Event"
                merged_trace['traceEvents'].append(event)
    
    with open(output_file, 'w') as f:
        json.dump(merged_trace, f, indent=2)

# Usage example
if __name__ == "__main__":
    # Customize barrier event names based on your application
    barrier_events = []
    trace_names = []
    
    merge_chrome_traces_with_barriers(
        trace_dir="./tensorboard_traces",
        output_file="./merged_synchronized_trace.json",
        barrier_events=barrier_events,
        trace_names=trace_names
    )