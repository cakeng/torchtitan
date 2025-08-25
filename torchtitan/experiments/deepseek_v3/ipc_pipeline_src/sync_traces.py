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
                                      trace_names: List[str] = None,
                                      whole_trace: bool = False):
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
    barrier_deltas, rank_final_times = find_barrier_sync_deltas(traces, barrier_events, whole_trace)
    
    if not barrier_deltas:
        print("No barrier events found, merging without synchronization")
        merge_simple(traces, output_file)
        return
    
    # Synchronize traces based on barriers
    sync_and_merge_traces(traces, barrier_deltas, rank_final_times, output_file, whole_trace)
    
    
    print(f"Merged {len(trace_files)} synchronized traces into {output_file}")

def find_barrier_sync_deltas(traces: Dict, barrier_events: List[str], whole_trace: bool = False) -> List[Tuple[str, float]]:
    """Find barrier events across all traces and their timestamps."""
    barrier_times = []
    earliest_times = {}
    ranks = []
    event_names = []

    for rank, trace_data in traces.items():
        if 'traceEvents' not in trace_data:
            print(f"Rank {rank}: No traceEvents found")
            continue

        if rank not in ranks:
            ranks.append(rank)
        
        # Look for barrier events - ONLY CPU events to avoid duplicates
        for event in trace_data['traceEvents']:
            event_name = event.get('name', '').lower()
            event_cat = event.get('cat', '').lower()
            if rank not in earliest_times:
                earliest_times[rank] = event['ts']
            else:
                earliest_times[rank] = min(earliest_times[rank], event['ts'])
            
            # Only consider CPU user_annotation events to avoid GPU duplicates
            if event_cat != 'user_annotation':
                continue

            is_barrier = any(barrier.lower() in event_name for barrier in barrier_events)
            
            if is_barrier and 'ts' in event:
                if event.get('name', 'Unknown') not in event_names:
                    event_names.append(event.get('name', 'Unknown'))
                barrier_times.append([rank, event['ts'], event['ts'] + event['dur'], 
                                    event['dur'], event.get('name', 'Unknown')])
    barrier_times.sort(key=lambda x: x[2])
    print(f"Found {len(barrier_times)} barrier events: {barrier_times}")

    barrier_deltas = {}
    for rank in ranks:
        earliest_barrier_time = None
        for barrier_time in barrier_times:
            if barrier_time[0] == rank:
                if earliest_barrier_time is None or barrier_time[2] < earliest_barrier_time:
                    earliest_barrier_time = barrier_time[2]
        barrier_deltas[rank] = earliest_barrier_time # Sets the earliest barrier time to 0
        earliest_times[rank] = earliest_barrier_time - earliest_times[rank]
    
    if whole_trace:
        biggest_delta = max(earliest_times.values())
        for rank in ranks:
            barrier_deltas[rank] = barrier_deltas[rank] - biggest_delta

    
    rank_final_times = {}
    for rank in ranks:
        for barrier_time in barrier_times:
            if barrier_time[0] == rank:
                barrier_time[1] -= barrier_deltas[rank]
                barrier_time[2] -= barrier_deltas[rank]
                if rank not in rank_final_times:
                    rank_final_times[rank] = barrier_time[2]
                else:
                    rank_final_times[rank] = max(rank_final_times[rank], barrier_time[2])
        
        prev_event_time = -1
        for event_name in event_names:
            for barrier_time in barrier_times:
                if barrier_time[0] == rank and barrier_time[4] == event_name:  
                    if prev_event_time == -1:
                        print(f"[Rank {rank} {event_name}] " + 
                              f"{barrier_time[1]} to {barrier_time[2]}, took {barrier_time[3]} us")
                    else:
                        print(f"[Rank {rank} {event_name}] " + 
                              f"{barrier_time[1] - prev_event_time} us after {prev_event_name}, " + 
                              f"{barrier_time[1]} to {barrier_time[2]}, took {barrier_time[3]} us")
                    prev_event_name = event_name
                    prev_event_time = barrier_time[2]
    for rank in ranks:
        avg_time = sum(rank_final_times.values()) / len(rank_final_times)
        delta_from_avg = abs(rank_final_times[rank] - avg_time)
        print(f"Rank {rank} final time: {rank_final_times[rank]}, delta from avg: {delta_from_avg}")
        if delta_from_avg > 1000:
            print(f"WARNING: Rank {rank} final time has {delta_from_avg} us delta compared to avg, possible sync issue!")

    return barrier_deltas, rank_final_times

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

def sync_and_merge_traces(traces: Dict, barrier_deltas: Dict, rank_final_times: Dict, output_file: str, whole_trace: bool = False) -> Dict:
    """Synchronize and merge traces by aligning them at barrier points."""
    merged_trace = {
        'traceEvents': [],
        'metadata': {
            'merged_from': list(traces.keys()),
            'timestamp': time.time()
        }
    }

    for rank, trace_data in traces.items():
        mbp_rank, dist_rank = get_ranks(rank)
        if 'traceEvents' not in trace_data:
            continue
        for event in trace_data['traceEvents']:
            if 'name' in event:
                event['name'] = f"[Rank {mbp_rank}-{dist_rank}] {event['name']}"
            else:
                event['name'] = f"[Rank {mbp_rank}-{dist_rank}] Event"
            if 'ts' in event:
                event['ts'] = event['ts'] - barrier_deltas[rank]
            if not whole_trace:
                if event['ts'] >= 0 and event['ts'] <= rank_final_times[rank]:
                    merged_trace['traceEvents'].append(event)
            else:
                merged_trace['traceEvents'].append(event)
    
    with open(output_file, 'w') as f:
        json.dump(merged_trace, f, indent=2)

def merge_simple(traces: Dict, output_file: str):
    """Simple merge without synchronization (fallback)."""
    merged_trace = {
        'traceEvents': [],
        'metadata': {
            'merged_from': list(traces.keys()),
            'timestamp': time.time()
        }
    }
    
    for rank, trace_data in traces.items():
        mbp_rank, dist_rank = get_ranks(rank)
        if 'traceEvents' in trace_data:
            for event in trace_data['traceEvents']:
                if 'name' in event:
                    event['name'] = f"[Rank {mbp_rank}-{dist_rank}] {event['name']}"
                else:
                    event['name'] = f"[Rank {mbp_rank}-{dist_rank}] Event"
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
        trace_names=trace_names,
        whole_trace=True # If false, will only output events between the first and last barriers.
    )