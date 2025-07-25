#!/usr/bin/env python3
"""
Analyze feasible paths between specific node pair (10.1.1.100, 10.2.1.9) 
across all UCSB mesh network traces.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment

def analyze_node_pair_feasibility():
    """Analyze which traces contain feasible paths between 10.1.1.100 and 10.2.1.9."""
    
    # Target node pair
    source_node = "10.1.1.100"
    dest_node = "10.2.1.9"
    
    print(f"Analyzing feasible paths between {source_node} and {dest_node}")
    print("=" * 60)
    
    # UCSB data directory
    ucsb_data_dir = Path("data/ucsb")
    
    # Find all trace directories
    trace_dirs = []
    for item in ucsb_data_dir.iterdir():
        if item.is_dir() and item.name.startswith("114"):
            trace_dirs.append(item)
    
    print(f"Found {len(trace_dirs)} trace directories")
    print()
    
    # Analyze each trace
    feasible_traces = []
    infeasible_traces = []
    
    for trace_dir in sorted(trace_dirs):
        trace_name = trace_dir.name
        print(f"Analyzing trace: {trace_name}")
        
        try:
            # Get all neighbortable files in the trace directory
            neighbortable_files = []
            for file in trace_dir.glob("neighbortable-*"):
                neighbortable_files.append(str(file))
            
            if not neighbortable_files:
                print(f"  ❌ No neighbortable files found")
                infeasible_traces.append((trace_name, "No neighbortable files"))
                continue
            
            # Sort files by timestamp
            neighbortable_files.sort()
            
            # Create environment for this trace
            env = UCSBMeshnetMemmapEnvironment(
                neighbortable_files=neighbortable_files,
                source=source_node,
                destination=dest_node,
                max_path_length=3,
                max_feasible_combinations=50
            )
            
            # Check if nodes exist in the network
            nodes = env.get_nodes()
            if source_node not in nodes or dest_node not in nodes:
                print(f"  ❌ One or both nodes not found in network")
                print(f"     Available nodes: {len(nodes)} nodes")
                print(f"     Sample nodes: {nodes[:5]}...")
                infeasible_traces.append((trace_name, "Nodes not found"))
                continue
            
            print(f"  ✅ Both nodes found in network")
            
            # Check connectivity by getting available arms for round 0
            try:
                available_arms = env.get_available_arms_for_round(0)
                print(f"  ✅ Available arms for round 0: {len(available_arms)}")
                
                # Get feasible combinations
                feasible_combinations = env.get_feasible_combinations(available_arms)
                
                if feasible_combinations:
                    # Convert arm combinations back to paths for analysis
                    path_lengths = []
                    for combination in feasible_combinations:
                        # Count unique links in the combination
                        path_lengths.append(len(combination))
                    
                    print(f"  ✅ Found {len(feasible_combinations)} feasible combinations")
                    print(f"     Path lengths (hops): {path_lengths}")
                    print(f"     Min hops: {min(path_lengths)}, Max hops: {max(path_lengths)}")
                    
                    feasible_traces.append((trace_name, len(feasible_combinations), path_lengths))
                else:
                    print(f"  ❌ No feasible combinations found")
                    infeasible_traces.append((trace_name, "No feasible combinations"))
                    
            except Exception as e:
                print(f"  ❌ Error checking connectivity: {e}")
                infeasible_traces.append((trace_name, f"Connectivity error: {e}"))
                
        except Exception as e:
            print(f"  ❌ Error analyzing trace: {e}")
            infeasible_traces.append((trace_name, f"Error: {e}"))
        
        print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"Total traces analyzed: {len(trace_dirs)}")
    print(f"Traces with feasible paths: {len(feasible_traces)}")
    print(f"Traces without feasible paths: {len(infeasible_traces)}")
    
    if feasible_traces:
        print("\nTraces with feasible paths:")
        print("-" * 40)
        for trace_name, num_paths, path_lengths in feasible_traces:
            print(f"{trace_name}: {num_paths} paths, lengths {path_lengths}")
        
        # Statistics
        all_path_counts = [num_paths for _, num_paths, _ in feasible_traces]
        all_path_lengths = []
        for _, _, lengths in feasible_traces:
            all_path_lengths.extend(lengths)
        
        print(f"\nPath count statistics:")
        print(f"  Average paths per trace: {np.mean(all_path_counts):.1f}")
        print(f"  Min paths: {min(all_path_counts)}, Max paths: {max(all_path_counts)}")
        
        print(f"\nPath length statistics:")
        print(f"  Average path length: {np.mean(all_path_lengths):.1f} hops")
        print(f"  Min length: {min(all_path_lengths)}, Max length: {max(all_path_lengths)}")
        print(f"  Length distribution: {dict(zip(*np.unique(all_path_lengths, return_counts=True)))}")
    
    if infeasible_traces:
        print("\nTraces without feasible paths:")
        print("-" * 40)
        for trace_name, reason in infeasible_traces:
            print(f"{trace_name}: {reason}")
    
    return feasible_traces, infeasible_traces

if __name__ == "__main__":
    feasible_traces, infeasible_traces = analyze_node_pair_feasibility() 