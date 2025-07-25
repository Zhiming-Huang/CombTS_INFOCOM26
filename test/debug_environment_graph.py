#!/usr/bin/env python3
"""
Debug script to check how the environment builds graphs for each round.
"""

import os
import sys
import numpy as np
import networkx as nx
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment

def debug_environment_graph():
    """Debug the environment's graph building process."""
    
    # Get UCSB data files
    ucsb_dir = Path("data/ucsb")
    trace_folders = [d for d in ucsb_dir.iterdir() if d.is_dir() and d.name.startswith("114")]
    
    if not trace_folders:
        print("No UCSB trace folders found!")
        return
    
    # Use the first folder
    trace_folder = trace_folders[0]
    print(f"Using trace folder: {trace_folder}")
    
    # Get trace files
    trace_files = sorted([f for f in trace_folder.iterdir() if f.name.startswith("neighbortable")])
    print(f"Found {len(trace_files)} trace files")
    
    # Use first 5 files for testing
    test_files = [str(f) for f in trace_files[:5]]
    
    # Test node pair
    src, dst = "10.1.1.102", "10.1.1.25"
    
    print(f"\nTesting node pair: {src} -> {dst}")
    print("=" * 80)
    
    # Create environment
    env = UCSBMeshnetMemmapEnvironment(
        neighbortable_files=test_files,
        source=src,
        destination=dst,
        routes_per_minute=4,
        max_path_length=3,
        max_paths_per_algorithm=25,
        seed=42
    )
    
    print(f"Environment created with {len(test_files)} trace files")
    print(f"Total arms (all edges across all timestamps): {len(env.arms)}")
    print(f"Environment rounds: {env.num_rounds}")
    
    # Test first few rounds
    for round_idx in range(5):
        print(f"\n--- Round {round_idx} ---")
        
        # Get available arms for this round
        available_arms = set(env.get_available_arms_for_round(round_idx))
        print(f"Available arms for round {round_idx}: {len(available_arms)}")
        
        # Get the actual edges for these arms
        available_edges = set(env.arm_to_link[arm] for arm in available_arms)
        print(f"Available edges for round {round_idx}: {len(available_edges)}")
        
        # Show some example edges
        edge_examples = list(available_edges)[:5]
        print(f"Example edges: {edge_examples}")
        
        # Build the graph that environment would use
        G = nx.Graph()
        G.add_nodes_from(env.nodes)
        G.add_edges_from(available_edges)
        
        print(f"Graph nodes: {len(G.nodes)}")
        print(f"Graph edges: {len(G.edges)}")
        
        # Check if source and destination are in the graph
        print(f"Source {src} in graph: {src in G}")
        print(f"Destination {dst} in graph: {dst in G}")
        
        if src in G and dst in G:
            # Check connectivity
            if nx.has_path(G, src, dst):
                print(f"Path exists between {src} and {dst}")
                
                # Find all simple paths
                try:
                    all_paths = list(nx.all_simple_paths(G, src, dst, cutoff=3))
                    print(f"Number of simple paths: {len(all_paths)}")
                    
                    # Show first few paths
                    for i, path in enumerate(all_paths[:3]):
                        print(f"  Path {i+1}: {path}")
                        
                except nx.NetworkXNoPath:
                    print("No path found")
            else:
                print(f"No path between {src} and {dst}")
        else:
            print("Source or destination not in graph")
        
        # Get environment's feasible combinations
        env_feasible = env.get_feasible_combinations(available_arms)
        print(f"Environment feasible combinations: {len(env_feasible)}")
        
        # Manual verification for this round's timestamp
        timestamp_idx = round_idx // env.routes_per_minute
        if timestamp_idx < len(test_files):
            neighbortable_file = test_files[timestamp_idx]
            print(f"Manual verification for {neighbortable_file}:")
            
            # Parse the file manually
            manual_edges = set()
            with open(neighbortable_file, 'r') as fin:
                for line in fin:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.strip().split()
                    src_node = parts[0]
                    for i in range(1, len(parts), 2):
                        dst_node = parts[i]
                        try:
                            ett = float(parts[i+1])
                            if ett < 1000:
                                manual_edges.add((src_node, dst_node))
                        except Exception:
                            continue
            
            print(f"  Manual edges: {len(manual_edges)}")
            
            # Build manual graph
            manual_G = nx.Graph()
            manual_G.add_nodes_from(env.nodes)
            manual_G.add_edges_from(manual_edges)
            
            if src in manual_G and dst in manual_G and nx.has_path(manual_G, src, dst):
                manual_paths = list(nx.all_simple_paths(manual_G, src, dst, cutoff=3))
                print(f"  Manual paths: {len(manual_paths)}")
                for i, path in enumerate(manual_paths[:3]):
                    print(f"    Path {i+1}: {path}")
            else:
                print("  No manual path found")
        
        print("-" * 40)

if __name__ == "__main__":
    debug_environment_graph() 