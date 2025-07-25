#!/usr/bin/env python3
"""
Debug Arms to Paths Conversion
===============================

This script debugs how the environment converts arms to paths.
"""

import os
import sys
import glob
import networkx as nx
from typing import List, Set, Tuple

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment

def debug_arms_to_paths():
    """Debug the arms to paths conversion."""
    print("Debugging Arms to Paths Conversion")
    print("=" * 80)
    
    # Setup
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'ucsb')
    period1_dir = os.path.join(data_dir, '1143927049-1143953729')
    
    # Get trace files
    neighbortable_files = glob.glob(os.path.join(period1_dir, 'neighbortable-*'))
    neighbortable_files = sorted(neighbortable_files)
    
    # Test node pair
    src = '10.1.1.102'
    dst = '10.1.1.25'
    
    print(f"Testing node pair: {src} -> {dst}")
    print(f"Using first 5 trace files for debugging")
    
    # Test first 5 files
    test_files = neighbortable_files[:5]
    
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
    print(f"Total arms: {env.num_arms}")
    print(f"Max combination size: {env.max_combination_size}")
    
    # Debug first round
    round_idx = 0
    print(f"\n--- Debugging Round {round_idx} ---")
    
    # Get available arms
    available_arms = set(env.get_available_arms_for_round(round_idx))
    print(f"Available arms: {len(available_arms)}")
    
    # Show some available arms
    print(f"Sample available arms:")
    for i, arm_idx in enumerate(list(available_arms)[:10]):
        u, v = env.arm_to_link[arm_idx]
        print(f"  Arm {arm_idx}: {u} -> {v}")
    
    # Get feasible combinations
    feasible_combinations = env.get_feasible_combinations(available_arms)
    print(f"Feasible combinations: {len(feasible_combinations)}")
    
    # Debug each feasible combination
    for i, path_arms in enumerate(feasible_combinations[:5]):  # Show first 5
        print(f"\nFeasible combination {i+1}:")
        print(f"  Arm indices: {path_arms}")
        
        # Convert back to links
        path_links = []
        for arm_idx in path_arms:
            u, v = env.arm_to_link[arm_idx]
            path_links.append((u, v))
        
        print(f"  Links: {path_links}")
        
        # Check if this forms a valid path
        if len(path_links) >= 1:
            # Build a simple graph from these links
            G = nx.Graph()
            for u, v in path_links:
                G.add_edge(u, v)
            
            # Check if there's a path from source to destination
            if nx.has_path(G, src, dst):
                path = nx.shortest_path(G, src, dst)
                print(f"  ✓ Valid path: {' -> '.join(path)}")
            else:
                print(f"  ✗ Not a valid path from {src} to {dst}")
        else:
            print(f"  ✗ No links in combination")

def debug_single_round():
    """Debug a single round in detail."""
    print(f"\n{'='*80}")
    print("DETAILED SINGLE ROUND DEBUG")
    print(f"{'='*80}")
    
    # Setup
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'ucsb')
    period1_dir = os.path.join(data_dir, '1143927049-1143953729')
    
    # Get trace files
    neighbortable_files = glob.glob(os.path.join(period1_dir, 'neighbortable-*'))
    neighbortable_files = sorted(neighbortable_files)
    
    # Test node pair
    src = '10.1.1.102'
    dst = '10.1.1.25'
    
    # Use the first file
    test_files = [neighbortable_files[0]]
    
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
    
    print(f"Testing file: {os.path.basename(test_files[0])}")
    
    # Get available arms for round 0
    available_arms = set(env.get_available_arms_for_round(0))
    print(f"Available arms: {len(available_arms)}")
    
    # Manually verify the graph
    print(f"\nManual verification:")
    nodes, edges = parse_neighbortable_file(test_files[0])
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    if nx.has_path(G, src, dst):
        all_paths = list(nx.all_simple_paths(G, src, dst, cutoff=3))
        print(f"Manual paths: {len(all_paths)}")
        for i, path in enumerate(all_paths):
            print(f"  Path {i+1}: {' -> '.join(path)}")
    else:
        print(f"No path found manually")
    
    # Get feasible combinations from environment
    feasible_combinations = env.get_feasible_combinations(available_arms)
    print(f"\nEnvironment feasible combinations: {len(feasible_combinations)}")
    
    # Check if any of the feasible combinations actually form valid paths
    valid_paths = 0
    for i, path_arms in enumerate(feasible_combinations):
        # Convert arms to links
        path_links = []
        for arm_idx in path_arms:
            u, v = env.arm_to_link[arm_idx]
            path_links.append((u, v))
        
        # Check if this forms a valid path
        if len(path_links) >= 1:
            G_test = nx.Graph()
            for u, v in path_links:
                G_test.add_edge(u, v)
            
            if nx.has_path(G_test, src, dst):
                valid_paths += 1
                if valid_paths <= 3:  # Show first 3 valid paths
                    path = nx.shortest_path(G_test, src, dst)
                    print(f"  Valid combination {i+1}: {' -> '.join(path)}")
    
    print(f"Valid paths found: {valid_paths}/{len(feasible_combinations)}")

def parse_neighbortable_file(file_path: str) -> Tuple[Set[str], List[Tuple[str, str]]]:
    """Parse a neighbortable file to extract nodes and edges."""
    nodes = set()
    edges = []
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        node = parts[0]
                        neighbor = parts[1]
                        nodes.add(node)
                        nodes.add(neighbor)
                        edges.append((node, neighbor))
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    
    return nodes, edges

def main():
    """Main function."""
    debug_arms_to_paths()
    debug_single_round()

if __name__ == "__main__":
    main() 