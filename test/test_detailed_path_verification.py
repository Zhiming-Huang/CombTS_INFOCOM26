#!/usr/bin/env python3
"""
Detailed verification script to show specific path examples and validate
the path reconstruction logic in the UCSB environment.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import networkx as nx
from src.environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment

def detailed_path_verification(env, round_idx):
    """
    Perform detailed verification of paths for a specific round.
    
    Args:
        env: UCSB environment instance
        round_idx: Round index to verify
    """
    print(f"\n{'='*80}")
    print(f"DETAILED PATH VERIFICATION - Round {round_idx}")
    print(f"{'='*80}")
    
    # Get available arms and feasible combinations
    available_arms = env.get_available_arms_for_round(round_idx)
    feasible_combinations = env.get_feasible_combinations(set(available_arms))
    
    print(f"Source: {env.source}")
    print(f"Destination: {env.destination}")
    print(f"Available arms: {len(available_arms)}")
    print(f"Feasible combinations: {len(feasible_combinations)}")
    
    # Build network graph
    adj_matrix = env.get_available_links_for_round(round_idx)
    G = nx.DiGraph()
    
    for i, node in enumerate(env.nodes):
        G.add_node(node)
    
    for i, u in enumerate(env.nodes):
        for j, v in enumerate(env.nodes):
            if adj_matrix[i, j]:
                G.add_edge(u, v)
    
    print(f"Network nodes: {len(G.nodes())}")
    print(f"Network edges: {len(G.edges())}")
    
    # Check connectivity
    if nx.has_path(G, env.source, env.destination):
        shortest_path = nx.shortest_path(G, env.source, env.destination)
        print(f"Shortest path exists: {shortest_path} ({len(shortest_path)-1} hops)")
    else:
        print("WARNING: No path exists from source to destination!")
        return
    
    # Verify each feasible combination in detail
    for combo_idx, combination in enumerate(feasible_combinations):
        print(f"\n--- Combination {combo_idx + 1} ---")
        print(f"Arm indices: {combination}")
        
        # Convert to links
        combo_links = []
        for arm in combination:
            if arm < len(env.arms):
                link = env.arm_to_link[arm]
                combo_links.append(link)
                print(f"  Arm {arm} -> Link {link}")
            else:
                print(f"  ERROR: Invalid arm index {arm}")
                continue
        
        # Reconstruct path
        path = reconstruct_path_from_links(combo_links, env.source, env.destination)
        
        if path:
            print(f"Reconstructed path: {path}")
            print(f"Path length: {len(path) - 1} hops")
            
            # Verify each link in the path exists
            path_valid = True
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if G.has_edge(u, v):
                    print(f"  ✓ Link {u} -> {v} exists")
                else:
                    print(f"  ✗ Link {u} -> {v} MISSING!")
                    path_valid = False
            
            if path_valid:
                print(f"  ✓ Path is valid")
                
                # Calculate expected reward
                expected_reward = env.get_expected_reward_for_path(combination, round_idx)
                print(f"  Expected reward: {expected_reward:.4f}")
            else:
                print(f"  ✗ Path is invalid")
        else:
            print(f"  ✗ Could not reconstruct path from links")
    
    # Show some network statistics
    print(f"\n--- Network Statistics ---")
    print(f"All simple paths from {env.source} to {env.destination}:")
    
    try:
        all_paths = list(nx.all_simple_paths(G, env.source, env.destination, cutoff=8))
        print(f"Found {len(all_paths)} simple paths (up to 8 hops)")
        
        # Group by path length
        path_lengths = {}
        for path in all_paths:
            length = len(path) - 1
            if length not in path_lengths:
                path_lengths[length] = []
            path_lengths[length].append(path)
        
        for length in sorted(path_lengths.keys()):
            paths = path_lengths[length]
            print(f"  {length}-hop paths: {len(paths)}")
            if length <= 3:  # Show first few short paths
                for i, path in enumerate(paths[:3]):
                    print(f"    {i+1}: {path}")
                if len(paths) > 3:
                    print(f"    ... and {len(paths) - 3} more")
    
    except Exception as e:
        print(f"Error finding all paths: {e}")

def reconstruct_path_from_links(links, source, destination):
    """
    Reconstruct a path from a list of links.
    
    Args:
        links: List of (u, v) tuples representing links
        source: Source node
        destination: Destination node
        
    Returns:
        list: Path as list of nodes, or None if invalid
    """
    if not links:
        return None
    
    # Create a graph from the links
    G = nx.DiGraph()
    for u, v in links:
        G.add_edge(u, v)
    
    # Check if path exists
    try:
        path = nx.shortest_path(G, source, destination)
        return path
    except nx.NetworkXNoPath:
        return None

def test_detailed_verification():
    """Main test function for detailed path verification."""
    
    # Get the largest dataset folder
    data_dir = "data/ucsb/1144393236-1144450070"
    neighbortable_files = []
    for f in os.listdir(data_dir):
        if f.startswith("neighbortable-"):
            neighbortable_files.append(os.path.join(data_dir, f))
    neighbortable_files.sort()
    
    print(f"Found {len(neighbortable_files)} neighbortable files")
    
    # Use the best node pair we found earlier
    source = "10.1.1.109"
    destination = "10.1.1.5"
    
    # Create environment
    env = UCSBMeshnetMemmapEnvironment(
        neighbortable_files=neighbortable_files[:5],  # Use first 5 files
        routes_per_minute=4,
        source=source,
        destination=destination,
        max_feasible_combinations=20,  # Limit for readability
        max_path_length=5,
        max_paths_per_algorithm=10
    )
    
    # Test a few rounds in detail
    for round_idx in range(min(3, env.num_rounds)):
        detailed_path_verification(env, round_idx)
    
    env.cleanup()
    
    print(f"\n{'='*80}")
    print("Detailed verification completed!")

if __name__ == "__main__":
    test_detailed_verification() 