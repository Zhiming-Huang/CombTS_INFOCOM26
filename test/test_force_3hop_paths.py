#!/usr/bin/env python3
"""
Test script to force inclusion of 3-hop paths by using very high max_paths_per_algorithm.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import networkx as nx
from src.environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment

def test_force_3hop_paths():
    """Test to force inclusion of 3-hop paths."""
    
    print("=" * 80)
    print("FORCING 3-HOP PATHS INCLUSION")
    print("=" * 80)
    
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
    
    # Test with very high max_paths_per_algorithm
    configs = [
        {
            "name": "Very High (1000 paths per algorithm)",
            "max_feasible_combinations": 500,
            "max_path_length": 3,
            "max_paths_per_algorithm": 1000
        },
        {
            "name": "Extreme (2000 paths per algorithm)",
            "max_feasible_combinations": 1000,
            "max_path_length": 3,
            "max_paths_per_algorithm": 2000
        },
        {
            "name": "Maximum (5000 paths per algorithm)",
            "max_feasible_combinations": 2000,
            "max_path_length": 3,
            "max_paths_per_algorithm": 5000
        }
    ]
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Configuration: {config['name']}")
        print(f"  max_feasible_combinations: {config['max_feasible_combinations']}")
        print(f"  max_path_length: {config['max_path_length']}")
        print(f"  max_paths_per_algorithm: {config['max_paths_per_algorithm']}")
        
        # Create environment
        env = UCSBMeshnetMemmapEnvironment(
            neighbortable_files=neighbortable_files[:5],  # Use first 5 files for quick test
            routes_per_minute=4,
            source=source,
            destination=destination,
            max_feasible_combinations=config['max_feasible_combinations'],
            max_path_length=config['max_path_length'],
            max_paths_per_algorithm=config['max_paths_per_algorithm']
        )
        
        # Analyze a few rounds
        total_paths = 0
        path_lengths = []
        
        for round_idx in range(min(3, env.num_rounds)):
            available_arms = env.get_available_arms_for_round(round_idx)
            if available_arms:
                feasible_combinations = env.get_feasible_combinations(set(available_arms))
                
                print(f"\n  Round {round_idx}: {len(feasible_combinations)} combinations")
                
                for combo_idx, combination in enumerate(feasible_combinations):
                    # Convert arm indices to links
                    combo_links = []
                    for arm in combination:
                        if arm < len(env.arms):
                            combo_links.append(env.arm_to_link[arm])
                    
                    # Reconstruct path
                    path = reconstruct_path_from_links(combo_links, env.source, env.destination)
                    if path:
                        path_length = len(path) - 1
                        path_lengths.append(path_length)
                        total_paths += 1
                        
                        if combo_idx < 10:  # Show first 10 paths
                            print(f"    Path {combo_idx + 1}: {path} ({path_length} hops)")
        
        # Analyze path length distribution
        if path_lengths:
            unique_lengths, counts = np.unique(path_lengths, return_counts=True)
            print(f"\n  Path length distribution:")
            for length, count in zip(unique_lengths, counts):
                percentage = (count / len(path_lengths)) * 100
                print(f"    {length}-hop paths: {count} ({percentage:.1f}%)")
        else:
            print(f"\n  No paths found!")
        
        env.cleanup()
    
    # Now let's manually check what paths exist and try to understand why 3-hop paths aren't being selected
    print(f"\n{'='*80}")
    print("MANUAL PATH ANALYSIS")
    print(f"{'='*80}")
    
    # Create environment for manual analysis
    env = UCSBMeshnetMemmapEnvironment(
        neighbortable_files=neighbortable_files[:1],  # Use just first file
        routes_per_minute=1,
        source=source,
        destination=destination
    )
    
    # Build network graph for first round
    round_idx = 0
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
    
    # Find all simple paths up to 3 hops
    try:
        all_paths = list(nx.all_simple_paths(G, env.source, env.destination, cutoff=3))
        print(f"\nAll simple paths from {env.source} to {env.destination} (up to 3 hops): {len(all_paths)}")
        
        # Group by path length
        path_lengths = {}
        for path in all_paths:
            length = len(path) - 1
            if length not in path_lengths:
                path_lengths[length] = []
            path_lengths[length].append(path)
        
        for length in sorted(path_lengths.keys()):
            paths = path_lengths[length]
            print(f"\n{length}-hop paths: {len(paths)}")
            for i, path in enumerate(paths[:3]):  # Show first 3
                print(f"  {i+1}: {path}")
            if len(paths) > 3:
                print(f"  ... and {len(paths) - 3} more")
        
        # Check if there are 1-hop paths (direct connection)
        if 1 in path_lengths:
            print(f"\n✓ Direct connection exists: {env.source} → {env.destination}")
        else:
            print(f"\n✗ No direct connection: {env.source} → {env.destination}")
        
        # Check if there are 3-hop paths
        if 3 in path_lengths:
            print(f"✓ 3-hop paths exist: {len(path_lengths[3])} paths")
            
            # Show some 3-hop paths
            print(f"\nSample 3-hop paths:")
            for i, path in enumerate(path_lengths[3][:5]):
                print(f"  {i+1}: {path}")
        else:
            print(f"✗ No 3-hop paths found")
            
    except Exception as e:
        print(f"Error finding paths: {e}")
    
    env.cleanup()

def reconstruct_path_from_links(links, source, destination):
    """Reconstruct a path from a list of links."""
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

if __name__ == "__main__":
    test_force_3hop_paths() 