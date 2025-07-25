#!/usr/bin/env python3
"""
Debug UCSB reward calculation and optimal path computation.
"""

import sys
import os
import numpy as np
import networkx as nx
from typing import List, Dict, Any

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment

def setup_ucsb_meshnet_environment(routes_per_minute: int = 4, pre_generate_rewards: bool = True):
    """Setup the UCSB MeshNet environment with memory-mapped data."""
    data_dir = os.path.join(project_root, "data", "ucsb", "1143927049-1143953729")
    neighbortable_files = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.startswith('neighbortable-')
    ]
    neighbortable_files = sorted(neighbortable_files, key=lambda x: int(x.split('-')[-1]))
    
    env = UCSBMeshnetMemmapEnvironment(
        neighbortable_files=neighbortable_files,
        routes_per_minute=routes_per_minute,
        pre_generate_rewards=pre_generate_rewards,
        use_persistent_files=False,
        seed=42
    )
    return env

def debug_rewards_and_paths():
    """Debug reward calculation and path finding."""
    print("UCSB Reward and Path Debug")
    print("=" * 50)
    
    # Setup environment
    env = setup_ucsb_meshnet_environment(routes_per_minute=2)
    print(f"Environment: {env.num_rounds} rounds, {len(env.get_nodes())} nodes")
    print(f"Source: {env.get_source_destination()[0]}")
    print(f"Destination: {env.get_source_destination()[1]}")
    
    nodes = env.get_nodes()
    source, destination = env.get_source_destination()
    
    # Check a few rounds
    for round_idx in [0, 100, 200, 300, 400]:
        print(f"\n=== Round {round_idx} ===")
        
        # Get available links
        adj = env.get_available_links_for_round(round_idx)
        print(f"Available links: {np.sum(adj)} out of {adj.shape[0] * adj.shape[1]}")
        
        # Build graph
        G = nx.Graph()
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if adj[i, j]:
                    G.add_edge(u, v, weight=1.0)
        
        print(f"Graph edges: {G.number_of_edges()}")
        
        # Check if path exists
        try:
            path = nx.shortest_path(G, source=source, target=destination)
            print(f"Optimal path: {path}")
            
            # Calculate optimal reward
            opt_reward = 0.0
            for k in range(len(path)-1):
                link_reward = env.get_reward_for_link(path[k], path[k+1], round_idx)
                opt_reward += link_reward
                print(f"  Link {path[k]} -> {path[k+1]}: reward = {link_reward:.3f}")
            print(f"Total optimal reward: {opt_reward:.3f}")
            
        except nx.NetworkXNoPath:
            print("No path found!")
            continue
        
        # Check some random paths
        print("Checking some random paths:")
        for i in range(3):
            try:
                # Try to find a different path
                paths = list(nx.all_shortest_paths(G, source, destination))
                if len(paths) > i + 1:
                    path = paths[i + 1]
                else:
                    # Generate a random path
                    path = [source]
                    current = source
                    visited = {source}
                    while current != destination and len(path) < 10:
                        neighbors = list(G.neighbors(current))
                        valid_neighbors = [n for n in neighbors if n not in visited]
                        if not valid_neighbors:
                            break
                        current = np.random.choice(valid_neighbors)
                        path.append(current)
                        visited.add(current)
                
                if len(path) > 1:
                    path_reward = 0.0
                    for k in range(len(path)-1):
                        link_reward = env.get_reward_for_link(path[k], path[k+1], round_idx)
                        path_reward += link_reward
                    print(f"  Path {i+1}: {path} -> reward = {path_reward:.3f}")
                    
            except Exception as e:
                print(f"  Error with path {i+1}: {e}")

def main():
    debug_rewards_and_paths()

if __name__ == "__main__":
    main() 