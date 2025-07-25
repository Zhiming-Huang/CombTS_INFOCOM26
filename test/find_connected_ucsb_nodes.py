#!/usr/bin/env python3
"""
Find connected node pairs in UCSB network for testing.
"""

import sys
import os
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple

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

def find_connected_node_pairs():
    """Find connected node pairs in UCSB network."""
    print("Finding Connected Node Pairs in UCSB Network")
    print("=" * 60)
    
    # Setup environment
    env = setup_ucsb_meshnet_environment(routes_per_minute=2)
    nodes = env.get_nodes()
    
    print(f"Total nodes: {len(nodes)}")
    
    # Group nodes by subnet
    subnets = {}
    for node in nodes:
        subnet = '.'.join(node.split('.')[:3])  # Get first 3 octets
        if subnet not in subnets:
            subnets[subnet] = []
        subnets[subnet].append(node)
    
    print(f"Subnets found: {list(subnets.keys())}")
    for subnet, subnet_nodes in subnets.items():
        print(f"  {subnet}: {len(subnet_nodes)} nodes")
    
    # Check connectivity within each subnet
    connected_pairs = []
    
    for round_idx in [0, 100, 200, 300, 400]:
        print(f"\n=== Round {round_idx} ===")
        
        # Get available links
        adj = env.get_available_links_for_round(round_idx)
        
        # Build graph
        G = nx.Graph()
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if adj[i, j]:
                    G.add_edge(u, v)
        
        # Find connected components
        components = list(nx.connected_components(G))
        print(f"Connected components: {len(components)}")
        
        for i, component in enumerate(components):
            print(f"  Component {i}: {len(component)} nodes")
            
            # Find nodes in this component that belong to the same subnet
            for subnet, subnet_nodes in subnets.items():
                subnet_nodes_in_component = [n for n in component if n in subnet_nodes]
                if len(subnet_nodes_in_component) >= 2:
                    print(f"    Subnet {subnet}: {len(subnet_nodes_in_component)} nodes")
                    
                    # Find pairs with paths
                    for j, node1 in enumerate(subnet_nodes_in_component):
                        for k, node2 in enumerate(subnet_nodes_in_component[j+1:], j+1):
                            if nx.has_path(G, node1, node2):
                                path = nx.shortest_path(G, node1, node2)
                                path_length = len(path) - 1
                                if path_length >= 2:  # At least 2 hops
                                    connected_pairs.append({
                                        'source': node1,
                                        'destination': node2,
                                        'path_length': path_length,
                                        'path': path,
                                        'round': round_idx,
                                        'subnet': subnet
                                    })
                                    print(f"      {node1} -> {node2}: {path_length} hops")
    
    # Find the best connected pair
    if connected_pairs:
        print(f"\n=== Best Connected Pairs ===")
        # Sort by path length (longer paths are better for testing)
        connected_pairs.sort(key=lambda x: x['path_length'], reverse=True)
        
        for i, pair in enumerate(connected_pairs[:5]):
            print(f"{i+1}. {pair['source']} -> {pair['destination']}")
            print(f"   Path length: {pair['path_length']} hops")
            print(f"   Path: {' -> '.join(pair['path'])}")
            print(f"   Subnet: {pair['subnet']}")
            print(f"   Round: {pair['round']}")
            print()
        
        # Return the best pair
        best_pair = connected_pairs[0]
        return best_pair['source'], best_pair['destination']
    else:
        print("No connected pairs found!")
        return None, None

def test_with_connected_nodes():
    """Test UCSB environment with connected node pair."""
    source, destination = find_connected_node_pairs()
    
    if source and destination:
        print(f"\n=== Testing with Connected Nodes ===")
        print(f"Source: {source}")
        print(f"Destination: {destination}")
        
        # Create environment with custom source/destination
        env = setup_ucsb_meshnet_environment(routes_per_minute=2)
        env.source = source
        env.destination = destination
        
        print(f"Environment source: {env.get_source_destination()[0]}")
        print(f"Environment destination: {env.get_source_destination()[1]}")
        
        # Test a few rounds
        for round_idx in [0, 100, 200]:
            adj = env.get_available_links_for_round(round_idx)
            G = nx.Graph()
            nodes = env.get_nodes()
            for i, u in enumerate(nodes):
                for j, v in enumerate(nodes):
                    if adj[i, j]:
                        G.add_edge(u, v)
            
            try:
                path = nx.shortest_path(G, source, destination)
                print(f"Round {round_idx}: Path found with {len(path)-1} hops")
            except nx.NetworkXNoPath:
                print(f"Round {round_idx}: No path found")

def main():
    test_with_connected_nodes()

if __name__ == "__main__":
    main() 