#!/usr/bin/env python3
"""
Debug UCSB network topology and connectivity.
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

def analyze_topology():
    """Analyze UCSB network topology."""
    print("UCSB Network Topology Analysis")
    print("=" * 50)
    
    # Setup environment
    env = setup_ucsb_meshnet_environment(routes_per_minute=2)
    nodes = env.get_nodes()
    source, destination = env.get_source_destination()
    
    print(f"Total nodes: {len(nodes)}")
    print(f"Source: {source}")
    print(f"Destination: {destination}")
    print(f"Source in nodes: {source in nodes}")
    print(f"Destination in nodes: {destination in nodes}")
    
    # Check a few rounds for connectivity
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
        
        print(f"Graph nodes: {G.number_of_nodes()}")
        print(f"Graph edges: {G.number_of_edges()}")
        
        # Check connectivity
        if source in G.nodes() and destination in G.nodes():
            print(f"Both source and destination are in the graph")
            
            # Check if they are in the same connected component
            if nx.has_path(G, source, destination):
                print(f"Path exists from {source} to {destination}")
                path = nx.shortest_path(G, source, destination)
                print(f"Shortest path: {path}")
                print(f"Path length: {len(path) - 1}")
            else:
                print(f"No path exists from {source} to {destination}")
                
                # Find connected components
                components = list(nx.connected_components(G))
                print(f"Number of connected components: {len(components)}")
                
                # Find which component contains source and destination
                source_component = None
                dest_component = None
                for i, component in enumerate(components):
                    if source in component:
                        source_component = i
                        print(f"Source {source} is in component {i} with {len(component)} nodes")
                    if destination in component:
                        dest_component = i
                        print(f"Destination {destination} is in component {i} with {len(component)} nodes")
                
                if source_component == dest_component and source_component is not None:
                    print(f"Both nodes are in the same component but no path exists!")
                else:
                    print(f"Nodes are in different components")
        else:
            print(f"Source or destination not in graph")
            if source not in G.nodes():
                print(f"Source {source} not in graph")
            if destination not in G.nodes():
                print(f"Destination {destination} not in graph")
    
    # Check what nodes are actually connected to source and destination
    print(f"\n=== Node Connectivity Analysis ===")
    for round_idx in [0, 100, 200]:
        print(f"\nRound {round_idx}:")
        adj = env.get_available_links_for_round(round_idx)
        
        # Find nodes connected to source
        source_idx = nodes.index(source) if source in nodes else -1
        if source_idx >= 0:
            source_neighbors = []
            for j, v in enumerate(nodes):
                if adj[source_idx, j]:
                    source_neighbors.append(v)
            print(f"Source {source} neighbors: {source_neighbors}")
        
        # Find nodes connected to destination
        dest_idx = nodes.index(destination) if destination in nodes else -1
        if dest_idx >= 0:
            dest_neighbors = []
            for j, v in enumerate(nodes):
                if adj[dest_idx, j]:
                    dest_neighbors.append(v)
            print(f"Destination {destination} neighbors: {dest_neighbors}")

def main():
    analyze_topology()

if __name__ == "__main__":
    main() 