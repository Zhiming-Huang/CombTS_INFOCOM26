#!/usr/bin/env python3
"""
Debug 3-hop paths for UCSB meshnet
"""

import os
import sys
import numpy as np
import networkx as nx
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def debug_3hop_paths():
    """Debug 3-hop paths for the current node pair"""
    
    # Current best node pair
    source = "10.1.1.109"
    destination = "10.1.1.5"
    
    print(f"Debugging 3-hop paths for {source} → {destination}")
    print("=" * 60)
    
    # Get the data directory
    data_dir = Path(__file__).parent.parent / "data" / "ucsb"
    target_folder = data_dir / "1144393236-1144450070"
    
    # Test on first file
    test_file = target_folder / "neighbortable-1144393236"
    
    print(f"Testing file: {test_file.name}")
    
    # Read the file and build graph manually
    G = nx.Graph()
    nodes_in_file = set()
    
    with open(test_file, 'r') as fin:
        for line in fin:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            src = parts[0]
            nodes_in_file.add(src)
            
            for i in range(1, len(parts), 2):
                dst = parts[i]
                try:
                    ett = float(parts[i+1])
                    G.add_edge(src, dst, weight=ett)
                    nodes_in_file.add(dst)
                except Exception:
                    continue
    
    print(f"Nodes in file: {len(nodes_in_file)}")
    print(f"Edges in graph: {G.number_of_edges()}")
    
    # Check if source and destination are in the graph
    print(f"Source {source} in graph: {source in G}")
    print(f"Destination {destination} in graph: {destination in G}")
    
    if source in G and destination in G:
        # Check connectivity
        try:
            shortest_path = nx.shortest_path(G, source, destination)
            print(f"Shortest path length: {len(shortest_path) - 1} hops")
            print(f"Shortest path: {' → '.join(shortest_path)}")
            
            # Find all simple paths up to 3 hops
            all_paths = []
            for path in nx.all_simple_paths(G, source, destination, cutoff=3):
                if len(path) <= 4:  # 3 hops = 4 nodes
                    all_paths.append(path)
            
            print(f"All paths up to 3 hops: {len(all_paths)}")
            
            # Filter for exactly 3-hop paths
            three_hop_paths = [path for path in all_paths if len(path) == 4]
            print(f"Exactly 3-hop paths: {len(three_hop_paths)}")
            
            if three_hop_paths:
                print("Sample 3-hop paths:")
                for i, path in enumerate(three_hop_paths[:10]):
                    print(f"  {i+1}: {' → '.join(path)}")
                if len(three_hop_paths) > 10:
                    print(f"  ... and {len(three_hop_paths) - 10} more")
            
        except nx.NetworkXNoPath:
            print("No path exists between source and destination")
    else:
        print("Source or destination not found in graph")
        
        # Show some sample nodes
        sample_nodes = list(nodes_in_file)[:10]
        print(f"Sample nodes in file: {sample_nodes}")

if __name__ == "__main__":
    debug_3hop_paths() 