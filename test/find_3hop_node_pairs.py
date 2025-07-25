#!/usr/bin/env python3
"""
Find node pairs that require 3-hop paths
"""

import os
import sys
import numpy as np
import networkx as nx
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def find_3hop_node_pairs():
    """Find node pairs that require 3-hop paths"""
    
    print("Finding node pairs that require 3-hop paths")
    print("=" * 60)
    
    # Get the data directory
    data_dir = Path(__file__).parent.parent / "data" / "ucsb"
    target_folder = data_dir / "1144393236-1144450070"
    
    # Use first file for testing
    test_file = target_folder / "neighbortable-1144393236"
    
    print(f"Testing file: {test_file.name}")
    
    # Build graph manually
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
    
    # Find all node pairs and their shortest path lengths
    nodes_list = list(nodes_in_file)
    three_hop_pairs = []
    
    print(f"\nAnalyzing node pairs...")
    
    for i, source in enumerate(nodes_list):
        if i % 10 == 0:  # Progress indicator
            print(f"  Processing node {i+1}/{len(nodes_list)}: {source}")
        
        for destination in nodes_list[i+1:]:  # Avoid duplicate pairs
            if source == destination:
                continue
                
            try:
                shortest_path = nx.shortest_path(G, source, destination)
                path_length = len(shortest_path) - 1  # Number of hops
                
                if path_length == 3:  # Exactly 3 hops
                    # Count total 3-hop paths
                    all_paths = []
                    for path in nx.all_simple_paths(G, source, destination, cutoff=3):
                        if len(path) == 4:  # 3 hops = 4 nodes
                            all_paths.append(path)
                    
                    three_hop_pairs.append({
                        'source': source,
                        'destination': destination,
                        'shortest_path': shortest_path,
                        'total_3hop_paths': len(all_paths)
                    })
                    
            except nx.NetworkXNoPath:
                continue
    
    # Sort by number of 3-hop paths
    three_hop_pairs.sort(key=lambda x: x['total_3hop_paths'], reverse=True)
    
    print(f"\nFound {len(three_hop_pairs)} node pairs with exactly 3-hop shortest paths")
    
    if three_hop_pairs:
        print(f"\nTop 10 node pairs with most 3-hop paths:")
        for i, pair in enumerate(three_hop_pairs[:10]):
            print(f"  {i+1}. {pair['source']} → {pair['destination']}")
            print(f"     Shortest path: {' → '.join(pair['shortest_path'])}")
            print(f"     Total 3-hop paths: {pair['total_3hop_paths']}")
            print()
        
        # Return the best pair
        best_pair = three_hop_pairs[0]
        print(f"Best node pair for 3-hop testing:")
        print(f"  Source: {best_pair['source']}")
        print(f"  Destination: {best_pair['destination']}")
        print(f"  Total 3-hop paths: {best_pair['total_3hop_paths']}")
        
        return best_pair['source'], best_pair['destination']
    else:
        print("No node pairs found with exactly 3-hop shortest paths")
        return None, None

if __name__ == "__main__":
    find_3hop_node_pairs() 