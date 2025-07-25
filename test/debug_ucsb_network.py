#!/usr/bin/env python3
"""
Debug script to examine UCSB network topology.
"""

import sys
import os
import numpy as np
import glob
import networkx as nx

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

def debug_ucsb_network():
    """Debug UCSB network topology."""
    
    # Find UCSB data files
    data_dir = os.path.join(project_root, 'data', 'ucsb')
    neighbortable_files = []
    
    # Look for neighbortable files in subdirectories
    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.isdir(subdir_path):
            files = glob.glob(os.path.join(subdir_path, 'neighbortable-*'))
            neighbortable_files.extend(sorted(files))
    
    if not neighbortable_files:
        print("No neighbortable files found!")
        return
    
    # Use first file for testing
    neighbortable_file = neighbortable_files[0]
    print(f"Using file: {neighbortable_file}")
    
    # Parse the neighbortable file
    nodes = set()
    edges = []
    
    with open(neighbortable_file, 'r') as fin:
        for line in fin:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            src = parts[0]
            nodes.add(src)
            
            for i in range(1, len(parts), 2):
                dst = parts[i]
                try:
                    ett = float(parts[i+1])
                    if ett < 1000:  # Valid ETT
                        nodes.add(dst)
                        edges.append((src, dst))
                except Exception:
                    continue
    
    # Create NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    print(f"Network analysis:")
    print(f"  Number of nodes: {len(nodes)}")
    print(f"  Number of edges: {len(edges)}")
    print(f"  Number of connected components: {nx.number_connected_components(G)}")
    
    # Check if source and destination are in the network
    source = "10.1.1.100"
    destination = "10.2.1.9"
    
    print(f"\nSource and destination analysis:")
    print(f"  Source '{source}' in network: {source in nodes}")
    print(f"  Destination '{destination}' in network: {destination in nodes}")
    
    if source in nodes and destination in nodes:
        # Check connectivity
        if nx.has_path(G, source, destination):
            shortest_path = nx.shortest_path(G, source, destination)
            print(f"  Shortest path length: {len(shortest_path) - 1} hops")
            print(f"  Shortest path: {' -> '.join(shortest_path)}")
            
            # Find all paths up to 3 hops
            all_paths = []
            for path in nx.all_simple_paths(G, source, destination, cutoff=3):
                all_paths.append(path)
            
            print(f"  Number of paths up to 3 hops: {len(all_paths)}")
            for i, path in enumerate(all_paths[:5]):  # Show first 5 paths
                print(f"    Path {i+1}: {' -> '.join(path)} (length: {len(path)-1})")
        else:
            print("  No path exists between source and destination!")
            
            # Find connected component of source
            source_component = nx.node_connected_component(G, source)
            print(f"  Source component size: {len(source_component)}")
            
            # Find connected component of destination
            dest_component = nx.node_connected_component(G, destination)
            print(f"  Destination component size: {len(dest_component)}")
    else:
        print("  Source or destination not in network!")
        
        # Show some nodes that are in the network
        print(f"  Sample nodes in network: {list(nodes)[:10]}")
    
    # Show network statistics
    print(f"\nNetwork statistics:")
    print(f"  Average degree: {sum(dict(G.degree()).values()) / len(G.nodes()):.2f}")
    print(f"  Diameter: {nx.diameter(G) if nx.is_connected(G) else 'N/A (disconnected)'}")
    print(f"  Average clustering coefficient: {nx.average_clustering(G):.3f}")

if __name__ == "__main__":
    debug_ucsb_network() 