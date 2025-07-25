#!/usr/bin/env python3
"""
Check the selected source and destination nodes in UCSB network.
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

def check_selected_nodes():
    """Check the selected source and destination nodes."""
    
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
    
    # Use first file for analysis
    neighbortable_file = neighbortable_files[0]
    print(f"Analyzing file: {neighbortable_file}")
    
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
    
    # Check the selected nodes
    source = "10.2.1.103"
    destination = "10.2.1.109"
    
    print(f"\nSelected nodes analysis:")
    print(f"  Source: {source}")
    print(f"  Destination: {destination}")
    
    # Check if nodes exist in network
    print(f"\nNode existence:")
    print(f"  Source '{source}' in network: {source in nodes}")
    print(f"  Destination '{destination}' in network: {destination in nodes}")
    
    if source in nodes and destination in nodes:
        # Check connectivity
        if nx.has_path(G, source, destination):
            shortest_path = nx.shortest_path(G, source, destination)
            print(f"\nPath analysis:")
            print(f"  Shortest path length: {len(shortest_path) - 1} hops")
            print(f"  Shortest path: {' -> '.join(shortest_path)}")
            
            # Find all paths up to 3 hops
            all_paths = []
            for path in nx.all_simple_paths(G, source, destination, cutoff=3):
                all_paths.append(path)
            
            print(f"  Number of paths up to 3 hops: {len(all_paths)}")
            
            # Show some paths
            for i, path in enumerate(all_paths[:5]):
                print(f"    Path {i+1}: {' -> '.join(path)} (length: {len(path)-1})")
            
            # Check node degrees
            source_degree = G.degree(source)
            dest_degree = G.degree(destination)
            print(f"\nNode degrees:")
            print(f"  Source degree: {source_degree}")
            print(f"  Destination degree: {dest_degree}")
            
            # Check neighbors
            source_neighbors = list(G.neighbors(source))
            dest_neighbors = list(G.neighbors(destination))
            print(f"\nNeighbors:")
            print(f"  Source neighbors: {source_neighbors}")
            print(f"  Destination neighbors: {dest_neighbors}")
            
        else:
            print("  No path exists between source and destination!")
    else:
        print("  Source or destination not in network!")
        
        # Show some nodes that are in the network
        print(f"  Sample nodes in network: {list(nodes)[:10]}")
    
    # Show network statistics
    print(f"\nNetwork statistics:")
    print(f"  Total nodes: {len(nodes)}")
    print(f"  Total edges: {len(edges)}")
    print(f"  Connected components: {nx.number_connected_components(G)}")
    
    # Show largest component
    largest_cc = max(nx.connected_components(G), key=len)
    print(f"  Largest component size: {len(largest_cc)}")
    print(f"  Largest component nodes: {list(largest_cc)[:10]}...")

if __name__ == "__main__":
    check_selected_nodes() 