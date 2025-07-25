#!/usr/bin/env python3
"""
Directly count 3-hop paths for UCSB meshnet
"""

import os
import sys
import numpy as np
import networkx as nx
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def count_3hop_paths_direct():
    """Directly count 3-hop paths for the current node pair"""
    
    # Current best node pair
    source = "10.1.1.109"
    destination = "10.1.1.5"
    
    print(f"Directly counting 3-hop paths for {source} → {destination}")
    print("=" * 60)
    
    # Get the data directory
    data_dir = Path(__file__).parent.parent / "data" / "ucsb"
    target_folder = data_dir / "1144393236-1144450070"
    
    # Get all neighbortable files
    neighbortable_files = sorted([f for f in target_folder.glob("neighbortable-*")])
    
    print(f"Found {len(neighbortable_files)} neighbortable files")
    
    # Test on first 10 files
    test_files = neighbortable_files[:10]
    
    total_3hop_paths = 0
    total_connections = 0
    
    for i, file_path in enumerate(test_files):
        print(f"\nProcessing file {i+1}/{len(test_files)}: {file_path.name}")
        
        try:
            # Build graph manually
            G = nx.Graph()
            
            with open(file_path, 'r') as fin:
                for line in fin:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.strip().split()
                    src = parts[0]
                    
                    for j in range(1, len(parts), 2):
                        dst = parts[j]
                        try:
                            ett = float(parts[j+1])
                            G.add_edge(src, dst, weight=ett)
                        except Exception:
                            continue
            
            # Check if source and destination are in the graph
            if source not in G or destination not in G:
                print(f"  Source or destination not in graph")
                continue
            
            # Check connectivity
            try:
                shortest_path = nx.shortest_path(G, source, destination)
                print(f"  Shortest path length: {len(shortest_path) - 1} hops")
                
                # Find all simple paths up to 3 hops
                all_paths = []
                for path in nx.all_simple_paths(G, source, destination, cutoff=3):
                    if len(path) <= 4:  # 3 hops = 4 nodes
                        all_paths.append(path)
                
                # Filter for exactly 3-hop paths
                three_hop_paths = [path for path in all_paths if len(path) == 4]
                
                print(f"  All paths up to 3 hops: {len(all_paths)}")
                print(f"  Exactly 3-hop paths: {len(three_hop_paths)}")
                
                if three_hop_paths:
                    print(f"  Sample 3-hop paths:")
                    for j, path in enumerate(three_hop_paths[:3]):  # Show first 3
                        print(f"    {j+1}: {' → '.join(path)}")
                    if len(three_hop_paths) > 3:
                        print(f"    ... and {len(three_hop_paths) - 3} more")
                
                total_3hop_paths += len(three_hop_paths)
                total_connections += 1
                
            except nx.NetworkXNoPath:
                print(f"  No path exists between source and destination")
                continue
                
        except Exception as e:
            print(f"  Error processing file: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Files with connections: {total_connections}/{len(test_files)}")
    print(f"Average 3-hop paths per file: {total_3hop_paths / total_connections:.1f}")
    print(f"Total 3-hop paths: {total_3hop_paths}")
    
    # Estimate for all files
    if total_connections > 0:
        estimated_total = (total_3hop_paths / total_connections) * len(neighbortable_files)
        print(f"Estimated total 3-hop paths (all {len(neighbortable_files)} files): {estimated_total:,.0f}")

if __name__ == "__main__":
    count_3hop_paths_direct() 