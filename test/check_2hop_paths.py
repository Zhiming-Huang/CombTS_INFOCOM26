#!/usr/bin/env python3
"""
Check 2-hop paths for the new node pair.
"""

import sys
import os
import numpy as np
import networkx as nx
from collections import Counter

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

def check_2hop_paths():
    """Check 2-hop paths for 10.1.1.109 → 10.1.1.5."""
    
    # Use the folder with most traces
    ucsb_dir = "data/ucsb/1144393236-1144450070"
    
    print("2-Hop Paths Analysis")
    print("=" * 30)
    print(f"Source: 10.1.1.109")
    print(f"Destination: 10.1.1.5")
    print(f"Max hops: 2")
    print(f"Analyzing folder: {ucsb_dir}")
    
    # Collect all neighbortable files
    neighbortable_files = []
    for f in os.listdir(ucsb_dir):
        if f.startswith("neighbortable-"):
            neighbortable_files.append(os.path.join(ucsb_dir, f))
    neighbortable_files.sort()
    
    print(f"Found {len(neighbortable_files)} neighbortable files")
    
    # Test first 10 files
    test_files = neighbortable_files[:10]
    print(f"Testing first {len(test_files)} files")
    
    # Source and destination
    source = "10.1.1.109"
    destination = "10.1.1.5"
    
    # Statistics
    path_counts = []
    all_2hop_paths = []
    connection_rounds = 0
    
    print("\nAnalyzing 2-hop paths...")
    for i, f in enumerate(test_files):
        print(f"\nFile {i+1}: {os.path.basename(f)}")
        
        # Build graph
        G = nx.Graph()
        with open(f, 'r') as fin:
            for line in fin:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                src = parts[0]
                for j in range(1, len(parts), 2):
                    if j + 1 < len(parts):
                        dst = parts[j]
                        try:
                            ett = float(parts[j+1])
                            if ett < 1000:
                                G.add_edge(src, dst)
                        except ValueError:
                            continue
        
        # Check for 2-hop paths
        try:
            # Find all simple paths with exactly 2 hops
            all_paths = list(nx.all_simple_paths(G, source, destination, cutoff=2))
            
            # Filter for exactly 2-hop paths
            two_hop_paths = [path for path in all_paths if len(path) == 3]  # 3 nodes = 2 hops
            
            if two_hop_paths:
                connection_rounds += 1
                path_count = len(two_hop_paths)
                path_counts.append(path_count)
                all_2hop_paths.extend(two_hop_paths)
                
                print(f"  2-hop paths found: {path_count}")
                print(f"  All 2-hop paths:")
                for j, path in enumerate(two_hop_paths):
                    print(f"    {j+1}. {' → '.join(path)} (2 hops)")
            else:
                print(f"  No 2-hop paths found")
                path_counts.append(0)
                
        except nx.NetworkXNoPath:
            print(f"  No path found")
            path_counts.append(0)
    
    # Calculate statistics
    print(f"\n" + "=" * 30)
    print(f"RESULTS")
    print("=" * 30)
    
    connection_rate = connection_rounds / len(test_files)
    print(f"Connection rate: {connection_rounds}/{len(test_files)} = {connection_rate:.1%}")
    
    if path_counts:
        connected_paths = [p for p in path_counts if p > 0]
        if connected_paths:
            avg_paths = np.mean(connected_paths)
            print(f"\n2-hop paths:")
            print(f"  Average paths per connected round: {avg_paths:.1f}")
            print(f"  Min paths: {np.min(connected_paths)}")
            print(f"  Max paths: {np.max(connected_paths)}")
            print(f"  Estimated feasible combinations per minute: {avg_paths * connection_rate:.1f}")
            
            # Show unique 2-hop paths
            unique_paths = set()
            for path in all_2hop_paths:
                unique_paths.add(tuple(path))
            
            print(f"\nUnique 2-hop paths found:")
            for i, path in enumerate(sorted(unique_paths)):
                print(f"  {i+1}. {' → '.join(path)}")
            
            print(f"\nTotal unique 2-hop paths: {len(unique_paths)}")
        else:
            print(f"\n2-hop paths: No paths found")
    
    return connection_rate

if __name__ == "__main__":
    check_2hop_paths() 