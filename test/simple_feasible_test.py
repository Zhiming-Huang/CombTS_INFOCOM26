#!/usr/bin/env python3
"""
Simple test to estimate feasible combinations for the new node pair.
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

def simple_feasible_test():
    """Simple test with limited hops and files."""
    
    # Use the folder with most traces
    ucsb_dir = "data/ucsb/1144393236-1144450070"
    
    print("Simple Feasible Combinations Test")
    print("=" * 40)
    print(f"Source: 10.1.1.109")
    print(f"Destination: 10.1.1.5")
    print(f"Max hops: 3, 4, 5 (limited)")
    print(f"Analyzing folder: {ucsb_dir}")
    
    # Collect all neighbortable files
    neighbortable_files = []
    for f in os.listdir(ucsb_dir):
        if f.startswith("neighbortable-"):
            neighbortable_files.append(os.path.join(ucsb_dir, f))
    neighbortable_files.sort()
    
    print(f"Found {len(neighbortable_files)} neighbortable files")
    
    # Test only first 10 files
    test_files = neighbortable_files[:10]
    print(f"Testing first {len(test_files)} files")
    
    # Source and destination
    source = "10.1.1.109"
    destination = "10.1.1.5"
    
    # Statistics for different hop limits
    hop_stats = {3: [], 4: [], 5: []}
    connection_rounds = 0
    
    print("\nTesting different hop limits...")
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
        
        # Test different hop limits
        file_connected = False
        for max_hops in [3, 4, 5]:
            try:
                all_paths = list(nx.all_simple_paths(G, source, destination, cutoff=max_hops))
                path_count = len(all_paths)
                hop_stats[max_hops].append(path_count)
                
                if path_count > 0:
                    file_connected = True
                    print(f"  {max_hops} hops: {path_count} paths")
                    
                    # Show some sample paths for 3 hops
                    if max_hops == 3 and path_count > 0:
                        print(f"    Sample paths:")
                        for j, path in enumerate(all_paths[:3]):
                            print(f"      {j+1}. {' → '.join(path)} ({len(path)-1} hops)")
                        if path_count > 3:
                            print(f"      ... and {path_count-3} more paths")
                else:
                    print(f"  {max_hops} hops: 0 paths")
                    
            except nx.NetworkXNoPath:
                hop_stats[max_hops].append(0)
                print(f"  {max_hops} hops: 0 paths")
        
        if file_connected:
            connection_rounds += 1
    
    # Calculate statistics
    print(f"\n" + "=" * 40)
    print(f"RESULTS")
    print("=" * 40)
    
    connection_rate = connection_rounds / len(test_files)
    print(f"Connection rate: {connection_rounds}/{len(test_files)} = {connection_rate:.1%}")
    
    for max_hops in [3, 4, 5]:
        if hop_stats[max_hops]:
            connected_paths = [p for p in hop_stats[max_hops] if p > 0]
            if connected_paths:
                avg_paths = np.mean(connected_paths)
                print(f"\n{max_hops} hops:")
                print(f"  Average paths per connected round: {avg_paths:.1f}")
                print(f"  Min paths: {np.min(connected_paths)}")
                print(f"  Max paths: {np.max(connected_paths)}")
                print(f"  Estimated feasible combinations per minute: {avg_paths * connection_rate:.1f}")
            else:
                print(f"\n{max_hops} hops: No paths found")
    
    # Estimate for 10 hops (rough extrapolation)
    print(f"\nROUGH ESTIMATION FOR 10 HOPS:")
    print("-" * 30)
    
    # Use 5-hop data as base and extrapolate
    if hop_stats[5]:
        connected_paths_5 = [p for p in hop_stats[5] if p > 0]
        if connected_paths_5:
            avg_5_hops = np.mean(connected_paths_5)
            # Rough estimate: each additional hop roughly doubles the paths
            estimated_10_hops = avg_5_hops * (2 ** 5)  # 5 more hops
            print(f"  Based on 5-hop average: {avg_5_hops:.1f} paths")
            print(f"  Estimated for 10 hops: {estimated_10_hops:.0f} paths per round")
            print(f"  Estimated feasible combinations per minute: {estimated_10_hops * connection_rate:.0f}")
    
    return connection_rate

if __name__ == "__main__":
    simple_feasible_test() 