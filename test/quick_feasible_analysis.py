#!/usr/bin/env python3
"""
Quick analysis of feasible combinations for the new node pair.
"""

import sys
import os
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

def quick_feasible_analysis():
    """Quick analysis of feasible combinations with smaller sample."""
    
    # Use the folder with most traces
    ucsb_dir = "data/ucsb/1144393236-1144450070"
    
    print("Quick Feasible Combinations Analysis")
    print("=" * 50)
    print(f"Source: 10.1.1.109")
    print(f"Destination: 10.1.1.5")
    print(f"Max hops: 10")
    print(f"Analyzing folder: {ucsb_dir}")
    
    # Collect all neighbortable files
    neighbortable_files = []
    for f in os.listdir(ucsb_dir):
        if f.startswith("neighbortable-"):
            neighbortable_files.append(os.path.join(ucsb_dir, f))
    neighbortable_files.sort()
    
    print(f"Found {len(neighbortable_files)} neighbortable files")
    
    # Use a much smaller sample for quick analysis
    sample_files = neighbortable_files[::100]  # Every 100th file
    print(f"Using {len(sample_files)} sample files for quick analysis")
    
    # Source and destination
    source = "10.1.1.109"
    destination = "10.1.1.5"
    
    # Statistics
    path_counts = []
    hop_lengths = []
    connection_rounds = 0
    total_rounds = len(sample_files)
    
    print("\nAnalyzing path counts...")
    for i, f in enumerate(tqdm(sample_files, desc="Analyzing files")):
        # Build graph for this file
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
                            if ett < 1000:  # Valid ETT
                                G.add_edge(src, dst)
                        except ValueError:
                            continue
        
        # Check if path exists
        try:
            # Use a smaller cutoff to avoid too many paths
            all_paths = list(nx.all_simple_paths(G, source, destination, cutoff=6))
            
            if all_paths:
                connection_rounds += 1
                path_count = len(all_paths)
                path_counts.append(path_count)
                
                # Analyze hop lengths
                for path in all_paths:
                    hop_length = len(path) - 1
                    hop_lengths.append(hop_length)
                
                if i < 3:  # Show details for first 3 files
                    print(f"\nFile {i+1}: {os.path.basename(f)}")
                    print(f"  Paths found: {path_count}")
                    print(f"  Shortest path: {' → '.join(min(all_paths, key=len))} ({len(min(all_paths, key=len))-1} hops)")
                    print(f"  Longest path: {' → '.join(max(all_paths, key=len))} ({len(max(all_paths, key=len))-1} hops)")
                    
                    # Show hop length distribution
                    hop_counts = Counter([len(path)-1 for path in all_paths])
                    print(f"  Hop distribution: {dict(hop_counts)}")
            else:
                if i < 3:  # Show details for first 3 files
                    print(f"\nFile {i+1}: {os.path.basename(f)} - No path found")
                    
        except nx.NetworkXNoPath:
            if i < 3:  # Show details for first 3 files
                print(f"\nFile {i+1}: {os.path.basename(f)} - No path found")
    
    # Calculate statistics
    if path_counts:
        print(f"\n" + "=" * 50)
        print(f"ANALYSIS RESULTS")
        print("=" * 50)
        
        print(f"Connection rate: {connection_rounds}/{total_rounds} = {connection_rounds/total_rounds:.1%}")
        print(f"Average paths per connected round: {np.mean(path_counts):.1f}")
        print(f"Median paths per connected round: {np.median(path_counts):.1f}")
        print(f"Min paths: {np.min(path_counts)}")
        print(f"Max paths: {np.max(path_counts)}")
        print(f"Standard deviation: {np.std(path_counts):.1f}")
        
        # Path count distribution
        print(f"\nPath count distribution:")
        path_count_dist = Counter(path_counts)
        for count, freq in sorted(path_count_dist.items()):
            print(f"  {count} paths: {freq} rounds ({freq/len(path_counts):.1%})")
        
        # Hop length analysis
        if hop_lengths:
            print(f"\nHop length analysis:")
            print(f"  Average hop length: {np.mean(hop_lengths):.1f}")
            print(f"  Median hop length: {np.median(hop_lengths):.1f}")
            print(f"  Min hops: {np.min(hop_lengths)}")
            print(f"  Max hops: {np.max(hop_lengths)}")
            
            hop_length_dist = Counter(hop_lengths)
            print(f"  Hop length distribution:")
            for hops, freq in sorted(hop_length_dist.items()):
                print(f"    {hops} hops: {freq} paths ({freq/len(hop_lengths):.1%})")
        
        # Estimate for full dataset
        print(f"\nESTIMATION FOR FULL DATASET")
        print("-" * 40)
        avg_paths_per_round = np.mean(path_counts)
        connection_rate = connection_rounds / total_rounds
        
        print(f"Estimated average feasible combinations per minute:")
        print(f"  = {avg_paths_per_round:.1f} paths × {connection_rate:.1%} connection rate")
        print(f"  = {avg_paths_per_round * connection_rate:.1f} feasible combinations per minute")
        
        # Extrapolate to 10 hops (rough estimate)
        print(f"\nROUGH EXTRAPOLATION TO 10 HOPS:")
        print(f"  Current analysis: 6 hops max")
        print(f"  Estimated for 10 hops: {avg_paths_per_round * 10:.0f} paths per round")
        print(f"  Estimated feasible combinations per minute: {avg_paths_per_round * 10 * connection_rate:.0f}")
        
        return avg_paths_per_round * connection_rate
    else:
        print("No paths found in any of the analyzed files!")
        return 0

def test_first_few_files():
    """Test the first few files to get a quick estimate."""
    print(f"\n" + "=" * 50)
    print(f"QUICK TEST OF FIRST FEW FILES")
    print("=" * 50)
    
    ucsb_dir = "data/ucsb/1144393236-1144450070"
    neighbortable_files = []
    for f in os.listdir(ucsb_dir):
        if f.startswith("neighbortable-"):
            neighbortable_files.append(os.path.join(ucsb_dir, f))
    neighbortable_files.sort()
    
    # Test first 5 files
    test_files = neighbortable_files[:5]
    source = "10.1.1.109"
    destination = "10.1.1.5"
    
    total_paths = 0
    connected_files = 0
    
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
        
        try:
            # Try different hop limits
            for max_hops in [3, 4, 5, 6]:
                try:
                    all_paths = list(nx.all_simple_paths(G, source, destination, cutoff=max_hops))
                    if all_paths:
                        print(f"  {max_hops} hops: {len(all_paths)} paths")
                        if max_hops == 6:  # Use 6 hops for estimation
                            total_paths += len(all_paths)
                            connected_files += 1
                    else:
                        print(f"  {max_hops} hops: 0 paths")
                        break
                except nx.NetworkXNoPath:
                    print(f"  {max_hops} hops: 0 paths")
                    break
        except Exception as e:
            print(f"  Error: {e}")
    
    if connected_files > 0:
        avg_paths = total_paths / connected_files
        connection_rate = connected_files / len(test_files)
        print(f"\nQuick estimate:")
        print(f"  Average paths (6 hops): {avg_paths:.1f}")
        print(f"  Connection rate: {connection_rate:.1%}")
        print(f"  Estimated feasible combinations per minute: {avg_paths * connection_rate:.1f}")

if __name__ == "__main__":
    quick_feasible_analysis()
    test_first_few_files() 