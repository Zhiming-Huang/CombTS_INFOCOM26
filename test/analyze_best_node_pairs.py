#!/usr/bin/env python3
"""
Analyze the best node pairs with most paths across all 948 files.
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

def analyze_best_node_pairs():
    """Analyze which node pair has the most paths on average across all files."""
    
    # Use the folder with most traces
    ucsb_dir = "data/ucsb/1144393236-1144450070"
    
    print("Analyzing Best Node Pairs Across 948 Files")
    print("=" * 60)
    print(f"Analyzing folder: {ucsb_dir}")
    
    # Collect all neighbortable files
    neighbortable_files = []
    for f in os.listdir(ucsb_dir):
        if f.startswith("neighbortable-"):
            neighbortable_files.append(os.path.join(ucsb_dir, f))
    neighbortable_files.sort()
    
    print(f"Found {len(neighbortable_files)} neighbortable files")
    
    # Collect all unique nodes first
    print("Collecting all unique nodes...")
    all_nodes = set()
    for f in tqdm(neighbortable_files[:10], desc="Scanning nodes"):  # Sample first 10 files to get all nodes
        with open(f, 'r') as fin:
            for line in fin:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                src = parts[0]
                all_nodes.add(src)
                for i in range(1, len(parts), 2):
                    if i + 1 < len(parts):
                        dst = parts[i]
                        try:
                            ett = float(parts[i+1])
                            if ett < 1000:  # Valid ETT
                                all_nodes.add(dst)
                        except ValueError:
                            continue
    
    all_nodes = sorted(all_nodes)
    print(f"Total unique nodes found: {len(all_nodes)}")
    
    # Initialize counters for node pairs
    node_pair_paths = defaultdict(list)  # (node1, node2) -> list of path counts
    node_pair_connected_rounds = defaultdict(int)  # (node1, node2) -> number of rounds with connection
    
    # Analyze a subset of files for efficiency (every 10th file)
    sample_files = neighbortable_files[::10]  # Sample every 10th file
    print(f"Analyzing {len(sample_files)} sample files (every 10th file)")
    
    # Use tqdm for progress bar
    for f in tqdm(sample_files, desc="Analyzing files"):
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
                for i in range(1, len(parts), 2):
                    if i + 1 < len(parts):
                        dst = parts[i]
                        try:
                            ett = float(parts[i+1])
                            if ett < 1000:  # Valid ETT
                                G.add_edge(src, dst)
                        except ValueError:
                            continue
        
        # Find connected components
        components = list(nx.connected_components(G))
        
        # For each component, analyze node pairs
        for component in components:
            component_nodes = list(component)
            if len(component_nodes) < 2:
                continue
            
            # Check all pairs within this component
            for i, node1 in enumerate(component_nodes):
                for j, node2 in enumerate(component_nodes[i+1:], i+1):
                    if node1 != node2:
                        # Count all paths between these nodes
                        try:
                            all_paths = list(nx.all_simple_paths(G, node1, node2, cutoff=10))  # Limit path length
                            path_count = len(all_paths)
                            
                            # Store the path count for this pair
                            pair = tuple(sorted([node1, node2]))  # Ensure consistent ordering
                            node_pair_paths[pair].append(path_count)
                            node_pair_connected_rounds[pair] += 1
                            
                        except nx.NetworkXNoPath:
                            continue
    
    print(f"\nAnalysis completed. Found {len(node_pair_paths)} connected node pairs.")
    
    # Calculate statistics for each node pair
    print("Calculating statistics...")
    node_pair_stats = []
    for pair, path_counts in tqdm(node_pair_paths.items(), desc="Processing pairs"):
        if len(path_counts) > 0:
            avg_paths = np.mean(path_counts)
            max_paths = np.max(path_counts)
            min_paths = np.min(path_counts)
            connected_rounds = node_pair_connected_rounds[pair]
            connection_rate = connected_rounds / len(sample_files)
            
            node_pair_stats.append({
                'pair': pair,
                'avg_paths': avg_paths,
                'max_paths': max_paths,
                'min_paths': min_paths,
                'connected_rounds': connected_rounds,
                'connection_rate': connection_rate,
                'total_path_counts': path_counts
            })
    
    # Sort by average number of paths (descending)
    node_pair_stats.sort(key=lambda x: x['avg_paths'], reverse=True)
    
    print(f"\nTop 20 Node Pairs by Average Path Count:")
    print("-" * 80)
    print(f"{'Rank':<4} {'Node Pair':<25} {'Avg Paths':<10} {'Max Paths':<10} {'Min Paths':<10} {'Connected':<10} {'Rate':<8}")
    print("-" * 80)
    
    for i, stats in enumerate(node_pair_stats[:20]):
        pair_str = f"{stats['pair'][0]} → {stats['pair'][1]}"
        print(f"{i+1:<4} {pair_str:<25} {stats['avg_paths']:<10.2f} {stats['max_paths']:<10} {stats['min_paths']:<10} {stats['connected_rounds']:<10} {stats['connection_rate']:<8.2%}")
    
    # Show detailed info for top 5 pairs
    print(f"\nDetailed Analysis of Top 5 Node Pairs:")
    print("=" * 60)
    
    for i, stats in enumerate(node_pair_stats[:5]):
        print(f"\n{i+1}. {stats['pair'][0]} → {stats['pair'][1]}")
        print(f"   Average paths: {stats['avg_paths']:.2f}")
        print(f"   Max paths: {stats['max_paths']}")
        print(f"   Min paths: {stats['min_paths']}")
        print(f"   Connected in {stats['connected_rounds']} out of {len(sample_files)} rounds ({stats['connection_rate']:.1%})")
        
        # Show path count distribution
        path_counts = stats['total_path_counts']
        unique_counts = Counter(path_counts)
        print(f"   Path count distribution: {dict(unique_counts)}")
    
    # Recommend the best pair
    if node_pair_stats:
        best_pair = node_pair_stats[0]
        print(f"\n" + "=" * 60)
        print(f"RECOMMENDED NODE PAIR:")
        print(f"Source: {best_pair['pair'][0]}")
        print(f"Destination: {best_pair['pair'][1]}")
        print(f"Average paths: {best_pair['avg_paths']:.2f}")
        print(f"Connection rate: {best_pair['connection_rate']:.1%}")
        print("=" * 60)
        
        return best_pair['pair'][0], best_pair['pair'][1]
    
    return None, None

def test_recommended_pair():
    """Test the recommended node pair with a small sample."""
    source, destination = analyze_best_node_pairs()
    
    if source and destination:
        print(f"\nTesting recommended pair: {source} → {destination}")
        
        # Test with a few files
        ucsb_dir = "data/ucsb/1144393236-1144450070"
        neighbortable_files = []
        for f in os.listdir(ucsb_dir):
            if f.startswith("neighbortable-"):
                neighbortable_files.append(os.path.join(ucsb_dir, f))
        neighbortable_files.sort()
        
        # Test first 5 files
        test_files = neighbortable_files[:5]
        
        for i, f in enumerate(tqdm(test_files, desc="Testing files")):
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
                all_paths = list(nx.all_simple_paths(G, source, destination, cutoff=10))
                print(f"\nFile {i+1}: {os.path.basename(f)} - Paths found: {len(all_paths)}")
                if len(all_paths) > 0:
                    shortest_path = min(all_paths, key=len)
                    print(f"  Shortest path: {' → '.join(shortest_path)} ({len(shortest_path)-1} hops)")
            except nx.NetworkXNoPath:
                print(f"\nFile {i+1}: {os.path.basename(f)} - No path found")

if __name__ == "__main__":
    test_recommended_pair() 