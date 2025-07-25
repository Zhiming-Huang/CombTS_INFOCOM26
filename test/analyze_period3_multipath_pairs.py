#!/usr/bin/env python3
"""
Analyze node pairs that have multiple paths in every timestamp of Period 3.
"""

import os
import sys
import numpy as np
import networkx as nx
from pathlib import Path
from collections import defaultdict

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def parse_neighbortable_file(file_path):
    """Parse neighbortable file and return edges."""
    edges = set()
    try:
        with open(file_path, 'r') as fin:
            for line in fin:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                src = parts[0]
                for i in range(1, len(parts), 2):
                    dst = parts[i]
                    try:
                        ett = float(parts[i+1])
                        if ett < 1000:
                            edges.add((src, dst))
                    except Exception:
                        continue
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    return edges

def find_multipath_pairs_in_period3():
    """Find node pairs that have multiple paths in every timestamp of Period 3."""
    
    # Get Period 3 data
    ucsb_dir = Path("data/ucsb")
    period3_folder = ucsb_dir / "1144393236-1144450070"
    
    if not period3_folder.exists():
        print("Period 3 folder not found!")
        return
    
    print(f"Analyzing Period 3: {period3_folder}")
    
    # Get all trace files
    trace_files = sorted([f for f in period3_folder.iterdir() if f.name.startswith("neighbortable")])
    print(f"Found {len(trace_files)} trace files")
    
    # Get all unique nodes from all files
    all_nodes = set()
    for file_path in trace_files:
        edges = parse_neighbortable_file(file_path)
        for src, dst in edges:
            all_nodes.add(src)
            all_nodes.add(dst)
    
    all_nodes = sorted(all_nodes)
    print(f"Total unique nodes: {len(all_nodes)}")
    
    # Analyze each timestamp
    timestamp_results = {}
    multipath_pairs_by_timestamp = {}
    
    for i, file_path in enumerate(trace_files):
        print(f"\nAnalyzing timestamp {i+1}/{len(trace_files)}: {file_path.name}")
        
        # Parse edges for this timestamp
        edges = parse_neighbortable_file(file_path)
        
        # Build graph
        G = nx.Graph()
        G.add_nodes_from(all_nodes)
        G.add_edges_from(edges)
        
        # Find all node pairs with multiple paths
        multipath_pairs = []
        
        # Test all possible node pairs
        for src_idx, src in enumerate(all_nodes):
            for dst_idx, dst in enumerate(all_nodes):
                if src_idx >= dst_idx:  # Avoid duplicates and self-loops
                    continue
                
                if src in G and dst in G and nx.has_path(G, src, dst):
                    try:
                        # Find all simple paths up to 3 hops
                        paths = list(nx.all_simple_paths(G, src, dst, cutoff=3))
                        if len(paths) >= 2:  # At least 2 paths
                            multipath_pairs.append((src, dst, len(paths)))
                    except nx.NetworkXNoPath:
                        continue
        
        multipath_pairs_by_timestamp[i] = multipath_pairs
        print(f"  Found {len(multipath_pairs)} node pairs with multiple paths")
        
        # Show some examples
        if multipath_pairs:
            print(f"  Examples:")
            for src, dst, path_count in multipath_pairs[:5]:
                print(f"    {src} -> {dst}: {path_count} paths")
    
    # Find node pairs that have multiple paths in ALL timestamps
    print(f"\n{'='*80}")
    print("FINDING NODE PAIRS WITH MULTIPLE PATHS IN ALL TIMESTAMPS")
    print(f"{'='*80}")
    
    # Get all unique node pairs that appear in any timestamp
    all_node_pairs = set()
    for pairs in multipath_pairs_by_timestamp.values():
        for src, dst, _ in pairs:
            all_node_pairs.add((src, dst))
    
    print(f"Total unique node pairs with multiple paths in any timestamp: {len(all_node_pairs)}")
    
    # Check which pairs have multiple paths in ALL timestamps
    stable_multipath_pairs = []
    
    for src, dst in all_node_pairs:
        multipath_in_all = True
        path_counts = []
        
        for timestamp_idx in range(len(trace_files)):
            pairs_in_timestamp = multipath_pairs_by_timestamp[timestamp_idx]
            found = False
            
            for pair_src, pair_dst, path_count in pairs_in_timestamp:
                if (pair_src == src and pair_dst == dst) or (pair_src == dst and pair_dst == src):
                    path_counts.append(path_count)
                    found = True
                    break
            
            if not found:
                multipath_in_all = False
                break
        
        if multipath_in_all:
            avg_paths = sum(path_counts) / len(path_counts)
            min_paths = min(path_counts)
            max_paths = max(path_counts)
            stable_multipath_pairs.append((src, dst, avg_paths, min_paths, max_paths))
    
    # Sort by average number of paths
    stable_multipath_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\nNode pairs with multiple paths in ALL {len(trace_files)} timestamps: {len(stable_multipath_pairs)}")
    
    if stable_multipath_pairs:
        print(f"\nTop 20 stable multipath pairs:")
        print(f"{'Source':<15} {'Destination':<15} {'Avg Paths':<10} {'Min':<5} {'Max':<5}")
        print("-" * 60)
        
        for src, dst, avg_paths, min_paths, max_paths in stable_multipath_pairs[:20]:
            print(f"{src:<15} {dst:<15} {avg_paths:<10.1f} {min_paths:<5} {max_paths:<5}")
        
        # Show detailed analysis for top 5 pairs
        print(f"\n{'='*80}")
        print("DETAILED ANALYSIS OF TOP 5 STABLE PAIRS")
        print(f"{'='*80}")
        
        for i, (src, dst, avg_paths, min_paths, max_paths) in enumerate(stable_multipath_pairs[:5]):
            print(f"\n{i+1}. {src} -> {dst}")
            print(f"   Average paths: {avg_paths:.1f}")
            print(f"   Path range: {min_paths} - {max_paths}")
            
            # Show path counts for first 10 timestamps
            print(f"   First 10 timestamps path counts:")
            for timestamp_idx in range(min(10, len(trace_files))):
                pairs_in_timestamp = multipath_pairs_by_timestamp[timestamp_idx]
                for pair_src, pair_dst, path_count in pairs_in_timestamp:
                    if (pair_src == src and pair_dst == dst) or (pair_src == dst and pair_dst == src):
                        print(f"     Timestamp {timestamp_idx+1}: {path_count} paths")
                        break
    else:
        print("No node pairs found with multiple paths in all timestamps.")
    
    return stable_multipath_pairs

if __name__ == "__main__":
    stable_pairs = find_multipath_pairs_in_period3() 
"""
Analyze node pairs that have multiple paths in every timestamp of Period 3.
"""

import os
import sys
import numpy as np
import networkx as nx
from pathlib import Path
from collections import defaultdict

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def parse_neighbortable_file(file_path):
    """Parse neighbortable file and return edges."""
    edges = set()
    try:
        with open(file_path, 'r') as fin:
            for line in fin:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                src = parts[0]
                for i in range(1, len(parts), 2):
                    dst = parts[i]
                    try:
                        ett = float(parts[i+1])
                        if ett < 1000:
                            edges.add((src, dst))
                    except Exception:
                        continue
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    return edges

def find_multipath_pairs_in_period3():
    """Find node pairs that have multiple paths in every timestamp of Period 3."""
    
    # Get Period 3 data
    ucsb_dir = Path("data/ucsb")
    period3_folder = ucsb_dir / "1144393236-1144450070"
    
    if not period3_folder.exists():
        print("Period 3 folder not found!")
        return
    
    print(f"Analyzing Period 3: {period3_folder}")
    
    # Get all trace files
    trace_files = sorted([f for f in period3_folder.iterdir() if f.name.startswith("neighbortable")])
    print(f"Found {len(trace_files)} trace files")
    
    # Get all unique nodes from all files
    all_nodes = set()
    for file_path in trace_files:
        edges = parse_neighbortable_file(file_path)
        for src, dst in edges:
            all_nodes.add(src)
            all_nodes.add(dst)
    
    all_nodes = sorted(all_nodes)
    print(f"Total unique nodes: {len(all_nodes)}")
    
    # Analyze each timestamp
    timestamp_results = {}
    multipath_pairs_by_timestamp = {}
    
    for i, file_path in enumerate(trace_files):
        print(f"\nAnalyzing timestamp {i+1}/{len(trace_files)}: {file_path.name}")
        
        # Parse edges for this timestamp
        edges = parse_neighbortable_file(file_path)
        
        # Build graph
        G = nx.Graph()
        G.add_nodes_from(all_nodes)
        G.add_edges_from(edges)
        
        # Find all node pairs with multiple paths
        multipath_pairs = []
        
        # Test all possible node pairs
        for src_idx, src in enumerate(all_nodes):
            for dst_idx, dst in enumerate(all_nodes):
                if src_idx >= dst_idx:  # Avoid duplicates and self-loops
                    continue
                
                if src in G and dst in G and nx.has_path(G, src, dst):
                    try:
                        # Find all simple paths up to 3 hops
                        paths = list(nx.all_simple_paths(G, src, dst, cutoff=3))
                        if len(paths) >= 2:  # At least 2 paths
                            multipath_pairs.append((src, dst, len(paths)))
                    except nx.NetworkXNoPath:
                        continue
        
        multipath_pairs_by_timestamp[i] = multipath_pairs
        print(f"  Found {len(multipath_pairs)} node pairs with multiple paths")
        
        # Show some examples
        if multipath_pairs:
            print(f"  Examples:")
            for src, dst, path_count in multipath_pairs[:5]:
                print(f"    {src} -> {dst}: {path_count} paths")
    
    # Find node pairs that have multiple paths in ALL timestamps
    print(f"\n{'='*80}")
    print("FINDING NODE PAIRS WITH MULTIPLE PATHS IN ALL TIMESTAMPS")
    print(f"{'='*80}")
    
    # Get all unique node pairs that appear in any timestamp
    all_node_pairs = set()
    for pairs in multipath_pairs_by_timestamp.values():
        for src, dst, _ in pairs:
            all_node_pairs.add((src, dst))
    
    print(f"Total unique node pairs with multiple paths in any timestamp: {len(all_node_pairs)}")
    
    # Check which pairs have multiple paths in ALL timestamps
    stable_multipath_pairs = []
    
    for src, dst in all_node_pairs:
        multipath_in_all = True
        path_counts = []
        
        for timestamp_idx in range(len(trace_files)):
            pairs_in_timestamp = multipath_pairs_by_timestamp[timestamp_idx]
            found = False
            
            for pair_src, pair_dst, path_count in pairs_in_timestamp:
                if (pair_src == src and pair_dst == dst) or (pair_src == dst and pair_dst == src):
                    path_counts.append(path_count)
                    found = True
                    break
            
            if not found:
                multipath_in_all = False
                break
        
        if multipath_in_all:
            avg_paths = sum(path_counts) / len(path_counts)
            min_paths = min(path_counts)
            max_paths = max(path_counts)
            stable_multipath_pairs.append((src, dst, avg_paths, min_paths, max_paths))
    
    # Sort by average number of paths
    stable_multipath_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\nNode pairs with multiple paths in ALL {len(trace_files)} timestamps: {len(stable_multipath_pairs)}")
    
    if stable_multipath_pairs:
        print(f"\nTop 20 stable multipath pairs:")
        print(f"{'Source':<15} {'Destination':<15} {'Avg Paths':<10} {'Min':<5} {'Max':<5}")
        print("-" * 60)
        
        for src, dst, avg_paths, min_paths, max_paths in stable_multipath_pairs[:20]:
            print(f"{src:<15} {dst:<15} {avg_paths:<10.1f} {min_paths:<5} {max_paths:<5}")
        
        # Show detailed analysis for top 5 pairs
        print(f"\n{'='*80}")
        print("DETAILED ANALYSIS OF TOP 5 STABLE PAIRS")
        print(f"{'='*80}")
        
        for i, (src, dst, avg_paths, min_paths, max_paths) in enumerate(stable_multipath_pairs[:5]):
            print(f"\n{i+1}. {src} -> {dst}")
            print(f"   Average paths: {avg_paths:.1f}")
            print(f"   Path range: {min_paths} - {max_paths}")
            
            # Show path counts for first 10 timestamps
            print(f"   First 10 timestamps path counts:")
            for timestamp_idx in range(min(10, len(trace_files))):
                pairs_in_timestamp = multipath_pairs_by_timestamp[timestamp_idx]
                for pair_src, pair_dst, path_count in pairs_in_timestamp:
                    if (pair_src == src and pair_dst == dst) or (pair_src == dst and pair_dst == src):
                        print(f"     Timestamp {timestamp_idx+1}: {path_count} paths")
                        break
    else:
        print("No node pairs found with multiple paths in all timestamps.")
    
    return stable_multipath_pairs

if __name__ == "__main__":
    stable_pairs = find_multipath_pairs_in_period3() 