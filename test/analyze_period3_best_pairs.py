#!/usr/bin/env python3
"""
Analyze node pairs with highest coverage in Period 3.
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

def analyze_period3_best_pairs():
    """Analyze node pairs with highest coverage in Period 3."""
    
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
    multipath_pairs_by_timestamp = {}
    
    for i, file_path in enumerate(trace_files):
        if i % 100 == 0:  # Show progress every 100 timestamps
            print(f"Analyzing timestamp {i+1}/{len(trace_files)}: {file_path.name}")
        
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
    
    # Count coverage for each node pair
    pair_coverage = defaultdict(list)
    
    for timestamp_idx, pairs in multipath_pairs_by_timestamp.items():
        for src, dst, path_count in pairs:
            pair_coverage[(src, dst)].append(path_count)
    
    # Calculate statistics for each pair
    pair_stats = []
    
    for (src, dst), path_counts in pair_coverage.items():
        coverage = len(path_counts) / len(trace_files) * 100
        avg_paths = sum(path_counts) / len(path_counts)
        min_paths = min(path_counts)
        max_paths = max(path_counts)
        
        pair_stats.append({
            'source': src,
            'destination': dst,
            'coverage': coverage,
            'avg_paths': avg_paths,
            'min_paths': min_paths,
            'max_paths': max_paths,
            'total_timestamps': len(path_counts)
        })
    
    # Sort by coverage (descending)
    pair_stats.sort(key=lambda x: x['coverage'], reverse=True)
    
    print(f"\n{'='*80}")
    print("NODE PAIRS WITH HIGHEST COVERAGE IN PERIOD 3")
    print(f"{'='*80}")
    
    print(f"Total unique node pairs with multiple paths in any timestamp: {len(pair_stats)}")
    print()
    
    # Show top 20 pairs by coverage
    print(f"Top 20 node pairs by coverage:")
    print(f"{'Source':<15} {'Destination':<15} {'Coverage':<10} {'Avg Paths':<10} {'Min':<5} {'Max':<5} {'Timestamps':<10}")
    print("-" * 85)
    
    for i, stats in enumerate(pair_stats[:20]):
        print(f"{stats['source']:<15} {stats['destination']:<15} {stats['coverage']:<10.1f} {stats['avg_paths']:<10.1f} {stats['min_paths']:<5} {stats['max_paths']:<5} {stats['total_timestamps']:<10}")
    
    # Show pairs with >90% coverage
    high_coverage_pairs = [p for p in pair_stats if p['coverage'] > 90]
    print(f"\nNode pairs with >90% coverage: {len(high_coverage_pairs)}")
    
    if high_coverage_pairs:
        print(f"\nHigh coverage pairs (>90%):")
        print(f"{'Source':<15} {'Destination':<15} {'Coverage':<10} {'Avg Paths':<10} {'Min':<5} {'Max':<5}")
        print("-" * 70)
        
        for stats in high_coverage_pairs:
            print(f"{stats['source']:<15} {stats['destination']:<15} {stats['coverage']:<10.1f} {stats['avg_paths']:<10.1f} {stats['min_paths']:<5} {stats['max_paths']:<5}")
    
    # Show pairs with >80% coverage
    medium_coverage_pairs = [p for p in pair_stats if 80 < p['coverage'] <= 90]
    print(f"\nNode pairs with 80-90% coverage: {len(medium_coverage_pairs)}")
    
    if medium_coverage_pairs:
        print(f"\nMedium coverage pairs (80-90%):")
        print(f"{'Source':<15} {'Destination':<15} {'Coverage':<10} {'Avg Paths':<10} {'Min':<5} {'Max':<5}")
        print("-" * 70)
        
        for stats in medium_coverage_pairs[:10]:  # Show first 10
            print(f"{stats['source']:<15} {stats['destination']:<15} {stats['coverage']:<10.1f} {stats['avg_paths']:<10.1f} {stats['min_paths']:<5} {stats['max_paths']:<5}")
        
        if len(medium_coverage_pairs) > 10:
            print(f"  ... and {len(medium_coverage_pairs) - 10} more pairs")
    
    # Show detailed analysis for top 5 pairs
    print(f"\n{'='*80}")
    print("DETAILED ANALYSIS OF TOP 5 PAIRS")
    print(f"{'='*80}")
    
    for i, stats in enumerate(pair_stats[:5]):
        print(f"\n{i+1}. {stats['source']} -> {stats['destination']}")
        print(f"   Coverage: {stats['coverage']:.1f}% ({stats['total_timestamps']}/{len(trace_files)} timestamps)")
        print(f"   Average paths: {stats['avg_paths']:.1f}")
        print(f"   Path range: {stats['min_paths']} - {stats['max_paths']}")
        
        # Show path counts for first 10 timestamps where this pair exists
        path_counts = pair_coverage[(stats['source'], stats['destination'])]
        print(f"   First 10 available timestamps path counts:")
        for j, count in enumerate(path_counts[:10]):
            print(f"     Timestamp {j+1}: {count} paths")
        if len(path_counts) > 10:
            print(f"     ... and {len(path_counts) - 10} more timestamps")
    
    return pair_stats

if __name__ == "__main__":
    pair_stats = analyze_period3_best_pairs() 
"""
Analyze node pairs with highest coverage in Period 3.
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

def analyze_period3_best_pairs():
    """Analyze node pairs with highest coverage in Period 3."""
    
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
    multipath_pairs_by_timestamp = {}
    
    for i, file_path in enumerate(trace_files):
        if i % 100 == 0:  # Show progress every 100 timestamps
            print(f"Analyzing timestamp {i+1}/{len(trace_files)}: {file_path.name}")
        
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
    
    # Count coverage for each node pair
    pair_coverage = defaultdict(list)
    
    for timestamp_idx, pairs in multipath_pairs_by_timestamp.items():
        for src, dst, path_count in pairs:
            pair_coverage[(src, dst)].append(path_count)
    
    # Calculate statistics for each pair
    pair_stats = []
    
    for (src, dst), path_counts in pair_coverage.items():
        coverage = len(path_counts) / len(trace_files) * 100
        avg_paths = sum(path_counts) / len(path_counts)
        min_paths = min(path_counts)
        max_paths = max(path_counts)
        
        pair_stats.append({
            'source': src,
            'destination': dst,
            'coverage': coverage,
            'avg_paths': avg_paths,
            'min_paths': min_paths,
            'max_paths': max_paths,
            'total_timestamps': len(path_counts)
        })
    
    # Sort by coverage (descending)
    pair_stats.sort(key=lambda x: x['coverage'], reverse=True)
    
    print(f"\n{'='*80}")
    print("NODE PAIRS WITH HIGHEST COVERAGE IN PERIOD 3")
    print(f"{'='*80}")
    
    print(f"Total unique node pairs with multiple paths in any timestamp: {len(pair_stats)}")
    print()
    
    # Show top 20 pairs by coverage
    print(f"Top 20 node pairs by coverage:")
    print(f"{'Source':<15} {'Destination':<15} {'Coverage':<10} {'Avg Paths':<10} {'Min':<5} {'Max':<5} {'Timestamps':<10}")
    print("-" * 85)
    
    for i, stats in enumerate(pair_stats[:20]):
        print(f"{stats['source']:<15} {stats['destination']:<15} {stats['coverage']:<10.1f} {stats['avg_paths']:<10.1f} {stats['min_paths']:<5} {stats['max_paths']:<5} {stats['total_timestamps']:<10}")
    
    # Show pairs with >90% coverage
    high_coverage_pairs = [p for p in pair_stats if p['coverage'] > 90]
    print(f"\nNode pairs with >90% coverage: {len(high_coverage_pairs)}")
    
    if high_coverage_pairs:
        print(f"\nHigh coverage pairs (>90%):")
        print(f"{'Source':<15} {'Destination':<15} {'Coverage':<10} {'Avg Paths':<10} {'Min':<5} {'Max':<5}")
        print("-" * 70)
        
        for stats in high_coverage_pairs:
            print(f"{stats['source']:<15} {stats['destination']:<15} {stats['coverage']:<10.1f} {stats['avg_paths']:<10.1f} {stats['min_paths']:<5} {stats['max_paths']:<5}")
    
    # Show pairs with >80% coverage
    medium_coverage_pairs = [p for p in pair_stats if 80 < p['coverage'] <= 90]
    print(f"\nNode pairs with 80-90% coverage: {len(medium_coverage_pairs)}")
    
    if medium_coverage_pairs:
        print(f"\nMedium coverage pairs (80-90%):")
        print(f"{'Source':<15} {'Destination':<15} {'Coverage':<10} {'Avg Paths':<10} {'Min':<5} {'Max':<5}")
        print("-" * 70)
        
        for stats in medium_coverage_pairs[:10]:  # Show first 10
            print(f"{stats['source']:<15} {stats['destination']:<15} {stats['coverage']:<10.1f} {stats['avg_paths']:<10.1f} {stats['min_paths']:<5} {stats['max_paths']:<5}")
        
        if len(medium_coverage_pairs) > 10:
            print(f"  ... and {len(medium_coverage_pairs) - 10} more pairs")
    
    # Show detailed analysis for top 5 pairs
    print(f"\n{'='*80}")
    print("DETAILED ANALYSIS OF TOP 5 PAIRS")
    print(f"{'='*80}")
    
    for i, stats in enumerate(pair_stats[:5]):
        print(f"\n{i+1}. {stats['source']} -> {stats['destination']}")
        print(f"   Coverage: {stats['coverage']:.1f}% ({stats['total_timestamps']}/{len(trace_files)} timestamps)")
        print(f"   Average paths: {stats['avg_paths']:.1f}")
        print(f"   Path range: {stats['min_paths']} - {stats['max_paths']}")
        
        # Show path counts for first 10 timestamps where this pair exists
        path_counts = pair_coverage[(stats['source'], stats['destination'])]
        print(f"   First 10 available timestamps path counts:")
        for j, count in enumerate(path_counts[:10]):
            print(f"     Timestamp {j+1}: {count} paths")
        if len(path_counts) > 10:
            print(f"     ... and {len(path_counts) - 10} more timestamps")
    
    return pair_stats

if __name__ == "__main__":
    pair_stats = analyze_period3_best_pairs() 