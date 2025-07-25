#!/usr/bin/env python3
"""
Analyze average path counts per timestamp for stable multipath node pairs.
"""

import os
import sys
import numpy as np
import networkx as nx
from pathlib import Path
from collections import defaultdict, Counter

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

def analyze_timestamp_path_counts():
    """Analyze path counts per timestamp for stable multipath node pairs."""
    
    # Get Period 1 data
    ucsb_dir = Path("data/ucsb")
    period1_folder = ucsb_dir / "1143927049-1143953729"
    
    if not period1_folder.exists():
        print("Period 1 folder not found!")
        return
    
    print(f"Analyzing path counts per timestamp for Period 1: {period1_folder}")
    
    # Get all trace files
    trace_files = sorted([f for f in period1_folder.iterdir() if f.name.startswith("neighbortable")])
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
    
    # Define the stable node pairs to analyze (top 5 from previous analysis)
    stable_pairs = [
        ("10.1.1.102", "10.1.1.25"),
        ("10.1.1.101", "10.1.1.102"),
        ("10.1.1.101", "10.1.1.25"),
        ("10.1.1.102", "10.1.1.103"),
        ("10.1.1.103", "10.1.1.25")
    ]
    
    # Analyze path counts for each pair across all timestamps
    pair_results = {}
    
    for pair_idx, (src, dst) in enumerate(stable_pairs):
        print(f"\n{'='*80}")
        print(f"ANALYZING PAIR {pair_idx+1}: {src} -> {dst}")
        print(f"{'='*80}")
        
        path_counts = []
        valid_timestamps = 0
        
        # Analyze each timestamp
        for timestamp_idx, file_path in enumerate(trace_files):
            # Parse edges for this timestamp
            edges = parse_neighbortable_file(file_path)
            
            # Build graph
            G = nx.Graph()
            G.add_nodes_from(all_nodes)
            G.add_edges_from(edges)
            
            if src in G and dst in G and nx.has_path(G, src, dst):
                try:
                    # Find all simple paths up to 3 hops
                    paths = list(nx.all_simple_paths(G, src, dst, cutoff=3))
                    
                    if len(paths) >= 2:  # At least 2 paths
                        path_counts.append(len(paths))
                        valid_timestamps += 1
                        
                        # Show statistics for first 10 timestamps
                        if timestamp_idx < 10:
                            print(f"Timestamp {timestamp_idx+1}: {len(paths)} paths")
                    else:
                        path_counts.append(0)
                        
                except nx.NetworkXNoPath:
                    path_counts.append(0)
            else:
                path_counts.append(0)
        
        # Calculate statistics
        if path_counts:
            valid_path_counts = [count for count in path_counts if count > 0]
            
            if valid_path_counts:
                avg_paths = sum(valid_path_counts) / len(valid_path_counts)
                min_paths = min(valid_path_counts)
                max_paths = max(valid_path_counts)
                std_paths = np.std(valid_path_counts)
                
                print(f"\nSTATISTICS FOR {src} -> {dst}")
                print(f"Valid timestamps: {valid_timestamps}/{len(trace_files)} ({valid_timestamps/len(trace_files)*100:.1f}%)")
                print(f"Average paths per timestamp: {avg_paths:.1f}")
                print(f"Min paths: {min_paths}")
                print(f"Max paths: {max_paths}")
                print(f"Standard deviation: {std_paths:.1f}")
                print(f"Path count range: {max_paths - min_paths}")
                
                # Show distribution
                path_counter = Counter(valid_path_counts)
                print(f"Path count distribution:")
                for count in sorted(path_counter.keys()):
                    percentage = path_counter[count] / len(valid_path_counts) * 100
                    print(f"  {count} paths: {path_counter[count]} timestamps ({percentage:.1f}%)")
                
                # Store results
                pair_results[(src, dst)] = {
                    'avg_paths': avg_paths,
                    'min_paths': min_paths,
                    'max_paths': max_paths,
                    'std_paths': std_paths,
                    'valid_timestamps': valid_timestamps,
                    'total_timestamps': len(trace_files),
                    'path_counts': path_counts,
                    'valid_path_counts': valid_path_counts
                }
            else:
                print(f"No valid paths found for {src} -> {dst}")
        else:
            print(f"No paths found for {src} -> {dst}")
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON OF ALL PAIRS")
    print(f"{'='*80}")
    print(f"{'Source':<15} {'Destination':<15} {'Avg Paths':<10} {'Min':<5} {'Max':<5} {'Std':<6} {'Valid %':<8}")
    print("-" * 80)
    
    for (src, dst), results in pair_results.items():
        valid_percentage = results['valid_timestamps'] / results['total_timestamps'] * 100
        print(f"{src:<15} {dst:<15} {results['avg_paths']:<10.1f} {results['min_paths']:<5} {results['max_paths']:<5} {results['std_paths']:<6.1f} {valid_percentage:<8.1f}%")
    
    # Show path count trends over time
    print(f"\n{'='*80}")
    print("PATH COUNT TRENDS OVER TIME")
    print(f"{'='*80}")
    
    # Show first 20 timestamps for top pair
    top_pair = list(pair_results.keys())[0]
    top_results = pair_results[top_pair]
    
    print(f"Path counts for {top_pair[0]} -> {top_pair[1]} (first 20 timestamps):")
    for i in range(min(20, len(top_results['path_counts']))):
        count = top_results['path_counts'][i]
        if count > 0:
            print(f"  Timestamp {i+1}: {count} paths")
        else:
            print(f"  Timestamp {i+1}: No paths")
    
    # Show overall statistics
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print(f"{'='*80}")
    
    all_avg_paths = [results['avg_paths'] for results in pair_results.values()]
    overall_avg = sum(all_avg_paths) / len(all_avg_paths)
    overall_min = min([results['min_paths'] for results in pair_results.values()])
    overall_max = max([results['max_paths'] for results in pair_results.values()])
    
    print(f"Average paths across all pairs: {overall_avg:.1f}")
    print(f"Overall min paths: {overall_min}")
    print(f"Overall max paths: {overall_max}")
    print(f"Path count variation: {overall_max - overall_min}")
    
    return pair_results

if __name__ == "__main__":
    results = analyze_timestamp_path_counts() 