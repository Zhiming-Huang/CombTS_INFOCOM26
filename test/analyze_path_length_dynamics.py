#!/usr/bin/env python3
"""
Analyze path length dynamics for stable multipath node pairs.
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

def analyze_path_length_dynamics():
    """Analyze path length dynamics for stable multipath node pairs."""
    
    # Get Period 1 data
    ucsb_dir = Path("data/ucsb")
    period1_folder = ucsb_dir / "1143927049-1143953729"
    
    if not period1_folder.exists():
        print("Period 1 folder not found!")
        return
    
    print(f"Analyzing path length dynamics for Period 1: {period1_folder}")
    
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
    
    # Analyze path length dynamics for each pair
    pair_results = {}
    
    for pair_idx, (src, dst) in enumerate(stable_pairs):
        print(f"\n{'='*80}")
        print(f"ANALYZING PAIR {pair_idx+1}: {src} -> {dst}")
        print(f"{'='*80}")
        
        path_lengths_by_timestamp = []
        path_counts_by_timestamp = []
        
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
                        # Calculate path lengths (number of hops)
                        path_lengths = [len(path) - 1 for path in paths]  # -1 because len(path) = hops + 1
                        
                        path_lengths_by_timestamp.append(path_lengths)
                        path_counts_by_timestamp.append(len(paths))
                        
                        # Show statistics for first 10 timestamps
                        if timestamp_idx < 10:
                            length_counter = Counter(path_lengths)
                            print(f"Timestamp {timestamp_idx+1}: {len(paths)} paths")
                            print(f"  Length distribution: {dict(length_counter)}")
                            print(f"  Min length: {min(path_lengths)}, Max length: {max(path_lengths)}")
                            print(f"  Avg length: {sum(path_lengths)/len(path_lengths):.2f}")
                    else:
                        path_lengths_by_timestamp.append([])
                        path_counts_by_timestamp.append(0)
                        
                except nx.NetworkXNoPath:
                    path_lengths_by_timestamp.append([])
                    path_counts_by_timestamp.append(0)
            else:
                path_lengths_by_timestamp.append([])
                path_counts_by_timestamp.append(0)
        
        # Calculate overall statistics
        all_path_lengths = []
        for lengths in path_lengths_by_timestamp:
            all_path_lengths.extend(lengths)
        
        if all_path_lengths:
            length_counter = Counter(all_path_lengths)
            avg_path_count = sum(path_counts_by_timestamp) / len(path_counts_by_timestamp)
            
            print(f"\nOVERALL STATISTICS FOR {src} -> {dst}")
            print(f"Average path count: {avg_path_count:.1f}")
            print(f"Total paths analyzed: {len(all_path_lengths)}")
            print(f"Length distribution: {dict(length_counter)}")
            print(f"Min length: {min(all_path_lengths)}, Max length: {max(all_path_lengths)}")
            print(f"Average length: {sum(all_path_lengths)/len(all_path_lengths):.2f}")
            
            # Calculate length stability
            length_variations = []
            for lengths in path_lengths_by_timestamp:
                if lengths:
                    length_variations.append(max(lengths) - min(lengths))
            
            if length_variations:
                print(f"Average length variation per timestamp: {sum(length_variations)/len(length_variations):.2f}")
                print(f"Max length variation: {max(length_variations)}")
            
            # Store results
            pair_results[(src, dst)] = {
                'avg_path_count': avg_path_count,
                'length_distribution': dict(length_counter),
                'min_length': min(all_path_lengths),
                'max_length': max(all_path_lengths),
                'avg_length': sum(all_path_lengths)/len(all_path_lengths),
                'path_lengths_by_timestamp': path_lengths_by_timestamp,
                'path_counts_by_timestamp': path_counts_by_timestamp
            }
        else:
            print(f"No paths found for {src} -> {dst}")
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON OF ALL PAIRS")
    print(f"{'='*80}")
    print(f"{'Source':<15} {'Destination':<15} {'Avg Paths':<10} {'Min Len':<8} {'Max Len':<8} {'Avg Len':<8} {'1-hop':<6} {'2-hop':<6} {'3-hop':<6}")
    print("-" * 100)
    
    for (src, dst), results in pair_results.items():
        length_dist = results['length_distribution']
        one_hop = length_dist.get(1, 0)
        two_hop = length_dist.get(2, 0)
        three_hop = length_dist.get(3, 0)
        
        print(f"{src:<15} {dst:<15} {results['avg_path_count']:<10.1f} {results['min_length']:<8} {results['max_length']:<8} {results['avg_length']:<8.2f} {one_hop:<6} {two_hop:<6} {three_hop:<6}")
    
    # Detailed length analysis for top pair
    print(f"\n{'='*80}")
    print("DETAILED LENGTH ANALYSIS FOR TOP PAIR")
    print(f"{'='*80}")
    
    top_pair = list(pair_results.keys())[0]
    top_results = pair_results[top_pair]
    
    print(f"Pair: {top_pair[0]} -> {top_pair[1]}")
    print(f"Length distribution: {top_results['length_distribution']}")
    
    # Show length changes over time
    print(f"\nLength changes over time (first 20 timestamps):")
    for i in range(min(20, len(top_results['path_lengths_by_timestamp']))):
        lengths = top_results['path_lengths_by_timestamp'][i]
        if lengths:
            length_counter = Counter(lengths)
            print(f"Timestamp {i+1}: {length_counter}")
    
    return pair_results

if __name__ == "__main__":
    results = analyze_path_length_dynamics() 