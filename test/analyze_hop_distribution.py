#!/usr/bin/env python3
"""
Analyze hop distribution (1-hop, 2-hop, 3-hop paths) per timestamp for stable multipath node pairs.
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

def analyze_hop_distribution():
    """Analyze hop distribution per timestamp for stable multipath node pairs."""
    
    # Get Period 1 data
    ucsb_dir = Path("data/ucsb")
    period1_folder = ucsb_dir / "1143927049-1143953729"
    
    if not period1_folder.exists():
        print("Period 1 folder not found!")
        return
    
    print(f"Analyzing hop distribution for Period 1: {period1_folder}")
    
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
    
    # Analyze hop distribution for each pair across all timestamps
    pair_results = {}
    
    for pair_idx, (src, dst) in enumerate(stable_pairs):
        print(f"\n{'='*80}")
        print(f"ANALYZING PAIR {pair_idx+1}: {src} -> {dst}")
        print(f"{'='*80}")
        
        hop_distributions = []
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
                        # Calculate hop distribution
                        hop_counts = Counter()
                        for path in paths:
                            hops = len(path) - 1  # -1 because len(path) = hops + 1
                            hop_counts[hops] += 1
                        
                        hop_distributions.append(hop_counts)
                        valid_timestamps += 1
                        
                        # Show statistics for first 10 timestamps
                        if timestamp_idx < 10:
                            print(f"Timestamp {timestamp_idx+1}: {len(paths)} paths")
                            print(f"  Hop distribution: {dict(hop_counts)}")
                            print(f"  1-hop: {hop_counts.get(1, 0)}, 2-hop: {hop_counts.get(2, 0)}, 3-hop: {hop_counts.get(3, 0)}")
                    else:
                        hop_distributions.append(Counter())
                        
                except nx.NetworkXNoPath:
                    hop_distributions.append(Counter())
            else:
                hop_distributions.append(Counter())
        
        # Calculate overall statistics
        if hop_distributions:
            # Aggregate all hop distributions
            total_hop_counts = Counter()
            hop_1_counts = []
            hop_2_counts = []
            hop_3_counts = []
            
            for hop_dist in hop_distributions:
                if hop_dist:  # Only count valid timestamps
                    total_hop_counts.update(hop_dist)
                    hop_1_counts.append(hop_dist.get(1, 0))
                    hop_2_counts.append(hop_dist.get(2, 0))
                    hop_3_counts.append(hop_dist.get(3, 0))
            
            if total_hop_counts:
                print(f"\nOVERALL STATISTICS FOR {src} -> {dst}")
                print(f"Valid timestamps: {valid_timestamps}/{len(trace_files)} ({valid_timestamps/len(trace_files)*100:.1f}%)")
                print(f"Total paths analyzed: {sum(total_hop_counts.values())}")
                print(f"Overall hop distribution: {dict(total_hop_counts)}")
                
                # Calculate averages per timestamp
                avg_hop_1 = sum(hop_1_counts) / len(hop_1_counts) if hop_1_counts else 0
                avg_hop_2 = sum(hop_2_counts) / len(hop_2_counts) if hop_2_counts else 0
                avg_hop_3 = sum(hop_3_counts) / len(hop_3_counts) if hop_3_counts else 0
                
                print(f"Average per timestamp:")
                print(f"  1-hop paths: {avg_hop_1:.1f}")
                print(f"  2-hop paths: {avg_hop_2:.1f}")
                print(f"  3-hop paths: {avg_hop_3:.1f}")
                
                # Calculate statistics for each hop count
                if hop_1_counts:
                    print(f"1-hop paths - Min: {min(hop_1_counts)}, Max: {max(hop_1_counts)}, Std: {np.std(hop_1_counts):.1f}")
                if hop_2_counts:
                    print(f"2-hop paths - Min: {min(hop_2_counts)}, Max: {max(hop_2_counts)}, Std: {np.std(hop_2_counts):.1f}")
                if hop_3_counts:
                    print(f"3-hop paths - Min: {min(hop_3_counts)}, Max: {max(hop_3_counts)}, Std: {np.std(hop_3_counts):.1f}")
                
                # Show percentage distribution
                total_paths = sum(total_hop_counts.values())
                print(f"Percentage distribution:")
                print(f"  1-hop: {total_hop_counts.get(1, 0)/total_paths*100:.1f}%")
                print(f"  2-hop: {total_hop_counts.get(2, 0)/total_paths*100:.1f}%")
                print(f"  3-hop: {total_hop_counts.get(3, 0)/total_paths*100:.1f}%")
                
                # Store results
                pair_results[(src, dst)] = {
                    'total_hop_counts': dict(total_hop_counts),
                    'avg_hop_1': avg_hop_1,
                    'avg_hop_2': avg_hop_2,
                    'avg_hop_3': avg_hop_3,
                    'hop_1_stats': {'min': min(hop_1_counts), 'max': max(hop_1_counts), 'std': np.std(hop_1_counts)} if hop_1_counts else {},
                    'hop_2_stats': {'min': min(hop_2_counts), 'max': max(hop_2_counts), 'std': np.std(hop_2_counts)} if hop_2_counts else {},
                    'hop_3_stats': {'min': min(hop_3_counts), 'max': max(hop_3_counts), 'std': np.std(hop_3_counts)} if hop_3_counts else {},
                    'valid_timestamps': valid_timestamps,
                    'total_timestamps': len(trace_files),
                    'hop_distributions': hop_distributions
                }
            else:
                print(f"No valid paths found for {src} -> {dst}")
        else:
            print(f"No paths found for {src} -> {dst}")
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON OF ALL PAIRS")
    print(f"{'='*80}")
    print(f"{'Source':<15} {'Destination':<15} {'Avg 1-hop':<10} {'Avg 2-hop':<10} {'Avg 3-hop':<10} {'Total':<8}")
    print("-" * 80)
    
    for (src, dst), results in pair_results.items():
        total_avg = results['avg_hop_1'] + results['avg_hop_2'] + results['avg_hop_3']
        print(f"{src:<15} {dst:<15} {results['avg_hop_1']:<10.1f} {results['avg_hop_2']:<10.1f} {results['avg_hop_3']:<10.1f} {total_avg:<8.1f}")
    
    # Detailed analysis for top pair
    print(f"\n{'='*80}")
    print("DETAILED HOP ANALYSIS FOR TOP PAIR")
    print(f"{'='*80}")
    
    top_pair = list(pair_results.keys())[0]
    top_results = pair_results[top_pair]
    
    print(f"Pair: {top_pair[0]} -> {top_pair[1]}")
    print(f"Overall hop distribution: {top_results['total_hop_counts']}")
    
    # Show hop distribution changes over time
    print(f"\nHop distribution changes over time (first 20 timestamps):")
    for i in range(min(20, len(top_results['hop_distributions']))):
        hop_dist = top_results['hop_distributions'][i]
        if hop_dist:
            print(f"  Timestamp {i+1}: {dict(hop_dist)}")
        else:
            print(f"  Timestamp {i+1}: No paths")
    
    # Show percentage breakdown
    print(f"\n{'='*80}")
    print("PERCENTAGE BREAKDOWN BY HOP COUNT")
    print(f"{'='*80}")
    
    for (src, dst), results in pair_results.items():
        total_paths = sum(results['total_hop_counts'].values())
        hop_1_pct = results['total_hop_counts'].get(1, 0) / total_paths * 100
        hop_2_pct = results['total_hop_counts'].get(2, 0) / total_paths * 100
        hop_3_pct = results['total_hop_counts'].get(3, 0) / total_paths * 100
        
        print(f"{src} -> {dst}:")
        print(f"  1-hop: {hop_1_pct:.1f}% ({results['total_hop_counts'].get(1, 0)} paths)")
        print(f"  2-hop: {hop_2_pct:.1f}% ({results['total_hop_counts'].get(2, 0)} paths)")
        print(f"  3-hop: {hop_3_pct:.1f}% ({results['total_hop_counts'].get(3, 0)} paths)")
        print()
    
    return pair_results

if __name__ == "__main__":
    results = analyze_hop_distribution() 