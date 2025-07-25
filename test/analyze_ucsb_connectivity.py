#!/usr/bin/env python3
"""
Analyze UCSB connectivity across different time periods
=======================================================

This script analyzes the connectivity of nodes across all traces in each time period
and finds source-destination pairs that remain connected throughout each period.
"""

import os
import sys
import glob
import networkx as nx
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional
import numpy as np

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)


def parse_neighbortable_file(file_path: str) -> Tuple[Set[str], List[Tuple[str, str]]]:
    """
    Parse a neighbortable file to extract nodes and edges.
    
    Args:
        file_path: Path to the neighbortable file
        
    Returns:
        Tuple of (nodes_set, edges_list)
    """
    nodes = set()
    edges = []
    
    try:
        with open(file_path, 'r') as fin:
            for line in fin:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                    
                src = parts[0]
                nodes.add(src)
                
                for i in range(1, len(parts), 2):
                    if i + 1 >= len(parts):
                        break
                    dst = parts[i]
                    try:
                        ett = float(parts[i+1])
                        if ett < 1000:  # Valid ETT
                            nodes.add(dst)
                            edges.append((src, dst))
                    except (ValueError, IndexError):
                        continue
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return set(), []
    
    return nodes, edges


def analyze_period_connectivity(period_dir: str, period_name: str, stability_threshold: float = 1.0):
    """
    Analyze connectivity for a specific time period.
    
    Args:
        period_dir: Directory containing trace files
        period_name: Name of the time period
        stability_threshold: Fraction of traces a pair must be stable in (default: 1.0 = all traces)
    """
    print(f"============================================================")
    print(f"Analyzing period: {period_name}")
    print(f"Directory: {period_dir}")
    print(f"Stability threshold: {stability_threshold:.1%}")
    print(f"============================================================")
    
    # Get all neighbortable files
    neighbortable_files = glob.glob(os.path.join(period_dir, 'neighbortable-*'))
    neighbortable_files = sorted(neighbortable_files)
    
    if not neighbortable_files:
        print(f"No neighbortable files found in {period_dir}")
        return
    
    print(f"Found {len(neighbortable_files)} trace files")
    
    # Analyze each trace file
    connectivity_data = []
    all_nodes = set()
    all_edges = set()
    
    for i, file_path in enumerate(neighbortable_files):
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(neighbortable_files)} files...")
        
        # Parse neighbortable file
        nodes, edges = parse_neighbortable_file(file_path)
        all_nodes.update(nodes)
        all_edges.update(edges)
        
        # Build NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        
        # Find largest connected component
        components = list(nx.connected_components(G))
        largest_cc = max(components, key=len) if components else set()
        
        connectivity_data.append({
            'graph': G,
            'largest_cc': largest_cc,
            'nodes': nodes,
            'edges': edges
        })
    
    print(f"\nPeriod Summary:")
    print(f"  Total unique nodes: {len(all_nodes)}")
    print(f"  Total edges: {len(all_edges)}")
    
    # Find nodes that are in the largest component across all traces
    if connectivity_data:
        common_largest_cc = set.intersection(*[data['largest_cc'] for data in connectivity_data])
        print(f"  Overall components: {len(list(nx.connected_components(connectivity_data[0]['graph'])))}")
        print(f"  Largest component size: {len(connectivity_data[0]['largest_cc'])}")
        print(f"  Nodes in largest component across all traces: {len(common_largest_cc)}")
        if common_largest_cc:
            print(f"  Sample nodes in common largest component: {list(common_largest_cc)[:5]}")
    
    # Find stable source-destination pairs
    print(f"\nStable Source-Destination Pairs (Multiple Paths):")
    
    if not connectivity_data:
        print("  No connectivity data available")
        return
    
    # Get all nodes from the largest component of the first trace
    candidate_nodes = list(connectivity_data[0]['largest_cc'])
    candidate_nodes.sort()  # Ensure deterministic ordering
    
    stable_pairs = []
    min_traces_required = int(len(connectivity_data) * stability_threshold)
    
    print(f"  Looking for pairs stable in at least {min_traces_required}/{len(connectivity_data)} traces...")
    
    # Check all possible pairs
    for i, src in enumerate(candidate_nodes):
        for dst in candidate_nodes[i+1:]:  # Avoid self-loops and duplicates
            path_counts = []
            path_lengths = []
            
            # Check this pair in all traces
            for trace_data in connectivity_data:
                G = trace_data['graph']
                
                if src in G and dst in G and nx.has_path(G, src, dst):
                    # Find all simple paths up to 3 hops
                    all_paths = list(nx.all_simple_paths(G, src, dst, cutoff=3))
                    
                    if len(all_paths) >= 2:  # At least 2 paths
                        path_counts.append(len(all_paths))
                        avg_length = sum(len(path) - 1 for path in all_paths) / len(all_paths)
                        path_lengths.append(avg_length)
                    else:
                        path_counts.append(0)
                        path_lengths.append(0)
                else:
                    path_counts.append(0)
                    path_lengths.append(0)
            
            # Calculate stability metrics
            stable_traces = sum(1 for count in path_counts if count >= 2)
            avg_path_count = np.mean([count for count in path_counts if count >= 2]) if any(count >= 2 for count in path_counts) else 0
            avg_path_length = np.mean([length for length in path_lengths if length > 0]) if any(length > 0 for length in path_lengths) else 0
            
            # Consider pair stable if it meets the threshold
            if stable_traces >= min_traces_required and avg_path_count > 0:
                stable_pairs.append({
                    'source': src,
                    'destination': dst,
                    'stable_traces': stable_traces,
                    'total_traces': len(connectivity_data),
                    'stability_ratio': stable_traces / len(connectivity_data),
                    'avg_path_count': avg_path_count,
                    'avg_path_length': avg_path_length
                })
    
    # Sort by average path count (descending), then by average path length (ascending)
    stable_pairs.sort(key=lambda x: (-x['avg_path_count'], x['avg_path_length']))
    
    print(f"  Found {len(stable_pairs)} stable pairs with multiple paths")
    
    # Show top 10 pairs
    for i, pair in enumerate(stable_pairs[:10]):
        stability_pct = pair['stability_ratio'] * 100
        print(f"  {i+1}. {pair['source']} -> {pair['destination']} "
              f"(stable in {pair['stable_traces']}/{pair['total_traces']} traces, "
              f"{stability_pct:.1f}%, avg paths: {pair['avg_path_count']:.1f}, "
              f"avg length: {pair['avg_path_length']:.2f})")
    
    return stable_pairs


def main():
    """Main function to analyze all UCSB time periods."""
    print("UCSB Connectivity Analysis")
    print("=" * 80)
    
    # Find UCSB data directory
    data_dir = os.path.join(project_root, 'data', 'ucsb')
    
    if not os.path.exists(data_dir):
        print(f"UCSB data directory not found: {data_dir}")
        return
    
    # Find all time period directories
    period_dirs = {}
    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.isdir(subdir_path):
            files = glob.glob(os.path.join(subdir_path, 'neighbortable-*'))
            if files:
                period_dirs[subdir] = {
                    'path': subdir_path,
                    'file_count': len(files)
                }
    
    if not period_dirs:
        print("No time period directories found with neighbortable files")
        return
    
    print(f"Found {len(period_dirs)} time periods:")
    for period, info in period_dirs.items():
        print(f"  {period}: {info['file_count']} files")
    
    # Analyze each period
    results = {}
    for period, info in period_dirs.items():
        results[period] = analyze_period_connectivity(info['path'], period)
    
    # Print summary for each period
    print(f"\n{'='*80}")
    print("SUMMARY ACROSS ALL PERIODS")
    print(f"{'='*80}")
    
    for period, result in results.items():
        if result:
            print(f"\n{period}:")
            print(f"  Files: {period_dirs[period]['file_count']}")
            print(f"  Stable pairs: {len(result)}")
            if result:
                best_pair = result[0]
                print(f"  Best pair: {best_pair['source']} -> {best_pair['destination']} "
                      f"(avg path length: {best_pair['avg_path_length']:.2f})")
    
    # Find common stable pairs across periods
    print(f"\n{'='*80}")
    print("RECOMMENDED NODE PAIRS")
    print(f"{'='*80}")
    
    for period, result in results.items():
        if result:
            print(f"\n{period}:")
            for i, pair in enumerate(result[:3]):
                print(f"  {i+1}. {pair['source']} -> {pair['destination']} "
                      f"(avg path length: {pair['avg_path_length']:.2f})")


if __name__ == "__main__":
    main() 