#!/usr/bin/env python3
"""
Analyze Period 2 Stability with Different Thresholds
===================================================

This script analyzes the second UCSB time period (1144393236-1144450070)
with different stability thresholds to find the most stable node pairs.
"""

import os
import glob
import networkx as nx
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional

# Add project root to Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_neighbortable_file(file_path: str) -> Tuple[Set[str], List[Tuple[str, str]]]:
    """Parse a neighbortable file to extract nodes and edges."""
    nodes = set()
    edges = []
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        node = parts[0]
                        neighbor = parts[1]
                        nodes.add(node)
                        nodes.add(neighbor)
                        edges.append((node, neighbor))
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    
    return nodes, edges

def analyze_period2_with_threshold(period_dir: str, stability_threshold: float):
    """
    Analyze period 2 with a specific stability threshold.
    
    Args:
        period_dir: Directory containing trace files
        stability_threshold: Fraction of traces a pair must be stable in
    """
    print(f"============================================================")
    print(f"Analyzing Period 2 with {stability_threshold:.1%} stability threshold")
    print(f"Directory: {period_dir}")
    print(f"============================================================")
    
    # Get all neighbortable files
    neighbortable_files = glob.glob(os.path.join(period_dir, 'neighbortable-*'))
    neighbortable_files = sorted(neighbortable_files)
    
    if not neighbortable_files:
        print(f"No neighbortable files found in {period_dir}")
        return []
    
    print(f"Found {len(neighbortable_files)} trace files")
    
    # Analyze each trace file
    connectivity_data = []
    all_nodes = set()
    all_edges = set()
    
    for i, file_path in enumerate(neighbortable_files):
        if (i + 1) % 100 == 0:
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
        return []
    
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
    """Main function to analyze period 2 with different thresholds."""
    print("Period 2 Stability Analysis with Different Thresholds")
    print("=" * 80)
    
    # Define the period 2 directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'ucsb')
    period2_dir = os.path.join(data_dir, '1144393236-1144450070')
    
    if not os.path.exists(period2_dir):
        print(f"Period 2 directory not found: {period2_dir}")
        return
    
    # Test different stability thresholds
    thresholds = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    
    all_results = {}
    
    for threshold in thresholds:
        print(f"\n{'='*80}")
        results = analyze_period2_with_threshold(period2_dir, threshold)
        all_results[threshold] = results
        
        if results:
            print(f"\nSummary for {threshold:.1%} threshold:")
            print(f"  Found {len(results)} stable pairs")
            if results:
                best_pair = results[0]
                print(f"  Best pair: {best_pair['source']} -> {best_pair['destination']}")
                print(f"    Stability: {best_pair['stability_ratio']:.1%}")
                print(f"    Avg paths: {best_pair['avg_path_count']:.1f}")
                print(f"    Avg length: {best_pair['avg_path_length']:.2f}")
        else:
            print(f"\nNo stable pairs found with {threshold:.1%} threshold")
    
    # Final recommendations
    print(f"\n{'='*80}")
    print("FINAL RECOMMENDATIONS")
    print(f"{'='*80}")
    
    for threshold in thresholds:
        if all_results[threshold]:
            best_pair = all_results[threshold][0]
            print(f"\n{threshold:.1%} threshold:")
            print(f"  Best pair: {best_pair['source']} -> {best_pair['destination']}")
            print(f"  Stability: {best_pair['stability_ratio']:.1%} ({best_pair['stable_traces']}/{best_pair['total_traces']})")
            print(f"  Avg paths: {best_pair['avg_path_count']:.1f}")
            print(f"  Avg length: {best_pair['avg_path_length']:.2f}")
    
    # Find the best compromise
    best_compromise = None
    for threshold in thresholds:
        if all_results[threshold]:
            pair = all_results[threshold][0]
            # Score based on stability and path diversity
            score = pair['stability_ratio'] * pair['avg_path_count']
            if best_compromise is None or score > best_compromise['score']:
                best_compromise = {
                    'threshold': threshold,
                    'pair': pair,
                    'score': score
                }
    
    if best_compromise:
        print(f"\nRECOMMENDED COMPROMISE:")
        print(f"  Threshold: {best_compromise['threshold']:.1%}")
        print(f"  Node pair: {best_compromise['pair']['source']} -> {best_compromise['pair']['destination']}")
        print(f"  Stability: {best_compromise['pair']['stability_ratio']:.1%}")
        print(f"  Avg paths: {best_compromise['pair']['avg_path_count']:.1f}")
        print(f"  Avg length: {best_compromise['pair']['avg_path_length']:.2f}")

if __name__ == "__main__":
    main() 