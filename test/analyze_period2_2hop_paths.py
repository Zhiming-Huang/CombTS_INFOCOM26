#!/usr/bin/env python3
"""
Analyze 2-hop paths between 10.1.1.100 and 10.1.1.102 in Period 2.
"""

import os
import sys
import numpy as np
import networkx as nx
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

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

def analyze_2hop_paths_period2():
    """Analyze 2-hop paths between 10.1.1.100 and 10.1.1.102 in Period 2."""
    
    # Get Period 2 data
    ucsb_dir = Path("data/ucsb")
    period2_folder = ucsb_dir / "1144373273-1144393193"
    
    if not period2_folder.exists():
        print("Period 2 folder not found!")
        return
    
    print(f"Analyzing Period 2: {period2_folder}")
    print(f"Node pair: 10.1.1.100 -> 10.1.1.102")
    print(f"Focus: 2-hop paths only")
    print()
    
    # Get all trace files
    trace_files = sorted([f for f in period2_folder.iterdir() if f.name.startswith("neighbortable")])
    print(f"Found {len(trace_files)} trace files")
    
    # Analyze each timestamp
    path_counts = []
    path_details = []
    intermediate_nodes = Counter()
    
    for i, file_path in enumerate(trace_files):
        print(f"Analyzing timestamp {i+1}/{len(trace_files)}: {file_path.name}")
        
        # Parse edges for this timestamp
        edges = parse_neighbortable_file(file_path)
        
        # Build graph
        G = nx.Graph()
        G.add_edges_from(edges)
        
        # Find all 2-hop paths between 10.1.1.100 and 10.1.1.102
        if "10.1.1.100" in G and "10.1.1.102" in G:
            try:
                # Find all simple paths up to 2 hops
                paths = list(nx.all_simple_paths(G, "10.1.1.100", "10.1.1.102", cutoff=2))
                
                # Filter only 2-hop paths (length = 3: source -> intermediate -> destination)
                two_hop_paths = [path for path in paths if len(path) == 3]
                
                path_counts.append(len(two_hop_paths))
                
                # Record intermediate nodes
                for path in two_hop_paths:
                    intermediate_node = path[1]  # Middle node
                    intermediate_nodes[intermediate_node] += 1
                
                # Store detailed path information for first few timestamps
                if i < 10:
                    path_details.append({
                        'timestamp': i+1,
                        'file': file_path.name,
                        'path_count': len(two_hop_paths),
                        'paths': two_hop_paths
                    })
                
                print(f"  Found {len(two_hop_paths)} 2-hop paths")
                
                # Show some examples for first few timestamps
                if i < 5 and two_hop_paths:
                    print(f"  Example paths:")
                    for j, path in enumerate(two_hop_paths[:5]):
                        print(f"    {j+1}. {' -> '.join(path)}")
                
            except nx.NetworkXNoPath:
                path_counts.append(0)
                print(f"  No 2-hop paths found")
        else:
            path_counts.append(0)
            print(f"  One or both nodes not in graph")
    
    # Analysis results
    print(f"\n{'='*80}")
    print("2-HOP PATH ANALYSIS RESULTS")
    print(f"{'='*80}")
    
    path_counts = np.array(path_counts)
    
    print(f"Total timestamps analyzed: {len(path_counts)}")
    print(f"Timestamps with 2-hop paths: {np.sum(path_counts > 0)}")
    print(f"Timestamps without 2-hop paths: {np.sum(path_counts == 0)}")
    print()
    
    print(f"Path count statistics:")
    print(f"  Mean: {np.mean(path_counts):.2f}")
    print(f"  Median: {np.median(path_counts):.2f}")
    print(f"  Min: {np.min(path_counts)}")
    print(f"  Max: {np.max(path_counts)}")
    print(f"  Std: {np.std(path_counts):.2f}")
    print()
    
    # Show path count distribution
    unique_counts, count_frequencies = np.unique(path_counts, return_counts=True)
    print(f"Path count distribution:")
    for count, freq in zip(unique_counts, count_frequencies):
        percentage = (freq / len(path_counts)) * 100
        print(f"  {count} paths: {freq} timestamps ({percentage:.1f}%)")
    print()
    
    # Show top intermediate nodes
    print(f"Top 10 intermediate nodes (total occurrences):")
    for node, count in intermediate_nodes.most_common(10):
        percentage = (count / len(path_counts)) * 100
        print(f"  {node}: {count} times ({percentage:.1f}% of timestamps)")
    print()
    
    # Show detailed analysis for first 10 timestamps
    print(f"Detailed analysis of first 10 timestamps:")
    for detail in path_details:
        print(f"\nTimestamp {detail['timestamp']} ({detail['file']}):")
        print(f"  2-hop paths: {detail['path_count']}")
        if detail['paths']:
            print(f"  Paths:")
            for j, path in enumerate(detail['paths'][:10]):  # Show first 10 paths
                print(f"    {j+1}. {' -> '.join(path)}")
            if len(detail['paths']) > 10:
                print(f"    ... and {len(detail['paths']) - 10} more paths")
    
    # Create visualization
    create_path_count_plot(path_counts, trace_files)
    
    return path_counts, intermediate_nodes

def create_path_count_plot(path_counts, trace_files):
    """Create a plot showing path count over time."""
    
    # Create time axis (timestamps)
    timestamps = range(1, len(path_counts) + 1)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot path counts over time
    plt.plot(timestamps, path_counts, 'b-', linewidth=1, alpha=0.7)
    plt.scatter(timestamps, path_counts, c='blue', s=10, alpha=0.6)
    
    # Add statistics lines
    mean_paths = np.mean(path_counts)
    median_paths = np.median(path_counts)
    plt.axhline(y=mean_paths, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_paths:.1f}')
    plt.axhline(y=median_paths, color='orange', linestyle='--', alpha=0.7, label=f'Median: {median_paths:.1f}')
    
    # Customize the plot
    plt.xlabel('Timestamp')
    plt.ylabel('Number of 2-hop paths')
    plt.title('2-hop paths between 10.1.1.100 and 10.1.1.102 in Period 2')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the plot
    output_dir = Path("output/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "period2_2hop_paths_analysis.pdf"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Path count plot saved to: {plot_path}")

if __name__ == "__main__":
    path_counts, intermediate_nodes = analyze_2hop_paths_period2() 