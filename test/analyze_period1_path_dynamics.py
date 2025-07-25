#!/usr/bin/env python3
"""
Analyze Path Dynamics for Period 1 Top Node Pairs
=================================================

This script analyzes how the path counts and lengths change over time
for the top node pairs in UCSB period 1.
"""

import os
import glob
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
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

def analyze_node_pair_dynamics(period_dir: str, node_pairs: List[Tuple[str, str]]):
    """
    Analyze the dynamics of specific node pairs over time.
    
    Args:
        period_dir: Directory containing trace files
        node_pairs: List of (source, destination) pairs to analyze
    """
    print(f"============================================================")
    print(f"Analyzing Path Dynamics for {len(node_pairs)} Node Pairs")
    print(f"Directory: {period_dir}")
    print(f"============================================================")
    
    # Get all neighbortable files
    neighbortable_files = glob.glob(os.path.join(period_dir, 'neighbortable-*'))
    neighbortable_files = sorted(neighbortable_files)
    
    if not neighbortable_files:
        print(f"No neighbortable files found in {period_dir}")
        return
    
    print(f"Found {len(neighbortable_files)} trace files")
    
    # Initialize data structures for each node pair
    dynamics_data = {}
    for src, dst in node_pairs:
        dynamics_data[(src, dst)] = {
            'path_counts': [],
            'path_lengths': [],
            'timestamps': [],
            'file_names': []
        }
    
    # Analyze each trace file
    for i, file_path in enumerate(neighbortable_files):
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(neighbortable_files)} files...")
        
        # Parse neighbortable file
        nodes, edges = parse_neighbortable_file(file_path)
        
        # Build NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        
        # Extract timestamp from filename
        filename = os.path.basename(file_path)
        timestamp = filename.replace('neighbortable-', '')
        
        # Analyze each node pair
        for src, dst in node_pairs:
            path_count = 0
            avg_path_length = 0
            
            if src in G and dst in G and nx.has_path(G, src, dst):
                # Find all simple paths up to 3 hops (same as connectivity analysis)
                all_paths = list(nx.all_simple_paths(G, src, dst, cutoff=3))
                
                if len(all_paths) >= 2:  # At least 2 paths
                    path_count = len(all_paths)
                    path_lengths = [len(path) - 1 for path in all_paths]
                    avg_path_length = sum(path_lengths) / len(path_lengths)
            
            # Store data
            dynamics_data[(src, dst)]['path_counts'].append(path_count)
            dynamics_data[(src, dst)]['path_lengths'].append(avg_path_length)
            dynamics_data[(src, dst)]['timestamps'].append(timestamp)
            dynamics_data[(src, dst)]['file_names'].append(filename)
    
    return dynamics_data

def print_dynamics_summary(dynamics_data: Dict):
    """Print a summary of the dynamics for each node pair."""
    print(f"\n{'='*80}")
    print("PATH DYNAMICS SUMMARY")
    print(f"{'='*80}")
    
    for (src, dst), data in dynamics_data.items():
        path_counts = data['path_counts']
        path_lengths = [length for length in data['path_lengths'] if length > 0]
        
        # Calculate statistics
        avg_path_count = np.mean([count for count in path_counts if count > 0])
        min_path_count = min(path_counts) if path_counts else 0
        max_path_count = max(path_counts) if path_counts else 0
        std_path_count = np.std([count for count in path_counts if count > 0])
        
        avg_path_length = np.mean(path_lengths) if path_lengths else 0
        min_path_length = min(path_lengths) if path_lengths else 0
        max_path_length = max(path_lengths) if path_lengths else 0
        std_path_length = np.std(path_lengths) if path_lengths else 0
        
        # Count stable periods
        stable_periods = sum(1 for count in path_counts if count >= 2)
        stability_ratio = stable_periods / len(path_counts) * 100
        
        print(f"\n{src} -> {dst}:")
        print(f"  Path Count Statistics:")
        print(f"    Average: {avg_path_count:.1f} ± {std_path_count:.1f}")
        print(f"    Range: {min_path_count} - {max_path_count}")
        print(f"    Stability: {stability_ratio:.1f}% ({stable_periods}/{len(path_counts)} periods)")
        
        if path_lengths:
            print(f"  Path Length Statistics:")
            print(f"    Average: {avg_path_length:.2f} ± {std_path_length:.2f}")
            print(f"    Range: {min_path_length:.2f} - {max_path_length:.2f}")
        else:
            print(f"  Path Length: No valid paths found")

def plot_dynamics(dynamics_data: Dict, output_dir: str = None):
    """Plot the dynamics for each node pair."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output', 'images')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    
    for (src, dst), data in dynamics_data.items():
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot path counts over time
        timestamps = range(len(data['path_counts']))
        ax1.plot(timestamps, data['path_counts'], 'b-', linewidth=1.5, alpha=0.8)
        ax1.set_ylabel('Number of Paths', fontsize=12)
        ax1.set_title(f'Path Count Dynamics: {src} -> {dst}', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Add statistics to the plot
        avg_count = np.mean([count for count in data['path_counts'] if count > 0])
        ax1.axhline(y=avg_count, color='r', linestyle='--', alpha=0.7, 
                   label=f'Average: {avg_count:.1f}')
        ax1.legend()
        
        # Plot path lengths over time
        valid_lengths = [(i, length) for i, length in enumerate(data['path_lengths']) if length > 0]
        if valid_lengths:
            indices, lengths = zip(*valid_lengths)
            ax2.plot(indices, lengths, 'g-', linewidth=1.5, alpha=0.8)
            ax2.set_ylabel('Average Path Length (hops)', fontsize=12)
            ax2.set_xlabel('Time Period', fontsize=12)
            ax2.set_title(f'Path Length Dynamics: {src} -> {dst}', fontsize=14)
            ax2.grid(True, alpha=0.3)
            
            # Add statistics to the plot
            avg_length = np.mean(lengths)
            ax2.axhline(y=avg_length, color='r', linestyle='--', alpha=0.7,
                       label=f'Average: {avg_length:.2f}')
            ax2.legend()
        
        plt.tight_layout()
        
        # Save the plot
        filename = f"period1_dynamics_{src.replace('.', '_')}_to_{dst.replace('.', '_')}.pdf"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Saved plot: {filepath}")
        plt.close()

def main():
    """Main function to analyze period 1 path dynamics."""
    print("Period 1 Path Dynamics Analysis")
    print("=" * 80)
    
    # Define the period 1 directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'ucsb')
    period1_dir = os.path.join(data_dir, '1143927049-1143953729')
    
    if not os.path.exists(period1_dir):
        print(f"Period 1 directory not found: {period1_dir}")
        return
    
    # Define the top node pairs to analyze
    top_node_pairs = [
        ('10.1.1.102', '10.1.1.25'),   # Best pair
        ('10.1.1.101', '10.1.1.102'),  # Second best
        ('10.1.1.101', '10.1.1.25'),   # Third best
        ('10.1.1.102', '10.1.1.103'),  # Fourth best
        ('10.1.1.103', '10.1.1.25'),   # Fifth best
        ('10.1.1.101', '10.1.1.103'),  # Sixth best
        ('10.1.1.100', '10.1.1.102'),  # Seventh best
        ('10.1.1.100', '10.1.1.25'),   # Eighth best
        ('10.1.1.102', '10.1.1.9'),    # Ninth best
        ('10.1.1.100', '10.1.1.101'),  # Tenth best
    ]
    
    print(f"Analyzing {len(top_node_pairs)} top node pairs:")
    for i, (src, dst) in enumerate(top_node_pairs, 1):
        print(f"  {i}. {src} -> {dst}")
    
    # Analyze dynamics
    dynamics_data = analyze_node_pair_dynamics(period1_dir, top_node_pairs)
    
    # Print summary
    print_dynamics_summary(dynamics_data)
    
    # Create plots
    print(f"\n{'='*80}")
    print("CREATING DYNAMICS PLOTS")
    print(f"{'='*80}")
    plot_dynamics(dynamics_data)
    
    print(f"\nAnalysis completed!")
    print(f"Plots saved to: output/images/")

if __name__ == "__main__":
    main() 