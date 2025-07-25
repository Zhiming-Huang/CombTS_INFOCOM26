#!/usr/bin/env python3
"""
Path Availability Analysis for UCSB Mesh Network
===============================================

Analyze and visualize the availability of paths between specific nodes
over time in the UCSB mesh network trace data.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
import networkx as nx
from collections import defaultdict
import pandas as pd

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment


def setup_plot_style():
    """Setup plot style following user preferences"""
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.figsize'] = (4, 3)


def get_all_simple_paths(graph: nx.Graph, source: str, target: str, max_hops: int = 3) -> List[List[str]]:
    """
    Get all simple paths between source and target with max hops.
    
    Args:
        graph: NetworkX graph
        source: Source node
        target: Target node
        max_hops: Maximum number of hops
        
    Returns:
        List of paths (each path is a list of nodes)
    """
    try:
        # NetworkX simple_paths_graph requires max path length (number of nodes)
        # max_hops = 3 means max 4 nodes in path (3 edges)
        paths = list(nx.all_simple_paths(graph, source, target, cutoff=max_hops))
        return paths
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def get_topology_at_round(env: UCSBMeshnetMemmapEnvironment, round_idx: int) -> nx.Graph:
    """
    Get NetworkX graph representation of topology at specific round.
    
    Args:
        env: UCSB environment
        round_idx: Round index
        
    Returns:
        NetworkX graph of available links
    """
    # Get available links as adjacency matrix
    adj_matrix = env.get_available_links_for_round(round_idx)
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add all nodes
    nodes = list(env.nodes)
    G.add_nodes_from(nodes)
    
    # Add edges based on adjacency matrix
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if i != j and adj_matrix[i, j] > 0:  # Link exists
                G.add_edge(u, v)
    
    return G


def analyze_path_availability(env: UCSBMeshnetMemmapEnvironment, 
                            source: str, 
                            target: str, 
                            max_hops: int = 3,
                            sample_interval: int = 10) -> Dict[str, Any]:
    """
    Analyze path availability over time.
    
    Args:
        env: UCSB environment
        source: Source node
        target: Target node  
        max_hops: Maximum hops allowed
        sample_interval: Sample every N rounds to reduce computation
        
    Returns:
        Dictionary with analysis results
    """
    print(f"Analyzing path availability from {source} to {target}")
    print(f"Max hops: {max_hops}, Total rounds: {env.num_rounds}")
    
    # Sample time points to analyze
    time_points = list(range(0, env.num_rounds, sample_interval))
    if env.num_rounds - 1 not in time_points:
        time_points.append(env.num_rounds - 1)
    
    print(f"Analyzing {len(time_points)} time points...")
    
    # Track paths and their availability
    all_paths = set()
    path_availability = defaultdict(list)
    total_available_paths = []
    graphs_over_time = []
    
    for i, round_idx in enumerate(time_points):
        if i % 20 == 0:
            print(f"Processing round {round_idx} ({i+1}/{len(time_points)})")
            
        # Get topology at this time point
        topology = get_topology_at_round(env, round_idx)
        graphs_over_time.append(topology.copy())
        
        # Find all paths at this time point
        current_paths = get_all_simple_paths(topology, source, target, max_hops)
        current_path_strs = [' -> '.join(path) for path in current_paths]
        
        # Update all known paths
        all_paths.update(current_path_strs)
        
        # Record availability for each known path
        for path_str in all_paths:
            path_availability[path_str].append(1 if path_str in current_path_strs else 0)
        
        total_available_paths.append(len(current_paths))
    
    print(f"Found {len(all_paths)} unique paths total")
    
    # Convert to consistent format
    path_availability_matrix = []
    path_names = sorted(list(all_paths))
    
    for path_name in path_names:
        # Pad with zeros if path wasn't available from the beginning
        availability = path_availability[path_name]
        if len(availability) < len(time_points):
            availability = [0] * (len(time_points) - len(availability)) + availability
        path_availability_matrix.append(availability)
    
    return {
        'time_points': time_points,
        'path_names': path_names,
        'path_availability_matrix': np.array(path_availability_matrix),
        'total_available_paths': total_available_paths,
        'source': source,
        'target': target,
        'max_hops': max_hops,
        'num_unique_paths': len(all_paths)
    }


def plot_path_availability(results: Dict[str, Any], output_path: str):
    """
    Plot path availability heatmap and total path count.
    
    Args:
        results: Results from analyze_path_availability
        output_path: Path to save the plot
    """
    setup_plot_style()
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    
    # Plot 1: Total available paths over time
    ax1.plot(results['time_points'], results['total_available_paths'], 
             'b-', linewidth=1.5, marker='o', markevery=len(results['time_points'])//10)
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Available Paths')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Available Paths: {results['source']} → {results['target']}")
    
    # Plot 2: Path availability heatmap
    if results['num_unique_paths'] > 0:
        # Limit to top 15 most available paths for readability
        path_scores = np.sum(results['path_availability_matrix'], axis=1)
        top_indices = np.argsort(path_scores)[-15:]  # Top 15 paths
        
        top_paths = [results['path_names'][i] for i in top_indices]
        top_availability = results['path_availability_matrix'][top_indices]
        
        # Create heatmap
        im = ax2.imshow(top_availability, aspect='auto', cmap='RdYlGn', 
                       interpolation='nearest', vmin=0, vmax=1)
        
        # Set labels
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Path')
        ax2.set_title('Path Availability Over Time')
        
        # Set ticks
        n_time_ticks = min(10, len(results['time_points']))
        time_tick_indices = np.linspace(0, len(results['time_points'])-1, n_time_ticks, dtype=int)
        ax2.set_xticks(time_tick_indices)
        ax2.set_xticklabels([str(results['time_points'][i]) for i in time_tick_indices], rotation=45)
        
        # Simplify path names for y-axis
        simplified_paths = []
        for path in top_paths:
            nodes = path.split(' -> ')
            if len(nodes) > 2:
                simplified = f"{nodes[0].split('.')[-1]}→...→{nodes[-1].split('.')[-1]}"
            else:
                simplified = f"{nodes[0].split('.')[-1]}→{nodes[-1].split('.')[-1]}"
            simplified_paths.append(simplified)
        
        ax2.set_yticks(range(len(top_paths)))
        ax2.set_yticklabels(simplified_paths, fontsize=8)
        
        # Add colorbar
        plt.colorbar(im, ax=ax2, label='Available')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    
    print(f"Path availability plot saved to: {output_path}")


def print_path_statistics(results: Dict[str, Any]):
    """Print detailed statistics about path availability."""
    print("\n" + "="*60)
    print("PATH AVAILABILITY STATISTICS")
    print("="*60)
    
    print(f"Source: {results['source']}")
    print(f"Target: {results['target']}")
    print(f"Max hops: {results['max_hops']}")
    print(f"Time points analyzed: {len(results['time_points'])}")
    print(f"Unique paths found: {results['num_unique_paths']}")
    
    if results['num_unique_paths'] > 0:
        # Calculate statistics
        availability_matrix = results['path_availability_matrix']
        path_scores = np.sum(availability_matrix, axis=1) / len(results['time_points']) * 100
        
        print(f"\nPath availability statistics:")
        print(f"  Average available paths per round: {np.mean(results['total_available_paths']):.2f}")
        print(f"  Max available paths: {np.max(results['total_available_paths'])}")
        print(f"  Min available paths: {np.min(results['total_available_paths'])}")
        
        # Most reliable paths
        top_indices = np.argsort(path_scores)[-5:]  # Top 5
        print(f"\nMost reliable paths (top 5):")
        for i, idx in enumerate(top_indices[::-1]):
            path_name = results['path_names'][idx]
            availability_pct = path_scores[idx]
            print(f"  {i+1}. {path_name} ({availability_pct:.1f}% available)")
    
    print("="*60)


def main():
    """Main analysis function."""
    # Parameters
    source = "10.1.1.100"
    target = "10.1.1.102" 
    trace_period = "1144393236-1144450070"
    max_hops = 3
    num_rounds = 5000  # Use subset for faster analysis
    
    print(f"UCSB Path Availability Analysis")
    print(f"Node pair: {source} -> {target}")
    print(f"Trace period: {trace_period}")
    print(f"Max hops: {max_hops}")
    print("-" * 60)
    
    # Get trace files for the period
    data_dir = os.path.join(project_root, 'data', 'ucsb', trace_period)
    if not os.path.exists(data_dir):
        print(f"Error: Trace directory {data_dir} not found")
        return
    
    # Get neighbortable files
    neighbortable_files = []
    for filename in sorted(os.listdir(data_dir)):
        if filename.startswith('neighbortable-'):
            neighbortable_files.append(os.path.join(data_dir, filename))
    
    if not neighbortable_files:
        print(f"Error: No neighbortable files found in {data_dir}")
        return
    
    print(f"Found {len(neighbortable_files)} neighbortable files")
    
    # Limit files for faster analysis
    max_files = min(100, len(neighbortable_files))  # Use first 100 files
    neighbortable_files = neighbortable_files[:max_files]
    
    # Setup environment
    env = UCSBMeshnetMemmapEnvironment(
        neighbortable_files=neighbortable_files,
        source=source,
        destination=target,
        max_path_length=max_hops,
        routes_per_minute=15,
        seed=42
    )
    
    # Analyze path availability
    results = analyze_path_availability(
        env=env,
        source=source,
        target=target,
        max_hops=max_hops,
        sample_interval=10  # Sample every 10 rounds
    )
    
    # Print statistics
    print_path_statistics(results)
    
    # Create output directory
    output_dir = os.path.join(project_root, 'output', 'images')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plot
    output_path = os.path.join(output_dir, f'path_availability_{source.replace(".", "_")}_to_{target.replace(".", "_")}.pdf')
    plot_path_availability(results, output_path)
    
    # Cleanup
    env.cleanup()
    
    print(f"\nAnalysis completed!")


if __name__ == "__main__":
    main() 