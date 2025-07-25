#!/usr/bin/env python3
"""
Network Topology Visualization for UCSB Mesh Network
===================================================

Visualize the network topology at specific time points in the UCSB mesh network trace data.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import List, Dict, Any, Tuple

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment


def setup_plot_style():
    """Setup plot style following user preferences"""
    sns.set_theme()
    sns.set_style("whitegrid")
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.figsize'] = (8, 6)


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


def plot_topology(G: nx.Graph, source: str, target: str, round_idx: int, output_path: str):
    """
    Plot network topology with highlighted source and target nodes.
    
    Args:
        G: NetworkX graph
        source: Source node
        target: Target node
        round_idx: Round index for title
        output_path: Path to save the plot
    """
    setup_plot_style()
    
    plt.figure(figsize=(12, 8))
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, edge_color='gray')
    
    # Prepare node colors
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node == source:
            node_colors.append('red')
            node_sizes.append(300)
        elif node == target:
            node_colors.append('blue')
            node_sizes.append(300)
        else:
            node_colors.append('lightblue')
            node_sizes.append(100)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    
    # Draw labels for source and target only
    special_nodes = {source: source.split('.')[-1], target: target.split('.')[-1]}
    nx.draw_networkx_labels(G, pos, labels=special_nodes, font_size=12, font_weight='bold')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label=f'Source ({source.split(".")[-1]})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label=f'Target ({target.split(".")[-1]})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=8, label='Other nodes')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Set labels (no title as per user preference)
    plt.xlabel('Network Layout', fontsize=10)
    plt.ylabel('', fontsize=10)
    
    # Remove axis ticks
    plt.xticks([])
    plt.yticks([])
    
    # Add network statistics as text
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)
    
    stats_text = f'Nodes: {num_nodes}\nEdges: {num_edges}\nDensity: {density:.3f}\nRound: {round_idx}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save plot without white margins
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    
    print(f"Topology plot saved to: {output_path}")


def analyze_topology_stats(env: UCSBMeshnetMemmapEnvironment, sample_rounds: List[int]) -> Dict[str, Any]:
    """
    Analyze topology statistics across multiple rounds.
    
    Args:
        env: UCSB environment
        sample_rounds: List of rounds to analyze
        
    Returns:
        Dictionary with topology statistics
    """
    source, destination = env.get_source_destination()
    
    stats = {
        'rounds': sample_rounds,
        'num_nodes': [],
        'num_edges': [],
        'density': [],
        'connectivity': [],
        'avg_degree': [],
        'source': source,
        'destination': destination
    }
    
    print(f"Analyzing topology statistics for {len(sample_rounds)} rounds...")
    
    for i, round_idx in enumerate(sample_rounds):
        if i % 10 == 0:
            print(f"Processing round {round_idx} ({i+1}/{len(sample_rounds)})")
            
        G = get_topology_at_round(env, round_idx)
        
        stats['num_nodes'].append(G.number_of_nodes())
        stats['num_edges'].append(G.number_of_edges())
        stats['density'].append(nx.density(G))
        stats['connectivity'].append(1 if nx.is_connected(G) else 0)
        stats['avg_degree'].append(np.mean([degree for node, degree in G.degree()]) if G.number_of_nodes() > 0 else 0)
    
    return stats


def plot_topology_evolution(stats: Dict[str, Any], output_path: str):
    """
    Plot topology evolution over time.
    
    Args:
        stats: Topology statistics from analyze_topology_stats
        output_path: Path to save the plot
    """
    setup_plot_style()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    rounds = stats['rounds']
    
    # Plot 1: Number of edges over time
    ax1.plot(rounds, stats['num_edges'], 'b-', linewidth=1.5, marker='o', 
             markevery=max(1, len(rounds)//10), markersize=4)
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Number of Edges')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Network density over time
    ax2.plot(rounds, stats['density'], 'g-', linewidth=1.5, marker='s',
             markevery=max(1, len(rounds)//10), markersize=4)
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Network Density')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Connectivity over time
    ax3.plot(rounds, stats['connectivity'], 'r-', linewidth=1.5, marker='^',
             markevery=max(1, len(rounds)//10), markersize=4)
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Connected (1/0)')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.1, 1.1)
    
    # Plot 4: Average degree over time
    ax4.plot(rounds, stats['avg_degree'], 'm-', linewidth=1.5, marker='D',
             markevery=max(1, len(rounds)//10), markersize=4)
    ax4.set_xlabel('Round')
    ax4.set_ylabel('Average Degree')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    
    print(f"Topology evolution plot saved to: {output_path}")


def main():
    """Main analysis function."""
    # Parameters
    source = "10.1.1.100"
    target = "10.1.1.102" 
    trace_period = "1144393236-1144450070"
    max_hops = 3
    
    print(f"UCSB Network Topology Visualization")
    print(f"Node pair: {source} -> {target}")
    print(f"Trace period: {trace_period}")
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
    
    # Setup environment
    env = UCSBMeshnetMemmapEnvironment(
        neighbortable_files=neighbortable_files,
        source=source,
        destination=target,
        max_path_length=max_hops,
        routes_per_minute=15,
        seed=42
    )
    
    # Create output directory
    output_dir = os.path.join(project_root, 'output', 'images')
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot topologies at different time points
    sample_rounds = [0, 1000, 5000, 10000, 14000]  # Sample different time points
    
    for round_idx in sample_rounds:
        if round_idx < env.num_rounds:
            print(f"\nGenerating topology plot for round {round_idx}...")
            G = get_topology_at_round(env, round_idx)
            
            output_path = os.path.join(output_dir, f'topology_round_{round_idx}.pdf')
            plot_topology(G, source, target, round_idx, output_path)
            
            print(f"Round {round_idx}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Analyze topology evolution
    print(f"\nAnalyzing topology evolution...")
    evolution_rounds = list(range(0, env.num_rounds, 100))  # Sample every 100 rounds
    stats = analyze_topology_stats(env, evolution_rounds)
    
    # Plot topology evolution
    evolution_output = os.path.join(output_dir, 'topology_evolution.pdf')
    plot_topology_evolution(stats, evolution_output)
    
    # Print summary statistics
    print(f"\n" + "="*60)
    print("TOPOLOGY STATISTICS SUMMARY")
    print("="*60)
    print(f"Average number of edges: {np.mean(stats['num_edges']):.1f}")
    print(f"Average network density: {np.mean(stats['density']):.3f}")
    print(f"Connectivity rate: {np.mean(stats['connectivity'])*100:.1f}%")
    print(f"Average degree: {np.mean(stats['avg_degree']):.2f}")
    print("="*60)
    
    # Cleanup
    env.cleanup()
    
    print(f"\nTopology analysis completed!")


if __name__ == "__main__":
    main() 