#!/usr/bin/env python3
"""
Example script showing how to add other types of real network data.
"""

import sys
import os
import numpy as np
import networkx as nx
from typing import Dict, Any

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.environments.simple_environment import SimpleEnvironment
from src.bandits.cts_g import CTSG
from src.bandits.cts_b import CTSB


def create_internet_topology():
    """
    Create an Internet-like topology using Barabasi-Albert model.
    """
    print("Creating Internet-like topology...")
    
    # Barabasi-Albert model simulates Internet growth
    graph = nx.barabasi_albert_graph(n=50, m=3, seed=42)
    
    # Add some random edges to ensure connectivity
    while not nx.is_connected(graph):
        nodes = list(graph.nodes())
        u, v = np.random.choice(nodes, 2, replace=False)
        if not graph.has_edge(u, v):
            graph.add_edge(u, v)
    
    print(f"Internet topology: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph


def create_social_network():
    """
    Create a social network-like topology using Watts-Strogatz model.
    """
    print("Creating social network topology...")
    
    # Watts-Strogatz model simulates social networks
    graph = nx.watts_strogatz_graph(n=30, k=4, p=0.3, seed=42)
    
    # Ensure connectivity
    while not nx.is_connected(graph):
        nodes = list(graph.nodes())
        u, v = np.random.choice(nodes, 2, replace=False)
        if not graph.has_edge(u, v):
            graph.add_edge(u, v)
    
    print(f"Social network: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph


def create_grid_network():
    """
    Create a grid network topology.
    """
    print("Creating grid network topology...")
    
    # 6x6 grid network
    graph = nx.grid_2d_graph(6, 6)
    
    # Convert to integer nodes
    mapping = {node: i for i, node in enumerate(graph.nodes())}
    graph = nx.relabel_nodes(graph, mapping)
    
    print(f"Grid network: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph


def test_network_performance(graph: nx.Graph, network_name: str):
    """
    Test algorithm performance on a given network topology.
    """
    print(f"\n" + "=" * 60)
    print(f"TESTING {network_name.upper()} NETWORK")
    print("=" * 60)
    
    # Create a simple environment with the given graph
    # Note: This is a simplified approach - in practice you'd create a proper environment class
    
    # Simulate environment behavior
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    
    # Create random arm means (simulating link quality)
    arm_means = np.random.beta(2, 5, num_edges)  # Beta distribution for realistic link quality
    
    # Set source and destination
    source = 0
    destination = num_nodes - 1
    
    print(f"Network: {num_nodes} nodes, {num_edges} edges")
    print(f"Source: {source}, Destination: {destination}")
    print(f"Average arm quality: {np.mean(arm_means):.3f}")
    
    # Simulate algorithm performance
    # This is a simplified simulation - in practice you'd use the full environment
    
    # Simulate CTS-G behavior
    cts_g_rewards = []
    cts_b_rewards = []
    
    for round_idx in range(50):
        # Simulate selection and reward
        # In practice, this would use the actual algorithm and environment
        
        # Random selection for demonstration
        selected_arms = np.random.choice(num_edges, size=min(3, num_edges), replace=False)
        
        # Calculate reward
        reward = sum(arm_means[arm] for arm in selected_arms)
        
        # Add some noise
        reward += np.random.normal(0, 0.1)
        reward = max(0, min(1, reward))  # Clip to [0,1]
        
        cts_g_rewards.append(reward)
        cts_b_rewards.append(reward)
    
    print(f"CTS-G average reward: {np.mean(cts_g_rewards):.3f}")
    print(f"CTS-B average reward: {np.mean(cts_b_rewards):.3f}")


def main():
    """
    Test different network topologies.
    """
    print("Testing Different Network Topologies")
    print("=" * 60)
    
    # Test Internet-like topology
    internet_graph = create_internet_topology()
    test_network_performance(internet_graph, "Internet")
    
    # Test social network topology
    social_graph = create_social_network()
    test_network_performance(social_graph, "Social Network")
    
    # Test grid network topology
    grid_graph = create_grid_network()
    test_network_performance(grid_graph, "Grid Network")
    
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("This demonstrates how to create different network topologies")
    print("for testing combinatorial bandit algorithms.")
    print("\nTo add real network data:")
    print("1. Create a new environment class (e.g., InternetEnvironment)")
    print("2. Load real topology data from files")
    print("3. Implement reward and availability models")
    print("4. Test with existing algorithms")


if __name__ == "__main__":
    main() 