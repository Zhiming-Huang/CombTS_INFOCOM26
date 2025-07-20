#!/usr/bin/env python3
"""
Test script for the Sleeping Combinatorial Bandits implementation.

This script runs basic tests to verify that the RoutingEnvironment and
CombTSAgent classes work correctly.
"""

import numpy as np
import networkx as nx
from routing_environment import RoutingEnvironment
from main_simulation import create_sample_network
from comb_ts_agent import CombTSAgent


def test_routing_environment():
    """Test the RoutingEnvironment class."""
    print("Testing RoutingEnvironment...")
    
    # Create a simple test network
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (0, 2)])
    
    availability_probs = {(0, 1): 0.8, (0, 2): 0.6, (1, 2): 0.9}
    reward_means = {(0, 1): 0.7, (0, 2): 0.5, (1, 2): 0.8}
    
    env = RoutingEnvironment(G, availability_probs, reward_means, source=0, target=2)
    
    # Test sampling available graph
    available_graph = env.sample_available_graph()
    assert isinstance(available_graph, nx.Graph)
    print(f"  ✓ Available graph sampled with {available_graph.number_of_edges()} edges")
    
    # Test getting feasible paths
    feasible_paths = env.get_feasible_paths(available_graph, max_paths=5)
    print(f"  ✓ Found {len(feasible_paths)} feasible paths")
    
    # Test reward generation
    if feasible_paths:
        path = feasible_paths[0]
        reward = env.get_reward(path)
        print(f"  ✓ Generated reward {reward} for path {path}")
    
    # Test path edge conversion
    test_path = [0, 1, 2]
    edges = env.get_path_edges(test_path)
    expected_edges = [(0, 1), (1, 2)]
    assert len(edges) == len(expected_edges)
    print(f"  ✓ Path to edges conversion works: {test_path} -> {edges}")
    
    print("RoutingEnvironment tests passed!\n")


def test_comb_ts_agent():
    """Test the CombTSAgent class."""
    print("Testing CombTSAgent...")
    
    # Create test environment first
    edge_availability_probs = {(0, 1): 1.0, (1, 2): 1.0, (0, 2): 1.0}
    edge_reward_probs = {(0, 1): 0.5, (1, 2): 0.5, (0, 2): 0.5}
    from routing_environment import RoutingEnvironment
    env = RoutingEnvironment(3, edge_availability_probs, edge_reward_probs, 0, 2)
    agent = CombTSAgent(env, alpha=1.0, beta=1.0)
    
    # Test edge mean sampling
    sampled_means = agent.sample_edge_means()
    assert len(sampled_means) == len(env.get_all_edges())
    print(f"  ✓ Sampled edge means: {sampled_means}")
    
    # Test path selection
    feasible_paths = [[0, 1, 2], [0, 2]]
    selected_path = agent.select_path(feasible_paths)
    assert selected_path in feasible_paths
    print(f"  ✓ Selected path: {selected_path}")
    
    # Test update
    initial_alpha = agent.edge_alpha[(0, 1)]
    agent.update([0, 1, 2], reward=2.0)
    new_alpha = agent.edge_alpha[(0, 1)]
    assert new_alpha > initial_alpha
    print(f"  ✓ Update works: alpha for (0,1) changed from {initial_alpha} to {new_alpha}")
    
    # Test edge estimates
    estimates = agent.get_edge_estimates()
    assert isinstance(estimates, dict)
    assert len(estimates) > 0
    print(f"  ✓ Edge estimates generated with {len(estimates)} edges")
    
    # Test confidence intervals
    confidence = agent.get_edge_confidence()
    assert isinstance(confidence, dict)
    assert len(confidence) > 0
    print(f"  ✓ Confidence intervals generated for {len(confidence)} edges")
    
    print("CombTSAgent tests passed!\n")


def test_integration():
    """Test integration between RoutingEnvironment and CombTSAgent."""
    print("Testing integration...")
    
    # Create sample network
    edge_availability_probs, edge_reward_probs = create_sample_network()
    source, target = 0, 2  # Hard-coded for simple 3-node network
    num_nodes = 3
    
    # Initialize environment and agent
    env = RoutingEnvironment(num_nodes, edge_availability_probs, edge_reward_probs, source, target)
    agent = CombTSAgent(env, alpha=1.0, beta=1.0)
    
    # Run a few simulation steps
    total_reward = 0.0
    successful_rounds = 0
    
    for round_num in range(10):
        # Sample available graph
        available_graph = env.sample_available_graph()
        
        # Get feasible paths
        feasible_paths = env.get_feasible_paths(available_graph)
        
        if feasible_paths:
            # Select path
            selected_path = agent.select_path(feasible_paths)
            
            # Get reward
            reward = env.get_reward(selected_path)
            total_reward += reward
            successful_rounds += 1
            
            # Update agent
            agent.update(selected_path, reward)
    
    print(f"  ✓ Completed {successful_rounds} successful rounds out of 10")
    print(f"  ✓ Total reward: {total_reward}")
    print(f"  ✓ Average reward: {total_reward / max(successful_rounds, 1):.3f}")
    
    # Check that agent learned something
    edge_estimates = agent.get_edge_estimates()
    total_observations = sum(agent.edge_alpha[edge] + agent.edge_beta[edge] - 2 for edge in env.get_all_edges())
    print(f"  ✓ Agent made {total_observations} total edge observations")
    print(f"  ✓ Agent learned estimates for {len(edge_estimates)} edges")
    
    print("Integration tests passed!\n")


def main():
    """Run all tests."""
    print("Running tests for Sleeping Combinatorial Bandits implementation")
    print("=" * 60)
    
    # Set random seed for reproducible tests
    np.random.seed(42)
    
    try:
        test_routing_environment()
        test_comb_ts_agent()
        test_integration()
        
        print("🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()