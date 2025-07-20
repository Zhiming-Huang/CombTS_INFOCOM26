#!/usr/bin/env python3
"""
Simple example of Sleeping Combinatorial Bandits for Network Routing

This demonstrates the basic usage of RoutingEnvironment and CombTSAgent
classes for solving network routing problems.
"""

import numpy as np
import networkx as nx
from routing_environment import RoutingEnvironment, create_sample_network
from comb_ts_agent import CombTSAgent

def main():
    """Run a simple example of the routing problem."""
    print("=== Sleeping Combinatorial Bandits for Network Routing ===\n")
    
    # Create a sample network
    print("1. Creating sample network...")
    G, availability_probs, reward_means, source, target = create_sample_network()
    print(f"   Network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"   Edges: {list(G.edges())}")
    
    # Create environment
    print(f"\n2. Setting up routing from node {source} to node {target}...")
    env = RoutingEnvironment(G, availability_probs, reward_means, source, target)
    
    # Create agent
    print("\n3. Initializing Thompson Sampling agent...")
    agent = CombTSAgent(list(G.edges()), alpha_prior=1.0, beta_prior=1.0)
    
    # Run a few rounds
    print(f"\n4. Running simulation rounds...")
    total_reward = 0.0
    successful_rounds = 0
    
    for round_num in range(10):
        # Sample available graph
        available_graph = env.sample_available_graph()
        
        # Get feasible paths
        feasible_paths = env.get_feasible_paths(available_graph)
        
        if not feasible_paths:
            print(f"   Round {round_num + 1}: No feasible paths available")
            continue
            
        # Agent selects path
        selected_path = agent.select_path(feasible_paths)
        
        # Get reward
        path_reward = env.get_reward(selected_path)
        
        # Update agent
        agent.update(selected_path, path_reward)
        
        total_reward += path_reward
        successful_rounds += 1
        
        print(f"   Round {round_num + 1}: Selected path {selected_path}, reward = {path_reward:.2f}")
    
    print(f"\n5. Results:")
    print(f"   Total rounds: 10")
    print(f"   Successful rounds: {successful_rounds}")
    print(f"   Total reward: {total_reward:.2f}")
    print(f"   Average reward per successful round: {total_reward/max(1, successful_rounds):.2f}")
    
    # Show agent's learned parameters
    print(f"\n6. Agent's learned edge parameters (α, β):")
    for edge in sorted(agent.edges):
        alpha = agent.alpha_params[edge]
        beta = agent.beta_params[edge]
        obs = agent.edge_observations[edge]
        mean_est = alpha / (alpha + beta)
        print(f"   Edge {edge}: α={alpha:.1f}, β={beta:.1f}, observations={obs}, est_mean={mean_est:.3f}")

if __name__ == "__main__":
    main()