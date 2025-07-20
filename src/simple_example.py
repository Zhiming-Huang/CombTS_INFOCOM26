#!/usr/bin/env python3
"""
Simple example of Sleeping Combinatorial Bandits for Network Routing

This demonstrates the basic usage of RoutingEnvironment and CombTSAgent
classes for solving network routing problems.
"""

import numpy as np
import networkx as nx
from routing_environment import RoutingEnvironment
from main_simulation import create_sample_network
from comb_ts_agent import CombTSAgent

def main():
    """Run a simple example of the routing problem."""
    print("=== Sleeping Combinatorial Bandits for Network Routing ===\n")
    
    # Create a sample network
    print("1. Creating sample network...")
    edge_availability_probs, edge_reward_probs = create_sample_network()
    source, target = 0, 2  # Hard-coded for simple 3-node network
    num_nodes = 3
    print(f"   Network has {num_nodes} nodes and {len(edge_availability_probs)} edges")
    print(f"   Edges: {list(edge_availability_probs.keys())}")
    
    # Create environment
    print(f"\n2. Setting up routing from node {source} to node {target}...")
    env = RoutingEnvironment(num_nodes, edge_availability_probs, edge_reward_probs, source, target)
    
    # Create agent
    print("\n3. Initializing Thompson Sampling agent...")
    agent = CombTSAgent(env, alpha=1.0, beta=1.0)
    
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
    for edge in sorted(env.get_all_edges()):
        alpha = agent.edge_alpha[edge]
        beta = agent.edge_beta[edge]
        # Calculate observations as increase from prior
        obs = (alpha + beta) - 2  # Starting with 1.0 + 1.0 = 2.0
        mean_est = alpha / (alpha + beta)
        print(f"   Edge {edge}: α={alpha:.1f}, β={beta:.1f}, observations={obs}, est_mean={mean_est:.3f}")

if __name__ == "__main__":
    main()