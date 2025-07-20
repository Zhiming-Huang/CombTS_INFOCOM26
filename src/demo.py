#!/usr/bin/env python3
"""
Demonstration of Sleeping Combinatorial Bandits for Network Routing

This script demonstrates how to use the RoutingEnvironment and CombTSAgent
classes to solve a network routing problem using Thompson Sampling.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Tuple
import time

from routing_environment import RoutingEnvironment
from main_simulation import create_sample_network
from comb_ts_agent import CombTSAgent


def run_simulation(num_rounds: int = 1000, max_paths: int = 10, seed: int = 42, track_history: bool = True) -> Dict:
    """
    Run a simulation of the sleeping combinatorial bandit routing problem.
    
    Args:
        num_rounds: Number of simulation rounds
        max_paths: Maximum number of paths to consider per round
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing simulation results
    """
    # Set random seed
    np.random.seed(seed)
    
    # Create sample network
    print("Creating sample network...")
    edge_availability_probs, edge_reward_probs = create_sample_network()
    source, target = 0, 2  # Hard-coded for simple 3-node network
    num_nodes = 3
    
    print(f"Network: {num_nodes} nodes, {len(edge_availability_probs)} edges")
    print(f"Source: {source}, Target: {target}")
    
    # Initialize environment and agent
    env = RoutingEnvironment(num_nodes, edge_availability_probs, edge_reward_probs, source, target)
    agent = CombTSAgent(env, alpha=1.0, beta=1.0)
    
    # Track simulation results
    results = {
        'rounds': [],
        'rewards': [],
        'cumulative_rewards': [],
        'selected_paths': [],
        'num_feasible_paths': [],
        'regret': [],
        'path_lengths': []
    }
    
    cumulative_reward = 0.0
    # Calculate best expected reward from all possible paths in full graph
    all_paths = env.get_feasible_paths(env.full_graph)
    best_expected_reward = 0.0
    if all_paths:
        best_expected_reward = max([sum(edge_reward_probs[edge] for edge in env.get_path_edges(path)) 
                                   for path in all_paths])
    cumulative_regret = 0.0
    
    print(f"\nStarting simulation for {num_rounds} rounds...")
    
    for round_num in range(num_rounds):
        # Sample available graph for this round
        available_graph = env.sample_available_graph()
        
        # Get feasible paths in current available graph
        feasible_paths = env.get_feasible_paths(available_graph)
        
        if not feasible_paths:
            # No paths available, skip this round
            reward = 0.0
            selected_path = []
            path_length = 0
        else:
            # Agent selects a path using Thompson Sampling
            selected_path = agent.select_path(feasible_paths)
            
            # Get reward from environment
            reward = env.get_reward(selected_path)
            path_length = len(selected_path) - 1 if len(selected_path) > 1 else 0
            
            # Update agent with observed reward
            agent.update(selected_path, reward)
        
        # Calculate regret (difference from optimal expected reward)
        if feasible_paths:
            expected_rewards = []
            for path in feasible_paths:
                path_edges = env.get_path_edges(path)
                expected_reward = sum(edge_reward_probs[edge] for edge in path_edges)
                expected_rewards.append(expected_reward)
            round_optimal = max(expected_rewards)
        else:
            round_optimal = 0.0
        
        instantaneous_regret = round_optimal - reward
        cumulative_regret += instantaneous_regret
        
        # Update tracking
        cumulative_reward += reward
        if track_history:
            results['rounds'].append(round_num)
            results['rewards'].append(reward)
            results['cumulative_rewards'].append(cumulative_reward)
            results['selected_paths'].append(selected_path.copy() if selected_path else [])
            results['num_feasible_paths'].append(len(feasible_paths))
            results['regret'].append(cumulative_regret)
            results['path_lengths'].append(path_length)
        
        # Print progress
        if (round_num + 1) % 100 == 0:
            avg_reward = cumulative_reward / (round_num + 1)
            print(f"Round {round_num + 1}: Avg reward = {avg_reward:.3f}, "
                  f"Cumulative regret = {cumulative_regret:.3f}")
    
    # Store final results
    results['final_reward'] = cumulative_reward
    results['final_regret'] = cumulative_regret
    results['average_reward'] = cumulative_reward / num_rounds
    results['average_regret'] = cumulative_regret / num_rounds
    
    print(f"\nSimulation completed!")
    print(f"Total cumulative reward: {cumulative_reward:.3f}")
    print(f"Average reward per round: {cumulative_reward / num_rounds:.3f}")
    print(f"Total cumulative regret: {cumulative_regret:.3f}")
    
    return results, agent, env


def plot_results(results: Dict, save_plots: bool = True):
    """
    Plot simulation results.
    
    Args:
        results: Results dictionary from run_simulation
        save_plots: Whether to save plots to files
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    rounds = results['rounds']
    
    # Plot cumulative rewards
    axes[0, 0].plot(rounds, results['cumulative_rewards'])
    axes[0, 0].set_title('Cumulative Rewards')
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Cumulative Reward')
    axes[0, 0].grid(True)
    
    # Plot instantaneous rewards (moving average)
    window_size = 50
    rewards = results['rewards']
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        axes[0, 1].plot(rounds[window_size-1:], moving_avg)
    axes[0, 1].set_title(f'Moving Average Reward (window={window_size})')
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Average Reward')
    axes[0, 1].grid(True)
    
    # Plot cumulative regret
    axes[1, 0].plot(rounds, results['regret'])
    axes[1, 0].set_title('Cumulative Regret')
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('Cumulative Regret')
    axes[1, 0].grid(True)
    
    # Plot number of feasible paths per round
    axes[1, 1].plot(rounds, results['num_feasible_paths'])
    axes[1, 1].set_title('Number of Feasible Paths per Round')
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('Number of Paths')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('simulation_results.png', dpi=300, bbox_inches='tight')
        print("Plot saved as 'simulation_results.png'")
    
    plt.show()


def analyze_learned_parameters(agent: CombTSAgent, env: RoutingEnvironment):
    """
    Analyze the learned parameters from the agent.
    
    Args:
        agent: The trained agent
        env: The environment
    """
    print("\n" + "="*50)
    print("LEARNED PARAMETERS ANALYSIS")
    print("="*50)
    
    edge_estimates = agent.get_edge_estimates()
    edge_confidence = agent.get_edge_confidence()
    
    total_observations = sum(agent.edge_alpha[edge] + agent.edge_beta[edge] - 2 
                           for edge in env.get_all_edges())
    
    print(f"Total observations: {total_observations}")
    print("\nEdge-wise analysis:")
    print(f"{'Edge':<15} {'Observations':<12} {'Posterior Mean':<15} {'Variance':<15}")
    print("-" * 70)
    
    for edge in sorted(env.get_all_edges()):
        obs = agent.edge_alpha[edge] + agent.edge_beta[edge] - 2
        mean = edge_estimates[edge]
        variance = edge_confidence[edge]
        
        print(f"{str(edge):<15} {obs:<12} {mean:<15.3f} {variance:<15.6f}")


def main():
    """Main execution function."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    print("Sleeping Combinatorial Bandits for Network Routing")
    print("=" * 55)
    
    # Run simulation
    start_time = time.time()
    results, agent, env = run_simulation(num_rounds=1000, max_paths=10, seed=42)
    end_time = time.time()
    
    print(f"\nSimulation time: {end_time - start_time:.2f} seconds")
    
    # Analyze results
    analyze_learned_parameters(agent, env)
    
    # Plot results (if matplotlib is available)
    try:
        plot_results(results, save_plots=True)
    except ImportError:
        print("\nMatplotlib not available, skipping plots.")
    except Exception as e:
        print(f"\nError creating plots: {e}")
        # Print traceback for debugging
        import traceback
        traceback.print_exc()
    
    # Print some example paths
    print("\n" + "="*50)
    print("EXAMPLE SELECTED PATHS (last 10 rounds)")
    print("="*50)
    
    for i, path in enumerate(results['selected_paths'][-10:], start=len(results['selected_paths'])-10):
        reward = results['rewards'][i]
        if path:
            print(f"Round {i+1}: Path {path} (length {len(path)-1}), Reward: {reward}")
        else:
            print(f"Round {i+1}: No feasible path, Reward: {reward}")


if __name__ == "__main__":
    main()