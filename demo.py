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

from routing_environment import RoutingEnvironment, create_sample_network
from comb_ts_agent import CombTSAgent


def run_simulation(num_rounds: int = 1000, max_paths: int = 10, seed: int = 42) -> Dict:
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
    graph, availability_probs, reward_means, source, target = create_sample_network()
    
    print(f"Network: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
    print(f"Source: {source}, Target: {target}")
    
    # Initialize environment and agent
    env = RoutingEnvironment(graph, availability_probs, reward_means, source, target)
    agent = CombTSAgent(list(graph.edges()), alpha_prior=1.0, beta_prior=1.0)
    
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
    best_expected_reward = max([sum(reward_means[tuple(sorted([path[i], path[i+1]]))] 
                                   for i in range(len(path)-1)) 
                               for path in env.get_feasible_paths(graph)])
    cumulative_regret = 0.0
    
    print(f"\nStarting simulation for {num_rounds} rounds...")
    
    for round_num in range(num_rounds):
        # Sample available graph for this round
        available_graph = env.sample_available_graph()
        
        # Get feasible paths in current available graph
        feasible_paths = env.get_feasible_paths(available_graph, max_paths=max_paths)
        
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
                expected_reward = sum(reward_means[edge] for edge in path_edges)
                expected_rewards.append(expected_reward)
            round_optimal = max(expected_rewards)
        else:
            round_optimal = 0.0
        
        instantaneous_regret = round_optimal - reward
        cumulative_regret += instantaneous_regret
        
        # Update tracking
        cumulative_reward += reward
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
    
    # Get final statistics from agent
    final_stats = agent.get_statistics()
    results['final_stats'] = final_stats
    
    print(f"\nSimulation completed!")
    print(f"Total cumulative reward: {cumulative_reward:.3f}")
    print(f"Average reward per round: {cumulative_reward / num_rounds:.3f}")
    print(f"Total cumulative regret: {cumulative_regret:.3f}")
    
    return results


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


def analyze_learned_parameters(results: Dict):
    """
    Analyze the learned parameters from the agent.
    
    Args:
        results: Results dictionary from run_simulation
    """
    print("\n" + "="*50)
    print("LEARNED PARAMETERS ANALYSIS")
    print("="*50)
    
    final_stats = results['final_stats']
    posterior_means = final_stats['posterior_means']
    confidence_intervals = final_stats['confidence_intervals']
    edge_observations = final_stats['edge_observations']
    
    print(f"Total observations: {final_stats['total_observations']}")
    print("\nEdge-wise analysis:")
    print(f"{'Edge':<15} {'Observations':<12} {'Posterior Mean':<15} {'95% CI':<20}")
    print("-" * 70)
    
    for edge in sorted(posterior_means.keys()):
        obs = edge_observations[edge]
        mean = posterior_means[edge]
        ci_low, ci_high = confidence_intervals[edge]
        
        print(f"{str(edge):<15} {obs:<12} {mean:<15.3f} "
              f"[{ci_low:.3f}, {ci_high:.3f}]")


def main():
    """Main execution function."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    print("Sleeping Combinatorial Bandits for Network Routing")
    print("=" * 55)
    
    # Run simulation
    start_time = time.time()
    results = run_simulation(num_rounds=1000, max_paths=10, seed=42)
    end_time = time.time()
    
    print(f"\nSimulation time: {end_time - start_time:.2f} seconds")
    
    # Analyze results
    analyze_learned_parameters(results)
    
    # Plot results (if matplotlib is available)
    try:
        plot_results(results, save_plots=True)
    except ImportError:
        print("\nMatplotlib not available, skipping plots.")
    except Exception as e:
        print(f"\nError creating plots: {e}")
    
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