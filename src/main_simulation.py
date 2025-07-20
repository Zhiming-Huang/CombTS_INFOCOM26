import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import random
from routing_environment import RoutingEnvironment
from comb_ts_agent import CombTSAgent


def create_sample_network() -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
    """
    Create a sample network with edge availability and reward probabilities.
    
    Returns:
        Tuple of (edge_availability_probs, edge_reward_probs)
    """
    # Define a simple network: 0 -> 1 -> 2, 0 -> 2
    edge_availability_probs = {
        (0, 1): 0.8,  # Edge 0->1 has 80% availability
        (1, 2): 0.7,  # Edge 1->2 has 70% availability
        (0, 2): 0.6,  # Edge 0->2 has 60% availability
    }
    
    edge_reward_probs = {
        (0, 1): 0.9,  # Edge 0->1 has 90% success probability
        (1, 2): 0.8,  # Edge 1->2 has 80% success probability
        (0, 2): 0.7,  # Edge 0->2 has 70% success probability
    }
    
    return edge_availability_probs, edge_reward_probs


def run_simulation(num_rounds: int = 1000, num_nodes: int = 3):
    """
    Run the main simulation of Sleeping Combinatorial Bandits for network routing.
    
    Args:
        num_rounds: Number of rounds to simulate
        num_nodes: Number of nodes in the network
    """
    print("Setting up network environment...")
    
    # Create network
    edge_availability_probs, edge_reward_probs = create_sample_network()
    
    # Initialize environment and agent
    environment = RoutingEnvironment(
        num_nodes=num_nodes,
        edge_availability_probs=edge_availability_probs,
        edge_reward_probs=edge_reward_probs,
        source=0,
        target=2
    )
    
    agent = CombTSAgent(environment, alpha=1.0, beta=1.0)
    
    print(f"Network edges: {list(edge_availability_probs.keys())}")
    print(f"Edge availability probabilities: {edge_availability_probs}")
    print(f"Edge reward probabilities: {edge_reward_probs}")
    print(f"Running simulation for {num_rounds} rounds...")
    
    # Track performance
    cumulative_rewards = []
    cumulative_regret = []
    path_selection_counts = {}
    
    # Calculate optimal expected reward for regret calculation
    optimal_paths = environment.get_feasible_paths(environment.full_graph)
    optimal_rewards = []
    for path in optimal_paths:
        expected_reward = sum(edge_reward_probs.get(edge, 0) for edge in environment.get_path_edges(path))
        optimal_rewards.append(expected_reward)
    optimal_expected_reward = max(optimal_rewards) if optimal_rewards else 0
    
    print(f"Optimal expected reward: {optimal_expected_reward}")
    
    total_reward = 0
    total_regret = 0
    
    for round_num in range(num_rounds):
        # Sample available graph
        available_graph = environment.sample_available_graph()
        
        # Get feasible paths
        feasible_paths = environment.get_feasible_paths(available_graph)
        
        if not feasible_paths:
            # No feasible paths in this round
            reward = 0
            regret = optimal_expected_reward
        else:
            # Select path using Thompson Sampling
            selected_path = agent.select_path(feasible_paths)
            
            # Get reward for selected path
            reward = environment.get_reward(selected_path)
            
            # Update agent
            agent.update(selected_path, reward)
            
            # Track path selection
            path_str = str(selected_path)
            path_selection_counts[path_str] = path_selection_counts.get(path_str, 0) + 1
            
            # Calculate regret
            regret = optimal_expected_reward - reward
        
        total_reward += reward
        total_regret += regret
        
        cumulative_rewards.append(total_reward)
        cumulative_regret.append(total_regret)
        
        # Print progress every 100 rounds
        if (round_num + 1) % 100 == 0:
            avg_reward = total_reward / (round_num + 1)
            avg_regret = total_regret / (round_num + 1)
            print(f"Round {round_num + 1}: Avg Reward = {avg_reward:.3f}, Avg Regret = {avg_regret:.3f}")
    
    # Print final results
    print("\n=== Final Results ===")
    print(f"Total reward: {total_reward}")
    print(f"Average reward: {total_reward / num_rounds:.3f}")
    print(f"Total regret: {total_regret}")
    print(f"Average regret: {total_regret / num_rounds:.3f}")
    
    # Print path selection statistics
    print("\n=== Path Selection Statistics ===")
    for path, count in sorted(path_selection_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / num_rounds) * 100
        print(f"Path {path}: {count} times ({percentage:.1f}%)")
    
    # Print final edge estimates
    print("\n=== Final Edge Estimates ===")
    edge_estimates = agent.get_edge_estimates()
    for edge, estimate in edge_estimates.items():
        true_prob = edge_reward_probs.get(edge, 0)
        print(f"Edge {edge}: Estimate = {estimate:.3f}, True = {true_prob:.3f}")
    
    return cumulative_rewards, cumulative_regret, path_selection_counts


def plot_results(cumulative_rewards: List[float], cumulative_regret: List[float]):
    """
    Plot the simulation results.
    
    Args:
        cumulative_rewards: List of cumulative rewards over time
        cumulative_regret: List of cumulative regret over time
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot cumulative rewards
    ax1.plot(cumulative_rewards, label='Cumulative Reward', color='blue')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Cumulative Reward')
    ax1.set_title('Cumulative Reward Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot cumulative regret
    ax2.plot(cumulative_regret, label='Cumulative Regret', color='red')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Cumulative Regret')
    ax2.set_title('Cumulative Regret Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simulation_results.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Run simulation
    cumulative_rewards, cumulative_regret, path_counts = run_simulation(num_rounds=1000)
    
    # Plot results
    plot_results(cumulative_rewards, cumulative_regret)