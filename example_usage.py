"""
Example usage of Sleeping Combinatorial Bandits for Network Routing

This script demonstrates how to use the RoutingEnvironment and CombTSAgent classes
with different network topologies and configurations.
"""

import numpy as np
import random
from routing_environment import RoutingEnvironment
from comb_ts_agent import CombTSAgent


def create_grid_network(size: int = 3) -> tuple:
    """
    Create a grid network topology.
    
    Args:
        size: Size of the grid (size x size)
        
    Returns:
        Tuple of (edge_availability_probs, edge_reward_probs)
    """
    edge_availability_probs = {}
    edge_reward_probs = {}
    
    # Create horizontal edges
    for i in range(size):
        for j in range(size - 1):
            node1 = i * size + j
            node2 = i * size + j + 1
            edge_availability_probs[(node1, node2)] = 0.8
            edge_reward_probs[(node1, node2)] = 0.7
    
    # Create vertical edges
    for i in range(size - 1):
        for j in range(size):
            node1 = i * size + j
            node2 = (i + 1) * size + j
            edge_availability_probs[(node1, node2)] = 0.75
            edge_reward_probs[(node1, node2)] = 0.65
    
    return edge_availability_probs, edge_reward_probs


def create_star_network(num_spokes: int = 5) -> tuple:
    """
    Create a star network topology.
    
    Args:
        num_spokes: Number of spoke nodes
        
    Returns:
        Tuple of (edge_availability_probs, edge_reward_probs)
    """
    edge_availability_probs = {}
    edge_reward_probs = {}
    
    center = 0
    for i in range(1, num_spokes + 1):
        # Edge from center to spoke
        edge_availability_probs[(center, i)] = 0.9
        edge_reward_probs[(center, i)] = 0.8
        
        # Edge from spoke to center
        edge_availability_probs[(i, center)] = 0.85
        edge_reward_probs[(i, center)] = 0.75
    
    return edge_availability_probs, edge_reward_probs


def run_single_round_example():
    """
    Demonstrate a single round of the algorithm.
    """
    print("=== Single Round Example ===")
    
    # Create a simple network
    edge_availability_probs = {
        (0, 1): 0.8,
        (1, 2): 0.7,
        (0, 2): 0.6,
    }
    
    edge_reward_probs = {
        (0, 1): 0.9,
        (1, 2): 0.8,
        (0, 2): 0.7,
    }
    
    # Initialize environment and agent
    environment = RoutingEnvironment(
        num_nodes=3,
        edge_availability_probs=edge_availability_probs,
        edge_reward_probs=edge_reward_probs,
        source=0,
        target=2
    )
    
    agent = CombTSAgent(environment)
    
    # Simulate one round
    print("1. Sampling available graph...")
    available_graph = environment.sample_available_graph()
    print(f"   Available edges: {list(available_graph.edges())}")
    
    print("2. Finding feasible paths...")
    feasible_paths = environment.get_feasible_paths(available_graph)
    print(f"   Feasible paths: {feasible_paths}")
    
    if feasible_paths:
        print("3. Selecting path using Thompson Sampling...")
        selected_path = agent.select_path(feasible_paths)
        print(f"   Selected path: {selected_path}")
        
        print("4. Getting reward...")
        reward = environment.get_reward(selected_path)
        print(f"   Reward: {reward}")
        
        print("5. Updating agent...")
        agent.update(selected_path, reward)
        print("   Agent updated!")
        
        print("6. Current edge estimates:")
        edge_estimates = agent.get_edge_estimates()
        for edge, estimate in edge_estimates.items():
            print(f"   Edge {edge}: {estimate:.3f}")
    else:
        print("   No feasible paths available in this round.")


def run_comparison_experiment():
    """
    Compare different network topologies.
    """
    print("\n=== Network Topology Comparison ===")
    
    topologies = {
        "Simple": create_simple_network(),
        "Grid": create_grid_network(3),
        "Star": create_star_network(4)
    }
    
    results = {}
    
    for name, (edge_avail, edge_reward) in topologies.items():
        print(f"\nTesting {name} topology...")
        
        # Determine number of nodes
        all_nodes = set()
        for edge in edge_avail.keys():
            all_nodes.add(edge[0])
            all_nodes.add(edge[1])
        num_nodes = max(all_nodes) + 1
        
        # Initialize environment and agent
        environment = RoutingEnvironment(
            num_nodes=num_nodes,
            edge_availability_probs=edge_avail,
            edge_reward_probs=edge_reward,
            source=0,
            target=num_nodes-1
        )
        
        agent = CombTSAgent(environment)
        
        # Run simulation
        total_reward = 0
        num_rounds = 100
        
        for _ in range(num_rounds):
            available_graph = environment.sample_available_graph()
            feasible_paths = environment.get_feasible_paths(available_graph)
            
            if feasible_paths:
                selected_path = agent.select_path(feasible_paths)
                reward = environment.get_reward(selected_path)
                agent.update(selected_path, reward)
                total_reward += reward
        
        avg_reward = total_reward / num_rounds
        results[name] = avg_reward
        print(f"   Average reward: {avg_reward:.3f}")
    
    print("\n=== Comparison Results ===")
    for name, avg_reward in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {avg_reward:.3f}")


def create_simple_network():
    """Create a simple 3-node network."""
    edge_availability_probs = {
        (0, 1): 0.8,
        (1, 2): 0.7,
        (0, 2): 0.6,
    }
    
    edge_reward_probs = {
        (0, 1): 0.9,
        (1, 2): 0.8,
        (0, 2): 0.7,
    }
    
    return edge_availability_probs, edge_reward_probs


def demonstrate_edge_learning():
    """
    Demonstrate how the agent learns edge qualities over time.
    """
    print("\n=== Edge Learning Demonstration ===")
    
    # Create environment with known edge qualities
    edge_availability_probs = {
        (0, 1): 1.0,  # Always available
        (1, 2): 1.0,
        (0, 2): 1.0,
    }
    
    edge_reward_probs = {
        (0, 1): 0.9,  # High quality
        (1, 2): 0.8,  # Medium quality
        (0, 2): 0.7,  # Low quality
    }
    
    environment = RoutingEnvironment(
        num_nodes=3,
        edge_availability_probs=edge_availability_probs,
        edge_reward_probs=edge_reward_probs,
        source=0,
        target=2
    )
    
    agent = CombTSAgent(environment)
    
    print("True edge qualities:")
    for edge, prob in edge_reward_probs.items():
        print(f"  Edge {edge}: {prob}")
    
    print("\nLearning progress:")
    for round_num in [10, 50, 100, 200]:
        # Run rounds
        for _ in range(round_num):
            available_graph = environment.sample_available_graph()
            feasible_paths = environment.get_feasible_paths(available_graph)
            
            if feasible_paths:
                selected_path = agent.select_path(feasible_paths)
                reward = environment.get_reward(selected_path)
                agent.update(selected_path, reward)
        
        # Show estimates
        edge_estimates = agent.get_edge_estimates()
        print(f"\nAfter {round_num} rounds:")
        for edge, estimate in edge_estimates.items():
            true_prob = edge_reward_probs[edge]
            error = abs(estimate - true_prob)
            print(f"  Edge {edge}: Estimate = {estimate:.3f}, True = {true_prob}, Error = {error:.3f}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Run examples
    run_single_round_example()
    run_comparison_experiment()
    demonstrate_edge_learning()