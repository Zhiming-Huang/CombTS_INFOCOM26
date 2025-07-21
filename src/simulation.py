import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple, Any
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

from src.bandits.comb_ts import CombTS
from src.environments.routing_environment import RoutingEnvironment


class RoutingSimulation:
    """
    Main simulation class for network routing with combinatorial bandits.
    
    This class orchestrates the interaction between the CombTS algorithm
    and the routing environment.
    """
    
    def __init__(self, network_topology: nx.Graph, link_means: Dict[int, float],
                 availability_rates: Dict[int, float] = None, 
                 source: int = 0, destination: int = 8,
                 alpha: float = 1.0, beta: float = 1.0):
        """
        Initialize the simulation.
        
        Args:
            network_topology: NetworkX graph representing the network
            link_means: Dictionary mapping link_id to mean reward
            availability_rates: Dictionary mapping link_id to availability probability
            source: Source node for routing
            destination: Destination node for routing
            alpha: Prior parameter for CombTS
            beta: Prior parameter for CombTS
        """
        self.network_topology = network_topology
        self.source = source
        self.destination = destination
        self.alpha = alpha
        self.beta = beta
        
        # Initialize environment and algorithm
        self.environment = RoutingEnvironment(network_topology, link_means, availability_rates)
        self.algorithm = CombTS(self.environment.num_links, alpha, beta)
        
        # Simulation tracking
        self.round = 0
        
        # Get the optimal path based on expected rewards (not shortest path)
        from src.utils.network_utils import get_optimal_path_info
        optimal_info = get_optimal_path_info()
        optimal_path = optimal_info['optimal_path']
        
        self.history = {
            'rounds': [],
            'selected_paths': [],
            'rewards': [],
            'available_links': [],
            'regrets': [],
            'optimal_path': optimal_path
        }
        
        # Calculate optimal expected reward
        self.optimal_expected_reward = optimal_info['expected_reward']
    
    def run_round(self) -> Dict[str, Any]:
        """
        Run one round of the simulation.
        
        Returns:
            Dictionary containing round results
        """
        self.round += 1
        
        # Step 1: Sample available links
        available_links = self.environment.sample_available_links()
        
        # Step 2: Get feasible paths
        feasible_paths = self.environment.get_feasible_paths(
            self.source, self.destination, available_links
        )
        
        # Step 3: Select path using CombTS
        selected_path = self.algorithm.select_combination(available_links, feasible_paths)
        
        # Step 4: Generate rewards
        rewards = {}
        total_reward = 0
        if selected_path:
            rewards = self.environment.generate_path_reward(selected_path)
            total_reward = sum(rewards.values())
        
        # Step 5: Update algorithm
        self.algorithm.update_posterior(selected_path, rewards)
        
        # Step 6: Calculate regret
        regret = self.optimal_expected_reward - total_reward
        
        # Step 7: Record history
        round_data = {
            'round': self.round,
            'available_links': available_links,
            'feasible_paths': feasible_paths,
            'selected_path': selected_path,
            'rewards': rewards,
            'total_reward': total_reward,
            'regret': regret
        }
        
        self.history['rounds'].append(round_data)
        self.history['selected_paths'].append(selected_path)
        self.history['rewards'].append(total_reward)
        self.history['available_links'].append(available_links)
        self.history['regrets'].append(regret)
        
        return round_data
    
    def run_simulation(self, num_rounds: int) -> Dict[str, Any]:
        """
        Run the complete simulation for specified number of rounds.
        
        Args:
            num_rounds: Number of rounds to run
            
        Returns:
            Dictionary containing simulation results
        """
        print(f"Starting simulation for {num_rounds} rounds...")
        print(f"Source: {self.source}, Destination: {self.destination}")
        print(f"Optimal path: {self.history['optimal_path']}")
        print(f"Optimal expected reward: {self.optimal_expected_reward:.3f}")
        print("-" * 50)
        
        for round_num in range(num_rounds):
            if round_num % 100 == 0:
                print(f"Round {round_num}/{num_rounds}")
            
            self.run_round()
        
        # Calculate cumulative statistics
        cumulative_rewards = np.cumsum(self.history['rewards'])
        cumulative_regrets = np.cumsum(self.history['regrets'])
        
        results = {
            'total_rounds': num_rounds,
            'cumulative_reward': cumulative_rewards[-1],
            'cumulative_regret': cumulative_regrets[-1],
            'average_reward': np.mean(self.history['rewards']),
            'average_regret': np.mean(self.history['regrets']),
            'cumulative_rewards': cumulative_rewards.tolist(),
            'cumulative_regrets': cumulative_regrets.tolist(),
            'final_arm_statistics': self.algorithm.get_arm_statistics(),
            'network_info': self.environment.get_network_info()
        }
        
        print(f"\nSimulation completed!")
        print(f"Total cumulative reward: {results['cumulative_reward']:.3f}")
        print(f"Total cumulative regret: {results['cumulative_regret']:.3f}")
        print(f"Average reward per round: {results['average_reward']:.3f}")
        print(f"Average regret per round: {results['average_regret']:.3f}")
        
        return results
    
    def plot_results(self, results: Dict[str, Any], save_path: str = None):
        """
        Plot simulation results.
        
        Args:
            results: Results from run_simulation
            save_path: Path to save the plot (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Cumulative rewards
        axes[0, 0].plot(results['cumulative_rewards'])
        axes[0, 0].set_title('Cumulative Rewards')
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Cumulative Reward')
        axes[0, 0].grid(True)
        
        # Plot 2: Cumulative regrets
        axes[0, 1].plot(results['cumulative_regrets'])
        axes[0, 1].set_title('Cumulative Regrets')
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Cumulative Regret')
        axes[0, 1].grid(True)
        
        # Plot 3: Rewards per round
        axes[1, 0].plot(self.history['rewards'])
        axes[1, 0].set_title('Rewards per Round')
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].grid(True)
        
        # Plot 4: Expected rewards for each arm
        arm_stats = results['final_arm_statistics']
        if 'expected_rewards' in arm_stats and arm_stats['expected_rewards']:
            arms = list(arm_stats['expected_rewards'].keys())
            expected_rewards = list(arm_stats['expected_rewards'].values())
            axes[1, 1].bar(arms, expected_rewards)
            axes[1, 1].set_title('Expected Rewards per Arm')
            axes[1, 1].set_xlabel('Arm ID')
            axes[1, 1].set_ylabel('Expected Reward')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "output"):
        """
        Save simulation results to files.
        
        Args:
            results: Results from run_simulation
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a serializable copy of results
        serializable_results = {}
        for key, value in results.items():
            if key == 'network_info':
                # Convert network_info to serializable format
                network_info = {}
                for k, v in value.items():
                    if k == 'edge_to_link_id':
                        network_info[k] = {str(kk): vv for kk, vv in v.items()}
                    elif k == 'link_id_to_edge':
                        network_info[k] = {str(kk): list(vv) for kk, vv in v.items()}
                    else:
                        network_info[k] = v
                serializable_results[key] = network_info
            else:
                serializable_results[key] = value
        
        # Save results as JSON
        results_file = os.path.join(output_dir, f"simulation_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        # Save plot
        plot_file = os.path.join(output_dir, f"simulation_plot_{timestamp}.png")
        self.plot_results(results, plot_file)
        
        print(f"Results saved to {output_dir}/")
        print(f"  - Results: {results_file}")
        print(f"  - Plot: {plot_file}")
    
    def get_network_visualization(self, save_path: str = None):
        """
        Create a visualization of the network topology.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(10, 8))
        
        pos = nx.spring_layout(self.network_topology)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.network_topology, pos, node_color='lightblue', 
                             node_size=500)
        nx.draw_networkx_labels(self.network_topology, pos)
        
        # Draw edges with labels
        edge_labels = {}
        for (u, v) in self.network_topology.edges():
            link_id = self.environment.edge_to_link_id[(u, v)]
            mean_reward = self.environment.link_means[link_id]
            edge_labels[(u, v)] = f"L{link_id}({mean_reward:.1f})"
        
        nx.draw_networkx_edges(self.network_topology, pos)
        nx.draw_networkx_edge_labels(self.network_topology, pos, edge_labels)
        
        plt.title("Network Topology with Link IDs and Mean Rewards")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Network visualization saved to {save_path}")
        
        plt.show() 