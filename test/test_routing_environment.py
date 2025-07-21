# %%

#!/usr/bin/env python3
"""
Test script for Routing Environment with 3x3 Mesh Network.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import List, Dict, Any
from tqdm import tqdm
from scipy import stats

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.bandits.cts_b import CTSB
from src.bandits.comb_ucb import CombUCB
from src.bandits.cts_g import CTSG
from src.bandits.cl_sg import CLSG
from src.bandits.bg_cts import BGCTS
from src.environments.routing_environment import RoutingEnvironment


def create_3x3_mesh_network():
    """
    Create a 3x3 mesh network topology.
    
    Returns:
        NetworkX graph representing the 3x3 mesh
    """
    # Create 3x3 mesh network
    G = nx.grid_2d_graph(3, 3)
    
    # Convert node labels from tuples to integers for easier handling
    node_mapping = {}
    for i, node in enumerate(G.nodes()):
        node_mapping[node] = i
    
    G = nx.relabel_nodes(G, node_mapping)
    
    return G


def setup_routing_environment():
    """
    Setup the routing environment with 3x3 mesh network.
    
    Returns:
        RoutingEnvironment instance
    """
    # Create 3x3 mesh network
    graph = create_3x3_mesh_network()
    
    # Define source and destination
    source = 0  # Top-left corner
    destination = 8  # Bottom-right corner
    
    # Define link availability rates (all links have 0.8 availability)
    link_availability_rates = {}
    for edge in graph.edges():
        link_availability_rates[edge] = 0.8
    
    # Define link reward means
    # One optimal path with 0.9 reward, others with 0.8 reward
    link_reward_means = {}
    
    # Define the optimal path: 0 -> 1 -> 2 -> 5 -> 8 (horizontal then vertical)
    optimal_edges = [(0, 1), (1, 2), (2, 5), (5, 8)]
    
    for edge in graph.edges():
        if edge in optimal_edges or (edge[1], edge[0]) in optimal_edges:
            link_reward_means[edge] = 0.9  # Optimal path
        else:
            link_reward_means[edge] = 0.8  # Suboptimal paths
    
    # Create routing environment
    env = RoutingEnvironment(
        graph=graph,
        source=source,
        destination=destination,
        link_availability_rates=link_availability_rates,
        link_reward_means=link_reward_means,
        num_rounds=10000,
        pre_generate_availability=True,
        pre_generate_rewards=True,
        rng=np.random.default_rng(42)
    )
    
    return env


def run_routing_simulation(num_rounds: int = 10000, num_runs: int = 5, 
                          progress_level: str = "normal", 
                          algorithms: List[str] = ["CTSB"],
                          main_seed: int = 42) -> Dict[str, Any]:
    """
    Run routing simulation with multiple algorithms.
    
    Args:
        num_rounds: Number of rounds per simulation
        num_runs: Number of independent runs
        progress_level: Progress tracking level
        algorithms: List of algorithms to test
        main_seed: Seed for the main random number generator
        
    Returns:
        Dictionary containing aggregated results
    """
    print(f"Running {num_runs} simulations with {num_rounds} rounds each...")
    print(f"Algorithms: {', '.join(algorithms)}")
    print(f"Progress level: {progress_level}")
    
    # Configure progress bars
    show_run_progress = progress_level in ["normal", "detailed"]
    show_round_progress = progress_level == "detailed"
    
    # Create progress bar for runs
    if show_run_progress:
        run_pbar = tqdm(range(num_runs), desc="Simulation runs", unit="run")
    else:
        run_pbar = range(num_runs)
    
    # Setup random generators
    main_rng = np.random.default_rng(main_seed)
    child_rngs = main_rng.spawn(1 + len(algorithms))
    env_rng = child_rngs[0]
    rng_dict = {alg: child_rngs[i+1] for i, alg in enumerate(algorithms)}
    
    # Initialize results storage
    results = {
        'num_rounds': num_rounds,
        'num_runs': num_runs,
        'algorithms': algorithms,
        'confidence_level': 0.95
    }
    
    # Storage for each algorithm
    for alg in algorithms:
        results[alg.lower()] = {
            'cumulative_regrets': np.zeros((num_runs, num_rounds)),
            'cumulative_rewards': np.zeros((num_runs, num_rounds)),
            'path_lengths': np.zeros((num_runs, num_rounds)),
            'path_success_rates': np.zeros((num_runs, num_rounds))
        }
    
    for run in run_pbar:
        if show_run_progress:
            run_pbar.set_postfix({"Run": f"{run + 1}/{num_runs}"})
        
        # Create environment for this run
        env = RoutingEnvironment(
            graph=create_3x3_mesh_network(),
            source=0,
            destination=8,
            link_availability_rates={(edge[0], edge[1]): 0.8 for edge in create_3x3_mesh_network().edges()},
            link_reward_means={(edge[0], edge[1]): 0.9 if edge in [(0, 1), (1, 2), (2, 5), (5, 8)] else 0.8 
                              for edge in create_3x3_mesh_network().edges()},
            num_rounds=num_rounds,
            pre_generate_availability=True,
            pre_generate_rewards=True,
            rng=env_rng
        )
        
        # Create algorithm instances
        algorithm_instances = {}
        if "CTSB" in algorithms:
            algorithm_instances["CTSB"] = CTSB(environment=env, rng=rng_dict["CTSB"])
        if "CombUCB" in algorithms:
            algorithm_instances["CombUCB"] = CombUCB(environment=env, rng=rng_dict["CombUCB"])
        if "CTS-G" in algorithms:
            algorithm_instances["CTS-G"] = CTSG(environment=env, rng=rng_dict["CTS-G"], gamma=0.1)
        if "CL-SG" in algorithms:
            algorithm_instances["CL-SG"] = CLSG(environment=env, rng=rng_dict["CL-SG"], gamma=0.1)
        if "BG-CTS" in algorithms:
            algorithm_instances["BG-CTS"] = BGCTS(environment=env, rnd_generator=rng_dict["BG-CTS"])
        
        # Run simulation for each algorithm
        for alg_name in algorithms:
            algorithm = algorithm_instances[alg_name]
            
            total_reward = 0
            total_regret = 0
            
            # Create progress bar for rounds
            if show_round_progress:
                round_pbar = tqdm(range(num_rounds), desc=f"Run {run + 1} {alg_name} rounds", 
                                 unit="round", leave=False)
            else:
                round_pbar = range(num_rounds)
            
            for round_num in round_pbar:
                # Select path
                selected_combination = algorithm.select_combination(round_num)
                rewards = {}
                total_round_reward = 0
                
                if selected_combination:
                    for arm in selected_combination:
                        rewards[arm] = env.get_reward_for_round(arm, round_num)
                    total_round_reward = sum(rewards.values())
                
                # Update algorithm
                algorithm.update_posterior(selected_combination, rewards, round_num)
                
                # Calculate regret
                available_arms = env.get_available_arms_for_round(round_num)
                optimal_combination = env.get_optimal_combination(available_arms)
                optimal_expected_reward = sum(env.arm_means[arm] for arm in optimal_combination) if optimal_combination else 0
                selected_expected_reward = sum(env.arm_means[arm] for arm in selected_combination) if selected_combination else 0
                regret = optimal_expected_reward - selected_expected_reward
                
                total_reward += total_round_reward
                total_regret += regret
                
                # Store results
                results[alg_name.lower()]['cumulative_regrets'][run, round_num] = total_regret
                results[alg_name.lower()]['cumulative_rewards'][run, round_num] = total_reward
                results[alg_name.lower()]['path_lengths'][run, round_num] = len(selected_combination)
                results[alg_name.lower()]['path_success_rates'][run, round_num] = 1.0 if selected_combination else 0.0
    
    # Calculate statistics
    for alg_name in algorithms:
        alg_data = results[alg_name.lower()]
        
        # Calculate averages and standard deviations
        avg_regrets = np.mean(alg_data['cumulative_regrets'], axis=0)
        std_regrets = np.std(alg_data['cumulative_regrets'], axis=0)
        
        # Calculate confidence intervals
        t_critical = stats.t.ppf((1 + 0.95) / 2, num_runs - 1)
        confidence_intervals = t_critical * std_regrets / np.sqrt(num_runs)
        
        # Update results
        results[alg_name.lower()].update({
            'avg_cumulative_regrets': avg_regrets,
            'std_cumulative_regrets': std_regrets,
            'confidence_intervals': confidence_intervals,
            'final_avg_regret': avg_regrets[-1],
            'final_std_regret': std_regrets[-1],
            'avg_path_length': np.mean(alg_data['path_lengths']),
            'avg_success_rate': np.mean(alg_data['path_success_rates'])
        })
    
    return results


def analyze_routing_results(results: Dict[str, Any]):
    """
    Analyze routing simulation results.
    
    Args:
        results: Results from run_routing_simulation
    """
    algorithms = results['algorithms']
    
    print(f"\nRouting Simulation Results:")
    print("=" * 60)
    
    for alg in algorithms:
        alg_data = results[alg.lower()]
        print(f"\n{alg}:")
        print(f"  Final regret: {alg_data['final_avg_regret']:.2f} ± {alg_data['final_std_regret']:.2f}")
        print(f"  Average path length: {alg_data['avg_path_length']:.2f}")
        print(f"  Success rate: {alg_data['avg_success_rate']:.3f}")
    
    # Compare algorithms
    if len(algorithms) > 1:
        print(f"\nAlgorithm Comparison:")
        print("-" * 40)
        
        final_regrets = {alg: results[alg.lower()]['final_avg_regret'] for alg in algorithms}
        best_alg = min(final_regrets, key=final_regrets.get)
        worst_alg = max(final_regrets, key=final_regrets.get)
        
        if best_alg != worst_alg:
            improvement = ((final_regrets[worst_alg] - final_regrets[best_alg]) / final_regrets[worst_alg]) * 100
            print(f"Best algorithm: {best_alg} (improvement: {improvement:.1f}%)")


def plot_routing_results(results: Dict[str, Any]):
    """
    Plot routing simulation results.
    
    Args:
        results: Results from run_routing_simulation
    """
    # Set seaborn style
    sns.set_theme()
    sns.set_style("whitegrid")
    
    # Set figure size and parameters
    plt.figure(figsize=(4, 3))
    
    # Try to use LaTeX rendering, fallback if not available
    try:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.size'] = 20
    except:
        plt.rcParams['text.usetex'] = False
        plt.rcParams['font.size'] = 12
    
    num_rounds = results['num_rounds']
    rounds = np.arange(1, num_rounds + 1)
    algorithms = results['algorithms']
    
    # Define markers and display names
    markers = {'CTSB': 'o', 'CombUCB': 's', 'CTS-G': '^', 'CL-SG': 'D', 'BG-CTS': 'P'}
    display_names = {'CTSB': 'CTS-B', 'CombUCB': 'CombUCB', 'CTS-G': 'CTS-G', 'CL-SG': 'CL-SG', 'BG-CTS': 'BG-CTS'}
    
    # Plot cumulative regret for each algorithm
    markevery = max(1, int(num_rounds / 10))
    
    for alg in algorithms:
        alg_data = results[alg.lower()]
        avg_regrets = alg_data['avg_cumulative_regrets']
        marker = markers.get(alg, 'o')
        label = display_names.get(alg, alg)
        plt.plot(rounds, avg_regrets, label=label, marker=marker, markevery=markevery, linewidth=1.5)
    
    # Set legend and formatting
    plt.legend(fontsize=10)
    
    # Create a ScalarFormatter object for scientific notation
    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    
    # Apply formatter to axes
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().xaxis.set_major_formatter(formatter)
    
    # Set tick font sizes
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # Grid and labels
    plt.grid(True)
    plt.xlabel('t', fontsize=10)
    plt.ylabel('Regret', fontsize=10)
    
    # Save plot
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, 'output', 'images')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'routing_environment_results.pdf')
    
    plt.savefig(output_path, bbox_inches='tight', format='pdf', dpi=300)
    plt.show()
    
    print(f"Routing results plot saved to: {output_path}")


def main():
    """Main function to run the routing environment test."""
    print("Routing Environment Test with 3x3 Mesh Network")
    print("=" * 60)
    
    # Create and visualize the environment
    env = setup_routing_environment()
    
    print(f"Network topology: 3x3 mesh")
    print(f"Source: {env.source}, Destination: {env.destination}")
    print(f"Number of links (arms): {env.num_arms}")
    print(f"Number of nodes: {env.graph.number_of_nodes()}")
    print(f"Max combination size: {env.max_combination_size}")
    
    # Visualize the network
    print("\nVisualizing network topology...")
    env.visualize_network()
    
    # Test optimal path
    available_arms = set(range(env.num_arms))
    optimal_combination = env.get_optimal_combination(available_arms)
    if optimal_combination:
        path_info = env.get_path_info(optimal_combination)
        print(f"\nOptimal path: {path_info['path']}")
        print(f"Optimal path edges: {path_info['edges']}")
        print(f"Optimal path length: {path_info['length']}")
        print(f"Expected reward: {path_info['expected_reward']:.3f}")
    
    # Run simulation
    print(f"\nRunning routing simulation...")
    results = run_routing_simulation(
        num_rounds=10000,
        num_runs=5,
        progress_level="normal",
        algorithms=["CTSB", "CombUCB", "CTS-G", "CL-SG", "BG-CTS"],
        main_seed=42
    )
    
    # Analyze results
    analyze_routing_results(results)
    
    # Plot results
    plot_routing_results(results)
    
    print("\nRouting environment test completed!")


if __name__ == "__main__":
    main()