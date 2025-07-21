#!/usr/bin/env python3
"""
Example: Routing Environment with 4x4 Mesh Network (High Availability)
====================================================================

This example demonstrates how to use the RoutingEnvironment for network routing problems
with a 4x4 mesh network topology. All links have high availability (0.9) representing
good wireless network conditions.

Features:
- 4x4 mesh network topology modeling using NetworkX
- High link availability (0.9) for good network conditions
- Link availability and reward modeling
- Path finding and optimization
- Multiple algorithm comparison
- Network visualization

Usage:
    python example_routing_environment_4x4.py
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


def create_4x4_mesh_network():
    """
    Create a 4x4 mesh network topology.
    
    Returns:
        NetworkX graph representing the 4x4 mesh
    """
    # Create 4x4 mesh network
    G = nx.grid_2d_graph(4, 4)
    
    # Convert node labels from tuples to integers for easier handling
    node_mapping = {}
    for i, node in enumerate(G.nodes()):
        node_mapping[node] = i
    
    G = nx.relabel_nodes(G, node_mapping)
    
    return G


def setup_routing_environment_4x4(link_availability_rate=0.9):
    """
    Setup the routing environment with 4x4 mesh network.
    
    Args:
        link_availability_rate: Availability rate for all links (default: 0.9)
    
    Returns:
        RoutingEnvironment instance
    """
    # Create 4x4 mesh network
    graph = create_4x4_mesh_network()
    
    # Define source and destination
    source = 0  # Top-left corner
    destination = 15  # Bottom-right corner
    
    # Set availability rate for all links
    link_availability_rates = {}
    for edge in graph.edges():
        link_availability_rates[edge] = link_availability_rate
    
    # Define link reward means
    # One optimal path with 0.9 reward, others with 0.8 reward
    link_reward_means = {}
    
    # Define the optimal path: 0 -> 1 -> 2 -> 3 -> 7 -> 11 -> 15 (horizontal then vertical)
    # This creates a path that goes right across the top, then down the right side
    optimal_edges = [(0, 1), (1, 2), (2, 3), (3, 7), (7, 11), (11, 15)]
    
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


def run_routing_simulation_4x4(num_rounds: int = 10000, num_runs: int = 5, 
                              progress_level: str = "normal", 
                              algorithms: List[str] = ["CTSB"],
                              main_seed: int = 42,
                              link_availability_rate: float = 0.9,
                              gamma_params: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Run routing simulation with multiple algorithms on 4x4 mesh.
    
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
    
    # Set default gamma parameters if not provided
    if gamma_params is None:
        gamma_params = {'CTS-G': 0.1, 'CL-SG': 0.1}
    
    # Initialize results storage
    results = {
        'num_rounds': num_rounds,
        'num_runs': num_runs,
        'algorithms': algorithms,
        'confidence_level': 0.95,
        'gamma_params': gamma_params
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
        graph = create_4x4_mesh_network()
        optimal_edges = [(0, 1), (1, 2), (2, 3), (3, 7), (7, 11), (11, 15)]
        
        env = RoutingEnvironment(
            graph=graph,
            source=0,
            destination=15,
            link_availability_rates={(edge[0], edge[1]): link_availability_rate for edge in graph.edges()},
            link_reward_means={(edge[0], edge[1]): 0.9 if edge in optimal_edges or (edge[1], edge[0]) in optimal_edges else 0.8 
                              for edge in graph.edges()},
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
            gamma_ctsg = gamma_params.get('CTS-G', 0.1)
            algorithm_instances["CTS-G"] = CTSG(environment=env, rng=rng_dict["CTS-G"], gamma=gamma_ctsg)
        if "CL-SG" in algorithms:
            gamma_clsg = gamma_params.get('CL-SG', 0.1)
            algorithm_instances["CL-SG"] = CLSG(environment=env, rng=rng_dict["CL-SG"], gamma=gamma_clsg)
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


def analyze_routing_results_4x4(results: Dict[str, Any]):
    """
    Analyze routing simulation results for 4x4 mesh.
    
    Args:
        results: Results from run_routing_simulation_4x4
    """
    algorithms = results['algorithms']
    
    print(f"\nRouting Simulation Results (4x4 Mesh Network - High Availability):")
    print("=" * 80)
    
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


def plot_routing_results_4x4(results: Dict[str, Any], link_availability_rate: float = 0.9, 
                            plot_options: Dict[str, Any] = None):
    """
    Plot routing simulation results for 4x4 mesh.
    
    Args:
        results: Results from run_routing_simulation_4x4
    """
    # Set default plot options if not provided
    if plot_options is None:
        plot_options = {
            'figsize': (4, 3),
            'use_latex': True,
            'font_size': 20,
            'tick_font_size': 10,
            'legend_font_size': 10,
            'line_width': 1.5,
            'marker_size': 6,
            'grid': True,
            'show_plot': True,
            'save_plot': True,
            'dpi': 300,
            'format': 'pdf'
        }
    
    # Set seaborn style
    sns.set_theme()
    sns.set_style("whitegrid")
    
    # Set figure size and parameters
    plt.figure(figsize=plot_options['figsize'])
    
    # Try to use LaTeX rendering, fallback if not available
    if plot_options['use_latex']:
        try:
            plt.rcParams['text.usetex'] = True
            plt.rcParams['font.size'] = plot_options['font_size']
        except:
            plt.rcParams['text.usetex'] = False
            plt.rcParams['font.size'] = 12
    else:
        plt.rcParams['text.usetex'] = False
        plt.rcParams['font.size'] = plot_options['font_size']
    
    num_rounds = results['num_rounds']
    rounds = np.arange(1, num_rounds + 1)
    algorithms = results['algorithms']
    
    # Define markers and display names
    markers = {'CTSB': 'o', 'CombUCB': 's', 'CTS-G': '^', 'CL-SG': 'D', 'BG-CTS': 'P'}
    display_names = {'CTSB': 'CTS-B', 'CombUCB': 'CombUCB', 'CTS-G': 'CTS-G', 'CL-SG': 'CL-SG', 'BG-CTS': 'BG-CTS'}
    
    # Add gamma information to algorithm names if available
    gamma_params = results.get('gamma_params', {})
    for alg in algorithms:
        if alg in gamma_params:
            gamma_val = gamma_params[alg]
            display_names[alg] = f"{display_names[alg]} (gamma={gamma_val})"
    
    # Plot cumulative regret for each algorithm
    markevery = max(1, int(num_rounds / 10))
    
    for alg in algorithms:
        alg_data = results[alg.lower()]
        avg_regrets = alg_data['avg_cumulative_regrets']
        marker = markers.get(alg, 'o')
        label = display_names.get(alg, alg)
        plt.plot(rounds, avg_regrets, label=label, marker=marker, markevery=markevery, 
                linewidth=plot_options['line_width'], markersize=plot_options['marker_size'])
    
    # Set legend and formatting
    plt.legend(fontsize=plot_options['legend_font_size'])
    
    # Create a ScalarFormatter object for scientific notation
    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    
    # Apply formatter to axes
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().xaxis.set_major_formatter(formatter)
    
    # Set tick font sizes
    plt.xticks(fontsize=plot_options['tick_font_size'])
    plt.yticks(fontsize=plot_options['tick_font_size'])
    
    # Grid and labels
    if plot_options['grid']:
        plt.grid(True)
    plt.xlabel('t', fontsize=plot_options['tick_font_size'])
    plt.ylabel('Regret', fontsize=plot_options['tick_font_size'])
    
    # Save plot
    if plot_options['save_plot']:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, 'output', 'images')
        os.makedirs(output_dir, exist_ok=True)
        
        # Update filename to include availability rate and gamma info
        availability_str = str(link_availability_rate).replace('.', '_')
        gamma_str = ""
        if gamma_params:
            gamma_vals = [f"{k}_{v}".replace('.', '_') for k, v in gamma_params.items()]
            gamma_str = "_" + "_".join(gamma_vals)
        
        output_path = os.path.join(output_dir, f'routing_environment_4x4_availability_{availability_str}{gamma_str}.{plot_options["format"]}')
        
        plt.savefig(output_path, bbox_inches='tight', format=plot_options['format'], dpi=plot_options['dpi'])
        print(f"4x4 routing results plot saved to: {output_path}")
    
    if plot_options['show_plot']:
        plt.show()


def main(link_availability_rate=0.9, gamma_params=None, plot_options=None, show_network=True):
    """
    Main function to run the 4x4 routing environment example.
    
    Args:
        link_availability_rate: Availability rate for all links (default: 0.9)
    """
    print("Routing Environment Example with 4x4 Mesh Network")
    print("=" * 80)
    
    # Create and visualize the environment
    env = setup_routing_environment_4x4(link_availability_rate)
    
    if show_network:
        print(f"Network topology: 4x4 mesh")
        print(f"Source: {env.source}, Destination: {env.destination}")
        print(f"Number of links (arms): {env.num_arms}")
        print(f"Number of nodes: {env.graph.number_of_nodes()}")
        print(f"Max combination size: {env.max_combination_size}")
        print(f"Link availability rate: {link_availability_rate} ({'high' if link_availability_rate >= 0.8 else 'medium' if link_availability_rate >= 0.6 else 'low'} availability)")
        print(f"Optimal path reward: 0.9, Suboptimal path reward: 0.8")
        
        # Show environment info
        env_info = env.get_environment_info()
        print(f"\nEnvironment Information:")
        print(f"  Arm means: {env_info['arm_means']}")
        print(f"  Availability rates: {env_info['arm_availability_rates']}")
        
        # Visualize the network
        print("\nVisualizing network topology...")
        env.visualize_network()
    
    # Test optimal path
    if show_network:
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
    results = run_routing_simulation_4x4(
        num_rounds=10000,
        num_runs=5,
        progress_level="normal",
        algorithms=["CTSB", "CombUCB", "CTS-G", "CL-SG", "BG-CTS"],
        main_seed=42,
        link_availability_rate=link_availability_rate,
        gamma_params=gamma_params
    )
    
    # Analyze results
    analyze_routing_results_4x4(results)
    
    # Plot results
    plot_routing_results_4x4(results, link_availability_rate, plot_options)
    
    print("\n4x4 routing environment example completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run 4x4 routing environment example with configurable parameters.")
    parser.add_argument('--availability', type=float, default=0.9, 
                       help='Link availability rate (default: 0.9)')
    parser.add_argument('--gamma-ctsg', type=float, default=0.1,
                       help='Gamma parameter for CTS-G algorithm (default: 0.1)')
    parser.add_argument('--gamma-clsg', type=float, default=0.1,
                       help='Gamma parameter for CL-SG algorithm (default: 0.1)')
    parser.add_argument('--rounds', type=int, default=10000,
                       help='Number of rounds per simulation (default: 10000)')
    parser.add_argument('--runs', type=int, default=5,
                       help='Number of independent runs (default: 5)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plot display')
    parser.add_argument('--no-save', action='store_true',
                       help='Disable plot saving')
    parser.add_argument('--format', type=str, default='pdf',
                       choices=['pdf', 'png', 'jpg'],
                       help='Output format (default: pdf)')
    parser.add_argument('--no-network', action='store_true',
                       help='Disable network topology display')
    
    args, unknown = parser.parse_known_args()
    
    # Validate availability rate
    if not 0.0 <= args.availability <= 1.0:
        print("Error: Availability rate must be between 0.0 and 1.0")
        exit(1)
    
    # Validate gamma parameters
    if args.gamma_ctsg <= 0 or args.gamma_clsg <= 0:
        print("Error: Gamma parameters must be positive")
        exit(1)
    
    # Setup gamma parameters
    gamma_params = {
        'CTS-G': args.gamma_ctsg,
        'CL-SG': args.gamma_clsg
    }
    
    # Setup plot options
    plot_options = {
        'figsize': (4, 3),
        'use_latex': True,
        'font_size': 20,
        'tick_font_size': 10,
        'legend_font_size': 10,
        'line_width': 1.5,
        'marker_size': 6,
        'grid': True,
        'show_plot': not args.no_plot,
        'save_plot': not args.no_save,
        'dpi': 300,
        'format': args.format
    }
    
    print(f"Configuration:")
    print(f"  Link availability rate: {args.availability}")
    print(f"  CTS-G gamma: {args.gamma_ctsg}")
    print(f"  CL-SG gamma: {args.gamma_clsg}")
    print(f"  Number of rounds: {args.rounds}")
    print(f"  Number of runs: {args.runs}")
    print(f"  Output format: {args.format}")
    print(f"  Show plot: {plot_options['show_plot']}")
    print(f"  Save plot: {plot_options['save_plot']}")
    print()
    
    main(link_availability_rate=args.availability, 
         gamma_params=gamma_params, 
         plot_options=plot_options) 