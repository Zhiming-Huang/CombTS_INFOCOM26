#!/usr/bin/env python3
"""
Example: Routing Environment with 4x4 Mesh Network (Memory-Mapped)
================================================================

This example demonstrates routing with a 4x4 mesh network topology using memory mapping
for efficient data storage. It tests all algorithms including multiple gamma values
for CTS-G and CL-SG algorithms.

Features:
- 4x4 mesh network topology modeling using NetworkX
- Memory-mapped data storage for efficient large-scale simulations
- Multiple gamma values for CTS-G and CL-SG algorithms
- All algorithm comparison (CTSB, CombUCB, CTS-G, CL-SG, BG-CTS)
- Path finding and optimization
- Network visualization

Usage:
    python example_routing_4x4_memmap.py
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
from src.utils.plotting import plot_all_routing_results


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


def setup_routing_environment_4x4(num_rounds: int = 10000):
    """
    Setup the routing environment with 4x4 mesh network.
    Uses the same parameters as test_routing_environment_4x4.py
    
    Args:
        num_rounds: Number of rounds for the environment
        
    Returns:
        RoutingEnvironment instance
    """
    # Create 4x4 mesh network
    graph = create_4x4_mesh_network()
    
    # Define source and destination
    source = 0  # Top-left corner
    destination = 15  # Bottom-right corner
    
    # For wireless mesh networks, typical availability rates are:
    # - Good conditions: 0.85-0.95
    # - Moderate conditions: 0.70-0.85
    # - Poor conditions: 0.50-0.70
    # We'll use 0.75 as a realistic value for moderate wireless conditions
    link_availability_rates = {}
    for edge in graph.edges():
        link_availability_rates[edge] = 0.75
    
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
        num_rounds=num_rounds,
        pre_generate_availability=True,
        pre_generate_rewards=True,
        rng=np.random.default_rng(42)
    )
    
    return env


def run_memory_mapped_routing_simulation(num_rounds: int = 10000, num_runs: int = 5, 
                                        progress_level: str = "normal", 
                                        main_seed: int = 42,
                                        gamma_values: List[float] = [0.01, 0.1, 0.5, 1.0]) -> Dict[str, Any]:
    """
    Run memory-mapped routing simulation with all algorithms including multiple gamma values.
    
    Args:
        num_rounds: Number of rounds per simulation
        num_runs: Number of independent runs
        progress_level: Progress tracking level ("minimal", "normal", "detailed")
        main_seed: Seed for the main random number generator
        gamma_values: List of gamma values for CTS-G and CL-SG algorithms
        
    Returns:
        Dictionary containing aggregated results with confidence intervals
    """
    print(f"Running {num_runs} simulations with {num_rounds} rounds each...")
    print(f"Gamma values for CTS-G and CL-SG: {gamma_values}")
    print(f"Progress level: {progress_level}")
    
    # Create output data directory for memory-mapped files
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_data_dir = os.path.join(project_root, 'output', 'data')
    os.makedirs(output_data_dir, exist_ok=True)
    
    # Define all algorithms to test
    base_algorithms = ["CTSB", "CombUCB", "BG-CTS"]
    gamma_algorithms = ["CTS-G", "CL-SG"]
    
    # Create memory-mapped files for each algorithm
    algorithm_files = {}
    algorithm_mmaps = {}
    
    # Base algorithms (no gamma parameter)
    for alg in base_algorithms:
        regrets_file = os.path.join(output_data_dir, f"{alg.lower()}_regrets.dat")
        rewards_file = os.path.join(output_data_dir, f"{alg.lower()}_rewards.dat")
        
        regrets_mmap = np.memmap(regrets_file, dtype=np.float64, mode='w+', 
                                 shape=(num_runs, num_rounds))
        rewards_mmap = np.memmap(rewards_file, dtype=np.float64, mode='w+', 
                                 shape=(num_runs, num_rounds))
        
        algorithm_files[alg] = (regrets_file, rewards_file)
        algorithm_mmaps[alg] = (regrets_mmap, rewards_mmap)
    
    # Gamma algorithms (multiple gamma values)
    for alg in gamma_algorithms:
        for gamma in gamma_values:
            alg_key = f"{alg.lower()}_gamma_{gamma}"
            regrets_file = os.path.join(output_data_dir, f"{alg_key}_regrets.dat")
            rewards_file = os.path.join(output_data_dir, f"{alg_key}_rewards.dat")
            
            regrets_mmap = np.memmap(regrets_file, dtype=np.float64, mode='w+', 
                                     shape=(num_runs, num_rounds))
            rewards_mmap = np.memmap(rewards_file, dtype=np.float64, mode='w+', 
                                     shape=(num_runs, num_rounds))
            
            algorithm_files[alg_key] = (regrets_file, rewards_file)
            algorithm_mmaps[alg_key] = (regrets_mmap, rewards_mmap)
    
    total_algorithms = len(base_algorithms) + len(gamma_algorithms) * len(gamma_values)
    print(f"Created memory-mapped arrays: {num_runs} runs × {num_rounds} rounds × {total_algorithms} algorithms")
    print(f"Storage location: {output_data_dir}")
    
    memory_usage = algorithm_mmaps[base_algorithms[0]][0].nbytes / (1024 * 1024)
    print(f"Memory usage: {memory_usage:.2f} MB per array")
    
    # Configure progress bars based on level
    show_run_progress = progress_level in ["normal", "detailed"]
    show_round_progress = progress_level == "detailed"
    
    # Create progress bar for runs
    if show_run_progress:
        run_pbar = tqdm(range(num_runs), desc="Simulation runs", unit="run")
    else:
        run_pbar = range(num_runs)
    
    # Setup a main random generator for fairness
    main_rng = np.random.default_rng(main_seed)
    total_alg_instances = len(base_algorithms) + len(gamma_algorithms) * len(gamma_values)
    child_rngs = main_rng.spawn(1 + total_alg_instances)
    env_rng = child_rngs[0]
    
    # Assign RNGs to algorithms
    rng_dict = {}
    rng_index = 1
    
    # Base algorithms
    for alg in base_algorithms:
        rng_dict[alg] = child_rngs[rng_index]
        rng_index += 1
    
    # Gamma algorithms
    for alg in gamma_algorithms:
        for gamma in gamma_values:
            alg_key = f"{alg}_gamma_{gamma}"
            rng_dict[alg_key] = child_rngs[rng_index]
            rng_index += 1
    
    # Create environment once for all runs
    print("Creating routing environment...")
    graph = create_4x4_mesh_network()
    optimal_edges = [(0, 1), (1, 2), (2, 3), (3, 7), (7, 11), (11, 15)]
    
    env = RoutingEnvironment(
        graph=graph,
        source=0,
        destination=15,
        link_availability_rates={(edge[0], edge[1]): 0.75 for edge in graph.edges()},
        link_reward_means={(edge[0], edge[1]): 0.9 if edge in optimal_edges or (edge[1], edge[0]) in optimal_edges else 0.8 
                          for edge in graph.edges()},
        num_rounds=num_rounds,
        pre_generate_availability=True,
        pre_generate_rewards=True,
        rng=env_rng
    )
    
    for run in run_pbar:
        if show_run_progress:
            run_pbar.set_postfix({"Run": f"{run + 1}/{num_runs}"})
        
        # Create algorithm instances
        algorithm_instances = {}
        
        # Base algorithms
        algorithm_instances["CTSB"] = CTSB(environment=env, rng=rng_dict["CTSB"])
        algorithm_instances["CombUCB"] = CombUCB(environment=env, rng=rng_dict["CombUCB"])
        algorithm_instances["BG-CTS"] = BGCTS(environment=env, rnd_generator=rng_dict["BG-CTS"])
        
        # Gamma algorithms
        for alg in gamma_algorithms:
            for gamma in gamma_values:
                alg_key = f"{alg}_gamma_{gamma}"
                if alg == "CTS-G":
                    algorithm_instances[alg_key] = CTSG(environment=env, rng=rng_dict[alg_key], gamma=gamma)
                elif alg == "CL-SG":
                    algorithm_instances[alg_key] = CLSG(environment=env, rng=rng_dict[alg_key], gamma=gamma)
        
        # Run simulation for each algorithm
        for alg_name in algorithm_instances.keys():
            algorithm = algorithm_instances[alg_name]
            
            # Determine the key for memory mapping
            if alg_name in base_algorithms:
                mmap_key = alg_name
            else:
                # For gamma algorithms, the alg_name is already in the correct format
                # e.g., "CTS-G_gamma_0.1" -> "cts-g_gamma_0.1"
                mmap_key = alg_name.lower()
            
            regrets_mmap, rewards_mmap = algorithm_mmaps[mmap_key]
            
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
                
                # Store results in memory-mapped arrays
                regrets_mmap[run, round_num] = total_regret
                rewards_mmap[run, round_num] = total_reward
    
    # Calculate statistics from memory-mapped data
    results = {
        'num_rounds': num_rounds,
        'num_runs': num_runs,
        'confidence_level': 0.95,
        'algorithm_files': algorithm_files,
        'algorithm_mmaps': algorithm_mmaps,
        'gamma_values': gamma_values
    }
    
    # Calculate statistics for each algorithm
    for alg_key in algorithm_mmaps.keys():
        regrets_mmap, rewards_mmap = algorithm_mmaps[alg_key]
        
        # Calculate average and standard deviation
        avg_cumulative_regrets = np.mean(regrets_mmap, axis=0)
        std_cumulative_regrets = np.std(regrets_mmap, axis=0)
        avg_cumulative_rewards = np.mean(rewards_mmap, axis=0)
        std_cumulative_rewards = np.std(rewards_mmap, axis=0)
        
        # Calculate confidence intervals
        confidence_interval = stats.t.interval(
            results['confidence_level'], 
            num_runs - 1, 
            loc=avg_cumulative_regrets, 
            scale=std_cumulative_regrets / np.sqrt(num_runs)
        )
        
        results[alg_key] = {
            'avg_cumulative_regrets': avg_cumulative_regrets,
            'std_cumulative_regrets': std_cumulative_regrets,
            'avg_cumulative_rewards': avg_cumulative_rewards,
            'std_cumulative_rewards': std_cumulative_rewards,
            'confidence_interval': confidence_interval,
            'final_regret': avg_cumulative_regrets[-1],
            'final_regret_std': std_cumulative_regrets[-1],
            'final_regret_ci': (confidence_interval[0][-1], confidence_interval[1][-1])
        }
    
    return results


def analyze_routing_results(results: Dict[str, Any]):
    """
    Analyze routing simulation results.
    
    Args:
        results: Results from run_memory_mapped_routing_simulation
    """
    print("\nRouting Simulation Results (4x4 Mesh Network):")
    print("=" * 80)
    
    # Get base algorithms and gamma values
    base_algorithms = ["CTSB", "CombUCB", "BG-CTS"]
    gamma_values = results.get('gamma_values', [0.01, 0.1, 0.5, 1.0])
    gamma_algorithms = ["CTS-G", "CL-SG"]
    
    num_runs = results['num_runs']
    confidence_level = results['confidence_level']
    
    # Print results for base algorithms
    print("\nBase Algorithms:")
    print("-" * 40)
    for alg in base_algorithms:
        if alg in results:
            alg_data = results[alg]
            final_regret = alg_data['final_regret']
            final_regret_std = alg_data['final_regret_std']
            final_regret_ci = alg_data['final_regret_ci']
            
            print(f"{alg}:")
            print(f"  Final regret: {final_regret:.2f} ± {final_regret_std:.2f}")
            print(f"  {confidence_level*100:.0f}% CI: [{final_regret_ci[0]:.2f}, {final_regret_ci[1]:.2f}]")

    
    # Print results for gamma algorithms (using default gamma=0.1)
    print(f"\nGamma Algorithms (γ=0.1):")
    print("-" * 40)
    for alg in gamma_algorithms:
        alg_key = f"{alg.lower()}_gamma_0.1"
        if alg_key in results:
            alg_data = results[alg_key]
            final_regret = alg_data['final_regret']
            final_regret_std = alg_data['final_regret_std']
            final_regret_ci = alg_data['final_regret_ci']
            
            print(f"{alg} (γ=0.1):")
            print(f"  Final regret: {final_regret:.2f} ± {final_regret_std:.2f}")
            print(f"  {confidence_level*100:.0f}% CI: [{final_regret_ci[0]:.2f}, {final_regret_ci[1]:.2f}]")
    
    # Find best algorithm among base algorithms
    if len(base_algorithms) > 1:
        print(f"\nAlgorithm Comparison:")
        print("-" * 40)
        
        base_regrets = {}
        for alg in base_algorithms:
            if alg.lower() in results:
                base_regrets[alg] = results[alg.lower()]['final_regret']
        
        if base_regrets:
            best_alg = min(base_regrets, key=base_regrets.get)
            best_regret = base_regrets[best_alg]
            
            for alg in base_regrets:
                if alg != best_alg:
                    regret = base_regrets[alg]
                    improvement = (regret - best_regret) / regret * 100
                    print(f"{alg} vs {best_alg}: {improvement:.1f}% improvement")


def cleanup_temp_files(results: Dict[str, Any], keep_files: bool = False):
    """
    Clean up temporary memory-mapped files.
    
    Args:
        results: Results from run_memory_mapped_routing_simulation
        keep_files: Whether to keep the memory-mapped files for later use
    """
    if keep_files:
        print("Memory-mapped files preserved for later analysis")
        return
        
    algorithm_files = results.get('algorithm_files', {})
    for alg, (regrets_file, rewards_file) in algorithm_files.items():
        try:
            os.remove(regrets_file)
            os.remove(rewards_file)
        except OSError:
            pass  # File might not exist or already be removed


def main(num_rounds=10000, num_runs=5, keep_memmap_files=False):
    """
    Main function to run the 4x4 routing environment example.
    
    Args:
        num_rounds: Number of rounds per simulation
        num_runs: Number of independent runs
        keep_memmap_files: Whether to keep memory-mapped files for later analysis
    """
    print("Routing Environment Example with 4x4 Mesh Network (Memory-Mapped)")
    print("=" * 80)
    
    # Create and visualize the environment
    env = setup_routing_environment_4x4(num_rounds)
    
    print(f"Network topology: 4x4 mesh")
    print(f"Source: {env.source}, Destination: {env.destination}")
    print(f"Number of links (arms): {env.num_arms}")
    print(f"Number of nodes: {env.graph.number_of_nodes()}")
    print(f"Max combination size: {env.max_combination_size}")
    print(f"Link availability rate: 0.75 (moderate wireless conditions)")
    print(f"Optimal path reward: 0.9, Suboptimal path reward: 0.8")
    
    # Show environment info
    env_info = env.get_environment_info()
    print(f"\nEnvironment Information:")
    print(f"  Arm means: {env_info['arm_means']}")
    print(f"  Availability rates: {env_info['arm_availability_rates']}")
    
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
    gamma_values = [0.01, 0.1, 0.5, 1.0]
    results = run_memory_mapped_routing_simulation(
        num_rounds=num_rounds,
        num_runs=num_runs,
        progress_level="normal",
        main_seed=42,
        gamma_values=gamma_values
    )
    
    # Analyze results
    analyze_routing_results(results)
    
    # Plot results using utils plotting functions
    print(f"\nGenerating plots...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, 'output', 'images')
    os.makedirs(output_dir, exist_ok=True)
    
    plot_all_routing_results(results, output_dir, gamma_values)
    
    # Clean up temporary files
    cleanup_temp_files(results, keep_files=keep_memmap_files)
    
    print("\n4x4 routing environment example completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run 4x4 routing environment example with memory mapping.")
    parser.add_argument('--rounds', type=int, default=10000,
                       help='Number of rounds per simulation (default: 10000)')
    parser.add_argument('--runs', type=int, default=5,
                       help='Number of independent runs (default: 5)')
    parser.add_argument('--keep-memmap', action='store_true',
                       help='Keep memory-mapped files for later analysis')
    
    args, unknown = parser.parse_known_args()
    
    print(f"Configuration:")
    print(f"  Number of rounds: {args.rounds}")
    print(f"  Number of runs: {args.runs}")
    print(f"  Keep memory-mapped files: {args.keep_memmap}")
    print()
    
    main(num_rounds=args.rounds, num_runs=args.runs, keep_memmap_files=args.keep_memmap) 