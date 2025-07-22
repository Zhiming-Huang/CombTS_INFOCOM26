#!/usr/bin/env python3
"""
Example: Real Network Environment with Memory-Mapped Data Storage
================================================================

This example demonstrates routing with real network topologies using memory mapping
for efficient data storage. It tests all algorithms on various real networks.

Supported Networks:
- Karate Club: Zachary's Karate Club network (34 nodes, 78 edges)
- Les Miserables: Character co-appearance network (77 nodes, 254 edges)
- Florentine Families: Family ties network (15 nodes, 20 edges)
- Internet AS: Simplified Internet AS-level topology (50 nodes)
- Power Grid: Simplified power grid topology (40 nodes)

Features:
- Real network topologies from empirical data
- Realistic link availability rates based on literature
- Memory-mapped data storage for efficient large-scale simulations
- Multiple gamma values for CTS-G and CL-SG algorithms
- All algorithm comparison (CTSB, CombUCB, CTS-G, CL-SG, BG-CTS)
- Path finding and optimization
- Network visualization

Usage:
    python example_real_network_memmap.py --network karate --rounds 5000 --runs 5
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
import argparse

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.bandits.cts_b import CTSB
from src.bandits.comb_ucb import CombUCB
from src.bandits.cts_g import CTSG
from src.bandits.cl_sg import CLSG
from src.bandits.bg_cts import BGCTS
from src.environments.real_network_environment import RealNetworkEnvironment
from src.utils.plotting import plot_all_routing_results


def setup_real_network_environment(network_type: str, num_rounds: int = 10000):
    """
    Setup the real network environment.
    
    Args:
        network_type: Type of real network ("karate", "les_miserables", "florentine", "internet_as", "power_grid")
        num_rounds: Number of rounds for the environment
        
    Returns:
        RealNetworkEnvironment instance
    """
    # Create real network environment
    env = RealNetworkEnvironment(
        network_type=network_type,
        num_rounds=num_rounds,
        pre_generate_availability=True,
        pre_generate_rewards=True,
        rng=np.random.default_rng(42)
    )
    
    return env


def run_memory_mapped_real_network_simulation(env: RealNetworkEnvironment, 
                                             num_runs: int = 5,
                                             gamma_values: List[float] = [0.01, 0.1, 0.5, 1.0],
                                             progress_level: str = "normal") -> Dict[str, Any]:
    """
    Run memory-mapped simulation on real network environment.
    
    Args:
        env: RealNetworkEnvironment instance
        num_runs: Number of simulation runs
        gamma_values: Gamma values for CTS-G and CL-SG algorithms
        progress_level: Progress bar level ("quiet", "normal", "verbose")
        
    Returns:
        Dictionary containing simulation results
    """
    print(f"Running {num_runs} simulations with {env.num_rounds} rounds each...")
    print(f"Gamma values for CTS-G and CL-SG: {gamma_values}")
    print(f"Progress level: {progress_level}")
    
    # Define algorithms
    base_algorithms = ["CTSB", "CombUCB", "BG-CTS"]
    gamma_algorithms = ["CTS-G", "CL-SG"]
    
    # Calculate total number of algorithm instances
    total_algorithms = len(base_algorithms) + len(gamma_algorithms) * len(gamma_values)
    print(f"Created memory-mapped arrays: {num_runs} runs × {env.num_rounds} rounds × {total_algorithms} algorithms")
    
    # Create output directory
    output_dir = os.path.join(project_root, "output", "data")
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate memory usage
    memory_per_array = num_runs * env.num_rounds * 8 / (1024 * 1024)  # 8 bytes per float64
    print(f"Storage location: {output_dir}")
    print(f"Memory usage: {memory_per_array:.2f} MB per array")
    
    # Initialize results dictionary
    results = {}
    
    # Create memory-mapped arrays for base algorithms
    for alg_name in base_algorithms:
        mmap_key = alg_name.lower()
        mmap_file = os.path.join(output_dir, f"{mmap_key}_regrets.dat")
        
        # Create memory-mapped array
        regrets_mmap = np.memmap(mmap_file, dtype=np.float64, mode='w+', 
                                shape=(num_runs, env.num_rounds))
        
        results[alg_name] = {
            'regrets': regrets_mmap,
            'file': mmap_file
        }
    
    # Create memory-mapped arrays for gamma algorithms
    for alg_name in gamma_algorithms:
        for gamma in gamma_values:
            mmap_key = f"{alg_name.lower()}_gamma_{gamma}"
            mmap_file = os.path.join(output_dir, f"{mmap_key}_regrets.dat")
            
            # Create memory-mapped array
            regrets_mmap = np.memmap(mmap_file, dtype=np.float64, mode='w+', 
                                    shape=(num_runs, env.num_rounds))
            
            results[f"{alg_name}_gamma_{gamma}"] = {
                'regrets': regrets_mmap,
                'file': mmap_file
            }
    
    # Setup progress bar
    if progress_level == "verbose":
        run_pbar = tqdm(range(num_runs), desc="Simulation runs", unit="run")
    elif progress_level == "normal":
        run_pbar = tqdm(range(num_runs), desc="Simulation runs")
    else:
        run_pbar = range(num_runs)
    
    # Create random number generators for fairness
    main_rng = np.random.default_rng(42)
    rngs = []
    
    # Spawn RNGs for each run
    for run in range(num_runs):
        rngs.append(main_rng.spawn(1)[0])
    
    print("Creating real network environment...")
    
    # Run simulations
    for run in run_pbar:
        # Create environment for this run
        run_env = RealNetworkEnvironment(
            network_type=env.network_type,
            source=env.source,
            destination=env.destination,
            link_availability_rates=env.link_availability_rates,
            link_reward_means=env.link_reward_means,
            num_rounds=env.num_rounds,
            max_combination_size=env.max_combination_size,
            pre_generate_availability=False,  # Generate on-the-fly for fairness
            pre_generate_rewards=False,
            rng=rngs[run]
        )
        
        # Initialize algorithms
        algorithms = {}
        
        # Base algorithms
        algorithms["CTSB"] = CTSB(run_env, alpha=1.0, beta=1.0)
        algorithms["CombUCB"] = CombUCB(run_env)
        algorithms["BG-CTS"] = BGCTS(run_env)
        
        # Gamma algorithms
        for alg_name in gamma_algorithms:
            for gamma in gamma_values:
                key = f"{alg_name}_gamma_{gamma}"
                if alg_name == "CTS-G":
                    algorithms[key] = CTSG(run_env, gamma=gamma)
                elif alg_name == "CL-SG":
                    algorithms[key] = CLSG(run_env, gamma=gamma)
        
        # Run simulation
        cumulative_regret = 0
        
        for round_num in range(env.num_rounds):
            # Get available arms
            available_arms = run_env.get_available_arms_for_round(round_num)
            
            # Get feasible combinations
            feasible_combinations = run_env.get_feasible_combinations(available_arms)
            
            # Get optimal combination for regret calculation
            optimal_combination = run_env.get_optimal_combination(available_arms)
            optimal_reward = 0
            if optimal_combination:
                optimal_rewards = run_env.get_reward_for_round(optimal_combination, round_num)
                optimal_reward = sum(optimal_rewards.values())
            
            # Run each algorithm
            for alg_name, algorithm in algorithms.items():
                # Select combination
                selected_combination = algorithm.select_combination(round_num)
                
                # Get rewards
                total_reward = 0
                if selected_combination:
                    rewards = run_env.get_reward_for_round(selected_combination, round_num)
                    total_reward = sum(rewards.values())
                    
                    # Update algorithm
                    algorithm.update_posterior(selected_combination, rewards, round_num)
                
                # Calculate regret
                regret = optimal_reward - total_reward
                cumulative_regret += regret
                
                # Store in memory-mapped array
                results[alg_name]['regrets'][run, round_num] = cumulative_regret
        
        # Update progress bar
        if progress_level != "quiet":
            if isinstance(run_pbar, tqdm):
                run_pbar.set_postfix({'Run': f"{run+1}/{num_runs}"})
    
    # Calculate statistics
    print("\nCalculating statistics...")
    final_results = {}
    
    for alg_name, data in results.items():
        regrets = data['regrets']
        
        # Calculate statistics
        final_regrets = regrets[:, -1]  # Last round regrets
        mean_regret = np.mean(final_regrets)
        std_regret = np.std(final_regrets)
        
        # Calculate confidence interval
        confidence_level = 0.95
        degrees_of_freedom = num_runs - 1
        t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
        margin_of_error = t_value * std_regret / np.sqrt(num_runs)
        ci_lower = mean_regret - margin_of_error
        ci_upper = mean_regret + margin_of_error
        
        final_results[alg_name] = {
            'mean_regret': mean_regret,
            'std_regret': std_regret,
            'confidence_interval': (ci_lower, ci_upper),
            'all_regrets': regrets.copy()
        }
    
    return final_results


def analyze_real_network_results(results: Dict[str, Any], network_type: str):
    """Analyze and print real network simulation results."""
    print(f"\nReal Network Simulation Results ({network_type.title()} Network):")
    print("=" * 80)
    
    # Separate base and gamma algorithms
    base_algorithms = ["CTSB", "CombUCB", "BG-CTS"]
    gamma_algorithms = ["CTS-G", "CL-SG"]
    
    print("\nBase Algorithms:")
    print("-" * 40)
    for alg in base_algorithms:
        if alg in results:
            data = results[alg]
            ci_lower, ci_upper = data['confidence_interval']
            print(f"{alg}:")
            print(f"  Final regret: {data['mean_regret']:.2f} ± {data['std_regret']:.2f}")
            print(f"  95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
    
    print(f"\nGamma Algorithms (γ=0.1):")
    print("-" * 40)
    for alg in gamma_algorithms:
        key = f"{alg}_gamma_0.1"
        if key in results:
            data = results[key]
            ci_lower, ci_upper = data['confidence_interval']
            print(f"{alg} (γ=0.1):")
            print(f"  Final regret: {data['mean_regret']:.2f} ± {data['std_regret']:.2f}")
            print(f"  95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
    
    print(f"\nAlgorithm Comparison:")
    print("-" * 40)
    
    # Find best algorithm
    best_alg = None
    best_regret = float('inf')
    
    for alg_name, data in results.items():
        if data['mean_regret'] < best_regret:
            best_regret = data['mean_regret']
            best_alg = alg_name
    
    if best_alg:
        print(f"Best performing algorithm: {best_alg} (regret: {best_regret:.2f})")


def cleanup_temp_files(results: Dict[str, Any], keep_files: bool = False):
    """Clean up temporary memory-mapped files."""
    if not keep_files:
        print("\nCleaning up temporary files...")
        for alg_name, data in results.items():
            if 'file' in data:
                try:
                    os.remove(data['file'])
                except FileNotFoundError:
                    pass
        print("Cleanup completed.")
    else:
        print("\nMemory-mapped files preserved for post-hoc analysis.")


def main():
    """Main function to run the real network simulation."""
    parser = argparse.ArgumentParser(description="Real Network Environment Simulation")
    parser.add_argument("--network", type=str, default="karate", 
                       choices=["karate", "les_miserables", "florentine", "internet_as", "power_grid"],
                       help="Network type to use")
    parser.add_argument("--rounds", type=int, default=5000,
                       help="Number of rounds per simulation")
    parser.add_argument("--runs", type=int, default=5,
                       help="Number of simulation runs")
    parser.add_argument("--keep-memmap", action="store_true",
                       help="Keep memory-mapped files for post-hoc analysis")
    parser.add_argument("--progress", type=str, default="normal",
                       choices=["quiet", "normal", "verbose"],
                       help="Progress bar level")
    
    args = parser.parse_args()
    
    print("Real Network Environment Example with Memory-Mapped Data Storage")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Network type: {args.network}")
    print(f"  Number of rounds: {args.rounds}")
    print(f"  Number of runs: {args.runs}")
    print(f"  Keep memory-mapped files: {args.keep_memmap}")
    
    # Setup environment
    env = setup_real_network_environment(args.network, args.rounds)
    
    # Get environment info
    env_info = env.get_environment_info()
    print(f"\nNetwork Information:")
    print(f"  Network type: {env_info['network_type']}")
    print(f"  Number of nodes: {env_info['num_nodes']}")
    print(f"  Number of edges: {env_info['num_edges']}")
    print(f"  Source: {env_info['source']}, Destination: {env_info['destination']}")
    print(f"  Max combination size: {env_info['max_combination_size']}")
    
    # Show network characteristics
    degrees = dict(env.graph.degree())
    avg_degree = np.mean(list(degrees.values()))
    print(f"  Average node degree: {avg_degree:.2f}")
    
    # Get optimal path info
    available_arms = set(range(env.num_arms))
    optimal_combination = env.get_optimal_combination(available_arms)
    if optimal_combination:
        path_info = env.get_path_info(optimal_combination)
        print(f"  Optimal path length: {path_info['length']}")
        print(f"  Expected optimal reward: {path_info['expected_reward']:.3f}")
    
    # Visualize network
    print(f"\nVisualizing network topology...")
    env.visualize_network()
    
    # Run simulation
    print(f"\nRunning real network simulation...")
    results = run_memory_mapped_real_network_simulation(
        env=env,
        num_runs=args.runs,
        progress_level=args.progress
    )
    
    # Analyze results
    analyze_real_network_results(results, args.network)
    
    # Generate plots
    print(f"\nGenerating plots...")
    output_dir = os.path.join(project_root, "output", "images")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create custom plotting function for real networks
    plot_real_network_results(results, output_dir, args.network)
    
    # Cleanup
    cleanup_temp_files(results, args.keep_memmap)
    
    print(f"\n{args.network.title()} real network simulation completed!")


def plot_real_network_results(results: Dict[str, Any], output_dir: str, network_type: str):
    """Generate plots for real network results."""
    from src.utils.plotting import setup_plot_style
    
    # Setup plot style
    setup_plot_style()
    
    # Main algorithm comparison (using gamma=0.1 for gamma algorithms)
    algorithms_to_plot = ["CTSB", "CombUCB", "CTS-G_gamma_0.1", "CL-SG_gamma_0.1", "BG-CTS"]
    available_algorithms = [alg for alg in algorithms_to_plot if alg in results]
    
    if len(available_algorithms) >= 2:
        plt.figure(figsize=(4, 3))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(available_algorithms)))
        rounds = np.arange(results[available_algorithms[0]]['all_regrets'].shape[1])
        
        for i, alg in enumerate(available_algorithms):
            regrets = results[alg]['all_regrets']
            mean_regrets = np.mean(regrets, axis=0)
            std_regrets = np.std(regrets, axis=0)
            
            # Calculate confidence intervals
            confidence_level = 0.95
            degrees_of_freedom = regrets.shape[0] - 1
            t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
            margin_of_error = t_value * std_regrets / np.sqrt(regrets.shape[0])
            ci_lower = mean_regrets - margin_of_error
            ci_upper = mean_regrets + margin_of_error
            
            # Plot with confidence intervals
            plt.plot(rounds, mean_regrets, label=alg, color=colors[i], 
                    linewidth=1.5, marker='o', markevery=len(rounds)//10)
            plt.fill_between(rounds, ci_lower, ci_upper, alpha=0.1, color=colors[i])
        
        plt.xlabel('t')
        plt.ylabel('Regret')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        output_path = os.path.join(output_dir, f"{network_type}_algorithm_comparison.pdf")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, format='pdf', dpi=300)
        plt.close()
        print(f"Algorithm comparison plot saved to: {output_path}")
    
    # Gamma comparison plots
    gamma_values = [0.01, 0.1, 0.5, 1.0]
    
    for alg_name in ["CTS-G", "CL-SG"]:
        available_gammas = []
        for gamma in gamma_values:
            key = f"{alg_name}_gamma_{gamma}"
            if key in results:
                available_gammas.append(gamma)
        
        if len(available_gammas) >= 2:
            plt.figure(figsize=(4, 3))
            
            colors = plt.cm.Set1(np.linspace(0, 1, len(available_gammas)))
            rounds = np.arange(results[f"{alg_name}_gamma_{available_gammas[0]}"]['all_regrets'].shape[1])
            
            for i, gamma in enumerate(available_gammas):
                key = f"{alg_name}_gamma_{gamma}"
                regrets = results[key]['all_regrets']
                mean_regrets = np.mean(regrets, axis=0)
                std_regrets = np.std(regrets, axis=0)
                
                # Calculate confidence intervals
                confidence_level = 0.95
                degrees_of_freedom = regrets.shape[0] - 1
                t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
                margin_of_error = t_value * std_regrets / np.sqrt(regrets.shape[0])
                ci_lower = mean_regrets - margin_of_error
                ci_upper = mean_regrets + margin_of_error
                
                # Plot with confidence intervals
                label = f"{alg_name} ($\\gamma={gamma}$)"
                plt.plot(rounds, mean_regrets, label=label, color=colors[i], 
                        linewidth=1.5, marker='o', markevery=len(rounds)//10)
                plt.fill_between(rounds, ci_lower, ci_upper, alpha=0.1, color=colors[i])
            
            plt.xlabel('t')
            plt.ylabel('Regret')
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)
            
            output_path = os.path.join(output_dir, f"{network_type}_{alg_name.lower()}_gamma_comparison.pdf")
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, format='pdf', dpi=300)
            plt.close()
            print(f"{alg_name} gamma comparison plot saved to: {output_path}")


if __name__ == "__main__":
    main() 