#!/usr/bin/env python3
"""
Example: Qurinet Real Wireless Mesh Network Environment with Memory-Mapped Data Storage
====================================================================================

This example demonstrates routing with real wireless mesh network topology and link
quality data from the Qurinet network deployed at Quail Ridge Natural Reserve.

Data Source: https://github.com/cjpatton/qr
Network: 20 nodes with dual adhoc interfaces
Frequency: 2.4GHz (channels 1, 6, 11)

Features:
- Real wireless mesh network topology from Qurinet deployment
- Realistic link availability rates based on signal strength and quality
- Memory-mapped data storage for efficient large-scale simulations
- Multiple gamma values for CTS-G and CL-SG algorithms
- All algorithm comparison (CTSB, CombUCB, CTS-G, CL-SG, BG-CTS)
- Path finding and optimization
- Network visualization

Usage:
    python example_qurinet_memmap.py --rounds 5000 --runs 5
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
from src.environments.qurinet_environment import QurinetEnvironment
from src.utils.plotting import setup_plot_style, plot_all_routing_results


def setup_qurinet_environment(num_rounds: int = 10000):
    """
    Setup the Qurinet real wireless mesh network environment.
    
    Args:
        num_rounds: Number of rounds for the environment
        
    Returns:
        QurinetEnvironment instance
    """
    # Create Qurinet environment
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "qurinet")
    
    env = QurinetEnvironment(
        data_dir=data_dir,
        date="2-May",
        num_rounds=num_rounds,
        pre_generate_availability=True,
        pre_generate_rewards=True,
        rng=np.random.default_rng(42)
    )
    
    return env


def run_memory_mapped_qurinet_simulation(env: QurinetEnvironment, 
                                        num_runs: int = 5,
                                        gamma_values: List[float] = [0.01, 0.1, 0.5, 1.0],
                                        progress_level: str = "normal",
                                        main_seed: int = 42) -> Dict[str, Any]:
    """
    Run memory-mapped simulation on Qurinet real network environment.
    
    Args:
        env: QurinetEnvironment instance
        num_runs: Number of simulation runs
        gamma_values: Gamma values for CTS-G and CL-SG algorithms
        progress_level: Progress bar level ("quiet", "normal", "verbose")
        main_seed: Seed for the main random number generator
        
    Returns:
        Dictionary containing simulation results
    """
    print(f"Running {num_runs} simulations with {env.num_rounds} rounds each...")
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
        regrets_file = os.path.join(output_data_dir, f"qurinet_{alg.lower()}_regrets.dat")
        rewards_file = os.path.join(output_data_dir, f"qurinet_{alg.lower()}_rewards.dat")
        
        regrets_mmap = np.memmap(regrets_file, dtype=np.float64, mode='w+', 
                                 shape=(num_runs, env.num_rounds))
        rewards_mmap = np.memmap(rewards_file, dtype=np.float64, mode='w+', 
                                 shape=(num_runs, env.num_rounds))
        
        algorithm_files[alg] = (regrets_file, rewards_file)
        algorithm_mmaps[alg] = (regrets_mmap, rewards_mmap)
    
    # Gamma algorithms (multiple gamma values)
    for alg in gamma_algorithms:
        for gamma in gamma_values:
            alg_key = f"{alg.lower()}_gamma_{gamma}"
            regrets_file = os.path.join(output_data_dir, f"qurinet_{alg_key}_regrets.dat")
            rewards_file = os.path.join(output_data_dir, f"qurinet_{alg_key}_rewards.dat")
            
            regrets_mmap = np.memmap(regrets_file, dtype=np.float64, mode='w+', 
                                     shape=(num_runs, env.num_rounds))
            rewards_mmap = np.memmap(rewards_file, dtype=np.float64, mode='w+', 
                                     shape=(num_runs, env.num_rounds))
            
            algorithm_files[alg_key] = (regrets_file, rewards_file)
            algorithm_mmaps[alg_key] = (regrets_mmap, rewards_mmap)
    
    # Setup progress tracking
    show_run_progress = progress_level in ["normal", "verbose"]
    show_round_progress = progress_level == "verbose"
    
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
    print("Creating Qurinet real network environment...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "qurinet")
    
    run_env = QurinetEnvironment(
        data_dir=data_dir,
        date="2-May",
        source=env.source,
        destination=env.destination,
        num_rounds=env.num_rounds,
        max_combination_size=env.max_combination_size,
        pre_generate_availability=True,  # Always pre-generate
        pre_generate_rewards=True,
        rng=env_rng
    )
    
    for run in run_pbar:
        if show_run_progress:
            run_pbar.set_postfix({"Run": f"{run + 1}/{num_runs}"})
        
        # Create algorithm instances
        algorithm_instances = {}
        
        # Base algorithms
        algorithm_instances["CTSB"] = CTSB(environment=run_env, rng=rng_dict["CTSB"])
        algorithm_instances["CombUCB"] = CombUCB(environment=run_env, rng=rng_dict["CombUCB"])
        algorithm_instances["BG-CTS"] = BGCTS(environment=run_env, rnd_generator=rng_dict["BG-CTS"])
        
        # Gamma algorithms
        for alg in gamma_algorithms:
            for gamma in gamma_values:
                alg_key = f"{alg}_gamma_{gamma}"
                if alg == "CTS-G":
                    algorithm_instances[alg_key] = CTSG(environment=run_env, rng=rng_dict[alg_key], gamma=gamma)
                elif alg == "CL-SG":
                    algorithm_instances[alg_key] = CLSG(environment=run_env, rng=rng_dict[alg_key], gamma=gamma)
        
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
                round_pbar = tqdm(range(env.num_rounds), desc=f"Run {run + 1} {alg_name} rounds", 
                                 unit="round", leave=False)
            else:
                round_pbar = range(env.num_rounds)
            
            for round_num in round_pbar:
                # Select path
                selected_combination = algorithm.select_combination(round_num)
                rewards = {}
                total_round_reward = 0
                # Use sampled rewards for algorithm update, but mean for regret
                if selected_combination:
                    rewards = run_env.get_reward_for_round(selected_combination, round_num)
                    total_round_reward = sum(run_env.arm_means[arm_id] for arm_id in selected_combination)
                # Update algorithm
                algorithm.update_posterior(selected_combination, rewards, round_num)
                # Calculate regret (mean-based)
                available_arms = run_env.get_available_arms_for_round(round_num)
                optimal_combination = run_env.get_optimal_combination(available_arms)
                optimal_reward = 0
                if optimal_combination:
                    optimal_reward = sum(run_env.arm_means[arm_id] for arm_id in optimal_combination)
                regret = optimal_reward - total_round_reward
                total_regret += regret
                total_reward += total_round_reward
                # Store results
                regrets_mmap[run, round_num] = total_regret
                rewards_mmap[run, round_num] = total_reward
    
    # Calculate statistics from memory-mapped data
    results = {
        'num_rounds': env.num_rounds,
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


def analyze_qurinet_results(results: Dict[str, Any]):
    """
    Analyze Qurinet simulation results.
    
    Args:
        results: Results from run_memory_mapped_qurinet_simulation
    """
    print("\nQurinet Real Wireless Mesh Network Simulation Results:")
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
            
            print(f"{alg}:")
            print(f"  Final regret: {final_regret:.2f} ± {final_regret_std:.2f}")
            print(f"  {confidence_level*100:.0f}% CI: [{final_regret_ci[0]:.2f}, {final_regret_ci[1]:.2f}]")
    
    # Find best performing algorithm
    print(f"\nAlgorithm Comparison:")
    print("-" * 40)
    best_alg = None
    best_regret = float('inf')
    
    for alg_name, alg_data in results.items():
        if isinstance(alg_data, dict) and 'final_regret' in alg_data and alg_data['final_regret'] < best_regret:
            best_regret = alg_data['final_regret']
            best_alg = alg_name
    
    if best_alg:
        print(f"Best performing algorithm: {best_alg} (regret: {best_regret:.2f})")


def cleanup_temp_files(results: Dict[str, Any], keep_files: bool = False):
    """Clean up temporary memory-mapped files."""
    if not keep_files:
        print("\nCleaning up temporary files...")
        algorithm_files = results.get('algorithm_files', {})
        for alg_name, (regrets_file, rewards_file) in algorithm_files.items():
            try:
                os.remove(regrets_file)
                os.remove(rewards_file)
            except FileNotFoundError:
                pass
        print("Cleanup completed.")
    else:
        print("\nMemory-mapped files preserved for post-hoc analysis.")


def plot_qurinet_results(results: Dict[str, Any], output_dir: str):
    """Generate plots for Qurinet results using the same functions as routing."""
    gamma_values = results.get('gamma_values', [0.01, 0.1, 0.5, 1.0])
    
    # Use the same plotting function as routing, but with qurinet-specific file names
    plot_all_routing_results(results, output_dir, gamma_values)
    
    # Rename files to have qurinet prefix
    import shutil
    old_names = [
        "routing_algorithm_comparison.pdf",
        "routing_ctsg_gamma_comparison.pdf", 
        "routing_clsg_gamma_comparison.pdf"
    ]
    new_names = [
        "qurinet_algorithm_comparison.pdf",
        "qurinet_ctsg_gamma_comparison.pdf",
        "qurinet_clsg_gamma_comparison.pdf"
    ]
    
    for old_name, new_name in zip(old_names, new_names):
        old_path = os.path.join(output_dir, old_name)
        new_path = os.path.join(output_dir, new_name)
        if os.path.exists(old_path):
            shutil.move(old_path, new_path)
            print(f"Renamed {old_name} to {new_name}")


def main():
    """Main function to run the Qurinet simulation."""
    parser = argparse.ArgumentParser(description="Qurinet Real Wireless Mesh Network Simulation")
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
    
    print("Qurinet Real Wireless Mesh Network Environment Example")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Number of rounds: {args.rounds}")
    print(f"  Number of runs: {args.runs}")
    print(f"  Keep memory-mapped files: {args.keep_memmap}")
    
    # Setup environment
    env = setup_qurinet_environment(args.rounds)
    
    # Print network statistics
    env.print_network_statistics()
    
    # Run simulation
    print(f"\nRunning Qurinet real network simulation...")
    results = run_memory_mapped_qurinet_simulation(
        env=env,
        num_runs=args.runs,
        progress_level=args.progress,
        main_seed=42
    )
    
    # Analyze results
    analyze_qurinet_results(results)
    
    # Generate plots
    print(f"\nGenerating plots...")
    output_dir = os.path.join(project_root, "output", "images")
    os.makedirs(output_dir, exist_ok=True)
    
    plot_qurinet_results(results, output_dir)
    
    # Cleanup
    cleanup_temp_files(results, args.keep_memmap)
    
    print(f"\nQurinet real wireless mesh network simulation completed!")


if __name__ == "__main__":
    main() 