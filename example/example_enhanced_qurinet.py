#!/usr/bin/env python3
"""
Example script for running the enhanced QurinetEnvironment with different dates.
This script demonstrates how to use the enhanced environment that can utilize
different date versions more effectively, especially the 29-April version with 32 nodes.
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Any
import time

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.environments.qurinet_environment_enhanced import QurinetEnvironmentEnhanced
from src.bandits.cts_b import CTSB
from src.bandits.comb_ucb import CombUCB
from src.bandits.cts_g import CTSG
from src.bandits.cl_sg import CLSG
from src.bandits.bg_cts import BGCTS
from src.utils.plotting import plot_all_results, plot_regret_comparison, plot_gamma_comparison


def setup_enhanced_qurinet_environment(date: str = "29-April", 
                                      num_rounds: int = 10000,
                                      rng: np.random.Generator = None) -> QurinetEnvironmentEnhanced:
    """
    Set up the enhanced Qurinet environment.
    
    Args:
        date: Date of the topology data
        num_rounds: Number of simulation rounds
        rng: Random number generator
    
    Returns:
        Enhanced Qurinet environment
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "qurinet")
    
    print(f"Setting up enhanced Qurinet environment for {date}...")
    print(f"Data directory: {data_dir}")
    
    env = QurinetEnvironmentEnhanced(
        data_dir=data_dir,
        date=date,
        num_rounds=num_rounds,
        pre_generate_availability=True,
        pre_generate_rewards=True,
        rng=rng,
        use_all_nodes=True
    )
    
    # Print environment statistics
    env.print_network_statistics()
    
    return env


def run_enhanced_qurinet_simulation(env: QurinetEnvironmentEnhanced,
                                  algorithms: Dict[str, Any],
                                  num_runs: int = 5,
                                  default_gamma: float = 0.01,
                                  gamma_values: List[float] = None) -> Dict[str, np.ndarray]:
    """
    Run simulation with enhanced Qurinet environment.
    
    Args:
        env: Enhanced Qurinet environment
        algorithms: Dictionary of algorithms to test
        num_runs: Number of simulation runs
        default_gamma: Default gamma value for gamma-dependent algorithms
    
    Returns:
        Dictionary containing cumulative regrets for each algorithm
    """
    num_rounds = env.num_rounds
    num_algorithms = len(algorithms)
    
    print(f"\nRunning enhanced Qurinet simulation...")
    print(f"Rounds: {num_rounds}, Runs: {num_runs}, Algorithms: {num_algorithms}, Gamma: {default_gamma}")
    
    # Set default gamma values if not provided
    if gamma_values is None:
        gamma_values = [0.01, 0.1, 0.5, 1.0]
    
    # Initialize results storage for all algorithms including gamma variants
    all_algorithm_names = list(algorithms.keys())
    for alg_name in algorithms.keys():
        if alg_name in ['CTS-G', 'CL-SG']:
            for gamma in gamma_values:
                all_algorithm_names.append(f"{alg_name.lower()}_gamma_{gamma}")
    
    cumulative_regrets = {alg_name: np.zeros((num_runs, num_rounds)) 
                         for alg_name in all_algorithm_names}
    
    # Run simulations
    for run_idx in range(num_runs):
        print(f"Run {run_idx + 1}/{num_runs}", end='\r')
        
        # Reset environment and algorithms for this run
        run_rng = env.rng.spawn(1)[0]
        
        # Create fresh algorithms for this run (including all gamma variants)
        run_algorithms = {}
        for alg_name, alg_class in algorithms.items():
            if alg_name in ['CTS-G', 'CL-SG']:
                # Create base algorithm with default gamma
                run_algorithms[alg_name] = alg_class(env, gamma=default_gamma, rng=run_rng)
                # Create gamma variants
                for gamma in gamma_values:
                    gamma_alg_name = f"{alg_name.lower()}_gamma_{gamma}"
                    run_algorithms[gamma_alg_name] = alg_class(env, gamma=gamma, rng=run_rng)
            elif alg_name == 'BG-CTS':
                run_algorithms[alg_name] = alg_class(env)  # BG-CTS doesn't accept rng
            else:
                run_algorithms[alg_name] = alg_class(env, rng=run_rng)
        
        # Run simulation for this run
        for round_idx in range(num_rounds):
            # Get available arms for this round
            available_arms = env.get_available_arms_for_round(round_idx)
            
            # Get optimal combination for regret calculation
            optimal_combination = env.get_optimal_combination(available_arms)
            if optimal_combination is None:
                continue
            
            optimal_reward = sum(env.arm_means[arm] for arm in optimal_combination)
            
            # Run each algorithm
            for alg_name, alg in run_algorithms.items():
                # Select combination
                selected_combination = alg.select_combination(round_idx)
                
                # Get rewards
                rewards = env.get_reward_for_round(selected_combination, round_idx)
                selected_reward = sum(rewards.values())
                
                # Calculate regret
                regret = optimal_reward - selected_reward
                
                # Update cumulative regret
                if round_idx == 0:
                    cumulative_regrets[alg_name][run_idx, round_idx] = regret
                else:
                    cumulative_regrets[alg_name][run_idx, round_idx] = (
                        cumulative_regrets[alg_name][run_idx, round_idx - 1] + regret
                    )
                
                # Update algorithm
                alg.update_posterior(selected_combination, rewards, round_idx)
        
        # Print progress
        if (run_idx + 1) % 2 == 0 or run_idx == num_runs - 1:
            print(f"  Completed run {run_idx + 1}")
        else:
            print(f"  Completed run {run_idx + 1}")
    
    return cumulative_regrets


def analyze_enhanced_qurinet_results(results: Dict[str, np.ndarray], 
                                   default_gamma: float = 0.01) -> None:
    """
    Analyze and print results from enhanced Qurinet simulation.
    
    Args:
        results: Dictionary containing cumulative regrets for each algorithm
        default_gamma: Default gamma value used
    """
    print(f"\n" + "=" * 60)
    print("ENHANCED QURINET SIMULATION RESULTS")
    print("=" * 60)
    
    # Calculate final regrets
    final_regrets = {}
    for alg_name, regrets in results.items():
        final_regrets[alg_name] = regrets[:, -1]
    
    # Print final regret statistics
    print(f"\nFinal Cumulative Regret (mean ± std):")
    print("-" * 50)
    
    # Sort algorithms by mean final regret
    sorted_algorithms = sorted(final_regrets.items(), 
                              key=lambda x: np.mean(x[1]))
    
    for alg_name, regrets in sorted_algorithms:
        mean_regret = np.mean(regrets)
        std_regret = np.std(regrets)
        print(f"{alg_name:<15}: {mean_regret:>8.2f} ± {std_regret:>6.2f}")
    
    # Find best algorithm
    best_alg_name = sorted_algorithms[0][0]
    best_mean_regret = np.mean(sorted_algorithms[0][1])
    
    print(f"\nBest performing algorithm: {best_alg_name} (mean regret: {best_mean_regret:.2f})")
    
    # Print gamma information for gamma-dependent algorithms
    gamma_algorithms = [name for name in results.keys() if name in ['CTS-G', 'CL-SG']]
    if gamma_algorithms:
        print(f"\nGamma-dependent algorithms used gamma = {default_gamma}:")
        for alg_name in gamma_algorithms:
            mean_regret = np.mean(final_regrets[alg_name])
            print(f"  {alg_name}: {mean_regret:.2f}")


def plot_enhanced_qurinet_results(results: Dict[str, np.ndarray],
                                date: str,
                                default_gamma: float = 0.01,
                                save_dir: str = "output/images") -> None:
    """
    Plot results from enhanced Qurinet simulation.
    
    Args:
        results: Dictionary containing cumulative regrets for each algorithm
        date: Date of the topology data
        default_gamma: Default gamma value used
        save_dir: Directory to save plots
    """
    print(f"\nGenerating plots for enhanced Qurinet ({date})...")
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare algorithm groups
    base_algorithms = ['CTSB', 'CombUCB', 'BG-CTS']
    gamma_algorithms = ['CTS-G', 'CL-SG']
    gamma_values = [0.01, 0.1, 0.5, 1.0]
    
    # Create file prefix
    file_prefix = f"enhanced_qurinet_{date.lower().replace('-', '_')}"
    
    # Convert results to the format expected by plotting functions
    num_rounds = results['CTSB'].shape[1]  # Get number of rounds from first algorithm
    plotting_results = {
        'num_rounds': num_rounds
    }
    
    # Add each algorithm's data
    for alg_name in results:
        alg_data = results[alg_name]  # Shape: (num_runs, num_rounds)
        
        # Calculate average cumulative regrets
        avg_cumulative_regrets = np.mean(alg_data, axis=0)
        
        # Calculate confidence intervals
        std_regrets = np.std(alg_data, axis=0)
        ci_lower = avg_cumulative_regrets - 1.96 * std_regrets / np.sqrt(alg_data.shape[0])
        ci_upper = avg_cumulative_regrets + 1.96 * std_regrets / np.sqrt(alg_data.shape[0])
        
        plotting_results[alg_name] = {
            'avg_cumulative_regrets': avg_cumulative_regrets,
            'confidence_interval': (ci_lower, ci_upper)
        }
    
    # Create a comprehensive algorithm comparison with all gamma values
    comprehensive_algorithms = base_algorithms.copy()
    for alg in gamma_algorithms:
        for gamma in gamma_values:
            comprehensive_algorithms.append(f"{alg.lower()}_gamma_{gamma}")
    
    # Plot 1: Comprehensive algorithm comparison (all algorithms with all gamma values)
    plot_regret_comparison(
        results=plotting_results,
        algorithms=comprehensive_algorithms,
        output_path=f"{save_dir}/{file_prefix}_comprehensive_algorithm_comparison.pdf",
        title="Comprehensive Algorithm Comparison (All Gamma Values)"
    )
    
    # Plot 2: Standard algorithm comparison (base algorithms + default gamma)
    standard_algorithms = base_algorithms.copy()
    for alg in gamma_algorithms:
        standard_algorithms.append(f"{alg.lower()}_gamma_{default_gamma}")
    
    plot_regret_comparison(
        results=plotting_results,
        algorithms=standard_algorithms,
        output_path=f"{save_dir}/{file_prefix}_algorithm_comparison.pdf",
        title="Algorithm Comparison (Default Gamma)"
    )
    
    # Plot 3 & 4: Gamma comparisons for each gamma algorithm
    for alg in gamma_algorithms:
        plot_gamma_comparison(
            results=plotting_results,
            algorithm_name=alg,
            gamma_values=gamma_values,
            output_path=f"{save_dir}/{file_prefix}_{alg.lower().replace('-', '')}_gamma_comparison.pdf",
            title=f"{alg} Algorithm: Regret vs Rounds"
        )
    
    print(f"Plots saved to {save_dir}/")


def main():
    """Main function to run enhanced Qurinet example."""
    parser = argparse.ArgumentParser(description='Enhanced Qurinet Environment Example')
    parser.add_argument('--date', type=str, default='29-April',
                       choices=['2-May', '9-May', '29-April', '24-May'],
                       help='Date of the topology data')
    parser.add_argument('--rounds', type=int, default=10000,
                       help='Number of simulation rounds')
    parser.add_argument('--runs', type=int, default=5,
                       help='Number of simulation runs')
    parser.add_argument('--default-gamma', type=float, default=0.01,
                       help='Default gamma value for gamma-dependent algorithms')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting')
    
    args = parser.parse_args()
    
    print("Enhanced Qurinet Environment Example")
    print("=" * 50)
    print(f"Date: {args.date}")
    print(f"Rounds: {args.rounds}")
    print(f"Runs: {args.runs}")
    print(f"Default gamma: {args.default_gamma}")
    
    # Set up environment
    env = setup_enhanced_qurinet_environment(
        date=args.date,
        num_rounds=args.rounds
    )
    
    # Define algorithms
    algorithms = {
        'CTSB': CTSB,
        'CombUCB': CombUCB,
        'CTS-G': CTSG,
        'CL-SG': CLSG,
        'BG-CTS': BGCTS
    }
    
    # Run simulation
    start_time = time.time()
    results = run_enhanced_qurinet_simulation(
        env=env,
        algorithms=algorithms,
        num_runs=args.runs,
        default_gamma=args.default_gamma,
        gamma_values=[0.01, 0.1, 0.5, 1.0]
    )
    end_time = time.time()
    
    print(f"\nSimulation completed in {end_time - start_time:.2f} seconds")
    
    # Analyze results
    analyze_enhanced_qurinet_results(results, args.default_gamma)
    
    # Plot results (unless disabled)
    if not args.no_plot:
        plot_enhanced_qurinet_results(
            results=results,
            date=args.date,
            default_gamma=args.default_gamma
        )
    
    print(f"\nEnhanced Qurinet example completed successfully!")
    print(f"Environment used: {env.get_environment_info()['environment_type']}")


if __name__ == "__main__":
    main() 