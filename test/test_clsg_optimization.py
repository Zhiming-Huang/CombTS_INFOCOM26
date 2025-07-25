#!/usr/bin/env python3
"""
Test different CL-SG parameter settings to find optimal configuration.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import time

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.bandits.cl_sg import CLSG
from src.bandits.cts_b import CTSB
from src.environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment
from src.utils.plotting import plot_all_results


def setup_ucsb_environment(num_rounds: int = 1000, max_path_length: int = 2):
    """Setup UCSB environment for testing."""
    data_dir = os.path.join(project_root, 'data', 'ucsb')
    trace_period = "1144393236-1144450070"
    trace_path = os.path.join(data_dir, trace_period)
    
    neighbortable_files = sorted([os.path.join(trace_path, f) for f in os.listdir(trace_path) 
                                 if f.startswith('neighbortable-')])
    
    routes_per_minute = max(1, num_rounds // len(neighbortable_files))
    
    env_config = {
        'neighbortable_files': neighbortable_files,
        'routes_per_minute': routes_per_minute,
        'source': '10.1.1.100',
        'destination': '10.1.1.102',
        'pre_generate_rewards': True,
        'use_persistent_files': False,
        'max_feasible_combinations': 50,
        'max_path_length': max_path_length,
        'max_paths_per_algorithm': 25
    }
    
    return env_config


def run_single_algorithm_test(alg_class, alg_name, env, num_rounds, num_runs, rng, **alg_params):
    """Run a single algorithm test."""
    # Ensure num_rounds doesn't exceed environment capacity
    actual_rounds = min(num_rounds, env.num_rounds)
    regrets_array = np.zeros((num_runs, actual_rounds))
    rewards_array = np.zeros((num_runs, actual_rounds))
    
    for run_idx in range(num_runs):
        run_rng = rng.spawn(1)[0]
        
        # Create algorithm instance
        alg = alg_class(environment=env, rng=run_rng, **alg_params)
        
        cumulative_regret = 0.0
        
        for round_idx in range(actual_rounds):
            # Get available arms for this round
            available_arms = env.get_available_arms_for_round(round_idx)
            feasible_combinations = env.get_feasible_combinations(available_arms)
            
            if not feasible_combinations:
                regrets_array[run_idx, round_idx] = cumulative_regret
                rewards_array[run_idx, round_idx] = 0.0
                continue
            
            # Select action
            selected_combination = alg.select_combination(round_idx)
            
            # Calculate total reward for the path
            total_reward = env.get_expected_reward_for_path(selected_combination, round_idx)
            
            # Get optimal reward for regret calculation
            optimal_reward = env.get_optimal_path_expected_reward(round_idx)
            
            # Update regret
            regret = optimal_reward - total_reward
            cumulative_regret += regret
            
            # Store results
            regrets_array[run_idx, round_idx] = cumulative_regret
            rewards_array[run_idx, round_idx] = total_reward
            
            # Update algorithm
            reward_dict = {}
            for arm in selected_combination:
                reward_dict[arm] = env.get_reward_for_round(arm, round_idx)
            alg.update_posterior(selected_combination, reward_dict, round_idx)
    
    return alg_name, regrets_array, rewards_array


def test_clsg_parameters():
    """Test different CL-SG parameter configurations."""
    print("Testing CL-SG parameter optimization...")
    print("=" * 60)
    
    # Setup environment
    env_config = setup_ucsb_environment(num_rounds=1000, max_path_length=2)
    main_rng = np.random.default_rng(42)
    env_rng = main_rng.spawn(1)[0]
    env = UCSBMeshnetMemmapEnvironment(rng=env_rng, **env_config)
    
    num_rounds = 1000
    num_runs = 5
    
    # Test configurations
    test_configs = [
        # (gamma, optimistic_init, description)
        (0.01, True, "CL-SG (γ=0.01, optimistic)"),
        (0.05, True, "CL-SG (γ=0.05, optimistic)"),
        (0.1, True, "CL-SG (γ=0.1, optimistic)"),
        (0.2, True, "CL-SG (γ=0.2, optimistic)"),
        (0.5, True, "CL-SG (γ=0.5, optimistic)"),
        (1.0, True, "CL-SG (γ=1.0, optimistic)"),
        (0.01, False, "CL-SG (γ=0.01, standard)"),
        (0.05, False, "CL-SG (γ=0.05, standard)"),
        (0.1, False, "CL-SG (γ=0.1, standard)"),
        (0.2, False, "CL-SG (γ=0.2, standard)"),
        (0.5, False, "CL-SG (γ=0.5, standard)"),
        (1.0, False, "CL-SG (γ=1.0, standard)"),
    ]
    
    # Also test CTSB as baseline
    baseline_configs = [
        (CTSB, "CTSB", {}, "CTSB (baseline)")
    ]
    
    results = {}
    
    # Test baseline
    print("Testing baseline algorithm...")
    for alg_class, alg_name, alg_params, description in baseline_configs:
        alg_name_key, regrets_array, rewards_array = run_single_algorithm_test(
            alg_class, alg_name, env, num_rounds, num_runs, main_rng, **alg_params
        )
        
        avg_final_regret = np.mean(regrets_array[:, -1])
        std_final_regret = np.std(regrets_array[:, -1])
        
        results[description] = {
            'avg_final_regret': avg_final_regret,
            'std_final_regret': std_final_regret,
            'regrets_array': regrets_array,
            'rewards_array': rewards_array
        }
        
        print(f"  {description}: {avg_final_regret:.2f} ± {std_final_regret:.2f}")
    
    print("\nTesting CL-SG configurations...")
    for gamma, optimistic_init, description in test_configs:
        alg_name_key, regrets_array, rewards_array = run_single_algorithm_test(
            CLSG, "CL-SG", env, num_rounds, num_runs, main_rng,
            gamma=gamma, optimistic_init=optimistic_init
        )
        
        avg_final_regret = np.mean(regrets_array[:, -1])
        std_final_regret = np.std(regrets_array[:, -1])
        
        results[description] = {
            'avg_final_regret': avg_final_regret,
            'std_final_regret': std_final_regret,
            'regrets_array': regrets_array,
            'rewards_array': rewards_array
        }
        
        print(f"  {description}: {avg_final_regret:.2f} ± {std_final_regret:.2f}")
    
    # Find best configuration
    best_config = min(results.items(), key=lambda x: x[1]['avg_final_regret'])
    print(f"\nBest configuration: {best_config[0]}")
    print(f"Best regret: {best_config[1]['avg_final_regret']:.2f} ± {best_config[1]['std_final_regret']:.2f}")
    
    # Compare with baseline
    baseline_regret = results["CTSB (baseline)"]['avg_final_regret']
    improvement = (baseline_regret - best_config[1]['avg_final_regret']) / baseline_regret * 100
    print(f"Improvement over CTSB: {improvement:.1f}%")
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot baseline
    baseline_data = results["CTSB (baseline)"]
    baseline_mean = np.mean(baseline_data['regrets_array'], axis=0)
    baseline_std = np.std(baseline_data['regrets_array'], axis=0)
    plt.plot(baseline_mean, label="CTSB (baseline)", linewidth=2, color='black')
    plt.fill_between(range(len(baseline_mean)), 
                     baseline_mean - baseline_std, 
                     baseline_mean + baseline_std, 
                     alpha=0.2, color='black')
    
    # Plot best CL-SG configuration
    best_data = results[best_config[0]]
    best_mean = np.mean(best_data['regrets_array'], axis=0)
    best_std = np.std(best_data['regrets_array'], axis=0)
    plt.plot(best_mean, label=f"Best CL-SG: {best_config[0]}", linewidth=2, color='red')
    plt.fill_between(range(len(best_mean)), 
                     best_mean - best_std, 
                     best_mean + best_std, 
                     alpha=0.2, color='red')
    
    plt.xlabel('Round')
    plt.ylabel('Cumulative Regret')
    plt.title('CL-SG Parameter Optimization Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_dir = os.path.join(project_root, 'output', 'images')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'clsg_parameter_optimization.pdf'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"\nPlot saved to: {os.path.join(output_dir, 'clsg_parameter_optimization.pdf')}")
    
    # Cleanup
    env.cleanup()
    
    return results, best_config


if __name__ == "__main__":
    test_clsg_parameters() 