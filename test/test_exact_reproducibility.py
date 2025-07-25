#!/usr/bin/env python3
"""
Exact reproducibility test to find the remaining source of randomness.
"""

import sys
import os
import numpy as np

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.bandits.cl_sg import CLSG
from src.bandits.cts_b import CTSB
from src.environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment


def run_single_clsg_test(seed, test_name):
    """Run a single CL-SG test and return detailed results."""
    print(f"\n{test_name} (seed={seed}):")
    
    # Setup environment
    data_dir = os.path.join(project_root, 'data', 'ucsb')
    trace_period = "1144393236-1144450070"
    trace_path = os.path.join(data_dir, trace_period)
    
    neighbortable_files = sorted([os.path.join(trace_path, f) for f in os.listdir(trace_path) 
                                 if f.startswith('neighbortable-')])
    
    env_config = {
        'neighbortable_files': neighbortable_files,
        'routes_per_minute': 15,
        'source': '10.1.1.100',
        'destination': '10.1.1.102',
        'pre_generate_rewards': True,
        'use_persistent_files': False,
        'max_feasible_combinations': 50,
        'max_path_length': 2,
        'max_paths_per_algorithm': 25
    }
    
    # Create main RNG
    main_rng = np.random.default_rng(seed)
    
    # Create environment
    env_rng = main_rng.spawn(1)[0]
    env = UCSBMeshnetMemmapEnvironment(rng=env_rng, **env_config)
    
    # Create CL-SG generator (simulate the algorithm at position 8 like in main script)
    all_alg_generators = main_rng.spawn(11)  # 11 total algorithms
    clsg_generator = all_alg_generators[8]   # CL-SG gamma=0.1 is at position 8
    
    # Run 3 simulated "runs" like in the main script
    regrets_list = []
    
    for run_idx in range(3):
        # Create run-specific generator
        run_generator = clsg_generator.spawn(1)[0]
        
        # Create CL-SG instance
        clsg = CLSG(environment=env, rng=run_generator, gamma=0.1, optimistic_init=True)
        
        cumulative_regret = 0.0
        
        # Run for 1000 rounds
        for round_idx in range(1000):
            # Get available arms
            available_arms = env.get_available_arms_for_round(round_idx)
            feasible_combinations = env.get_feasible_combinations(available_arms)
            
            if feasible_combinations:
                # Select action
                selected_combination = clsg.select_combination(round_idx)
                
                # Calculate total reward
                total_reward = env.get_expected_reward_for_path(selected_combination, round_idx)
                
                # Get optimal reward
                optimal_reward = env.get_optimal_path_expected_reward(round_idx)
                
                # Update regret
                regret = optimal_reward - total_reward
                cumulative_regret += regret
                
                # Update algorithm
                reward_dict = {}
                for arm in selected_combination:
                    reward_dict[arm] = env.get_reward_for_round(arm, round_idx)
                clsg.update_posterior(selected_combination, reward_dict, round_idx)
        
        regrets_list.append(cumulative_regret)
    
    # Calculate statistics
    regrets = np.array(regrets_list)
    mean_regret = np.mean(regrets)
    std_regret = np.std(regrets)
    
    print(f"  Individual regrets: {[f'{r:.6f}' for r in regrets]}")
    print(f"  Mean: {mean_regret:.6f}, Std: {std_regret:.6f}")
    
    # Cleanup
    env.cleanup()
    
    return mean_regret, std_regret, regrets


def test_exact_reproducibility():
    """Test exact reproducibility with detailed analysis."""
    print("Exact Reproducibility Test")
    print("=" * 60)
    
    seed = 2023
    
    # Run the same test multiple times
    results = []
    for i in range(5):
        mean_regret, std_regret, regrets = run_single_clsg_test(seed, f"Test Run {i+1}")
        results.append((mean_regret, std_regret, regrets))
    
    print("\n" + "=" * 60)
    print("Summary:")
    for i, (mean_regret, std_regret, regrets) in enumerate(results):
        print(f"Run {i+1}: Mean={mean_regret:.6f}, Std={std_regret:.6f}")
    
    # Check if all results are identical
    first_mean = results[0][0]
    all_identical = all(abs(r[0] - first_mean) < 1e-10 for r in results)
    
    print(f"\nAll results identical: {all_identical}")
    if not all_identical:
        print("There is still some source of randomness!")
        
        # Print differences
        for i, (mean_regret, std_regret, regrets) in enumerate(results[1:], 1):
            diff = abs(mean_regret - first_mean)
            print(f"  Run {i+1} vs Run 1: diff = {diff:.10f}")


if __name__ == "__main__":
    test_exact_reproducibility() 