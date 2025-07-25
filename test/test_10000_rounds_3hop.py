#!/usr/bin/env python3
"""
Test script to run 10,000 rounds with up to 3-hop paths in UCSB environment.
Tests multiple bandit algorithms and compares their performance.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
from src.environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment
from src.bandits.cts_b import CTSB
from src.bandits.cts_g import CTSG
from src.bandits.bg_cts import BGCTS
from src.bandits.cl_sg import CLSG
from src.bandits.comb_ucb import CombUCB
from src.utils.plotting import plot_regret_comparison

def run_10000_rounds_3hop_test():
    """Run 10,000 rounds test with up to 3-hop paths."""
    
    print("=" * 80)
    print("UCSB ENVIRONMENT - 10,000 ROUNDS TEST WITH 3-HOP PATHS")
    print("=" * 80)
    
    # Get the largest dataset folder
    data_dir = "data/ucsb/1144393236-1144450070"
    neighbortable_files = []
    for f in os.listdir(data_dir):
        if f.startswith("neighbortable-"):
            neighbortable_files.append(os.path.join(data_dir, f))
    neighbortable_files.sort()
    
    print(f"Found {len(neighbortable_files)} neighbortable files")
    
    # Use the best node pair we found earlier
    source = "10.1.1.109"
    destination = "10.1.1.5"
    
    # Calculate minimum routes_per_minute to reach 10,000 rounds
    min_routes_per_minute = max(1, int(np.ceil(10000 / len(neighbortable_files))))
    print(f"Using {min_routes_per_minute} routes per minute to reach ~{len(neighbortable_files) * min_routes_per_minute} rounds")
    
    # Create environment with 3-hop path limit
    print(f"\nCreating environment with 3-hop path limit...")
    start_time = time.time()
    
    env = UCSBMeshnetMemmapEnvironment(
        neighbortable_files=neighbortable_files,
        routes_per_minute=min_routes_per_minute,
        source=source,
        destination=destination,
        max_feasible_combinations=100,  # Allow more combinations for 3-hop paths
        max_path_length=3,              # Limit to 3 hops
        max_paths_per_algorithm=50,     # Allow more paths per algorithm
        rng=np.random.Generator(np.random.PCG64(seed=42))   # Use advanced numpy generator
    )
    
    env_init_time = time.time() - start_time
    print(f"Environment initialization time: {env_init_time:.2f} seconds")
    print(f"Total rounds: {env.num_rounds}")
    
    # Test algorithms with proper naming convention
    algorithms = {
        'CTSB': CTSB(env, alpha=1.0, beta=1.0),
        'cts-g_gamma_0.1': CTSG(env, gamma=0.1),
        'BG-CTS': BGCTS(env, sigma=1.0, sigma_prior=1.0, lamda=0.0),
        'cl-sg_gamma_0.1': CLSG(env, gamma=0.1),
        'CombUCB': CombUCB(env)
    }
    
    # Run simulation for each algorithm
    results = {}
    
    for alg_name, algorithm in algorithms.items():
        print(f"\n{'='*60}")
        print(f"Testing {alg_name}")
        print(f"{'='*60}")
        
        # Reset environment state
        regrets = []
        cumulative_regret = 0.0
        total_combinations = 0
        rounds_with_combinations = 0
        
        start_time = time.time()
        
        for round_idx in range(env.num_rounds):
            if round_idx % 1000 == 0:
                print(f"  Round {round_idx}/{env.num_rounds} ({(round_idx/env.num_rounds)*100:.1f}%)")
            
            # Select action using algorithm
            selected_combination = algorithm.select_combination(round_idx)
            
            if not selected_combination:
                regrets.append(cumulative_regret)
                continue
            
            # Calculate reward and regret
            selected_reward = env.get_expected_reward_for_path(selected_combination, round_idx)
            optimal_reward = env.get_optimal_path_expected_reward(round_idx)
            
            regret = optimal_reward - selected_reward
            cumulative_regret += regret
            regrets.append(cumulative_regret)
            
            total_combinations += len(selected_combination)
            rounds_with_combinations += 1
            
            # Update algorithm with observed rewards
            rewards = {}
            for arm in selected_combination:
                reward = env.get_reward_for_round(arm, round_idx)
                rewards[arm] = reward
            
            # Update algorithm (handle different update methods)
            if hasattr(algorithm, 'update_posterior'):
                algorithm.update_posterior(selected_combination, rewards, round_idx)
            elif hasattr(algorithm, 'update'):
                for arm, reward in rewards.items():
                    algorithm.update(arm, reward)
        
        simulation_time = time.time() - start_time
        
        print(f"  Simulation completed in {simulation_time:.2f} seconds")
        print(f"  Final cumulative regret: {cumulative_regret:.4f}")
        print(f"  Rounds with combinations: {rounds_with_combinations}/{env.num_rounds}")
        print(f"  Average combinations per round: {total_combinations/max(1, rounds_with_combinations):.1f}")
        
        # Store results
        results[alg_name] = {
            'regrets': regrets,
            'final_regret': cumulative_regret,
            'rounds_with_combinations': rounds_with_combinations,
            'avg_combinations': total_combinations / max(1, rounds_with_combinations),
            'simulation_time': simulation_time
        }
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Algorithm':<12} {'Final Regret':<12} {'Rounds w/ Comb':<12} {'Avg Comb':<8} {'Time (s)':<8}")
    print("-" * 80)
    
    for alg_name, result in results.items():
        print(f"{alg_name:<12} {result['final_regret']:<12.4f} "
              f"{result['rounds_with_combinations']:<12} "
              f"{result['avg_combinations']:<8.1f} "
              f"{result['simulation_time']:<8.2f}")
    
    # Generate plots using utils plotting functions
    print(f"\n{'='*80}")
    print("GENERATING PLOTS")
    print(f"{'='*80}")
    
    # Prepare data for utils plotting
    from src.utils.plotting import plot_algorithm_comparison
    
    # Create results dictionary in the format expected by utils
    utils_results = {
        'num_rounds': env.num_rounds,
        'algorithms': {}
    }
    
    for alg_name, result in results.items():
        utils_results['algorithms'][alg_name] = {
            'avg_cumulative_regrets': result['regrets'],
            'confidence_interval': (result['regrets'], result['regrets'])  # Single run, no confidence interval
        }
    
    # Use utils plotting function
    plot_algorithm_comparison(
        utils_results,
        algorithms=list(results.keys()),
        output_path="output/images/ucsb_10000_rounds_3hop_comparison.pdf",
        title="UCSB Environment: Algorithm Comparison (10,000 Rounds, 3-Hop Paths)"
    )
    
    print("Plot saved to: output/images/ucsb_10000_rounds_3hop_comparison.pdf")
    
    # Save detailed results
    save_detailed_results(results, env.num_rounds)
    
    env.cleanup()
    
    print(f"\n{'='*80}")
    print("TEST COMPLETED!")
    print(f"{'='*80}")



def save_detailed_results(results, num_rounds):
    """Save detailed results to file."""
    import json
    
    # Prepare data for saving
    save_data = {
        'num_rounds': num_rounds,
        'algorithms': {}
    }
    
    for alg_name, result in results.items():
        save_data['algorithms'][alg_name] = {
            'final_regret': result['final_regret'],
            'rounds_with_combinations': result['rounds_with_combinations'],
            'avg_combinations': result['avg_combinations'],
            'simulation_time': result['simulation_time'],
            'regrets': result['regrets']
        }
    
    # Save to file
    output_file = "output/data/ucsb_10000_rounds_3hop_results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"Detailed results saved to: {output_file}")

if __name__ == "__main__":
    run_10000_rounds_3hop_test() 