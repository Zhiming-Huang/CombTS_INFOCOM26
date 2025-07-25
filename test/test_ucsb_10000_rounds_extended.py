#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment
from src.bandits.cts_b import CTSB
from src.bandits.comb_ucb import CombUCB
from src.bandits.cts_g import CTSG
from src.bandits.cl_sg import CLSG
from src.bandits.bg_cts import BGCTS
from src.utils.plotting import plot_all_routing_results

def collect_all_ucsb_files():
    """Collect all available UCSB neighbortable files from all time periods"""
    ucsb_dir = "data/ucsb"
    all_files = []
    
    # Check all subdirectories
    for subdir in os.listdir(ucsb_dir):
        subdir_path = os.path.join(ucsb_dir, subdir)
        if os.path.isdir(subdir_path) and subdir.startswith('114'):
            print(f"Scanning directory: {subdir}")
            for f in os.listdir(subdir_path):
                if f.startswith("neighbortable-"):
                    all_files.append(os.path.join(subdir_path, f))
    
    all_files.sort()
    print(f"Total neighbortable files found: {len(all_files)}")
    return all_files

def run_extended_simulation(env, algorithm, algorithm_name, num_rounds=10000):
    """Run extended simulation for a single algorithm and return regret data"""
    print(f"Running {algorithm_name} for {num_rounds} rounds...")
    
    cumulative_regret = 0.0
    regrets = []
    cumulative_regrets = []
    
    # Progress reporting intervals for extended rounds
    report_intervals = [100, 500, 1000, 2000, 5000, 10000]
    
    for t in range(num_rounds):
        # Get optimal expected reward for this round
        optimal_expected_reward = env.get_optimal_path_expected_reward(t)
        
        # Select combination using algorithm
        selected_arms = algorithm.select_combination(t)
        
        if not selected_arms:
            selected_expected_reward = 0.0
            rewards_dict = {}
        else:
            # Calculate expected reward for selected path
            selected_expected_reward = env.get_expected_reward_for_path(selected_arms, t)
            
            # Get actual rewards for algorithm update
            rewards_dict = {}
            for arm in selected_arms:
                reward = env.get_reward_for_round(arm, t)
                rewards_dict[arm] = reward
        
        # Update algorithm
        algorithm.update_posterior(selected_arms, rewards_dict, t)
        
        # Calculate regret
        regret = optimal_expected_reward - selected_expected_reward
        cumulative_regret += regret
        regrets.append(regret)
        cumulative_regrets.append(cumulative_regret)
        
        # Report progress
        if t + 1 in report_intervals:
            avg_regret = np.mean(regrets)
            print(f"  Round {t+1}: Cumulative Regret = {cumulative_regret:.4f}, "
                  f"Avg Regret = {avg_regret:.4f}")
    
    return np.array(regrets), np.array(cumulative_regrets)

def main():
    print("UCSB Environment Extended 10000-Round Algorithm Test")
    print("=" * 70)
    
    # Collect all available UCSB files
    all_neighbortable_files = collect_all_ucsb_files()
    
    if len(all_neighbortable_files) == 0:
        print("No neighbortable files found!")
        return
    
    # Calculate how many files we need for 10000 rounds
    # Each file generates 40 rounds (routes_per_minute=40)
    files_needed = 10000 // 40  # 250 files needed for 10000 rounds
    
    if len(all_neighbortable_files) < files_needed:
        print(f"Warning: Only {len(all_neighbortable_files)} files available, "
              f"can generate {len(all_neighbortable_files) * 40} rounds maximum")
        files_to_use = all_neighbortable_files
    else:
        files_to_use = all_neighbortable_files[:files_needed]
    
    print(f"Using {len(files_to_use)} files to generate {len(files_to_use) * 40} rounds")
    
    # Create environment with 40 rounds per timestamp
    env = UCSBMeshnetMemmapEnvironment(
        neighbortable_files=files_to_use,
        routes_per_minute=40,  # Increased from 4 to 40
        source="10.1.1.109",
        destination="10.1.1.5",
        pre_generate_rewards=True,
        seed=42
    )
    
    print(f"Environment created with {env.num_rounds} rounds")
    print(f"Source: {env.source}, Destination: {env.destination}")
    print(f"Number of nodes: {len(env.nodes)}")
    print()
    
    # Define gamma values to test
    gamma_values = [0.01, 0.1, 0.5, 1.0]
    
    # Define algorithms to test
    algorithms = {
        "CTSB": CTSB(environment=env, rng=np.random.default_rng(1)),
        "CombUCB": CombUCB(environment=env, rng=np.random.default_rng(2)),
        "BG-CTS": BGCTS(environment=env, rnd_generator=np.random.default_rng(3)),
    }
    
    # Add gamma algorithms for each gamma value
    for gamma in gamma_values:
        algorithms[f"cts-g_gamma_{gamma}"] = CTSG(
            environment=env, 
            rng=np.random.default_rng(10 + int(gamma * 100)), 
            gamma=gamma
        )
        algorithms[f"cl-sg_gamma_{gamma}"] = CLSG(
            environment=env, 
            rng=np.random.default_rng(20 + int(gamma * 100)), 
            gamma=gamma
        )
    
    print(f"Testing {len(algorithms)} algorithms:")
    for name in algorithms.keys():
        print(f"  - {name}")
    print()
    
    # Run extended simulations
    num_rounds = min(10000, env.num_rounds)
    results = {}
    
    for name, alg in algorithms.items():
        print(f"\n{'='*50}")
        regrets, cumulative_regrets = run_extended_simulation(env, alg, name, num_rounds)
        results[name] = {
            'regrets': regrets,
            'cumulative_regrets': cumulative_regrets,
            'avg_cumulative_regrets': cumulative_regrets
        }
        print(f"{'='*50}")
    
    # Add metadata for plotting
    results['num_rounds'] = num_rounds
    results['num_runs'] = 1
    results['gamma_values'] = gamma_values
    
    # Generate plots
    print("\nGenerating plots...")
    plot_all_routing_results(
        results=results,
        output_dir="output/images",
        gamma_values=gamma_values,
        file_prefix="ucsb_extended_10000",
        default_gamma=0.1
    )
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for name, data in results.items():
        if name not in ['num_rounds', 'num_runs', 'gamma_values']:
            final_cumulative = data['cumulative_regrets'][-1]
            avg_regret = np.mean(data['regrets'])
            final_1000_avg = np.mean(data['regrets'][-1000:]) if len(data['regrets']) >= 1000 else avg_regret
            final_5000_avg = np.mean(data['regrets'][-5000:]) if len(data['regrets']) >= 5000 else avg_regret
            print(f"{name:20s}: Final Cumulative = {final_cumulative:10.4f}, "
                  f"Overall Avg = {avg_regret:6.4f}, "
                  f"Final 1000 Avg = {final_1000_avg:6.4f}, "
                  f"Final 5000 Avg = {final_5000_avg:6.4f}")
    
    # Cleanup
    env.cleanup()
    print(f"\nExtended 10000-round test completed with {num_rounds} actual rounds!")

if __name__ == "__main__":
    main() 