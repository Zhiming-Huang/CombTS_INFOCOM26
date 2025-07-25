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

def run_long_simulation(env, algorithm, algorithm_name, num_rounds=10000):
    """Run long simulation for a single algorithm and return regret data"""
    print(f"Running {algorithm_name} for {num_rounds} rounds...")
    
    cumulative_regret = 0.0
    regrets = []
    cumulative_regrets = []
    
    # Progress reporting intervals
    report_intervals = [100, 500, 1000, 2000, 5000, 10000]
    if num_rounds < 10000:
        report_intervals = [50, 100, 200, 500, 1000]
    
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
    print("UCSB Environment Long-Run Algorithm Test (10000+ rounds)")
    print("=" * 70)
    
    # Setup environment with all available files
    ucsb_dir = "data/ucsb/1143927049-1143953729"
    neighbortable_files = []
    for f in os.listdir(ucsb_dir):
        if f.startswith("neighbortable-"):
            neighbortable_files.append(os.path.join(ucsb_dir, f))
    neighbortable_files.sort()
    
    print(f"Found {len(neighbortable_files)} neighbortable files")
    
    env = UCSBMeshnetMemmapEnvironment(
        neighbortable_files=neighbortable_files,
        routes_per_minute=4,
        source="10.1.1.109",
        destination="10.1.1.5",
        pre_generate_rewards=True,
        seed=42
    )
    
    print(f"Environment created with {env.num_rounds} rounds")
    print(f"Source: {env.source}, Destination: {env.destination}")
    print(f"Number of nodes: {len(env.nodes)}")
    print()
    
    # Define algorithms to test
    algorithms = {
        "CTSB": CTSB(environment=env, rng=np.random.default_rng(1)),
        "CombUCB": CombUCB(environment=env, rng=np.random.default_rng(2)),
        "BG-CTS": BGCTS(environment=env, rnd_generator=np.random.default_rng(3)),
        "cts-g_gamma_0.1": CTSG(environment=env, rng=np.random.default_rng(4), gamma=0.1),
        "cl-sg_gamma_0.1": CLSG(environment=env, rng=np.random.default_rng(5), gamma=0.1),
    }
    
    # Run long simulations
    num_rounds = env.num_rounds
    results = {}
    
    for name, alg in algorithms.items():
        print(f"\n{'='*50}")
        regrets, cumulative_regrets = run_long_simulation(env, alg, name, num_rounds)
        results[name] = {
            'regrets': regrets,
            'cumulative_regrets': cumulative_regrets,
            'avg_cumulative_regrets': cumulative_regrets
        }
        print(f"{'='*50}")
    
    # Add metadata for plotting
    results['num_rounds'] = num_rounds
    results['num_runs'] = 1
    results['gamma_values'] = [0.1]
    
    # Generate plots
    print("\nGenerating plots...")
    plot_all_routing_results(
        results=results,
        output_dir="output/images",
        gamma_values=[0.1],
        file_prefix="ucsb_long_run",
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
            print(f"{name:15s}: Final Cumulative = {final_cumulative:10.4f}, "
                  f"Overall Avg = {avg_regret:6.4f}, "
                  f"Final 1000 Avg = {final_1000_avg:6.4f}")
    
    # Cleanup
    env.cleanup()
    print("\nLong-run test completed!")

if __name__ == "__main__":
    main() 