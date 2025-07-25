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

def run_advanced_simulation(env, algorithm, algorithm_name, num_rounds=10000):
    """Run advanced simulation for a single algorithm and return regret data"""
    print(f"Running {algorithm_name} for {num_rounds} rounds...")
    
    cumulative_regret = 0.0
    regrets = []
    cumulative_regrets = []
    
    # Progress reporting intervals for 10000 rounds
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
    print("UCSB Environment Advanced 10000-Round Algorithm Test")
    print("=" * 70)
    
    # Use the folder with most traces
    ucsb_dir = "data/ucsb/1144393236-1144450070"
    neighbortable_files = []
    for f in os.listdir(ucsb_dir):
        if f.startswith("neighbortable-"):
            neighbortable_files.append(os.path.join(ucsb_dir, f))
    neighbortable_files.sort()
    
    print(f"Using folder with most traces: {ucsb_dir}")
    print(f"Found {len(neighbortable_files)} neighbortable files")
    
    # Calculate routes_per_minute needed for 10000 rounds
    # 10000 / 948 ≈ 10.54, so we need at least 11 routes per minute
    routes_per_minute = 11
    expected_rounds = len(neighbortable_files) * routes_per_minute
    print(f"Using {routes_per_minute} routes per minute")
    print(f"Expected total rounds: {expected_rounds}")
    
    # Create main generator with advanced seed
    main_seed = 42
    main_generator = np.random.Generator(np.random.PCG64(main_seed))
    print(f"Created main generator with seed {main_seed}")
    
    # Spawn child generators for environment and algorithms
    env_generator = main_generator.spawn(1)[0]
    print("Spawned environment generator")
    
    # Create environment with spawned generator
    env = UCSBMeshnetMemmapEnvironment(
        neighbortable_files=neighbortable_files,
        routes_per_minute=routes_per_minute,
        source="10.1.1.109",
        destination="10.1.1.5",
        pre_generate_rewards=True,
        rng=env_generator
    )
    
    print(f"Environment created with {env.num_rounds} rounds")
    print(f"Source: {env.source}, Destination: {env.destination}")
    print(f"Number of nodes: {len(env.nodes)}")
    print()
    
    # Spawn generators for each algorithm
    algorithm_generators = {}
    algorithm_seeds = [1, 2, 3, 4, 5]  # Seeds for different algorithms
    
    for i, seed in enumerate(algorithm_seeds):
        child_generator = main_generator.spawn(1)[0]
        algorithm_generators[f"alg_{i}"] = child_generator
        print(f"Spawned algorithm generator {i} with seed {seed}")
    
    # Define algorithms to test (only gamma=0.1)
    algorithms = {
        "CTSB": CTSB(environment=env, rng=algorithm_generators["alg_0"]),
        "CombUCB": CombUCB(environment=env, rng=algorithm_generators["alg_1"]),
        "BG-CTS": BGCTS(environment=env, rnd_generator=algorithm_generators["alg_2"]),
        "cts-g_gamma_0.1": CTSG(environment=env, rng=algorithm_generators["alg_3"], gamma=0.1),
        "cl-sg_gamma_0.1": CLSG(environment=env, rng=algorithm_generators["alg_4"], gamma=0.1),
    }
    
    print(f"\nTesting {len(algorithms)} algorithms:")
    for name in algorithms.keys():
        print(f"  - {name}")
    print()
    
    # Run simulations
    num_rounds = min(10000, env.num_rounds)
    results = {}
    
    for name, alg in algorithms.items():
        print(f"\n{'='*50}")
        regrets, cumulative_regrets = run_advanced_simulation(env, alg, name, num_rounds)
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
        file_prefix="ucsb_advanced_10000",
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
    print(f"\nAdvanced 10000-round test completed with {num_rounds} actual rounds!")
    print("Used advanced NumPy generators with spawn function for reproducibility!")

if __name__ == "__main__":
    main() 