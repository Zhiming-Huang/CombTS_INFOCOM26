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

def test_algorithm_regret(env, algorithm, algorithm_name, num_rounds=10):
    """
    Test a single algorithm and verify regret calculation
    """
    print(f"\n=== Testing {algorithm_name} ===")
    
    cumulative_regret = 0.0
    regrets = []
    
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
        
        # Print detailed information for first few rounds
        if t < 3:
            print(f"Round {t}:")
            print(f"  Optimal expected reward: {optimal_expected_reward:.4f}")
            print(f"  Selected arms: {selected_arms}")
            print(f"  Selected expected reward: {selected_expected_reward:.4f}")
            print(f"  Regret: {regret:.4f}")
            print(f"  Cumulative regret: {cumulative_regret:.4f}")
            print()
    
    print(f"Final cumulative regret for {algorithm_name}: {cumulative_regret:.4f}")
    print(f"Average regret per round: {np.mean(regrets):.4f}")
    print(f"Regret range: [{np.min(regrets):.4f}, {np.max(regrets):.4f}]")
    
    return cumulative_regret, regrets

def main():
    print("UCSB Environment Algorithm Regret Verification")
    print("=" * 60)
    
    # Setup environment
    ucsb_dir = "data/ucsb/1143927049-1143953729"
    neighbortable_files = []
    for f in os.listdir(ucsb_dir):
        if f.startswith("neighbortable-"):
            neighbortable_files.append(os.path.join(ucsb_dir, f))
    neighbortable_files.sort()
    
    # Use a small subset for testing
    neighbortable_files = neighbortable_files[:5]  # Use first 5 files
    
    print(f"Using {len(neighbortable_files)} neighbortable files")
    
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
    
    # Test each algorithm
    algorithms = {
        "CTSB": CTSB(environment=env, rng=np.random.default_rng(1)),
        "CombUCB": CombUCB(environment=env, rng=np.random.default_rng(2)),
        "BG-CTS": BGCTS(environment=env, rnd_generator=np.random.default_rng(3)),
        "CTS-G (γ=0.1)": CTSG(environment=env, rng=np.random.default_rng(4), gamma=0.1),
        "CL-SG (γ=0.1)": CLSG(environment=env, rng=np.random.default_rng(5), gamma=0.1),
    }
    
    results = {}
    for name, alg in algorithms.items():
        cumulative_regret, regrets = test_algorithm_regret(env, alg, name, num_rounds=20)
        results[name] = {
            'cumulative_regret': cumulative_regret,
            'average_regret': np.mean(regrets),
            'regrets': regrets
        }
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        print(f"{name:15s}: Cumulative Regret = {result['cumulative_regret']:8.4f}, "
              f"Avg Regret = {result['average_regret']:6.4f}")
    
    # Cleanup
    env.cleanup()
    print("\nTest completed!")

if __name__ == "__main__":
    main() 