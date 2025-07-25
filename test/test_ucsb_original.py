#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment
from src.bandits.cts_b import CTSB
import matplotlib.pyplot as plt

def test_ucsb_original():
    print("Testing original UCSB environment with CTS-B algorithm")
    
    # Find UCSB data files
    ucsb_dir = "data/ucsb/1143927049-1143953729"
    neighbortable_files = []
    for f in os.listdir(ucsb_dir):
        if f.startswith("neighbortable-"):
            neighbortable_files.append(os.path.join(ucsb_dir, f))
    neighbortable_files.sort()
    
    print(f"Found {len(neighbortable_files)} neighbortable files")
    print(f"Using files: {neighbortable_files[:3]}...{neighbortable_files[-3:]}")
    
    # Use a subset for faster testing
    neighbortable_files = neighbortable_files[:10]  # Use first 10 files
    
    # Create environment
    print("Creating UCSB environment...")
    env = UCSBMeshnetMemmapEnvironment(
        neighbortable_files=neighbortable_files,
        routes_per_minute=4,
        source="10.1.1.109",  # Use connected nodes
        destination="10.1.1.5",
        pre_generate_rewards=True,
        seed=42
    )
    
    print(f"Environment created with {env.num_rounds} rounds")
    print(f"Source: {env.source}, Destination: {env.destination}")
    print(f"Number of nodes: {len(env.nodes)}")
    
    # Test environment
    print("\nTesting environment...")
    round_idx = 0
    available_arms = env.get_available_arms_for_round(round_idx)
    print(f"Available arms for round {round_idx}: {len(available_arms)}")
    
    feasible_combinations = env.get_feasible_combinations(available_arms)
    print(f"Feasible combinations: {len(feasible_combinations)}")
    
    if len(feasible_combinations) > 0:
        print(f"First few combinations: {feasible_combinations[:3]}")
        
        # Test CTS-B algorithm
        print("\nTesting CTS-B algorithm...")
        algorithm = CTSB(env, alpha=1.0, beta=1.0)
        
        regrets = []
        cumulative_regret = 0
        
        for t in range(min(100, env.num_rounds)):  # Test 100 rounds
            # Select combination using the algorithm
            arm_set = algorithm.select_combination(t)
            
            if not arm_set:
                print(f"No feasible combination selected for round {t}")
                continue
                
            # Get expected reward for the selected path
            selected_path_expected_reward = env.get_expected_reward_for_path(arm_set, t)
            
            # Get optimal path expected reward
            optimal_path_expected_reward = env.get_optimal_path_expected_reward(t)
            
            # Calculate regret as difference between optimal and selected expected rewards
            regret = optimal_path_expected_reward - selected_path_expected_reward
            cumulative_regret += regret
            regrets.append(cumulative_regret)
            
            # Get actual rewards for algorithm update
            rewards_dict = {}
            for arm in arm_set:
                reward = env.get_reward_for_round(arm, t)
                rewards_dict[arm] = reward
            
            # Update algorithm
            algorithm.update_posterior(arm_set, rewards_dict, t)
            
            if t % 20 == 0:
                print(f"Round {t}: Arms {arm_set}, Selected Expected Reward {selected_path_expected_reward:.4f}, Optimal Expected Reward {optimal_path_expected_reward:.4f}, Regret {regret:.4f}, Cumulative Regret {cumulative_regret:.4f}")
        
        print(f"\nFinal cumulative regret: {cumulative_regret:.4f}")
        
        # Plot results
        plt.figure(figsize=(8, 6))
        plt.plot(regrets, 'b-', linewidth=2, label='CTS-B')
        plt.xlabel('Round')
        plt.ylabel('Cumulative Regret')
        plt.title('CTS-B Algorithm on UCSB Meshnet')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('output/images/ucsb_original_ctsb_test.pdf', dpi=300, bbox_inches='tight')
        plt.show()
        
    else:
        print("No feasible combinations found!")
    
    # Cleanup
    env.cleanup()
    print("\nTest completed!")

if __name__ == "__main__":
    test_ucsb_original() 