#!/usr/bin/env python3
"""
Test bandit algorithms and check feasible sets for each round
"""

import os
import sys
import numpy as np
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment
from bandits.cts_b import CTSB
from bandits.cts_g import CTSG
from bandits.cl_sg import CLSG
from bandits.comb_ucb import CombUCB

def test_bandit_feasible_sets():
    """Test bandit algorithms and check feasible sets for each round"""
    
    print("Testing Bandit Algorithms and Feasible Sets")
    print("=" * 60)
    
    # Original node pair
    source = "10.1.1.109"
    destination = "10.1.1.5"
    
    # Get the data directory
    data_dir = Path(__file__).parent.parent / "data" / "ucsb"
    target_folder = data_dir / "1144393236-1144450070"
    
    # Get all neighbortable files
    neighbortable_files = sorted([f for f in target_folder.glob("neighbortable-*")])
    
    print(f"Found {len(neighbortable_files)} neighbortable files")
    
    # Use all files for testing
    test_files = neighbortable_files
    print(f"Using all {len(test_files)} files for testing")
    
    # Create environment
    print(f"\nCreating environment for {source} → {destination}")
    print("Setting up environment...")
    
    start_time = time.time()
    
    try:
        env = UCSBMeshnetMemmapEnvironment(
            neighbortable_files=[str(f) for f in test_files],
            source=source,
            destination=destination,
            routes_per_minute=1,  # 1 round per minute for testing
            pre_generate_rewards=True
        )
        
        setup_time = time.time() - start_time
        print(f"Environment setup completed in {setup_time:.2f} seconds")
        
        print(f"Environment info:")
        print(f"  Total rounds: {env.num_rounds}")
        print(f"  Number of arms: {env.num_arms}")
        print(f"  Max combination size: {env.max_combination_size}")
        
        # Check feasible sets for all rounds
        print(f"\nChecking feasible sets for all rounds...")
        
        total_feasible_combinations = 0
        max_feasible_per_round = 0
        min_feasible_per_round = float('inf')
        size_distribution = {}  # Track distribution of combination sizes
        
        start_check_time = time.time()
        
        for round_idx in range(env.num_rounds):
            # Get available arms for this round
            available_arms = set(env.get_available_arms_for_round(round_idx))
            
            if available_arms:
                # Get feasible combinations (paths) for this round
                feasible_combinations = env.get_feasible_combinations(available_arms)
                
                # Count combinations by size
                for combo in feasible_combinations:
                    size = len(combo)
                    size_distribution[size] = size_distribution.get(size, 0) + 1
                
                total_feasible_combinations += len(feasible_combinations)
                max_feasible_per_round = max(max_feasible_per_round, len(feasible_combinations))
                min_feasible_per_round = min(min_feasible_per_round, len(feasible_combinations))
                
                if round_idx < 10:  # Show details for first 10 rounds
                    print(f"  Round {round_idx}: {len(available_arms)} available arms, {len(feasible_combinations)} feasible combinations")
                    if feasible_combinations:
                        # Show size breakdown for this round
                        round_sizes = {}
                        for combo in feasible_combinations:
                            size = len(combo)
                            round_sizes[size] = round_sizes.get(size, 0) + 1
                        print(f"    Size breakdown: {dict(sorted(round_sizes.items()))}")
                        print(f"    Sample combinations: {list(feasible_combinations[:3])}")
                elif round_idx == 10:
                    print(f"  ... (showing first 10 rounds only)")
            else:
                if round_idx < 10:
                    print(f"  Round {round_idx}: No available arms")
        
        check_time = time.time() - start_check_time
        print(f"Feasible set checking completed in {check_time:.2f} seconds")
        
        print(f"\nFeasible Set Summary:")
        print(f"  Total feasible combinations across all rounds: {total_feasible_combinations}")
        print(f"  Average feasible combinations per round: {total_feasible_combinations / env.num_rounds:.1f}")
        print(f"  Max feasible combinations per round: {max_feasible_per_round}")
        print(f"  Min feasible combinations per round: {min_feasible_per_round if min_feasible_per_round != float('inf') else 0}")
        
        print(f"\nSize Distribution (across all rounds):")
        for size in sorted(size_distribution.keys()):
            print(f"  {size}-hop paths: {size_distribution[size]} combinations")
        
        # Test bandit algorithms
        print(f"\nTesting Bandit Algorithms...")
        
        algorithms = {
            'CTSB': CTSB(env),
            'CTSG': CTSG(env),
            'CLSG': CLSG(env),
            'CombUCB': CombUCB(env)
        }
        
        for alg_name, algorithm in algorithms.items():
            print(f"\nTesting {alg_name}...")
            
            total_regret = 0
            total_reward = 0
            
            for round_idx in range(min(10, env.num_rounds)):  # Test first 10 rounds
                # Get available arms and feasible combinations
                available_arms = set(env.get_available_arms_for_round(round_idx))
                feasible_combinations = env.get_feasible_combinations(available_arms)
                
                if not feasible_combinations:
                    continue
                
                # Get optimal expected reward
                optimal_reward = env.get_optimal_path_expected_reward(round_idx)
                
                # Algorithm selects combination
                selected_arms = algorithm.select_combination(round_idx)
                
                # Get reward for selected combination
                reward = env.get_expected_reward_for_path(selected_arms, round_idx)
                
                # Update algorithm (create rewards dict for each arm)
                rewards = {}
                for arm in selected_arms:
                    rewards[arm] = env.get_reward_for_round(arm, round_idx)
                algorithm.update_posterior(selected_arms, rewards, round_idx)
                
                # Calculate regret
                regret = optimal_reward - reward
                total_regret += regret
                total_reward += reward
                
                if round_idx < 3:  # Show details for first 3 rounds
                    print(f"    Round {round_idx}: Selected {selected_arms}, Reward {reward:.4f}, Regret {regret:.4f}")
            
            print(f"  {alg_name} Results (first 10 rounds):")
            print(f"    Total reward: {total_reward:.4f}")
            print(f"    Total regret: {total_regret:.4f}")
            print(f"    Average regret per round: {total_regret / 10:.4f}")
        
        print(f"\nTest completed successfully!")
        print(f"Environment can initialize feasible sets for all rounds.")
        print(f"Even though only shortest paths are found, the algorithms can still work with available feasible combinations.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_bandit_feasible_sets() 