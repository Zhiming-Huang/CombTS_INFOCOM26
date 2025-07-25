#!/usr/bin/env python3
"""
Debug script to analyze regret calculation issues.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment
from bandits.cts_b import CTSB

def debug_regret_calculation():
    """Debug the regret calculation to understand why it might become zero or negative."""
    
    print("Debugging regret calculation...")
    print("=" * 50)
    
    # Setup environment (use a small number of files for debugging)
    data_dir = Path("data/ucsb")
    
    # Find the trace directory with the most files
    trace_dirs = {}
    for subdir in data_dir.iterdir():
        if subdir.is_dir():
            files = list(subdir.glob("neighbortable-*"))
            trace_dirs[subdir.name] = len(files)
    
    selected_trace_dir = max(trace_dirs, key=trace_dirs.get)
    selected_trace_path = data_dir / selected_trace_dir
    
    # Use only first 10 files for debugging
    neighbortable_files = sorted(list(selected_trace_path.glob("neighbortable-*")))[:10]
    neighbortable_files = [str(f) for f in neighbortable_files]
    
    print(f"Using trace directory: {selected_trace_dir}")
    print(f"Using {len(neighbortable_files)} files for debugging")
    print()
    
    # Create environment
    env = UCSBMeshnetMemmapEnvironment(
        neighbortable_files=neighbortable_files,
        source="10.2.1.100",
        destination="10.2.1.20",
        max_path_length=3,
        max_feasible_combinations=50
    )
    
    print(f"Environment created with {env.num_rounds} rounds")
    print()
    
    # Create algorithm
    alg = CTSB(environment=env, rng=np.random.default_rng(42))
    
    # Debug first few rounds
    print("Debugging first 10 rounds:")
    print("-" * 50)
    
    cumulative_regret = 0.0
    
    for round_idx in range(min(10, env.num_rounds)):
        print(f"\nRound {round_idx}:")
        
        # Get available arms
        available_arms = env.get_available_arms_for_round(round_idx)
        print(f"  Available arms: {len(available_arms)}")
        
        # Get feasible combinations
        feasible_combinations = env.get_feasible_combinations(available_arms)
        print(f"  Feasible combinations: {len(feasible_combinations)}")
        
        if not feasible_combinations:
            print("  ❌ No feasible combinations - skipping round")
            continue
        
        # Select action
        selected_combination = alg.select_combination(round_idx)
        print(f"  Selected combination: {selected_combination}")
        
        # Calculate rewards
        total_reward = env.get_expected_reward_for_path(selected_combination, round_idx)
        optimal_reward = env.get_optimal_path_expected_reward(round_idx)
        
        print(f"  Total reward: {total_reward:.4f}")
        print(f"  Optimal reward: {optimal_reward:.4f}")
        
        # Calculate regret
        regret = optimal_reward - total_reward
        cumulative_regret += regret
        
        print(f"  Instant regret: {regret:.4f}")
        print(f"  Cumulative regret: {cumulative_regret:.4f}")
        
        # Check for issues
        if regret < 0:
            print(f"  ⚠️  WARNING: Negative instant regret!")
        if cumulative_regret < 0:
            print(f"  ⚠️  WARNING: Negative cumulative regret!")
        if regret == 0:
            print(f"  ℹ️  INFO: Zero instant regret (algorithm found optimal path)")
        
        # Update algorithm
        reward_dict = {}
        for arm in selected_combination:
            reward_dict[arm] = env.get_reward_for_round(arm, round_idx)
        alg.update_posterior(selected_combination, reward_dict, round_idx)
    
    print(f"\nFinal cumulative regret: {cumulative_regret:.4f}")
    
    # Analyze reward distributions
    print(f"\nAnalyzing reward distributions...")
    print("-" * 50)
    
    all_rewards = []
    all_optimal_rewards = []
    
    for round_idx in range(min(50, env.num_rounds)):
        available_arms = env.get_available_arms_for_round(round_idx)
        feasible_combinations = env.get_feasible_combinations(available_arms)
        
        if feasible_combinations:
            # Sample some combinations
            for i, combination in enumerate(feasible_combinations[:5]):  # Sample first 5
                reward = env.get_expected_reward_for_path(combination, round_idx)
                all_rewards.append(reward)
            
            optimal_reward = env.get_optimal_path_expected_reward(round_idx)
            all_optimal_rewards.append(optimal_reward)
    
    if all_rewards:
        print(f"Reward statistics:")
        print(f"  All rewards - Min: {min(all_rewards):.4f}, Max: {max(all_rewards):.4f}, Mean: {np.mean(all_rewards):.4f}")
        print(f"  Optimal rewards - Min: {min(all_optimal_rewards):.4f}, Max: {max(all_optimal_rewards):.4f}, Mean: {np.mean(all_optimal_rewards):.4f}")
        
        # Check if rewards are all the same
        if len(set(all_rewards)) == 1:
            print(f"  ⚠️  WARNING: All rewards are identical ({all_rewards[0]:.4f})")
        if len(set(all_optimal_rewards)) == 1:
            print(f"  ⚠️  WARNING: All optimal rewards are identical ({all_optimal_rewards[0]:.4f})")

if __name__ == "__main__":
    debug_regret_calculation() 