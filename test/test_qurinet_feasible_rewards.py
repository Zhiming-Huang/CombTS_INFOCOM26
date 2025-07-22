#!/usr/bin/env python3
"""
Test script to analyze the reward distribution of the 5 feasible combinations
in the Qurinet environment.
"""

import sys
import os
import numpy as np
from typing import Set, List, Dict, Any

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.environments.qurinet_environment import QurinetEnvironment


def analyze_feasible_combinations_rewards():
    """
    Analyze the reward distribution of all feasible combinations in Qurinet environment.
    """
    print("Qurinet Feasible Combinations Reward Analysis")
    print("=" * 60)
    
    # Create Qurinet environment
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "qurinet")
    
    env = QurinetEnvironment(
        data_dir=data_dir,
        date="2-May",
        num_rounds=1000,  # Use more rounds for better statistics
        pre_generate_availability=True,
        pre_generate_rewards=True,
        rng=np.random.default_rng(42)
    )
    
    # Set availability rate manually
    env.availability_rate = 0.9
    
    print(f"Environment: {env.num_arms} arms, {env.num_rounds} rounds")
    print(f"Source: {env.source}, Destination: {env.destination}")
    print()
    
    # Get available arms and feasible combinations for first round
    available_arms = env.get_available_arms_for_round(0)
    feasible_combinations = env.get_feasible_combinations(available_arms)
    
    print(f"Available arms: {sorted(available_arms)}")
    print(f"Number of feasible combinations: {len(feasible_combinations)}")
    print()
    
    # Analyze each feasible combination
    combination_stats = {}
    
    for i, combination in enumerate(feasible_combinations):
        print(f"Feasible Combination {i+1}: {combination}")
        print("-" * 40)
        
        # Get expected reward (from arm means)
        expected_reward = sum(env.arm_means[arm] for arm in combination)
        print(f"Expected reward (from arm means): {expected_reward:.4f}")
        
        # Get individual arm means
        print("Individual arm means:")
        for arm in sorted(combination):
            print(f"  Arm {arm}: {env.arm_means[arm]:.4f}")
        
        # Simulate rewards over multiple rounds
        rewards = []
        for round_idx in range(1000):
            reward_dict = env.get_reward_for_round(combination, round_idx)
            total_reward = sum(reward_dict.values())
            rewards.append(total_reward)
        
        # Calculate statistics
        rewards = np.array(rewards)
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        min_reward = np.min(rewards)
        max_reward = np.max(rewards)
        
        print(f"Simulated reward statistics (1000 rounds):")
        print(f"  Mean: {mean_reward:.4f}")
        print(f"  Std:  {std_reward:.4f}")
        print(f"  Min:  {min_reward:.4f}")
        print(f"  Max:  {max_reward:.4f}")
        
        # Analyze reward distribution
        unique_rewards, counts = np.unique(rewards, return_counts=True)
        print(f"Reward distribution:")
        for reward_val, count in zip(unique_rewards, counts):
            percentage = count / len(rewards) * 100
            print(f"  {reward_val:.1f}: {count} times ({percentage:.1f}%)")
        
        # Store statistics
        combination_stats[i] = {
            'combination': combination,
            'expected_reward': expected_reward,
            'simulated_mean': mean_reward,
            'simulated_std': std_reward,
            'rewards': rewards,
            'arm_means': {arm: env.arm_means[arm] for arm in combination}
        }
        
        print()
    
    # Compare all combinations
    print("=" * 60)
    print("COMPARISON OF ALL FEASIBLE COMBINATIONS")
    print("=" * 60)
    
    print(f"{'Combination':<15} {'Expected':<10} {'Simulated':<10} {'Std':<8} {'Min':<8} {'Max':<8}")
    print("-" * 70)
    
    for i in range(len(feasible_combinations)):
        stats = combination_stats[i]
        print(f"{str(stats['combination']):<15} "
              f"{stats['expected_reward']:<10.4f} "
              f"{stats['simulated_mean']:<10.4f} "
              f"{stats['simulated_std']:<8.4f} "
              f"{np.min(stats['rewards']):<8.4f} "
              f"{np.max(stats['rewards']):<8.4f}")
    
    # Find optimal combination
    optimal_idx = max(combination_stats.keys(), 
                     key=lambda k: combination_stats[k]['simulated_mean'])
    optimal_stats = combination_stats[optimal_idx]
    
    print(f"\nOptimal combination: {optimal_stats['combination']}")
    print(f"Optimal expected reward: {optimal_stats['expected_reward']:.4f}")
    print(f"Optimal simulated mean: {optimal_stats['simulated_mean']:.4f}")
    
    # Analyze why certain combinations are better
    print(f"\n" + "=" * 60)
    print("DETAILED ANALYSIS OF OPTIMAL COMBINATION")
    print("=" * 60)
    
    print(f"Optimal combination: {optimal_stats['combination']}")
    print("Arm details:")
    for arm in sorted(optimal_stats['combination']):
        arm_mean = optimal_stats['arm_means'][arm]
        print(f"  Arm {arm}: mean = {arm_mean:.4f}")
    
    # Check if this matches what CTS-G selects
    print(f"\n" + "=" * 60)
    print("VERIFICATION: WHAT CTS-G SELECTS")
    print("=" * 60)
    
    # Create a simple test to see what CTS-G would select
    from src.bandits.cts_g import CTSG
    
    cts_g = CTSG(env, gamma=0.01, rng=np.random.default_rng(42))
    
    # Run for a few rounds to see what CTS-G selects
    cts_g_selections = []
    for round_idx in range(20):
        selection = cts_g.select_combination(round_idx)
        cts_g_selections.append(selection)
        
        # Update with rewards
        rewards = env.get_reward_for_round(selection, round_idx)
        cts_g.update_posterior(selection, rewards, round_idx)
    
    # Count selections
    from collections import Counter
    selection_counter = Counter(tuple(sorted(s)) for s in cts_g_selections)
    
    print("CTS-G selections over 20 rounds:")
    for selection, count in selection_counter.most_common():
        print(f"  {selection}: {count} times")
        
        # Find the corresponding combination index
        for i, comb in enumerate(feasible_combinations):
            if set(comb) == set(selection):
                stats = combination_stats[i]
                print(f"    Expected reward: {stats['expected_reward']:.4f}")
                print(f"    Simulated mean: {stats['simulated_mean']:.4f}")
                break


def analyze_arm_quality_distribution():
    """
    Analyze the distribution of arm qualities in the Qurinet environment.
    """
    print(f"\n" + "=" * 60)
    print("ARM QUALITY DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    # Create environment
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "qurinet")
    
    env = QurinetEnvironment(
        data_dir=data_dir,
        date="2-May",
        num_rounds=100,
        pre_generate_availability=True,
        pre_generate_rewards=True,
        rng=np.random.default_rng(42)
    )
    
    env.availability_rate = 0.9
    
    # Analyze all arms
    arm_means = env.arm_means
    print(f"Arm means type: {type(arm_means)}")
    print(f"Arm means shape: {arm_means.shape if hasattr(arm_means, 'shape') else 'N/A'}")
    
    # Convert to list if it's a numpy array
    if isinstance(arm_means, np.ndarray):
        arm_means_list = arm_means.tolist()
    else:
        arm_means_list = list(arm_means.values())
    
    print(f"Arm quality statistics:")
    print(f"  Number of arms: {len(arm_means_list)}")
    print(f"  Mean quality: {np.mean(arm_means_list):.4f}")
    print(f"  Std quality: {np.std(arm_means_list):.4f}")
    print(f"  Min quality: {np.min(arm_means_list):.4f}")
    print(f"  Max quality: {np.max(arm_means_list):.4f}")
    
    # Show top 10 arms
    if isinstance(arm_means, np.ndarray):
        # For numpy array, create (index, value) pairs
        arm_pairs = [(i, arm_means[i]) for i in range(len(arm_means))]
    else:
        arm_pairs = list(arm_means.items())
    
    sorted_arms = sorted(arm_pairs, key=lambda x: x[1], reverse=True)
    print(f"\nTop 10 arms by quality:")
    for i, (arm, mean) in enumerate(sorted_arms[:10]):
        print(f"  {i+1}. Arm {arm}: {mean:.4f}")
    
    # Show bottom 10 arms
    print(f"\nBottom 10 arms by quality:")
    for i, (arm, mean) in enumerate(sorted_arms[-10:]):
        print(f"  {len(sorted_arms)-9+i}. Arm {arm}: {mean:.4f}")


if __name__ == "__main__":
    analyze_feasible_combinations_rewards()
    analyze_arm_quality_distribution() 