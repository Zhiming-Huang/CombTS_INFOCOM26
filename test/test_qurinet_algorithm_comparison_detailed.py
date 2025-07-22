#!/usr/bin/env python3
"""
Detailed comparison of CTS-G and CTSB algorithms in Qurinet environment.
"""

import sys
import os
import numpy as np
from typing import Set, List, Dict, Any

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.bandits.cts_b import CTSB
from src.bandits.cts_g import CTSG
from src.environments.qurinet_environment import QurinetEnvironment


def test_algorithm_comparison_detailed():
    """
    Detailed comparison of CTS-G and CTSB algorithms.
    """
    print("Detailed Algorithm Comparison in Qurinet Environment")
    print("=" * 70)
    
    # Create Qurinet environment
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "qurinet")
    
    env = QurinetEnvironment(
        data_dir=data_dir,
        date="2-May",
        num_rounds=15,
        pre_generate_availability=True,
        pre_generate_rewards=True,
        rng=np.random.default_rng(42)
    )
    
    # Set availability rate manually
    env.availability_rate = 0.9
    
    print(f"Environment: {env.num_arms} arms, {env.num_rounds} rounds")
    print(f"Source: {env.source}, Destination: {env.destination}")
    print()
    
    # Create algorithms with same RNG for fair comparison
    rng = np.random.default_rng(42)
    cts_b = CTSB(env, rng=rng)
    cts_g = CTSG(env, gamma=0.01, rng=rng)
    
    # Track statistics
    cts_b_stats = []
    cts_g_stats = []
    
    for round_idx in range(15):
        print(f"Round {round_idx + 1}:")
        print("-" * 40)
        
        # Get environment info
        available_arms = env.get_available_arms_for_round(round_idx)
        feasible_combinations = env.get_feasible_combinations(available_arms)
        
        print(f"Available arms: {len(available_arms)}")
        print(f"Feasible combinations: {len(feasible_combinations)}")
        
        # Show first few feasible combinations
        print("First 3 feasible combinations:")
        for i, comb in enumerate(feasible_combinations[:3]):
            reward = sum(env.arm_means[arm] for arm in comb)
            print(f"  {i}: {comb} (expected reward: {reward:.3f})")
        
        # Get algorithm selections
        cts_b_selection = cts_b.select_combination(round_idx)
        cts_g_selection = cts_g.select_combination(round_idx)
        
        print(f"\nSelections:")
        print(f"  CTS-B: {cts_b_selection}")
        print(f"  CTS-G: {cts_g_selection}")
        
        # Get algorithm statistics
        cts_b_arm_stats = cts_b.get_arm_statistics()
        cts_g_arm_stats = cts_g.get_arm_statistics(round_idx)
        
        # Show posterior samples for selected arms
        print(f"\nPosterior samples for selected arms:")
        for arm in cts_b_selection:
            if arm in cts_b_arm_stats.get('expected_rewards', {}):
                print(f"  CTS-B arm {arm}: {cts_b_arm_stats['expected_rewards'][arm]:.3f}")
        
        for arm in cts_g_selection:
            if arm in cts_g_arm_stats.get('gaussian_samples', {}):
                print(f"  CTS-G arm {arm}: {cts_g_arm_stats['gaussian_samples'][arm]:.3f}")
        
        # Get rewards and update
        cts_b_rewards = env.get_reward_for_round(cts_b_selection, round_idx)
        cts_g_rewards = env.get_reward_for_round(cts_g_selection, round_idx)
        
        print(f"\nRewards:")
        print(f"  CTS-B: {cts_b_rewards}")
        print(f"  CTS-G: {cts_g_rewards}")
        
        # Update algorithms
        cts_b.update_posterior(cts_b_selection, cts_b_rewards, round_idx)
        cts_g.update_posterior(cts_g_selection, cts_g_rewards, round_idx)
        
        # Store statistics
        cts_b_stats.append({
            'selection': cts_b_selection,
            'rewards': cts_b_rewards,
            'arm_stats': cts_b_arm_stats
        })
        
        cts_g_stats.append({
            'selection': cts_g_selection,
            'rewards': cts_g_rewards,
            'arm_stats': cts_g_arm_stats
        })
        
        print()
    
    # Final analysis
    print("=" * 70)
    print("FINAL ANALYSIS")
    print("=" * 70)
    
    # Calculate cumulative rewards
    cts_b_cumulative = sum(sum(stats['rewards'].values()) for stats in cts_b_stats)
    cts_g_cumulative = sum(sum(stats['rewards'].values()) for stats in cts_g_stats)
    
    print(f"Cumulative rewards:")
    print(f"  CTS-B: {cts_b_cumulative:.3f}")
    print(f"  CTS-G: {cts_g_cumulative:.3f}")
    
    # Analyze selection patterns
    cts_b_selections = [stats['selection'] for stats in cts_b_stats]
    cts_g_selections = [stats['selection'] for stats in cts_g_stats]
    
    print(f"\nSelection patterns:")
    print(f"  CTS-B unique selections: {len(set(tuple(sorted(s)) for s in cts_b_selections))}")
    print(f"  CTS-G unique selections: {len(set(tuple(sorted(s)) for s in cts_g_selections))}")
    
    # Show most common selections
    from collections import Counter
    cts_b_counter = Counter(tuple(sorted(s)) for s in cts_b_selections)
    cts_g_counter = Counter(tuple(sorted(s)) for s in cts_g_selections)
    
    print(f"\nMost common selections:")
    print(f"  CTS-B: {cts_b_counter.most_common(3)}")
    print(f"  CTS-G: {cts_g_counter.most_common(3)}")


def test_posterior_differences():
    """
    Test the differences in posterior computation between CTS-G and CTS-B.
    """
    print("\n" + "=" * 70)
    print("POSTERIOR COMPUTATION DIFFERENCES")
    print("=" * 70)
    
    # Create environment
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "qurinet")
    
    env = QurinetEnvironment(
        data_dir=data_dir,
        date="2-May",
        num_rounds=10,
        pre_generate_availability=True,
        pre_generate_rewards=True,
        rng=np.random.default_rng(42)
    )
    
    env.availability_rate = 0.9
    
    # Create algorithms
    rng = np.random.default_rng(42)
    cts_b = CTSB(env, rng=rng)
    cts_g = CTSG(env, gamma=0.01, rng=rng)
    
    print("Testing posterior computation for arm 4 (which CTS-G often selects):")
    print("-" * 60)
    
    for round_idx in range(10):
        print(f"\nRound {round_idx + 1}:")
        
        # Get posterior samples for arm 4
        cts_b_sample = rng.beta(cts_b.alpha_posterior[4], cts_b.beta_posterior[4])
        
        # For CTS-G, we need to compute the Gaussian sample manually
        empirical_mean = cts_g.empirical_means[4]
        variance = cts_g.gamma * np.log(round_idx + 1) / (cts_g.pull_counts[4] + 1)
        cts_g_sample = rng.normal(empirical_mean, np.sqrt(variance))
        
        print(f"  Arm 4 posterior samples:")
        print(f"    CTS-B (Beta): {cts_b_sample:.4f}")
        print(f"    CTS-G (Gaussian): {cts_g_sample:.4f}")
        print(f"    Difference: {abs(cts_b_sample - cts_g_sample):.4f}")
        
        # Update with some reward
        reward = env.get_reward_for_round({4}, round_idx)[4]
        cts_b.update_posterior({4}, {4: reward}, round_idx)
        cts_g.update_posterior({4}, {4: reward}, round_idx)
        
        print(f"  Observed reward: {reward:.3f}")
        print(f"  CTS-B posterior params: alpha={cts_b.alpha_posterior[4]:.2f}, beta={cts_b.beta_posterior[4]:.2f}")
        print(f"  CTS-G posterior params: mean={cts_g.empirical_means[4]:.4f}, pulls={cts_g.pull_counts[4]}")


if __name__ == "__main__":
    test_algorithm_comparison_detailed()
    test_posterior_differences() 