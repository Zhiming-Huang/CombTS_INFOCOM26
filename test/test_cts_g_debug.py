#!/usr/bin/env python3
"""
Debug CTS-G algorithm to understand its selection behavior.
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from environments.qurinet_environment import QurinetEnvironment
from bandits.cts_g import CTSG

def debug_cts_g():
    """Debug CTS-G algorithm selection behavior."""
    
    print("Debugging CTS-G Algorithm")
    print("=" * 50)
    
    # Create environment with 100% availability
    env = QurinetEnvironment(
        data_dir="data/qurinet",
        date="2-May",
        num_rounds=100,
        pre_generate_availability=True,
        pre_generate_rewards=True,
        availability_rate=1.0
    )
    
    cts_g = CTSG(env, gamma=0.01)
    
    print(f"Environment: {env.num_arms} arms")
    print(f"Gamma: {cts_g.gamma}")
    
    # Test for several rounds
    for round_idx in range(5):
        print(f"\n=== Round {round_idx} ===")
        
        available_arms = env.get_available_arms_for_round(round_idx)
        feasible_combinations = env.get_feasible_combinations(available_arms)
        
        print(f"Available arms: {len(available_arms)}")
        print(f"Feasible combinations: {len(feasible_combinations)}")
        
        # Show all feasible combinations with their rewards
        print(f"\nAll feasible combinations:")
        for i, comb in enumerate(feasible_combinations):
            reward = sum(env.arm_means[arm_id] for arm_id in comb)
            print(f"  {i}: {comb} (reward: {reward:.3f})")
        
        # Get CTS-G posterior samples
        print(f"\nCTS-G posterior samples:")
        posterior_samples = {}
        for arm in available_arms:
            sample = cts_g._compute_gaussian_sample(arm, round_idx)
            posterior_samples[arm] = sample
            empirical_mean = cts_g.empirical_means[arm]
            pull_count = cts_g.pull_counts[arm]
            print(f"  Arm {arm}: sample={sample:.3f}, empirical_mean={empirical_mean:.3f}, pulls={pull_count}")
        
        # Show combination scores
        print(f"\nCombination scores (sum of posterior samples):")
        best_combination = None
        best_score = float('-inf')
        
        for i, comb in enumerate(feasible_combinations):
            score = sum(posterior_samples[arm] for arm in comb)
            print(f"  {i}: {comb} -> score={score:.3f}")
            if score > best_score:
                best_score = score
                best_combination = comb
        
        print(f"\nBest combination by CTS-G: {best_combination} (score: {best_score:.3f})")
        
        # Get actual selection
        selection = cts_g.select_combination(round_idx)
        print(f"Actual selection: {selection}")
        
        # Check if they match
        if selection == best_combination:
            print("✓ Selection matches best combination")
        else:
            print("✗ Selection does NOT match best combination!")
            print(f"  Expected: {best_combination}")
            print(f"  Got: {selection}")
        
        # Update CTS-G
        rewards = env.get_reward_for_round(selection, round_idx)
        cts_g.update_posterior(selection, rewards, round_idx)
        
        print(f"Updated empirical means:")
        for arm in selection:
            print(f"  Arm {arm}: {cts_g.empirical_means[arm]:.3f}")

def test_cts_g_initialization():
    """Test CTS-G initialization and early rounds."""
    
    print("\n\nTesting CTS-G Initialization")
    print("=" * 50)
    
    # Create environment
    env = QurinetEnvironment(
        data_dir="data/qurinet",
        date="2-May",
        num_rounds=100,
        pre_generate_availability=True,
        pre_generate_rewards=True,
        availability_rate=1.0
    )
    
    cts_g = CTSG(env, gamma=0.01)
    
    print(f"Initial state:")
    print(f"  Empirical means: {cts_g.empirical_means}")
    print(f"  Pull counts: {cts_g.pull_counts}")
    print(f"  Total rewards: {cts_g.total_rewards}")
    
    # Test first round
    round_idx = 0
    available_arms = env.get_available_arms_for_round(round_idx)
    feasible_combinations = env.get_feasible_combinations(available_arms)
    
    print(f"\nRound {round_idx}:")
    print(f"Available arms: {len(available_arms)}")
    print(f"Feasible combinations: {len(feasible_combinations)}")
    
    # Show posterior samples for first round
    print(f"\nPosterior samples for round {round_idx}:")
    for arm in available_arms:
        sample = cts_g._compute_gaussian_sample(arm, round_idx)
        empirical_mean = cts_g.empirical_means[arm]
        variance = cts_g.gamma * np.log(round_idx + 1) / (cts_g.pull_counts[arm] + 1)
        print(f"  Arm {arm}: sample={sample:.3f}, mean={empirical_mean:.3f}, variance={variance:.3f}")
    
    # Show combination scores
    print(f"\nCombination scores:")
    for i, comb in enumerate(feasible_combinations):
        score = sum(cts_g._compute_gaussian_sample(arm, round_idx) for arm in comb)
        reward = sum(env.arm_means[arm_id] for arm_id in comb)
        print(f"  {i}: {comb} -> score={score:.3f}, reward={reward:.3f}")

if __name__ == "__main__":
    test_cts_g_initialization()
    debug_cts_g() 