#!/usr/bin/env python3
"""
Test script to verify that the synchronization issue between environment and test program has been fixed.
"""

import sys
import os
sys.path.append('.')

import numpy as np
from src.environments.simple_environment import SimpleEnvironment
from src.bandits.comb_ts import CombTS


def test_synchronization_fix():
    """Test that environment and test program are now synchronized."""
    print("=== Testing Synchronization Fix ===")
    
    # Create environment with pre-generated matrices
    env = SimpleEnvironment(
        num_arms=10,
        num_rounds=100,
        pre_generate_rewards=True,
        seed=42
    )
    
    algorithm = CombTS(environment=env, alpha=1.0, beta=1.0)
    
    print("Testing round-by-round synchronization:")
    
    for round_idx in range(5):
        print(f"\n--- Round {round_idx} ---")
        
        # Get available arms directly from matrix
        available_arms = env.get_available_arms_for_round(round_idx)
        print(f"Available arms (from matrix): {available_arms}")
        
        # Algorithm selects combination
        selected_combination = algorithm.select_combination()
        print(f"Selected combination: {selected_combination}")
        
        # Get rewards from matrix
        rewards = {}
        if selected_combination:
            for arm in selected_combination:
                rewards[arm] = env.get_reward_for_round(arm, round_idx)
        print(f"Rewards: {rewards}")
        
        # Update algorithm
        algorithm.update_posterior(selected_combination, rewards)
        
        # Calculate regret using same available arms
        optimal_combination = env.get_optimal_combination(available_arms)
        print(f"Optimal combination: {optimal_combination}")
        
        # Calculate expected rewards
        optimal_expected = sum(env.arm_means[arm] for arm in optimal_combination)
        selected_expected = sum(env.arm_means[arm] for arm in selected_combination) if selected_combination else 0
        
        regret = optimal_expected - selected_expected
        print(f"Regret: {regret}")
        
        # Verify consistency
        available_arms_again = env.get_available_arms_for_round(round_idx)
        print(f"Available arms consistency: {available_arms == available_arms_again}")


def test_old_vs_new_approach():
    """Compare old approach (with counters) vs new approach (direct matrix access)."""
    print("\n=== Comparing Old vs New Approach ===")
    
    # Test old approach (deprecated)
    print("Old approach (with counters):")
    env_old = SimpleEnvironment(
        num_arms=10,
        num_rounds=10,
        pre_generate_rewards=True,
        seed=42
    )
    
    old_available_arms = []
    for round_idx in range(5):
        # This increments the internal counter
        available_arms = env_old.sample_available_arms_once()
        old_available_arms.append(available_arms)
        print(f"  Round {round_idx}: {available_arms}")
    
    # Test new approach (direct matrix access)
    print("\nNew approach (direct matrix access):")
    env_new = SimpleEnvironment(
        num_arms=10,
        num_rounds=10,
        pre_generate_rewards=True,
        seed=42
    )
    
    new_available_arms = []
    for round_idx in range(5):
        # This directly accesses the matrix without counter
        available_arms = env_new.get_available_arms_for_round(round_idx)
        new_available_arms.append(available_arms)
        print(f"  Round {round_idx}: {available_arms}")
    
    # Compare results
    print(f"\nResults are identical: {old_available_arms == new_available_arms}")


def test_counter_independence():
    """Test that different parts of the code can access different rounds independently."""
    print("\n=== Testing Counter Independence ===")
    
    env = SimpleEnvironment(
        num_arms=10,
        num_rounds=20,
        pre_generate_rewards=True,
        seed=42
    )
    
    # Simulate algorithm accessing rounds
    algorithm_rounds = []
    for round_idx in range(3):
        available_arms = env.get_available_arms_for_round(round_idx)
        algorithm_rounds.append(available_arms)
        print(f"Algorithm round {round_idx}: {available_arms}")
    
    # Simulate test program accessing same rounds
    test_rounds = []
    for round_idx in range(3):
        available_arms = env.get_available_arms_for_round(round_idx)
        test_rounds.append(available_arms)
        print(f"Test round {round_idx}: {available_arms}")
    
    # Verify consistency
    print(f"Algorithm and test rounds are consistent: {algorithm_rounds == test_rounds}")
    
    # Test accessing different rounds
    print(f"Round 0: {env.get_available_arms_for_round(0)}")
    print(f"Round 5: {env.get_available_arms_for_round(5)}")
    print(f"Round 0 again: {env.get_available_arms_for_round(0)}")
    print("Round 0 is consistent: {env.get_available_arms_for_round(0) == env.get_available_arms_for_round(0)}")


def test_reward_consistency():
    """Test that rewards are consistent when accessed multiple times."""
    print("\n=== Testing Reward Consistency ===")
    
    env = SimpleEnvironment(
        num_arms=10,
        num_rounds=10,
        pre_generate_rewards=True,
        seed=42
    )
    
    # Test reward consistency
    for round_idx in range(3):
        for arm in range(3):
            reward1 = env.get_reward_for_round(arm, round_idx)
            reward2 = env.get_reward_for_round(arm, round_idx)
            print(f"Arm {arm}, Round {round_idx}: {reward1} == {reward2} ({reward1 == reward2})")


def test_full_simulation():
    """Test a complete simulation with the new approach."""
    print("\n=== Testing Full Simulation ===")
    
    env = SimpleEnvironment(
        num_arms=10,
        num_rounds=100,
        pre_generate_rewards=True,
        seed=42
    )
    
    algorithm = CombTS(environment=env, alpha=1.0, beta=1.0)
    
    total_regret = 0
    
    for round_idx in range(10):
        # Get available arms for this round
        available_arms = env.get_available_arms_for_round(round_idx)
        
        # Algorithm selects combination
        selected_combination = algorithm.select_combination()
        
        # Get rewards
        rewards = {}
        if selected_combination:
            for arm in selected_combination:
                rewards[arm] = env.get_reward_for_round(arm, round_idx)
        
        # Update algorithm
        algorithm.update_posterior(selected_combination, rewards)
        
        # Calculate regret
        optimal_combination = env.get_optimal_combination(available_arms)
        optimal_expected = sum(env.arm_means[arm] for arm in optimal_combination)
        selected_expected = sum(env.arm_means[arm] for arm in selected_combination) if selected_combination else 0
        regret = optimal_expected - selected_expected
        
        total_regret += regret
        
        if round_idx < 5:  # Show first 5 rounds
            print(f"Round {round_idx}: Regret = {regret:.2f}, Total = {total_regret:.2f}")
    
    print(f"Final total regret: {total_regret:.2f}")


def main():
    """Run all synchronization tests."""
    print("Testing Synchronization Fix")
    print("=" * 40)
    
    try:
        test_synchronization_fix()
        test_old_vs_new_approach()
        test_counter_independence()
        test_reward_consistency()
        test_full_simulation()
        
        print("\n" + "=" * 40)
        print("All tests completed!")
        print("\nKey fixes:")
        print("- Environment no longer maintains internal counter")
        print("- Test program directly specifies round index")
        print("- No more synchronization issues between environment and test")
        print("- Consistent access to availability and rewards matrices")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 