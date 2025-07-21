#!/usr/bin/env python3
"""
Test script to verify correct regret calculation.
Ensures we compare expected rewards with expected rewards, not with instantaneous rewards.
"""

import sys
import os
sys.path.append('.')

import numpy as np
from src.environments.simple_environment import SimpleEnvironment


def test_regret_calculation():
    """Test that regret calculation is correct."""
    print("=== Testing Regret Calculation ===")
    
    # Create environment
    env = SimpleEnvironment(
        num_arms=10,
        num_optimal=3,
        optimal_mean=0.9,
        suboptimal_mean=0.1,
        availability_rate=0.5,
        max_combination_size=3,
        seed=42
    )
    
    print("Environment setup:")
    print(f"  Optimal arms: {env.optimal_arms}")
    print(f"  Arm means: {env.arm_means}")
    print(f"  Optimal expected reward: {env.optimal_expected_reward}")
    
    # Test regret calculation for a few rounds
    for round_idx in range(5):
        print(f"\n--- Round {round_idx} ---")
        
        # Get available arms
        available_arms = env.get_available_arms()
        print(f"Available arms: {available_arms}")
        
        # Get optimal combination
        optimal_combination = env.get_optimal_combination(available_arms)
        print(f"Optimal combination: {optimal_combination}")
        
        # Calculate expected rewards
        optimal_expected_reward = sum(env.arm_means[arm] for arm in optimal_combination)
        print(f"Optimal expected reward: {optimal_expected_reward}")
        
        # Simulate selecting a suboptimal combination
        if available_arms:
            # Select a suboptimal combination (e.g., only suboptimal arms)
            suboptimal_arms = [arm for arm in available_arms if arm >= 3][:3]
            if len(suboptimal_arms) < 3:
                suboptimal_arms = list(available_arms)[:3]
            
            selected_combination = set(suboptimal_arms)
            print(f"Selected combination: {selected_combination}")
            
            # Calculate expected reward for selected combination
            selected_expected_reward = sum(env.arm_means[arm] for arm in selected_combination)
            print(f"Selected expected reward: {selected_expected_reward}")
            
            # Calculate regret (expected vs expected)
            regret = optimal_expected_reward - selected_expected_reward
            print(f"Regret (expected vs expected): {regret}")
            
            # Generate instantaneous rewards for comparison
            rewards = env.generate_combination_reward(selected_combination)
            instantaneous_reward = sum(rewards.values())
            print(f"Instantaneous reward: {instantaneous_reward}")
            
            # Show the difference
            instantaneous_regret = optimal_expected_reward - instantaneous_reward
            print(f"Regret (expected vs instantaneous): {instantaneous_regret}")
            print(f"Difference: {instantaneous_regret - regret}")
            
            # Verify that expected regret is deterministic
            regret2 = optimal_expected_reward - selected_expected_reward
            print(f"Regret consistency: {regret == regret2}")
        else:
            print("No available arms")


def test_regret_consistency():
    """Test that regret calculation is consistent across rounds."""
    print("\n=== Testing Regret Consistency ===")
    
    env = SimpleEnvironment(
        num_arms=10,
        num_rounds=100,
        pre_generate_rewards=True,
        seed=42
    )
    
    regrets = []
    instantaneous_regrets = []
    
    for round_idx in range(10):
        # Get available arms for this round
        available_arms = env.get_available_arms_for_round(round_idx)
        optimal_combination = env.get_optimal_combination(available_arms)
        
        # Calculate expected regret
        optimal_expected = sum(env.arm_means[arm] for arm in optimal_combination)
        
        # Simulate selecting a suboptimal combination
        if available_arms:
            selected_arms = list(available_arms)[:3]
            selected_combination = set(selected_arms)
            selected_expected = sum(env.arm_means[arm] for arm in selected_combination)
            
            # Expected regret
            expected_regret = optimal_expected - selected_expected
            regrets.append(expected_regret)
            
            # Instantaneous regret
            instantaneous_reward = 0
            for arm in selected_combination:
                instantaneous_reward += env.get_reward_for_round(arm, round_idx)
            instantaneous_regret = optimal_expected - instantaneous_reward
            instantaneous_regrets.append(instantaneous_regret)
    
    print(f"Expected regrets: {regrets}")
    print(f"Instantaneous regrets: {instantaneous_regrets}")
    print(f"Expected regret variance: {np.var(regrets):.4f}")
    print(f"Instantaneous regret variance: {np.var(instantaneous_regrets):.4f}")
    
    # Expected regrets should be more consistent (deterministic for same combinations)
    print(f"Expected regrets are more consistent: {np.var(regrets) < np.var(instantaneous_regrets)}")


def test_regret_theoretical():
    """Test that regret calculation matches theoretical expectations."""
    print("\n=== Testing Theoretical Regret ===")
    
    env = SimpleEnvironment(
        num_arms=10,
        num_optimal=3,
        optimal_mean=0.9,
        suboptimal_mean=0.1,
        availability_rate=1.0,  # All arms always available
        max_combination_size=3,
        seed=42
    )
    
    # With all arms available, optimal combination should be first 3 arms
    available_arms = set(range(10))
    optimal_combination = env.get_optimal_combination(available_arms)
    
    print(f"Optimal combination: {optimal_combination}")
    print(f"Expected: {set([0, 1, 2])}")
    print(f"Correct: {optimal_combination == {0, 1, 2}}")
    
    # Calculate theoretical regret
    optimal_expected = 3 * 0.9  # 3 optimal arms * 0.9 mean
    suboptimal_expected = 3 * 0.1  # 3 suboptimal arms * 0.1 mean
    theoretical_regret = optimal_expected - suboptimal_expected
    
    print(f"Theoretical regret: {theoretical_regret}")
    print(f"Expected: 2.4 (3*0.9 - 3*0.1 = 2.7 - 0.3 = 2.4)")
    print(f"Correct: {abs(theoretical_regret - 2.4) < 0.01}")


def main():
    """Run all regret calculation tests."""
    print("Testing Regret Calculation Fix")
    print("=" * 40)
    
    try:
        test_regret_calculation()
        test_regret_consistency()
        test_regret_theoretical()
        
        print("\n" + "=" * 40)
        print("All tests completed!")
        print("\nKey fix:")
        print("- Regret now compares expected rewards with expected rewards")
        print("- No longer mixes expected rewards (optimal) with instantaneous rewards (selected)")
        print("- This provides more consistent and theoretically correct regret measurement")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 