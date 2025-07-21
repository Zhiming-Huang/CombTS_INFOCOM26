#!/usr/bin/env python3
"""
Test script for pre-generated available arms functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environments.simple_environment import SimpleEnvironment
from src.bandits.comb_ts import CombTS
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def test_pre_generated_environment():
    """Test the pre-generated environment functionality."""
    print("Testing Pre-Generated Environment")
    print("=" * 50)
    
    # Test with pre-generated arms
    num_rounds = 1000
    env_pre = SimpleEnvironment(num_rounds=num_rounds)
    print(f"Environment with {num_rounds} pre-generated rounds:")
    print(f"  Pre-generated: {env_pre.get_environment_info()['pre_generated']}")
    print(f"  Current round: {env_pre.get_current_round()}")
    
    # Test without pre-generation (backward compatibility)
    env_old = SimpleEnvironment()
    print(f"\nEnvironment without pre-generation:")
    print(f"  Pre-generated: {env_old.get_environment_info()['pre_generated']}")
    print(f"  Current round: {env_old.get_current_round()}")
    
    # Test consistency of pre-generated arms
    print(f"\nTesting consistency of pre-generated arms...")
    env_pre.reset_round_counter()
    
    # Get arms for first few rounds
    arms_round_1 = env_pre.sample_available_arms_once()
    env_pre.reset_available_arms()
    
    arms_round_2 = env_pre.sample_available_arms_once()
    env_pre.reset_available_arms()
    
    arms_round_3 = env_pre.sample_available_arms_once()
    env_pre.reset_available_arms()
    
    print(f"  Round 1 available arms: {arms_round_1}")
    print(f"  Round 2 available arms: {arms_round_2}")
    print(f"  Round 3 available arms: {arms_round_3}")
    
    # Test that arms are different across rounds (as expected)
    print(f"  Arms different across rounds: {arms_round_1 != arms_round_2 != arms_round_3}")
    
    # Test CombTS with pre-generated environment
    print(f"\nTesting CombTS with pre-generated environment...")
    algorithm = CombTS(env_pre)
    
    total_reward = 0
    total_regret = 0
    
    for round_num in range(10):
        env_pre.reset_available_arms()
        available_arms = env_pre.sample_available_arms_once()
        
        # Algorithm selection
        selected_combination = algorithm.select_combination()
        
        # Generate rewards
        rewards = env_pre.generate_combination_reward(selected_combination)
        total_reward += sum(rewards.values())
        
        # Calculate regret
        optimal_combination = env_pre.get_optimal_combination(available_arms)
        optimal_reward = sum(env_pre.arm_means[arm] for arm in optimal_combination)
        actual_reward = sum(rewards.values())
        regret = optimal_reward - actual_reward
        total_regret += regret
        
        # Update algorithm
        algorithm.update_posterior(selected_combination, rewards)
        
        print(f"  Round {round_num + 1}: Reward = {actual_reward:.1f}, Regret = {regret:.1f}")
    
    print(f"\nFinal Results:")
    print(f"  Total reward: {total_reward}")
    print(f"  Total regret: {total_regret:.1f}")
    print(f"  Average reward per round: {total_reward / 10:.3f}")
    print(f"  Average regret per round: {total_regret / 10:.3f}")
    
    # Test round counter
    print(f"\nRound counter test:")
    print(f"  Current round: {env_pre.get_current_round()}")
    env_pre.reset_round_counter()
    print(f"  After reset: {env_pre.get_current_round()}")
    
    print("\n✓ Pre-generated environment test completed successfully!")


def test_performance_comparison():
    """Compare performance between pre-generated and on-demand environments."""
    print("\n" + "=" * 50)
    print("Performance Comparison Test")
    print("=" * 50)
    
    num_rounds = 1000
    
    # Test pre-generated environment
    print("Testing pre-generated environment...")
    env_pre = SimpleEnvironment(num_rounds=num_rounds)
    env_pre.reset_round_counter()  # Reset to start from round 0
    algorithm_pre = CombTS(env_pre)
    
    total_reward_pre = 0
    total_regret_pre = 0
    
    for round_num in range(num_rounds):
        if round_num >= len(env_pre._all_available_arms):
            print(f"  Stopping at round {round_num} (exceeds pre-generated rounds)")
            break
        env_pre.reset_available_arms()
        available_arms = env_pre.sample_available_arms_once()
        
        selected_combination = algorithm_pre.select_combination()
        rewards = env_pre.generate_combination_reward(selected_combination)
        total_reward_pre += sum(rewards.values())
        
        optimal_combination = env_pre.get_optimal_combination(available_arms)
        optimal_reward = sum(env_pre.arm_means[arm] for arm in optimal_combination)
        actual_reward = sum(rewards.values())
        regret = optimal_reward - actual_reward
        total_regret_pre += regret
        
        algorithm_pre.update_posterior(selected_combination, rewards)
        
        if (round_num + 1) % 200 == 0:
            print(f"  Round {round_num + 1}: Reward = {total_reward_pre}, Regret = {total_regret_pre:.1f}")
    
    # Test on-demand environment
    print("\nTesting on-demand environment...")
    env_old = SimpleEnvironment()
    algorithm_old = CombTS(env_old)
    
    total_reward_old = 0
    total_regret_old = 0
    
    for round_num in range(num_rounds):
        env_old.reset_available_arms()
        available_arms = env_old.sample_available_arms_once()
        
        selected_combination = algorithm_old.select_combination()
        rewards = env_old.generate_combination_reward(selected_combination)
        total_reward_old += sum(rewards.values())
        
        optimal_combination = env_old.get_optimal_combination(available_arms)
        optimal_reward = sum(env_old.arm_means[arm] for arm in optimal_combination)
        actual_reward = sum(rewards.values())
        regret = optimal_reward - actual_reward
        total_regret_old += regret
        
        algorithm_old.update_posterior(selected_combination, rewards)
        
        if (round_num + 1) % 200 == 0:
            print(f"  Round {round_num + 1}: Reward = {total_reward_old}, Regret = {total_regret_old:.1f}")
    
    print(f"\nPerformance Comparison Results:")
    print(f"  Pre-generated - Total reward: {total_reward_pre}, Total regret: {total_regret_pre:.1f}")
    print(f"  On-demand     - Total reward: {total_reward_old}, Total regret: {total_regret_old:.1f}")
    print(f"  Reward difference: {total_reward_pre - total_reward_old}")
    print(f"  Regret difference: {total_regret_pre - total_regret_old:.1f}")
    
    print("\n✓ Performance comparison completed!")


if __name__ == "__main__":
    test_pre_generated_environment()
    test_performance_comparison() 