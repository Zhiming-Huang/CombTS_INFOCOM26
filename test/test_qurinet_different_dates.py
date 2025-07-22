#!/usr/bin/env python3
"""
Test script to compare different dates of Qurinet data.
"""

import sys
import os
import numpy as np
from typing import Dict, Any

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.environments.qurinet_environment import QurinetEnvironment


def test_qurinet_different_dates():
    """
    Test and compare different dates of Qurinet data.
    """
    print("Qurinet Different Dates Comparison")
    print("=" * 60)
    
    # Available dates
    dates = ["2-May", "9-May", "29-April", "24-May"]
    
    # Create environment for each date
    environments = {}
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "qurinet")
    
    for date in dates:
        try:
            print(f"\nTesting date: {date}")
            print("-" * 40)
            
            env = QurinetEnvironment(
                data_dir=data_dir,
                date=date,
                num_rounds=100,
                pre_generate_availability=True,
                pre_generate_rewards=True,
                rng=np.random.default_rng(42)
            )
            
            # Set availability rate
            env.availability_rate = 0.9
            
            # Get environment info
            info = env.get_environment_info()
            
            print(f"Network: {info['num_nodes']} nodes, {info['num_edges']} edges")
            print(f"Source: {info['source']}, Destination: {info['destination']}")
            print(f"Arms: {info['num_arms']}")
            print(f"Average reward: {info['avg_reward']:.3f}")
            print(f"Average availability: {info['avg_availability']:.3f}")
            
            # Get feasible combinations
            available_arms = env.get_available_arms_for_round(0)
            feasible_combinations = env.get_feasible_combinations(available_arms)
            
            print(f"Feasible combinations: {len(feasible_combinations)}")
            
            # Show first few combinations
            for i, comb in enumerate(feasible_combinations[:3]):
                expected_reward = sum(env.arm_means[arm] for arm in comb)
                print(f"  {i+1}: {comb} (expected reward: {expected_reward:.3f})")
            
            environments[date] = env
            
        except Exception as e:
            print(f"Error with date {date}: {e}")
            continue
    
    # Compare environments
    print(f"\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    print(f"{'Date':<12} {'Nodes':<6} {'Edges':<6} {'Arms':<6} {'Avg Reward':<10} {'Feasible':<8}")
    print("-" * 60)
    
    for date, env in environments.items():
        info = env.get_environment_info()
        available_arms = env.get_available_arms_for_round(0)
        feasible_combinations = env.get_feasible_combinations(available_arms)
        
        print(f"{date:<12} {info['num_nodes']:<6} {info['num_edges']:<6} {info['num_arms']:<6} "
              f"{info['avg_reward']:<10.3f} {len(feasible_combinations):<8}")
    
    # Test algorithm performance on different dates
    print(f"\n" + "=" * 60)
    print("ALGORITHM PERFORMANCE COMPARISON")
    print("=" * 60)
    
    from src.bandits.cts_g import CTSG
    from src.bandits.cts_b import CTSB
    
    for date, env in environments.items():
        print(f"\nDate: {date}")
        print("-" * 30)
        
        # Test CTS-G
        cts_g = CTSG(env, gamma=0.01, rng=np.random.default_rng(42))
        
        # Run for 20 rounds
        total_reward = 0
        for round_idx in range(20):
            selection = cts_g.select_combination(round_idx)
            rewards = env.get_reward_for_round(selection, round_idx)
            total_reward += sum(rewards.values())
            cts_g.update_posterior(selection, rewards, round_idx)
        
        print(f"CTS-G total reward (20 rounds): {total_reward:.3f}")
        
        # Test CTS-B
        cts_b = CTSB(env, rng=np.random.default_rng(42))
        
        # Run for 20 rounds
        total_reward = 0
        for round_idx in range(20):
            selection = cts_b.select_combination(round_idx)
            rewards = env.get_reward_for_round(selection, round_idx)
            total_reward += sum(rewards.values())
            cts_b.update_posterior(selection, rewards, round_idx)
        
        print(f"CTS-B total reward (20 rounds): {total_reward:.3f}")


def test_24_may_with_power():
    """
    Test the 24-May version which includes transmission power information.
    """
    print(f"\n" + "=" * 60)
    print("TESTING 24-MAY WITH TRANSMISSION POWER")
    print("=" * 60)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "qurinet")
    
    try:
        env = QurinetEnvironment(
            data_dir=data_dir,
            date="24-May",
            num_rounds=100,
            pre_generate_availability=True,
            pre_generate_rewards=True,
            rng=np.random.default_rng(42)
        )
        
        env.availability_rate = 0.9
        
        info = env.get_environment_info()
        print(f"Network: {info['num_nodes']} nodes, {info['num_edges']} edges")
        print(f"Source: {info['source']}, Destination: {info['destination']}")
        print(f"Arms: {info['num_arms']}")
        
        # Show some arm means
        print(f"\nFirst 10 arm means:")
        for i in range(min(10, env.num_arms)):
            print(f"  Arm {i}: {env.arm_means[i]:.4f}")
        
        # Get feasible combinations
        available_arms = env.get_available_arms_for_round(0)
        feasible_combinations = env.get_feasible_combinations(available_arms)
        
        print(f"\nFeasible combinations: {len(feasible_combinations)}")
        
        # Show all combinations
        for i, comb in enumerate(feasible_combinations):
            expected_reward = sum(env.arm_means[arm] for arm in comb)
            print(f"  {i+1}: {comb} (expected reward: {expected_reward:.3f})")
        
    except Exception as e:
        print(f"Error testing 24-May: {e}")


if __name__ == "__main__":
    test_qurinet_different_dates()
    test_24_may_with_power() 