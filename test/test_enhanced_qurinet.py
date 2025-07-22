#!/usr/bin/env python3
"""
Test script for the enhanced QurinetEnvironment.
"""

import sys
import os
import numpy as np
from typing import Dict, Any

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.environments.qurinet_environment_enhanced import QurinetEnvironmentEnhanced


def test_enhanced_qurinet_29_april():
    """
    Test the enhanced QurinetEnvironment with 29-April data.
    """
    print("Enhanced QurinetEnvironment 29-April Test")
    print("=" * 60)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "qurinet")
    
    try:
        # Create enhanced environment
        env = QurinetEnvironmentEnhanced(
            data_dir=data_dir,
            date="29-April",
            num_rounds=100,
            pre_generate_availability=True,
            pre_generate_rewards=True,
            rng=np.random.default_rng(42),
            use_all_nodes=True
        )
        
        # Print network statistics
        env.print_network_statistics()
        
        # Get environment info
        info = env.get_environment_info()
        
        print(f"\nEnhanced Environment Info:")
        print(f"  Environment type: {info['environment_type']}")
        print(f"  Sites data size: {info['sites_data_size']}")
        print(f"  Number of nodes: {info['num_nodes']}")
        print(f"  Number of edges: {info['num_edges']}")
        print(f"  Number of arms: {info['num_arms']}")
        print(f"  Source: {info['source']}, Destination: {info['destination']}")
        
        # Get feasible combinations
        available_arms = env.get_available_arms_for_round(0)
        feasible_combinations = env.get_feasible_combinations(available_arms)
        
        print(f"\nFeasible combinations: {len(feasible_combinations)}")
        
        # Show all combinations
        for i, comb in enumerate(feasible_combinations):
            expected_reward = sum(env.arm_means[arm] for arm in comb)
            path_info = env.get_path_info(comb)
            print(f"  {i+1}: {comb} (expected reward: {expected_reward:.3f}, path: {path_info['path']})")
        
        # Show arm means
        print(f"\nArm means (first 15):")
        for i in range(min(15, env.num_arms)):
            print(f"  Arm {i}: {env.arm_means[i]:.4f}")
        
        # Test algorithm performance
        print(f"\n" + "=" * 40)
        print("ALGORITHM PERFORMANCE TEST")
        print("=" * 40)
        
        from src.bandits.cts_g import CTSG
        from src.bandits.cts_b import CTSB
        
        # Test CTS-G
        cts_g = CTSG(env, gamma=0.01, rng=np.random.default_rng(42))
        
        # Run for 20 rounds
        total_reward = 0
        selections = []
        for round_idx in range(20):
            selection = cts_g.select_combination(round_idx)
            selections.append(selection)
            rewards = env.get_reward_for_round(selection, round_idx)
            total_reward += sum(rewards.values())
            cts_g.update_posterior(selection, rewards, round_idx)
        
        print(f"CTS-G total reward (20 rounds): {total_reward:.3f}")
        
        # Count selections
        from collections import Counter
        selection_counter = Counter(tuple(sorted(s)) for s in selections)
        print(f"CTS-G selections:")
        for selection, count in selection_counter.most_common():
            print(f"  {selection}: {count} times")
        
        # Test CTS-B
        cts_b = CTSB(env, rng=np.random.default_rng(42))
        
        # Run for 20 rounds
        total_reward = 0
        selections = []
        for round_idx in range(20):
            selection = cts_b.select_combination(round_idx)
            selections.append(selection)
            rewards = env.get_reward_for_round(selection, round_idx)
            total_reward += sum(rewards.values())
            cts_b.update_posterior(selection, rewards, round_idx)
        
        print(f"\nCTS-B total reward (20 rounds): {total_reward:.3f}")
        
        # Count selections
        selection_counter = Counter(tuple(sorted(s)) for s in selections)
        print(f"CTS-B selections:")
        for selection, count in selection_counter.most_common():
            print(f"  {selection}: {count} times")
        
    except Exception as e:
        print(f"Error testing enhanced 29-April environment: {e}")
        import traceback
        traceback.print_exc()


def test_enhanced_qurinet_24_may():
    """
    Test the enhanced QurinetEnvironment with 24-May data.
    """
    print(f"\n" + "=" * 60)
    print("Enhanced QurinetEnvironment 24-May Test")
    print("=" * 60)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "qurinet")
    
    try:
        # Create enhanced environment
        env = QurinetEnvironmentEnhanced(
            data_dir=data_dir,
            date="24-May",
            num_rounds=100,
            pre_generate_availability=True,
            pre_generate_rewards=True,
            rng=np.random.default_rng(42),
            use_all_nodes=True
        )
        
        # Print network statistics
        env.print_network_statistics()
        
        # Get environment info
        info = env.get_environment_info()
        
        print(f"\nEnhanced Environment Info:")
        print(f"  Environment type: {info['environment_type']}")
        print(f"  Sites data size: {info['sites_data_size']}")
        print(f"  Number of nodes: {info['num_nodes']}")
        print(f"  Number of edges: {info['num_edges']}")
        print(f"  Number of arms: {info['num_arms']}")
        print(f"  Source: {info['source']}, Destination: {info['destination']}")
        
        # Get feasible combinations
        available_arms = env.get_available_arms_for_round(0)
        feasible_combinations = env.get_feasible_combinations(available_arms)
        
        print(f"\nFeasible combinations: {len(feasible_combinations)}")
        
        # Show all combinations
        for i, comb in enumerate(feasible_combinations):
            expected_reward = sum(env.arm_means[arm] for arm in comb)
            path_info = env.get_path_info(comb)
            print(f"  {i+1}: {comb} (expected reward: {expected_reward:.3f}, path: {path_info['path']})")
        
    except Exception as e:
        print(f"Error testing enhanced 24-May environment: {e}")
        import traceback
        traceback.print_exc()


def compare_enhanced_vs_original():
    """
    Compare enhanced vs original QurinetEnvironment.
    """
    print(f"\n" + "=" * 60)
    print("COMPARISON: Enhanced vs Original QurinetEnvironment")
    print("=" * 60)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "qurinet")
    
    # Create both environments
    from src.environments.qurinet_environment import QurinetEnvironment
    
    env_original = QurinetEnvironment(
        data_dir=data_dir,
        date="29-April",
        num_rounds=100,
        pre_generate_availability=True,
        pre_generate_rewards=True,
        rng=np.random.default_rng(42)
    )
    
    env_enhanced = QurinetEnvironmentEnhanced(
        data_dir=data_dir,
        date="29-April",
        num_rounds=100,
        pre_generate_availability=True,
        pre_generate_rewards=True,
        rng=np.random.default_rng(42),
        use_all_nodes=True
    )
    
    # Get info
    info_original = env_original.get_environment_info()
    info_enhanced = env_enhanced.get_environment_info()
    
    print(f"{'Metric':<25} {'Original':<15} {'Enhanced':<15} {'Difference':<15}")
    print("-" * 75)
    print(f"{'Nodes':<25} {info_original['num_nodes']:<15} {info_enhanced['num_nodes']:<15} {info_enhanced['num_nodes'] - info_original['num_nodes']:<15}")
    print(f"{'Edges':<25} {info_original['num_edges']:<15} {info_enhanced['num_edges']:<15} {info_enhanced['num_edges'] - info_original['num_edges']:<15}")
    print(f"{'Arms':<25} {info_original['num_arms']:<15} {info_enhanced['num_arms']:<15} {info_enhanced['num_arms'] - info_original['num_arms']:<15}")
    print(f"{'Avg Reward':<25} {info_original['avg_reward']:<15.3f} {info_enhanced['avg_reward']:<15.3f} {info_enhanced['avg_reward'] - info_original['avg_reward']:<15.3f}")
    print(f"{'Avg Availability':<25} {info_original['avg_availability']:<15.3f} {info_enhanced['avg_availability']:<15.3f} {info_enhanced['avg_availability'] - info_original['avg_availability']:<15.3f}")
    
    # Compare feasible combinations
    available_arms_original = env_original.get_available_arms_for_round(0)
    feasible_original = env_original.get_feasible_combinations(available_arms_original)
    
    available_arms_enhanced = env_enhanced.get_available_arms_for_round(0)
    feasible_enhanced = env_enhanced.get_feasible_combinations(available_arms_enhanced)
    
    print(f"{'Feasible Combos':<25} {len(feasible_original):<15} {len(feasible_enhanced):<15} {len(feasible_enhanced) - len(feasible_original):<15}")
    
    # Compare arm means
    print(f"\nArm means comparison (first 10):")
    print(f"{'Arm':<5} {'Original':<10} {'Enhanced':<10} {'Diff':<10}")
    print("-" * 35)
    for i in range(min(10, env_original.num_arms, env_enhanced.num_arms)):
        diff = env_enhanced.arm_means[i] - env_original.arm_means[i]
        print(f"{i:<5} {env_original.arm_means[i]:<10.4f} {env_enhanced.arm_means[i]:<10.4f} {diff:<10.4f}")


def test_different_dates_enhanced():
    """
    Test enhanced environment with different dates.
    """
    print(f"\n" + "=" * 60)
    print("ENHANCED ENVIRONMENT - DIFFERENT DATES COMPARISON")
    print("=" * 60)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "qurinet")
    
    dates = ["2-May", "9-May", "29-April", "24-May"]
    
    print(f"{'Date':<12} {'Nodes':<6} {'Edges':<6} {'Arms':<6} {'Avg Reward':<10} {'Feasible':<8}")
    print("-" * 60)
    
    for date in dates:
        try:
            env = QurinetEnvironmentEnhanced(
                data_dir=data_dir,
                date=date,
                num_rounds=100,
                pre_generate_availability=True,
                pre_generate_rewards=True,
                rng=np.random.default_rng(42),
                use_all_nodes=True
            )
            
            info = env.get_environment_info()
            available_arms = env.get_available_arms_for_round(0)
            feasible_combinations = env.get_feasible_combinations(available_arms)
            
            print(f"{date:<12} {info['num_nodes']:<6} {info['num_edges']:<6} {info['num_arms']:<6} "
                  f"{info['avg_reward']:<10.3f} {len(feasible_combinations):<8}")
            
        except Exception as e:
            print(f"{date:<12} Error: {e}")


if __name__ == "__main__":
    test_enhanced_qurinet_29_april()
    test_enhanced_qurinet_24_may()
    compare_enhanced_vs_original()
    test_different_dates_enhanced() 