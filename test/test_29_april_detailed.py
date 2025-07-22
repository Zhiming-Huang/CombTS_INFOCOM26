#!/usr/bin/env python3
"""
Detailed analysis of 29-April Qurinet data.
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, Any

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.environments.qurinet_environment import QurinetEnvironment


def analyze_29_april_data():
    """
    Analyze the 29-April Qurinet data in detail.
    """
    print("29-April Qurinet Data Analysis")
    print("=" * 60)
    
    # Load the CSV data directly
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "qurinet")
    
    # Read the sites file
    sites_file = os.path.join(data_dir, "sites_29-April.csv")
    sites_df = pd.read_csv(sites_file)
    
    print(f"29-April sites data:")
    print(f"Number of sites: {len(sites_df)}")
    print(f"Columns: {list(sites_df.columns)}")
    print()
    
    print("Sites data:")
    print(sites_df.to_string(index=False))
    print()
    
    # Analyze the data
    print("Data Analysis:")
    print(f"Unique adhoc0 values: {sorted(sites_df['adhoc0'].unique())}")
    print(f"Unique adhoc1 values: {sorted(sites_df['adhoc1'].unique())}")
    print()
    
    # Count sites by adhoc configuration
    print("Sites by adhoc0 configuration:")
    adhoc0_counts = sites_df['adhoc0'].value_counts().sort_index()
    for adhoc, count in adhoc0_counts.items():
        print(f"  adhoc0={adhoc}: {count} sites")
    print()
    
    print("Sites by adhoc1 configuration:")
    adhoc1_counts = sites_df['adhoc1'].value_counts().sort_index()
    for adhoc, count in adhoc1_counts.items():
        print(f"  adhoc1={adhoc}: {count} sites")
    print()
    
    # Show sites with different configurations
    print("Sites with adhoc0=6 (different from others):")
    adhoc0_6_sites = sites_df[sites_df['adhoc0'] == 6]
    print(adhoc0_6_sites.to_string(index=False))
    print()
    
    # Compare with 2-May data
    print("Comparison with 2-May data:")
    sites_2may_file = os.path.join(data_dir, "sites.csv")
    sites_2may_df = pd.read_csv(sites_2may_file)
    
    print(f"2-May sites: {len(sites_2may_df)}")
    print(f"29-April sites: {len(sites_df)}")
    print(f"Additional sites in 29-April: {len(sites_df) - len(sites_2may_df)}")
    print()
    
    # Find new sites in 29-April
    sites_2may_set = set(sites_2may_df['site'])
    sites_29april_set = set(sites_df['site'])
    
    new_sites = sites_29april_set - sites_2may_set
    print(f"New sites in 29-April: {sorted(new_sites)}")
    print()
    
    # Show new sites data
    if new_sites:
        print("New sites data:")
        new_sites_data = sites_df[sites_df['site'].isin(new_sites)]
        print(new_sites_data.to_string(index=False))
        print()


def test_29_april_environment():
    """
    Test the 29-April environment specifically.
    """
    print(f"\n" + "=" * 60)
    print("TESTING 29-APRIL ENVIRONMENT")
    print("=" * 60)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "qurinet")
    
    try:
        env = QurinetEnvironment(
            data_dir=data_dir,
            date="29-April",
            num_rounds=100,
            pre_generate_availability=True,
            pre_generate_rewards=True,
            rng=np.random.default_rng(42)
        )
        
        env.availability_rate = 0.9
        
        # Get environment info
        info = env.get_environment_info()
        
        print(f"Environment created successfully!")
        print(f"Network: {info['num_nodes']} nodes, {info['num_edges']} edges")
        print(f"Source: {info['source']}, Destination: {info['destination']}")
        print(f"Arms: {info['num_arms']}")
        print(f"Average reward: {info['avg_reward']:.3f}")
        print(f"Average availability: {info['avg_availability']:.3f}")
        
        # Get feasible combinations
        available_arms = env.get_available_arms_for_round(0)
        feasible_combinations = env.get_feasible_combinations(available_arms)
        
        print(f"\nFeasible combinations: {len(feasible_combinations)}")
        
        # Show all combinations
        for i, comb in enumerate(feasible_combinations):
            expected_reward = sum(env.arm_means[arm] for arm in comb)
            print(f"  {i+1}: {comb} (expected reward: {expected_reward:.3f})")
        
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
        print(f"Error testing 29-April environment: {e}")
        import traceback
        traceback.print_exc()


def compare_with_2may():
    """
    Compare 29-April with 2-May environments.
    """
    print(f"\n" + "=" * 60)
    print("COMPARISON: 29-APRIL vs 2-MAY")
    print("=" * 60)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "qurinet")
    
    # Create both environments
    env_2may = QurinetEnvironment(
        data_dir=data_dir,
        date="2-May",
        num_rounds=100,
        pre_generate_availability=True,
        pre_generate_rewards=True,
        rng=np.random.default_rng(42)
    )
    
    env_29april = QurinetEnvironment(
        data_dir=data_dir,
        date="29-April",
        num_rounds=100,
        pre_generate_availability=True,
        pre_generate_rewards=True,
        rng=np.random.default_rng(42)
    )
    
    # Set availability rates
    env_2may.availability_rate = 0.9
    env_29april.availability_rate = 0.9
    
    # Get info
    info_2may = env_2may.get_environment_info()
    info_29april = env_29april.get_environment_info()
    
    print(f"{'Metric':<20} {'2-May':<15} {'29-April':<15} {'Difference':<15}")
    print("-" * 70)
    print(f"{'Nodes':<20} {info_2may['num_nodes']:<15} {info_29april['num_nodes']:<15} {info_29april['num_nodes'] - info_2may['num_nodes']:<15}")
    print(f"{'Edges':<20} {info_2may['num_edges']:<15} {info_29april['num_edges']:<15} {info_29april['num_edges'] - info_2may['num_edges']:<15}")
    print(f"{'Arms':<20} {info_2may['num_arms']:<15} {info_29april['num_arms']:<15} {info_29april['num_arms'] - info_2may['num_arms']:<15}")
    print(f"{'Avg Reward':<20} {info_2may['avg_reward']:<15.3f} {info_29april['avg_reward']:<15.3f} {info_29april['avg_reward'] - info_2may['avg_reward']:<15.3f}")
    print(f"{'Avg Availability':<20} {info_2may['avg_availability']:<15.3f} {info_29april['avg_availability']:<15.3f} {info_29april['avg_availability'] - info_2may['avg_availability']:<15.3f}")
    
    # Compare feasible combinations
    available_arms_2may = env_2may.get_available_arms_for_round(0)
    feasible_2may = env_2may.get_feasible_combinations(available_arms_2may)
    
    available_arms_29april = env_29april.get_available_arms_for_round(0)
    feasible_29april = env_29april.get_feasible_combinations(available_arms_29april)
    
    print(f"{'Feasible Combos':<20} {len(feasible_2may):<15} {len(feasible_29april):<15} {len(feasible_29april) - len(feasible_2may):<15}")
    
    # Compare arm means
    print(f"\nArm means comparison (first 10):")
    print(f"{'Arm':<5} {'2-May':<10} {'29-April':<10} {'Diff':<10}")
    print("-" * 35)
    for i in range(min(10, env_2may.num_arms, env_29april.num_arms)):
        diff = env_29april.arm_means[i] - env_2may.arm_means[i]
        print(f"{i:<5} {env_2may.arm_means[i]:<10.4f} {env_29april.arm_means[i]:<10.4f} {diff:<10.4f}")


if __name__ == "__main__":
    analyze_29_april_data()
    test_29_april_environment()
    compare_with_2may() 