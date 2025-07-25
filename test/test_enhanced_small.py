#!/usr/bin/env python3
"""
Test script for the enhanced QurinetEnvironment with smaller network.
"""

import sys
import os
import numpy as np

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.environments.qurinet_environment_enhanced import QurinetEnvironmentEnhanced


def test_enhanced_small():
    """Test enhanced environment with smaller network."""
    print("Testing Enhanced QurinetEnvironment (Small Network)")
    print("=" * 60)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "qurinet")
    
    try:
        print("Creating enhanced environment with 2-May data (smaller network)...")
        
        # Use 2-May data which should be smaller
        env = QurinetEnvironmentEnhanced(
            data_dir=data_dir,
            date="2-May",  # Use smaller network
            num_rounds=100,
            pre_generate_availability=True,
            pre_generate_rewards=True,
            rng=np.random.default_rng(42),
            use_all_nodes=True
        )
        
        print("Environment created successfully!")
        
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
        
        # Test getting feasible combinations
        print(f"\nTesting feasible combinations...")
        available_arms = env.get_available_arms_for_round(0)
        feasible_combinations = env.get_feasible_combinations(available_arms)
        
        print(f"Feasible combinations: {len(feasible_combinations)}")
        
        # Show first few combinations
        for i, comb in enumerate(feasible_combinations[:5]):
            expected_reward = sum(env.arm_means[arm] for arm in comb)
            path_info = env.get_path_info(comb)
            print(f"  {i+1}: {comb} (expected reward: {expected_reward:.3f}, path: {path_info['path']})")
        
        if len(feasible_combinations) > 5:
            print(f"  ... and {len(feasible_combinations) - 5} more combinations")
        
        # Test algorithm performance
        print(f"\n" + "=" * 40)
        print("ALGORITHM PERFORMANCE TEST")
        print("=" * 40)
        
        from src.bandits.cts_g import CTSG
        from src.bandits.cts_b import CTSB
        
        # Test CTS-G
        cts_g = CTSG(env, gamma=0.01, rng=np.random.default_rng(42))
        
        # Run for 10 rounds
        total_reward = 0
        selections = []
        for round_idx in range(10):
            selection = cts_g.select_combination(round_idx)
            selections.append(selection)
            rewards = env.get_reward_for_round(selection, round_idx)
            total_reward += sum(rewards.values())
            cts_g.update_posterior(selection, rewards, round_idx)
        
        print(f"CTS-G total reward (10 rounds): {total_reward:.3f}")
        
        # Count selections
        from collections import Counter
        selection_counter = Counter(tuple(sorted(s)) for s in selections)
        print(f"CTS-G selections:")
        for selection, count in selection_counter.most_common():
            print(f"  {selection}: {count} times")
        
        # Test CTS-B
        cts_b = CTSB(env, rng=np.random.default_rng(42))
        
        # Run for 10 rounds
        total_reward = 0
        selections = []
        for round_idx in range(10):
            selection = cts_b.select_combination(round_idx)
            selections.append(selection)
            rewards = env.get_reward_for_round(selection, round_idx)
            total_reward += sum(rewards.values())
            cts_b.update_posterior(selection, rewards, round_idx)
        
        print(f"\nCTS-B total reward (10 rounds): {total_reward:.3f}")
        
        # Count selections
        selection_counter = Counter(tuple(sorted(s)) for s in selections)
        print(f"CTS-B selections:")
        for selection, count in selection_counter.most_common():
            print(f"  {selection}: {count} times")
        
        print(f"\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_enhanced_small() 