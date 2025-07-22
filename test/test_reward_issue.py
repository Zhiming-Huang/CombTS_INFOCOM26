#!/usr/bin/env python3
"""
Test reward issue in environment.
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from environments.qurinet_environment import QurinetEnvironment

def test_reward_issue():
    """Test if rewards are being returned correctly."""
    
    print("Testing Reward Issue")
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
    
    print(f"Environment arm means:")
    for i in range(min(10, env.num_arms)):
        print(f"  Arm {i}: {env.arm_means[i]:.3f}")
    
    # Test a combination
    combination = {4, 8}
    round_num = 0
    
    print(f"\nTesting combination: {combination}")
    print(f"Expected mean rewards:")
    for arm_id in combination:
        print(f"  Arm {arm_id}: {env.arm_means[arm_id]:.3f}")
    
    # Get rewards
    rewards = env.get_reward_for_round(combination, round_num)
    print(f"\nActual rewards from environment:")
    for arm_id in combination:
        print(f"  Arm {arm_id}: {rewards[arm_id]:.3f}")
    
    # Check if rewards matrix is being used
    if hasattr(env, 'rewards_matrix'):
        print(f"\nRewards matrix shape: {env.rewards_matrix.shape}")
        print(f"Rewards matrix values for combination {combination} at round {round_num}:")
        for arm_id in combination:
            matrix_value = env.rewards_matrix[round_num, arm_id]
            print(f"  Arm {arm_id}: {matrix_value:.3f}")
    
    # Test without pre-generated rewards
    print(f"\nTesting without pre-generated rewards:")
    env2 = QurinetEnvironment(
        data_dir="data/qurinet",
        date="2-May",
        num_rounds=100,
        pre_generate_availability=True,
        pre_generate_rewards=False,  # Don't pre-generate rewards
        availability_rate=1.0
    )
    
    rewards2 = env2.get_reward_for_round(combination, round_num)
    print(f"Rewards without pre-generation:")
    for arm_id in combination:
        print(f"  Arm {arm_id}: {rewards2[arm_id]:.3f}")

if __name__ == "__main__":
    test_reward_issue() 