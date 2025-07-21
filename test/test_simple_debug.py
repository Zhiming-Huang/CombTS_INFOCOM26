#!/usr/bin/env python3
"""
Debug script to test basic functionality of CombTS with simple environment.
"""

import sys
import os
sys.path.append('.')

import numpy as np
from src.bandits.comb_ts import CombTS
from src.environments.simple_environment import SimpleEnvironment


def test_basic_functionality():
    """Test basic functionality of CombTS with simple environment."""
    print("Testing basic functionality...")
    
    # Create environment
    env = SimpleEnvironment(
        num_arms=10,
        num_optimal=3,
        optimal_mean=0.9,
        suboptimal_mean=0.1,
        availability_rate=0.5,
        max_combination_size=3
    )
    
    # Create algorithm
    algorithm = CombTS(environment=env, alpha=1.0, beta=1.0)
    
    print(f"Environment info: {env.get_environment_info()}")
    print(f"Algorithm num_arms: {algorithm.num_arms}")
    
    # Test a few rounds
    for round_num in range(5):
        print(f"\nRound {round_num + 1}:")
        
        # Reset available arms for this round
        env.reset_available_arms()
        
        # Select combination (this will use the consistent available arms)
        selected_combination = algorithm.select_combination()
        
        # Get the available arms that were used by the algorithm
        available_arms = env.sample_available_arms_once()
        print(f"  Available arms: {available_arms}")
        
        # Get feasible combinations
        feasible_combinations = env.get_feasible_combinations(available_arms)
        print(f"  Number of feasible combinations: {len(feasible_combinations)}")
        
        print(f"  Selected combination: {selected_combination}")
        
        # Generate rewards
        rewards = {}
        total_reward = 0
        if selected_combination:
            rewards = env.generate_combination_reward(selected_combination)
            total_reward = sum(rewards.values())
        
        print(f"  Rewards: {rewards}")
        print(f"  Total reward: {total_reward}")
        
        # Update algorithm
        algorithm.update_posterior(selected_combination, rewards)
        
        # Get optimal combination for comparison
        optimal_combination = env.get_optimal_combination(available_arms)
        optimal_reward = sum(env.arm_means[arm] for arm in optimal_combination)
        regret = optimal_reward - total_reward
        
        print(f"  Optimal combination: {optimal_combination}")
        print(f"  Optimal reward: {optimal_reward}")
        print(f"  Regret: {regret}")
    
    # Show final statistics
    stats = algorithm.get_arm_statistics()
    print(f"\nFinal arm statistics:")
    print(f"  Number of arms played: {stats['played_arms']}")
    print(f"  Expected rewards: {stats['expected_rewards']}")


if __name__ == "__main__":
    test_basic_functionality() 