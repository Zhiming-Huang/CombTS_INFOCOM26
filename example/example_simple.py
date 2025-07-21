#!/usr/bin/env python3
"""
Simple example demonstrating the new CombTS interface with environment.
"""

import sys
import os
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt

from src.bandits.comb_ts import CombTS
from src.environments.simple_environment import SimpleEnvironment


def main():
    """Demonstrate the new CombTS interface."""
    print("CombTS with Environment Interface Demo")
    print("=" * 40)
    
    # Create simple environment
    env = SimpleEnvironment(
        num_arms=10,
        num_optimal=3,
        optimal_mean=0.9,
        suboptimal_mean=0.1,
        availability_rate=0.5,
        max_combination_size=3
    )
    
    # Create CombTS algorithm with environment
    algorithm = CombTS(environment=env, alpha=1.0, beta=1.0)
    
    print(f"Environment: {env.num_arms} arms, {env.num_optimal} optimal")
    print(f"Algorithm: {algorithm.num_arms} arms")
    
    # Run simulation
    print("\nRunning simulation...")
    cumulative_rewards = []
    cumulative_regrets = []
    total_reward = 0
    total_regret = 0
    
    for round_num in range(1000):
        # Reset available arms for this round
        env.reset_available_arms()
        
        # Select combination (this will use the consistent available arms)
        selected_combination = algorithm.select_combination()
        
        # Generate rewards
        rewards = {}
        total_round_reward = 0
        if selected_combination:
            rewards = env.generate_combination_reward(selected_combination)
            total_round_reward = sum(rewards.values())
        
        # Update algorithm
        algorithm.update_posterior(selected_combination, rewards)
        
        # Calculate regret using the same available arms that were used by the algorithm
        available_arms = env.sample_available_arms_once()
        optimal_combination = env.get_optimal_combination(available_arms)
        optimal_reward = sum(env.arm_means[arm] for arm in optimal_combination)
        regret = optimal_reward - total_round_reward
        
        # Update cumulative values
        total_reward += total_round_reward
        total_regret += regret
        
        cumulative_rewards.append(total_reward)
        cumulative_regrets.append(total_regret)
        
        if round_num % 200 == 0 and round_num > 0:
            print(f"Round {round_num}: Reward = {total_reward:.1f}, Regret = {total_regret:.1f}")
    
    # Show final results
    print(f"\nFinal Results:")
    print(f"Total reward: {total_reward:.1f}")
    print(f"Total regret: {total_regret:.1f}")
    print(f"Average reward per round: {total_reward / 1000:.3f}")
    print(f"Average regret per round: {total_regret / 1000:.3f}")
    
    # Show arm statistics
    stats = algorithm.get_arm_statistics()
    print(f"\nArm Statistics:")
    print(f"Arms played: {stats['played_arms']}/{stats['num_arms']}")
    
    print("\nExpected rewards vs true means:")
    for arm in range(env.num_arms):
        true_mean = env.arm_means[arm]
        expected_reward = stats['expected_rewards'].get(arm, "Not played")
        print(f"  Arm {arm}: Expected = {expected_reward}, True = {true_mean:.1f}")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(cumulative_rewards)
    plt.title('Cumulative Rewards')
    plt.xlabel('Round')
    plt.ylabel('Cumulative Reward')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(cumulative_regrets)
    plt.title('Cumulative Regrets')
    plt.xlabel('Round')
    plt.ylabel('Cumulative Regret')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('output/simple_example_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nDemo completed! Check output/simple_example_results.png for plots.")


if __name__ == "__main__":
    main() 