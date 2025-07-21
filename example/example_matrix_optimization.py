#!/usr/bin/env python3
"""
Example script demonstrating matrix optimization for availability arms.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from environments.simple_environment import SimpleEnvironment
from bandits.comb_ts import CombTS
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def demo_matrix_optimization():
    """Demonstrate matrix optimization functionality."""
    print("Matrix Optimization Demo")
    print("=" * 50)
    
    # Create environment with matrix optimization
    num_arms = 20
    num_rounds = 1000
    seed = 42
    
    print(f"Creating environment with {num_arms} arms and {num_rounds} rounds...")
    env = SimpleEnvironment(
        num_arms=num_arms,
        num_rounds=num_rounds,
        seed=seed
    )
    
    # Display environment info
    env_info = env.get_environment_info()
    print(f"\nEnvironment Information:")
    print(f"  Matrix shape: {env_info['shape']}")
    print(f"  Memory usage: {env_info['memory_usage_mb']:.2f} MB")
    print(f"  Actual availability rate: {env_info['availability_rate_actual']:.3f}")
    print(f"  Total available entries: {env_info['total_available']}")
    
    # Initialize algorithm
    algorithm = CombTS(env)
    
    # Run simulation
    print(f"\nRunning simulation...")
    total_reward = 0
    total_regret = 0
    rewards_per_round = []
    regrets_per_round = []
    
    for round_idx in range(num_rounds):
        # Get available arms for this round (direct matrix access)
        available_arms = env.get_available_arms_for_round(round_idx)
        
        # Algorithm selection
        selected_combination = algorithm.select_combination()
        
        # Generate rewards
        rewards = env.generate_combination_reward(selected_combination)
        round_reward = sum(rewards.values())
        total_reward += round_reward
        rewards_per_round.append(round_reward)
        
        # Calculate regret
        optimal_combination = env.get_optimal_combination(available_arms)
        optimal_reward = sum(env.arm_means[arm] for arm in optimal_combination)
        regret = optimal_reward - round_reward
        total_regret += regret
        regrets_per_round.append(regret)
        
        # Update algorithm
        algorithm.update_posterior(selected_combination, rewards)
        
        # Progress report
        if (round_idx + 1) % 200 == 0:
            print(f"  Round {round_idx + 1}: Reward = {total_reward}, Regret = {total_regret:.1f}")
    
    # Final results
    print(f"\nFinal Results:")
    print(f"  Total reward: {total_reward}")
    print(f"  Total regret: {total_regret:.1f}")
    print(f"  Average reward per round: {total_reward / num_rounds:.3f}")
    print(f"  Average regret per round: {total_regret / num_rounds:.3f}")
    
    # Plot results
    plot_results(rewards_per_round, regrets_per_round, num_rounds)
    
    return total_reward, total_regret


def demo_backward_compatibility():
    """Demonstrate backward compatibility."""
    print(f"\n" + "=" * 50)
    print("Backward Compatibility Demo")
    print("=" * 50)
    
    # Test with matrix approach using old interface
    env = SimpleEnvironment(num_arms=10, num_rounds=100, seed=42)
    env.reset_round_counter()
    
    print("Testing old interface with matrix approach:")
    for round_idx in range(5):
        # Old interface
        available_arms_old = env.sample_available_arms_once()
        env.reset_available_arms()
        
        # New interface
        available_arms_new = env.get_available_arms_for_round(round_idx)
        
        print(f"  Round {round_idx}: {available_arms_old} == {available_arms_new}: {available_arms_old == available_arms_new}")
    
    # Test on-demand approach
    env_ondemand = SimpleEnvironment(num_arms=10, seed=42)
    print(f"\nTesting on-demand approach:")
    for _ in range(3):
        available_arms = env_ondemand.sample_available_arms_once()
        env_ondemand.reset_available_arms()
        print(f"  Available arms: {available_arms}")


def demo_memory_efficiency():
    """Demonstrate memory efficiency for different matrix sizes."""
    print(f"\n" + "=" * 50)
    print("Memory Efficiency Demo")
    print("=" * 50)
    
    test_cases = [
        (10, 100),      # Small
        (50, 1000),     # Medium
        (100, 5000),    # Large
        (200, 10000),   # Very large
    ]
    
    print("Matrix size comparison:")
    print(f"{'Arms':<6} {'Rounds':<8} {'Shape':<15} {'Memory (MB)':<12} {'Generation (s)':<15}")
    print("-" * 60)
    
    for num_arms, num_rounds in test_cases:
        import time
        start_time = time.time()
        env = SimpleEnvironment(num_arms=num_arms, num_rounds=num_rounds, seed=42)
        generation_time = time.time() - start_time
        
        matrix_info = env.get_availability_matrix_info()
        
        print(f"{num_arms:<6} {num_rounds:<8} {str(matrix_info['shape']):<15} "
              f"{matrix_info['memory_usage_mb']:<12.2f} {generation_time:<15.4f}")


def plot_results(rewards_per_round, regrets_per_round, num_rounds):
    """Plot simulation results."""
    plt.figure(figsize=(15, 10))
    
    # Plot cumulative rewards
    plt.subplot(2, 3, 1)
    cumulative_rewards = np.cumsum(rewards_per_round)
    plt.plot(cumulative_rewards)
    plt.title('Cumulative Rewards')
    plt.xlabel('Round')
    plt.ylabel('Cumulative Reward')
    
    # Plot cumulative regrets
    plt.subplot(2, 3, 2)
    cumulative_regrets = np.cumsum(regrets_per_round)
    plt.plot(cumulative_regrets)
    plt.title('Cumulative Regrets')
    plt.xlabel('Round')
    plt.ylabel('Cumulative Regret')
    
    # Plot rewards per round
    plt.subplot(2, 3, 3)
    plt.plot(rewards_per_round[:100])  # First 100 rounds
    plt.title('Rewards per Round (First 100)')
    plt.xlabel('Round')
    plt.ylabel('Reward')
    
    # Plot regrets per round
    plt.subplot(2, 3, 4)
    plt.plot(regrets_per_round[:100])  # First 100 rounds
    plt.title('Regrets per Round (First 100)')
    plt.xlabel('Round')
    plt.ylabel('Regret')
    
    # Plot moving average rewards
    plt.subplot(2, 3, 5)
    window_size = 50
    moving_avg_rewards = np.convolve(rewards_per_round, np.ones(window_size)/window_size, mode='valid')
    plt.plot(moving_avg_rewards)
    plt.title(f'Moving Average Rewards (Window={window_size})')
    plt.xlabel('Round')
    plt.ylabel('Average Reward')
    
    # Plot moving average regrets
    plt.subplot(2, 3, 6)
    moving_avg_regrets = np.convolve(regrets_per_round, np.ones(window_size)/window_size, mode='valid')
    plt.plot(moving_avg_regrets)
    plt.title(f'Moving Average Regrets (Window={window_size})')
    plt.xlabel('Round')
    plt.ylabel('Average Regret')
    
    plt.tight_layout()
    plt.savefig('output/matrix_optimization_results.png', dpi=300, bbox_inches='tight')
    print("Results plot saved to output/matrix_optimization_results.png")


def demo_advanced_features():
    """Demonstrate advanced features."""
    print(f"\n" + "=" * 50)
    print("Advanced Features Demo")
    print("=" * 50)
    
    # Test with different seeds for reproducibility
    print("Testing reproducibility with different seeds:")
    for seed in [42, 123, 456]:
        env = SimpleEnvironment(num_arms=10, num_rounds=50, seed=seed)
        available_arms_round_0 = env.get_available_arms_for_round(0)
        print(f"  Seed {seed}: Round 0 arms = {available_arms_round_0}")
    
    # Test matrix statistics
    env = SimpleEnvironment(num_arms=20, num_rounds=100, seed=42)
    matrix_info = env.get_availability_matrix_info()
    
    print(f"\nMatrix statistics:")
    print(f"  Shape: {matrix_info['shape']}")
    print(f"  Total available: {matrix_info['total_available']}")
    print(f"  Availability rate: {matrix_info['availability_rate_actual']:.3f}")
    print(f"  Memory usage: {matrix_info['memory_usage_mb']:.2f} MB")
    
    # Test random access
    print(f"\nRandom access test:")
    test_rounds = [0, 25, 50, 75, 99]
    for round_idx in test_rounds:
        available_arms = env.get_available_arms_for_round(round_idx)
        print(f"  Round {round_idx}: {len(available_arms)} arms available")


if __name__ == "__main__":
    # Run all demos
    demo_matrix_optimization()
    demo_backward_compatibility()
    demo_memory_efficiency()
    demo_advanced_features()
    
    print(f"\n" + "=" * 50)
    print("✓ Matrix optimization demo completed successfully!")
    print("Key benefits:")
    print("  - Fast random access to any round's available arms")
    print("  - Memory efficient for large simulations")
    print("  - Reproducible results with seed")
    print("  - Backward compatible with existing code")
    print("  - No need for reset_available_arms() calls") 