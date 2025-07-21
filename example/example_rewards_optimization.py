#!/usr/bin/env python3
"""
Example script demonstrating rewards optimization functionality.

This script shows how to use pre-generated rewards for improved performance
in both in-memory and memory-mapped environments.
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from environments.simple_environment import SimpleEnvironment
from environments.simple_environment_memmap import SimpleEnvironmentMemmap


def example_basic_usage():
    """Demonstrate basic usage of pre-generated rewards."""
    print("=== Basic Usage Example ===")
    
    # Create environment with pre-generated rewards
    env = SimpleEnvironment(
        num_arms=10,
        num_rounds=1000,
        pre_generate_rewards=True,  # Enable pre-generated rewards
        seed=42
    )
    
    # Get environment info
    info = env.get_environment_info()
    print(f"Environment configured with {info['num_arms']} arms for {env.num_rounds} rounds")
    print(f"Rewards matrix generated: {info['rewards_matrix_generated']}")
    print(f"Rewards matrix shape: {info['shape']}")
    print(f"Memory usage: {info['memory_usage_mb']:.2f} MB")
    
    # Simulate a few rounds
    print("\nSimulating rounds with pre-generated rewards:")
    for round_idx in range(5):
        # Get available arms for this round
        available_arms = env.get_available_arms_for_round(round_idx)
        print(f"Round {round_idx}: Available arms = {available_arms}")
        
        if available_arms:
            # Get optimal combination
            optimal_combo = env.get_optimal_combination(available_arms)
            print(f"  Optimal combination: {optimal_combo}")
            
            # Get rewards for optimal combination
            total_reward = 0
            for arm in optimal_combo:
                reward = env.get_reward_for_round(arm, round_idx)
                total_reward += reward
                print(f"    Arm {arm}: Reward = {reward}")
            print(f"  Total reward: {total_reward}")


def example_memory_mapped_usage():
    """Demonstrate memory-mapped rewards for large-scale simulations."""
    print("\n=== Memory-Mapped Usage Example ===")
    
    # Create memory-mapped environment for large simulation
    env = SimpleEnvironmentMemmap(
        num_arms=50,
        num_rounds=10000,
        pre_generate_rewards=True,
        chunk_size=1000,  # Generate in chunks
        use_persistent_files=False,  # Use temporary files
        seed=42
    )
    
    # Get environment info
    info = env.get_environment_info()
    print(f"Large-scale environment: {info['num_arms']} arms × {env.num_rounds} rounds")
    print(f"Rewards matrix generated: {info['rewards_matrix_generated']}")
    print(f"Memory usage: {info['memory_usage_mb']:.2f} MB")
    print(f"File path: {info['file_path']}")
    
    # Simulate some rounds
    print("\nSimulating rounds with memory-mapped rewards:")
    total_reward = 0
    start_time = time.time()
    
    for round_idx in range(100):  # Simulate 100 rounds
        available_arms = env.get_available_arms_for_round(round_idx)
        if available_arms:
            # Get optimal combination
            optimal_combo = env.get_optimal_combination(available_arms)
            
            # Get rewards for optimal combination
            round_reward = 0
            for arm in optimal_combo:
                reward = env.get_reward_for_round(arm, round_idx)
                round_reward += reward
            
            total_reward += round_reward
            
            if round_idx % 20 == 0:  # Print every 20th round
                print(f"Round {round_idx}: {len(available_arms)} available, "
                      f"optimal combo {optimal_combo}, reward = {round_reward}")
    
    simulation_time = time.time() - start_time
    print(f"\nSimulation completed in {simulation_time:.3f}s")
    print(f"Total reward: {total_reward}")
    
    # Cleanup
    env.cleanup()


def example_performance_comparison():
    """Compare performance between different approaches."""
    print("\n=== Performance Comparison Example ===")
    
    num_arms = 20
    num_rounds = 5000
    
    # Test 1: On-demand rewards
    print("1. Testing on-demand rewards...")
    env_ondemand = SimpleEnvironment(num_arms=num_arms, seed=42)
    
    start_time = time.time()
    total_rewards = 0
    for round_idx in range(num_rounds):
        for arm in range(num_arms):
            reward = env_ondemand.generate_reward(arm)
            total_rewards += reward
    ondemand_time = time.time() - start_time
    
    print(f"   Time: {ondemand_time:.3f}s, Total rewards: {total_rewards}")
    
    # Test 2: Pre-generated rewards (in-memory)
    print("2. Testing pre-generated rewards (in-memory)...")
    env_pregenerated = SimpleEnvironment(
        num_arms=num_arms,
        num_rounds=num_rounds,
        pre_generate_rewards=True,
        seed=42
    )
    
    start_time = time.time()
    total_rewards = 0
    for round_idx in range(num_rounds):
        for arm in range(num_arms):
            reward = env_pregenerated.get_reward_for_round(arm, round_idx)
            total_rewards += reward
    pregenerated_time = time.time() - start_time
    
    print(f"   Time: {pregenerated_time:.3f}s, Total rewards: {total_rewards}")
    
    # Test 3: Memory-mapped rewards
    print("3. Testing memory-mapped rewards...")
    env_memmap = SimpleEnvironmentMemmap(
        num_arms=num_arms,
        num_rounds=num_rounds,
        pre_generate_rewards=True,
        chunk_size=1000,
        use_persistent_files=False,
        seed=42
    )
    
    start_time = time.time()
    total_rewards = 0
    for round_idx in range(num_rounds):
        for arm in range(num_arms):
            reward = env_memmap.get_reward_for_round(arm, round_idx)
            total_rewards += reward
    memmap_time = time.time() - start_time
    
    print(f"   Time: {memmap_time:.3f}s, Total rewards: {total_rewards}")
    
    # Calculate speedups
    speedup_pregenerated = ondemand_time / pregenerated_time
    speedup_memmap = ondemand_time / memmap_time
    
    print(f"\nPerformance Summary:")
    print(f"  On-demand vs Pre-generated speedup: {speedup_pregenerated:.2f}x")
    print(f"  On-demand vs Memory-mapped speedup: {speedup_memmap:.2f}x")
    
    # Cleanup
    env_memmap.cleanup()


def example_visualization():
    """Create visualizations of rewards patterns."""
    print("\n=== Rewards Visualization Example ===")
    
    # Create environment with pre-generated rewards
    env = SimpleEnvironment(
        num_arms=10,
        num_rounds=500,
        pre_generate_rewards=True,
        seed=42
    )
    
    # Collect data for visualization
    print("Collecting rewards data...")
    rewards_by_arm = {arm: [] for arm in range(10)}
    rewards_by_round = []
    
    for round_idx in range(500):
        round_rewards = []
        for arm in range(10):
            reward = env.get_reward_for_round(arm, round_idx)
            rewards_by_arm[arm].append(reward)
            round_rewards.append(reward)
        rewards_by_round.append(sum(round_rewards))
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Average rewards by arm
    arm_means = [np.mean(rewards_by_arm[arm]) for arm in range(10)]
    axes[0, 0].bar(range(10), arm_means)
    axes[0, 0].set_title('Average Rewards by Arm')
    axes[0, 0].set_xlabel('Arm ID')
    axes[0, 0].set_ylabel('Average Reward')
    axes[0, 0].axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='Optimal (0.9)')
    axes[0, 0].axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Suboptimal (0.1)')
    axes[0, 0].legend()
    
    # Plot 2: Total rewards by round
    axes[0, 1].plot(range(500), rewards_by_round)
    axes[0, 1].set_title('Total Rewards by Round')
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Total Reward')
    
    # Plot 3: Rewards distribution for optimal vs suboptimal arms
    optimal_rewards = []
    suboptimal_rewards = []
    
    for arm in range(3):  # First 3 arms are optimal
        optimal_rewards.extend(rewards_by_arm[arm])
    for arm in range(3, 10):  # Remaining arms are suboptimal
        suboptimal_rewards.extend(rewards_by_arm[arm])
    
    axes[1, 0].hist([optimal_rewards, suboptimal_rewards], 
                    label=['Optimal Arms', 'Suboptimal Arms'], 
                    alpha=0.7, bins=20)
    axes[1, 0].set_title('Rewards Distribution')
    axes[1, 0].set_xlabel('Reward')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Plot 4: Heatmap of rewards (first 50 rounds, all arms)
    rewards_matrix = np.array([rewards_by_arm[arm][:50] for arm in range(10)])
    im = axes[1, 1].imshow(rewards_matrix, cmap='YlOrRd', aspect='auto')
    axes[1, 1].set_title('Rewards Heatmap (First 50 Rounds)')
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('Arm')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('rewards_optimization_example.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'rewards_optimization_example.png'")


def example_backward_compatibility():
    """Demonstrate backward compatibility with existing code."""
    print("\n=== Backward Compatibility Example ===")
    
    # Create environment with pre-generated rewards
    env = SimpleEnvironment(
        num_arms=10,
        num_rounds=100,
        pre_generate_rewards=True,
        seed=42
    )
    
    # Use the old interface (should still work)
    print("Using old interface with pre-generated rewards:")
    env.reset_round_counter()
    
    for round_idx in range(5):
        # Old interface methods
        available_arms = env.sample_available_arms_once()
        print(f"Round {round_idx}: Available arms = {available_arms}")
        
        if available_arms:
            optimal_combo = env.get_optimal_combination(available_arms)
            rewards = env.generate_combination_reward(optimal_combo)
            total_reward = sum(rewards.values())
            print(f"  Optimal combo: {optimal_combo}")
            print(f"  Rewards: {rewards}")
            print(f"  Total reward: {total_reward}")
        
        env.reset_available_arms()
    
    print("Backward compatibility confirmed!")


def main():
    """Run all examples."""
    print("Rewards Optimization Examples")
    print("=" * 50)
    
    try:
        # Run all examples
        example_basic_usage()
        example_memory_mapped_usage()
        example_performance_comparison()
        example_visualization()
        example_backward_compatibility()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nKey benefits of pre-generated rewards:")
        print("- Improved performance through reduced random number generation")
        print("- Consistent rewards across multiple simulations")
        print("- Memory-efficient storage with memory mapping")
        print("- Backward compatibility with existing code")
        
    except Exception as e:
        print(f"\nError during examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 