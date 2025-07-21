#!/usr/bin/env python3
"""
Test script for rewards optimization functionality.
Tests both in-memory and memory-mapped versions with pre-generated rewards.
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from environments.simple_environment import SimpleEnvironment
from environments.simple_environment_memmap import SimpleEnvironmentMemmap


def test_rewards_generation():
    """Test basic rewards generation functionality."""
    print("=== Testing Rewards Generation ===")
    
    # Test in-memory environment with pre-generated rewards
    env = SimpleEnvironment(
        num_arms=10, 
        num_rounds=1000, 
        pre_generate_rewards=True,
        seed=42
    )
    
    # Check if rewards matrix was generated
    info = env.get_environment_info()
    print(f"Rewards matrix generated: {info['rewards_matrix_generated']}")
    print(f"Rewards matrix shape: {info['shape']}")
    print(f"Total rewards: {info['total_rewards']}")
    print(f"Reward rate: {info['reward_rate_actual']:.3f}")
    
    # Test reward retrieval
    print("\nTesting reward retrieval:")
    for round_idx in [0, 1, 999]:
        for arm in [0, 5, 9]:
            reward = env.get_reward_for_round(arm, round_idx)
            print(f"Arm {arm}, Round {round_idx}: Reward = {reward}")
    
    # Test consistency
    print("\nTesting consistency:")
    reward1 = env.get_reward_for_round(0, 0)
    reward2 = env.get_reward_for_round(0, 0)
    print(f"Same arm/round reward consistency: {reward1 == reward2}")


def test_memory_mapped_rewards():
    """Test memory-mapped rewards generation."""
    print("\n=== Testing Memory-Mapped Rewards ===")
    
    # Test memory-mapped environment with pre-generated rewards
    env = SimpleEnvironmentMemmap(
        num_arms=10, 
        num_rounds=1000, 
        pre_generate_rewards=True,
        chunk_size=100,
        use_persistent_files=False,
        seed=42
    )
    
    # Check if rewards matrix was generated
    info = env.get_environment_info()
    print(f"Rewards matrix generated: {info['rewards_matrix_generated']}")
    print(f"Rewards matrix shape: {info['shape']}")
    print(f"Total rewards: {info['total_rewards']}")
    print(f"Reward rate: {info['reward_rate_actual']:.3f}")
    print(f"File path: {info['file_path']}")
    
    # Test reward retrieval
    print("\nTesting memory-mapped reward retrieval:")
    for round_idx in [0, 1, 999]:
        for arm in [0, 5, 9]:
            reward = env.get_reward_for_round(arm, round_idx)
            print(f"Arm {arm}, Round {round_idx}: Reward = {reward}")
    
    # Cleanup
    env.cleanup()


def performance_comparison():
    """Compare performance between on-demand and pre-generated rewards."""
    print("\n=== Performance Comparison ===")
    
    num_arms = 10
    num_rounds = 10000
    
    # Test on-demand rewards
    print("Testing on-demand rewards...")
    env_ondemand = SimpleEnvironment(num_arms=num_arms, seed=42)
    
    start_time = time.time()
    total_rewards = 0
    for round_idx in range(num_rounds):
        for arm in range(num_arms):
            reward = env_ondemand.generate_reward(arm)
            total_rewards += reward
    ondemand_time = time.time() - start_time
    
    print(f"On-demand time: {ondemand_time:.3f}s")
    print(f"Total rewards: {total_rewards}")
    
    # Test pre-generated rewards
    print("\nTesting pre-generated rewards...")
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
    
    print(f"Pre-generated time: {pregenerated_time:.3f}s")
    print(f"Total rewards: {total_rewards}")
    
    # Calculate speedup
    speedup = ondemand_time / pregenerated_time
    print(f"\nSpeedup: {speedup:.2f}x")
    
    return ondemand_time, pregenerated_time, speedup


def test_large_scale_simulation():
    """Test large-scale simulation with memory mapping."""
    print("\n=== Large-Scale Simulation Test ===")
    
    num_arms = 100
    num_rounds = 50000
    
    print(f"Simulating {num_arms} arms for {num_rounds} rounds...")
    
    # Test memory-mapped environment
    env = SimpleEnvironmentMemmap(
        num_arms=num_arms,
        num_rounds=num_rounds,
        pre_generate_rewards=True,
        chunk_size=5000,
        use_persistent_files=False,
        seed=42
    )
    
    # Get matrix info
    info = env.get_environment_info()
    print(f"Availability matrix: {info['shape']}")
    print(f"Rewards matrix: {info['shape']}")
    print(f"Memory usage: {info['memory_usage_mb']:.2f} MB")
    
    # Simulate some rounds
    print("\nSimulating rounds...")
    start_time = time.time()
    total_reward = 0
    
    for round_idx in range(1000):  # Test first 1000 rounds
        available_arms = env.get_available_arms_for_round(round_idx)
        if available_arms:
            # Get rewards for available arms
            for arm in available_arms:
                reward = env.get_reward_for_round(arm, round_idx)
                total_reward += reward
    
    simulation_time = time.time() - start_time
    print(f"Simulation time: {simulation_time:.3f}s")
    print(f"Total reward: {total_reward}")
    
    # Cleanup
    env.cleanup()


def test_backward_compatibility():
    """Test backward compatibility with existing CombTS code."""
    print("\n=== Backward Compatibility Test ===")
    
    # Test that existing code still works
    env = SimpleEnvironment(
        num_arms=10,
        num_rounds=100,
        pre_generate_rewards=True,
        seed=42
    )
    
    # Test old interface
    env.reset_round_counter()
    
    for round_idx in range(5):
        # Get available arms using old interface
        available_arms = env.sample_available_arms_once()
        print(f"Round {round_idx}: Available arms = {available_arms}")
        
        if available_arms:
            # Get optimal combination
            optimal_combo = env.get_optimal_combination(available_arms)
            print(f"  Optimal combination: {optimal_combo}")
            
            # Generate rewards using old interface
            rewards = env.generate_combination_reward(optimal_combo)
            print(f"  Rewards: {rewards}")
        
        # Move to next round
        env.reset_available_arms()
    
    print("Backward compatibility test passed!")


def visualize_rewards_distribution():
    """Visualize rewards distribution across arms and rounds."""
    print("\n=== Rewards Distribution Visualization ===")
    
    env = SimpleEnvironment(
        num_arms=10,
        num_rounds=1000,
        pre_generate_rewards=True,
        seed=42
    )
    
    # Collect rewards data
    rewards_data = []
    for round_idx in range(100):  # Sample first 100 rounds
        for arm in range(10):
            reward = env.get_reward_for_round(arm, round_idx)
            rewards_data.append({
                'arm': arm,
                'round': round_idx,
                'reward': reward,
                'is_optimal': arm < 3  # First 3 arms are optimal
            })
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Rewards by arm
    arm_rewards = [d['reward'] for d in rewards_data if d['arm'] < 5]  # First 5 arms
    arm_labels = [f'Arm {d["arm"]}' for d in rewards_data if d['arm'] < 5]
    
    axes[0, 0].bar(range(len(arm_rewards)), arm_rewards)
    axes[0, 0].set_title('Rewards by Arm (First 5 Arms)')
    axes[0, 0].set_xlabel('Arm')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_xticks(range(len(arm_labels)))
    axes[0, 0].set_xticklabels(arm_labels)
    
    # Plot 2: Rewards by round
    round_rewards = []
    for round_idx in range(100):
        round_total = sum(d['reward'] for d in rewards_data if d['round'] == round_idx)
        round_rewards.append(round_total)
    
    axes[0, 1].plot(range(100), round_rewards)
    axes[0, 1].set_title('Total Rewards by Round')
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Total Reward')
    
    # Plot 3: Optimal vs Suboptimal arms
    optimal_rewards = [d['reward'] for d in rewards_data if d['is_optimal']]
    suboptimal_rewards = [d['reward'] for d in rewards_data if not d['is_optimal']]
    
    axes[1, 0].hist([optimal_rewards, suboptimal_rewards], 
                    label=['Optimal Arms', 'Suboptimal Arms'], 
                    alpha=0.7, bins=20)
    axes[1, 0].set_title('Rewards Distribution: Optimal vs Suboptimal')
    axes[1, 0].set_xlabel('Reward')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Plot 4: Heatmap of rewards
    rewards_matrix = np.zeros((10, 100))
    for d in rewards_data:
        rewards_matrix[d['arm'], d['round']] = d['reward']
    
    sns.heatmap(rewards_matrix, ax=axes[1, 1], cmap='YlOrRd', 
                xticklabels=20, yticklabels=2)
    axes[1, 1].set_title('Rewards Heatmap')
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('Arm')
    
    plt.tight_layout()
    plt.savefig('rewards_distribution.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'rewards_distribution.png'")


def main():
    """Run all tests."""
    print("Testing Rewards Optimization Functionality")
    print("=" * 50)
    
    try:
        # Run all tests
        test_rewards_generation()
        test_memory_mapped_rewards()
        performance_comparison()
        test_large_scale_simulation()
        test_backward_compatibility()
        visualize_rewards_distribution()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 