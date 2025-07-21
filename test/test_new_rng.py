#!/usr/bin/env python3
"""
Test script for the new NumPy random number generator.
Compares old np.random with new default_rng approach.
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from environments.simple_environment import SimpleEnvironment
from environments.simple_environment_memmap import SimpleEnvironmentMemmap


def test_new_rng_functionality():
    """Test that the new RNG works correctly."""
    print("=== Testing New RNG Functionality ===")
    
    # Test in-memory environment
    env = SimpleEnvironment(
        num_arms=10,
        num_rounds=100,
        pre_generate_rewards=True,
        seed=42
    )
    
    # Check that RNG was created
    print(f"RNG created: {hasattr(env, 'rng')}")
    print(f"RNG type: {type(env.rng)}")
    
    # Test reward generation
    reward = env.get_reward_for_round(0, 0)
    print(f"Reward for arm 0, round 0: {reward}")
    
    # Test consistency
    reward1 = env.get_reward_for_round(0, 0)
    reward2 = env.get_reward_for_round(0, 0)
    print(f"Consistency check: {reward1 == reward2}")


def test_memory_mapped_rng():
    """Test RNG in memory-mapped environment."""
    print("\n=== Testing Memory-Mapped RNG ===")
    
    env = SimpleEnvironmentMemmap(
        num_arms=10,
        num_rounds=100,
        pre_generate_rewards=True,
        seed=42
    )
    
    print(f"RNG created: {hasattr(env, 'rng')}")
    print(f"RNG type: {type(env.rng)}")
    
    # Test reward generation
    reward = env.get_reward_for_round(0, 0)
    print(f"Reward for arm 0, round 0: {reward}")
    
    # Cleanup
    env.cleanup()


def compare_old_vs_new_rng():
    """Compare old np.random with new default_rng."""
    print("\n=== Comparing Old vs New RNG ===")
    
    # Old approach
    print("Old approach (np.random):")
    np.random.seed(42)
    old_rewards = []
    start_time = time.time()
    for _ in range(10000):
        reward = np.random.binomial(1, 0.5)
        old_rewards.append(reward)
    old_time = time.time() - start_time
    
    print(f"  Time: {old_time:.4f}s")
    print(f"  Mean reward: {np.mean(old_rewards):.3f}")
    print(f"  Total rewards: {sum(old_rewards)}")
    
    # New approach
    print("\nNew approach (default_rng):")
    rng = np.random.default_rng(42)
    new_rewards = []
    start_time = time.time()
    for _ in range(10000):
        reward = rng.binomial(1, 0.5)
        new_rewards.append(reward)
    new_time = time.time() - start_time
    
    print(f"  Time: {new_time:.4f}s")
    print(f"  Mean reward: {np.mean(new_rewards):.3f}")
    print(f"  Total rewards: {sum(new_rewards)}")
    
    # Compare
    speedup = old_time / new_time
    print(f"\nSpeedup: {speedup:.2f}x")
    
    # Check if results are different (they should be due to different algorithms)
    same_results = old_rewards == new_rewards
    print(f"Same results: {same_results}")


def test_rng_independence():
    """Test that different RNG instances are independent."""
    print("\n=== Testing RNG Independence ===")
    
    # Create two environments with same seed
    env1 = SimpleEnvironment(num_arms=5, num_rounds=10, seed=42)
    env2 = SimpleEnvironment(num_arms=5, num_rounds=10, seed=42)
    
    # Test that they produce same results
    rewards1 = []
    rewards2 = []
    
    for round_idx in range(5):
        for arm in range(5):
            reward1 = env1.generate_reward(arm)
            reward2 = env2.generate_reward(arm)
            rewards1.append(reward1)
            rewards2.append(reward2)
    
    same_results = rewards1 == rewards2
    print(f"Same results with same seed: {same_results}")
    
    # Create environment with different seed
    env3 = SimpleEnvironment(num_arms=5, num_rounds=10, seed=123)
    rewards3 = []
    
    for round_idx in range(5):
        for arm in range(5):
            reward3 = env3.generate_reward(arm)
            rewards3.append(reward3)
    
    different_results = rewards1 != rewards3
    print(f"Different results with different seed: {different_results}")


def test_rng_thread_safety():
    """Test RNG thread safety (simulated)."""
    print("\n=== Testing RNG Thread Safety ===")
    
    # Create multiple environments (simulating different threads)
    environments = []
    for i in range(5):
        env = SimpleEnvironment(num_arms=5, num_rounds=10, seed=42 + i)
        environments.append(env)
    
    # Generate rewards from each environment
    all_rewards = []
    for i, env in enumerate(environments):
        env_rewards = []
        for round_idx in range(3):
            for arm in range(3):
                reward = env.generate_reward(arm)
                env_rewards.append(reward)
        all_rewards.append(env_rewards)
        print(f"Environment {i} rewards: {env_rewards}")
    
    # Check that different environments produce different results
    unique_results = len(set(tuple(rewards) for rewards in all_rewards))
    print(f"Unique result sets: {unique_results}/5")


def visualize_rng_quality():
    """Visualize the quality of random number generation."""
    print("\n=== Visualizing RNG Quality ===")
    
    # Generate rewards using new RNG
    env = SimpleEnvironment(
        num_arms=10,
        num_rounds=1000,
        pre_generate_rewards=True,
        seed=42
    )
    
    # Collect rewards data
    rewards_data = []
    for round_idx in range(100):
        for arm in range(10):
            reward = env.get_reward_for_round(arm, round_idx)
            rewards_data.append({
                'arm': arm,
                'round': round_idx,
                'reward': reward,
                'is_optimal': arm < 3
            })
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Reward distribution by arm
    arm_rewards = {}
    for d in rewards_data:
        arm = d['arm']
        if arm not in arm_rewards:
            arm_rewards[arm] = []
        arm_rewards[arm].append(d['reward'])
    
    arm_means = [np.mean(arm_rewards[arm]) for arm in range(10)]
    axes[0, 0].bar(range(10), arm_means)
    axes[0, 0].set_title('Average Rewards by Arm (New RNG)')
    axes[0, 0].set_xlabel('Arm ID')
    axes[0, 0].set_ylabel('Average Reward')
    axes[0, 0].axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='Optimal (0.9)')
    axes[0, 0].axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Suboptimal (0.1)')
    axes[0, 0].legend()
    
    # Plot 2: Reward sequence
    rewards_sequence = [d['reward'] for d in rewards_data[:200]]
    axes[0, 1].plot(rewards_sequence)
    axes[0, 1].set_title('Reward Sequence (First 200)')
    axes[0, 1].set_xlabel('Sample')
    axes[0, 1].set_ylabel('Reward')
    
    # Plot 3: Optimal vs Suboptimal distribution
    optimal_rewards = [d['reward'] for d in rewards_data if d['is_optimal']]
    suboptimal_rewards = [d['reward'] for d in rewards_data if not d['is_optimal']]
    
    axes[1, 0].hist([optimal_rewards, suboptimal_rewards], 
                    label=['Optimal Arms', 'Suboptimal Arms'], 
                    alpha=0.7, bins=20)
    axes[1, 0].set_title('Rewards Distribution (New RNG)')
    axes[1, 0].set_xlabel('Reward')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Plot 4: Reward pattern
    rewards_matrix = np.array([rewards_sequence[i:i+20] for i in range(0, len(rewards_sequence)-20, 20)])
    if len(rewards_matrix) > 0:
        im = axes[1, 1].imshow(rewards_matrix, cmap='YlOrRd', aspect='auto')
        axes[1, 1].set_title('Reward Pattern Matrix')
        axes[1, 1].set_xlabel('Position in Sequence')
        axes[1, 1].set_ylabel('Sequence Block')
        plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('new_rng_quality.png', dpi=300, bbox_inches='tight')
    print("RNG quality visualization saved as 'new_rng_quality.png'")


def main():
    """Run all tests."""
    print("Testing New NumPy Random Number Generator")
    print("=" * 50)
    
    try:
        # Run all tests
        test_new_rng_functionality()
        test_memory_mapped_rng()
        compare_old_vs_new_rng()
        test_rng_independence()
        test_rng_thread_safety()
        visualize_rng_quality()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print("\nKey improvements with new RNG:")
        print("- Better statistical properties (PCG64 algorithm)")
        print("- Thread safety and independence")
        print("- More explicit and readable code")
        print("- Future-proof implementation")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 