#!/usr/bin/env python3
"""
Test script for matrix optimization functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environments.simple_environment import SimpleEnvironment
from src.bandits.comb_ts import CombTS
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time


def test_matrix_generation():
    """Test the matrix generation functionality."""
    print("Testing Matrix Generation")
    print("=" * 50)
    
    # Test with different sizes
    test_cases = [
        (10, 100),    # Small
        (50, 1000),   # Medium
        (100, 5000),  # Large
    ]
    
    for num_arms, num_rounds in test_cases:
        print(f"\nTesting {num_arms} arms, {num_rounds} rounds:")
        
        # Time matrix generation
        start_time = time.time()
        env = SimpleEnvironment(num_arms=num_arms, num_rounds=num_rounds, seed=42)
        generation_time = time.time() - start_time
        
        # Get matrix info
        matrix_info = env.get_availability_matrix_info()
        
        print(f"  Generation time: {generation_time:.4f}s")
        print(f"  Matrix shape: {matrix_info['shape']}")
        print(f"  Memory usage: {matrix_info['memory_usage_mb']:.2f} MB")
        print(f"  Actual availability rate: {matrix_info['availability_rate_actual']:.3f}")
        
        # Test accessing specific rounds
        test_rounds = [0, num_rounds//2, num_rounds-1]
        for round_idx in test_rounds:
            available_arms = env.get_available_arms_for_round(round_idx)
            print(f"  Round {round_idx}: {len(available_arms)} available arms")


def test_performance_comparison():
    """Compare performance between matrix and on-demand approaches."""
    print("\n" + "=" * 50)
    print("Performance Comparison")
    print("=" * 50)
    
    num_arms, num_rounds = 50, 1000
    
    # Test matrix approach
    print("Testing matrix approach...")
    start_time = time.time()
    env_matrix = SimpleEnvironment(num_arms=num_arms, num_rounds=num_rounds, seed=42)
    matrix_init_time = time.time() - start_time
    
    # Test on-demand approach
    print("Testing on-demand approach...")
    start_time = time.time()
    env_ondemand = SimpleEnvironment(num_arms=num_arms, seed=42)
    ondemand_init_time = time.time() - start_time
    
    print(f"\nInitialization times:")
    print(f"  Matrix approach: {matrix_init_time:.4f}s")
    print(f"  On-demand approach: {ondemand_init_time:.4f}s")
    
    # Test access times
    print(f"\nAccess time comparison (1000 rounds):")
    
    # Matrix approach access time
    start_time = time.time()
    for round_idx in range(1000):
        available_arms = env_matrix.get_available_arms_for_round(round_idx)
    matrix_access_time = time.time() - start_time
    
    # On-demand approach access time
    start_time = time.time()
    for _ in range(1000):
        available_arms = env_ondemand.get_available_arms()
    ondemand_access_time = time.time() - start_time
    
    print(f"  Matrix access time: {matrix_access_time:.4f}s")
    print(f"  On-demand access time: {ondemand_access_time:.4f}s")
    print(f"  Speedup: {ondemand_access_time / matrix_access_time:.2f}x")


def test_algorithm_integration():
    """Test CombTS integration with matrix approach."""
    print("\n" + "=" * 50)
    print("Algorithm Integration Test")
    print("=" * 50)
    
    # Create environment with matrix
    env = SimpleEnvironment(num_arms=20, num_rounds=500, seed=42)
    algorithm = CombTS(env)
    
    total_reward = 0
    total_regret = 0
    
    print("Running algorithm with matrix approach...")
    for round_idx in range(100):
        # Get available arms for this round
        available_arms = env.get_available_arms_for_round(round_idx)
        
        # Algorithm selection
        selected_combination = algorithm.select_combination()
        
        # Generate rewards
        rewards = env.generate_combination_reward(selected_combination)
        total_reward += sum(rewards.values())
        
        # Calculate regret
        optimal_combination = env.get_optimal_combination(available_arms)
        optimal_reward = sum(env.arm_means[arm] for arm in optimal_combination)
        actual_reward = sum(rewards.values())
        regret = optimal_reward - actual_reward
        total_regret += regret
        
        # Update algorithm
        algorithm.update_posterior(selected_combination, rewards)
        
        if (round_idx + 1) % 20 == 0:
            print(f"  Round {round_idx + 1}: Reward = {total_reward}, Regret = {total_regret:.1f}")
    
    print(f"\nFinal Results:")
    print(f"  Total reward: {total_reward}")
    print(f"  Total regret: {total_regret:.1f}")
    print(f"  Average reward per round: {total_reward / 100:.3f}")
    print(f"  Average regret per round: {total_regret / 100:.3f}")


def test_backward_compatibility():
    """Test backward compatibility with old interface."""
    print("\n" + "=" * 50)
    print("Backward Compatibility Test")
    print("=" * 50)
    
    # Test with matrix approach
    env_matrix = SimpleEnvironment(num_arms=10, num_rounds=100, seed=42)
    env_matrix.reset_round_counter()
    
    print("Testing matrix approach with old interface:")
    for round_idx in range(5):
        available_arms_old = env_matrix.sample_available_arms_once()
        env_matrix.reset_available_arms()
        
        available_arms_new = env_matrix.get_available_arms_for_round(round_idx)
        
        print(f"  Round {round_idx}: Old={available_arms_old}, New={available_arms_new}")
        print(f"    Match: {available_arms_old == available_arms_new}")
    
    # Test without matrix (on-demand)
    env_ondemand = SimpleEnvironment(num_arms=10, seed=42)
    print(f"\nTesting on-demand approach:")
    for _ in range(3):
        available_arms = env_ondemand.sample_available_arms_once()
        env_ondemand.reset_available_arms()
        print(f"  Available arms: {available_arms}")


def visualize_matrix():
    """Visualize the availability matrix."""
    print("\n" + "=" * 50)
    print("Matrix Visualization")
    print("=" * 50)
    
    # Create environment with small matrix for visualization
    env = SimpleEnvironment(num_arms=20, num_rounds=100, seed=42)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot availability matrix
    plt.subplot(2, 2, 1)
    plt.imshow(env.availability_matrix, cmap='Blues', aspect='auto')
    plt.title('Availability Matrix')
    plt.xlabel('Round')
    plt.ylabel('Arm')
    plt.colorbar()
    
    # Plot availability rate over rounds
    plt.subplot(2, 2, 2)
    availability_rate_per_round = np.mean(env.availability_matrix, axis=0)
    plt.plot(availability_rate_per_round)
    plt.title('Availability Rate per Round')
    plt.xlabel('Round')
    plt.ylabel('Availability Rate')
    plt.axhline(y=env.availability_rate, color='r', linestyle='--', label='Expected')
    plt.legend()
    
    # Plot number of available arms per round
    plt.subplot(2, 2, 3)
    num_available_per_round = np.sum(env.availability_matrix, axis=0)
    plt.plot(num_available_per_round)
    plt.title('Number of Available Arms per Round')
    plt.xlabel('Round')
    plt.ylabel('Number of Available Arms')
    
    # Plot arm availability frequency
    plt.subplot(2, 2, 4)
    arm_availability_freq = np.mean(env.availability_matrix, axis=1)
    plt.bar(range(env.num_arms), arm_availability_freq)
    plt.title('Arm Availability Frequency')
    plt.xlabel('Arm ID')
    plt.ylabel('Availability Frequency')
    
    plt.tight_layout()
    plt.savefig('output/matrix_visualization.png', dpi=300, bbox_inches='tight')
    print("Matrix visualization saved to output/matrix_visualization.png")


if __name__ == "__main__":
    test_matrix_generation()
    test_performance_comparison()
    test_algorithm_integration()
    test_backward_compatibility()
    visualize_matrix()
    
    print("\n" + "=" * 50)
    print("✓ All tests completed successfully!") 