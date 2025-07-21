#!/usr/bin/env python3
"""
Test script for memory-mapped environment functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environments.simple_environment_memmap import SimpleEnvironmentMemmap
from src.bandits.comb_ts import CombTS
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tempfile


def test_memmap_generation():
    """Test memory-mapped matrix generation."""
    print("Testing Memory-Mapped Matrix Generation")
    print("=" * 50)
    
    # Test with different sizes
    test_cases = [
        (10, 100),      # Small
        (50, 1000),     # Medium
        (100, 5000),    # Large
        (200, 10000),   # Very large
    ]
    
    for num_arms, num_rounds in test_cases:
        print(f"\nTesting {num_arms} arms, {num_rounds} rounds:")
        
        # Time matrix generation
        start_time = time.time()
        env = SimpleEnvironmentMemmap(num_arms=num_arms, num_rounds=num_rounds, 
                                    seed=42, chunk_size=1000)
        generation_time = time.time() - start_time
        
        # Get matrix info
        matrix_info = env.get_availability_matrix_info()
        
        print(f"  Generation time: {generation_time:.4f}s")
        print(f"  Matrix shape: {matrix_info['shape']}")
        print(f"  File size: {matrix_info['file_size_mb']:.2f} MB")
        print(f"  Actual availability rate: {matrix_info['availability_rate_actual']:.3f}")
        print(f"  Matrix file: {matrix_info['matrix_file']}")
        
        # Test accessing specific rounds
        test_rounds = [0, num_rounds//2, num_rounds-1]
        for round_idx in test_rounds:
            available_arms = env.get_available_arms_for_round(round_idx)
            print(f"  Round {round_idx}: {len(available_arms)} available arms")


def test_performance_comparison():
    """Compare performance between memory and memory-mapped approaches."""
    print("\n" + "=" * 50)
    print("Performance Comparison: Memory vs Memory-Mapped")
    print("=" * 50)
    
    num_arms, num_rounds = 100, 5000
    
    # Test memory-mapped approach
    print("Testing memory-mapped approach...")
    start_time = time.time()
    env_memmap = SimpleEnvironmentMemmap(num_arms=num_arms, num_rounds=num_rounds, 
                                        seed=42, chunk_size=1000)
    memmap_init_time = time.time() - start_time
    
    # Test memory approach (import here to avoid conflicts)
    from src.environments.simple_environment import SimpleEnvironment
    print("Testing memory approach...")
    start_time = time.time()
    env_memory = SimpleEnvironment(num_arms=num_arms, num_rounds=num_rounds, seed=42)
    memory_init_time = time.time() - start_time
    
    print(f"\nInitialization times:")
    print(f"  Memory-mapped: {memmap_init_time:.4f}s")
    print(f"  Memory: {memory_init_time:.4f}s")
    print(f"  Ratio: {memmap_init_time / memory_init_time:.2f}x")
    
    # Test access times
    print(f"\nAccess time comparison (1000 rounds):")
    
    # Memory-mapped access time
    start_time = time.time()
    for round_idx in range(1000):
        available_arms = env_memmap.get_available_arms_for_round(round_idx)
    memmap_access_time = time.time() - start_time
    
    # Memory access time
    start_time = time.time()
    for round_idx in range(1000):
        available_arms = env_memory.get_available_arms_for_round(round_idx)
    memory_access_time = time.time() - start_time
    
    print(f"  Memory-mapped access time: {memmap_access_time:.4f}s")
    print(f"  Memory access time: {memory_access_time:.4f}s")
    print(f"  Speedup: {memory_access_time / memmap_access_time:.2f}x")


def test_algorithm_integration():
    """Test CombTS integration with memory-mapped environment."""
    print("\n" + "=" * 50)
    print("Algorithm Integration Test")
    print("=" * 50)
    
    # Create environment with memory mapping
    env = SimpleEnvironmentMemmap(num_arms=20, num_rounds=500, seed=42, chunk_size=100)
    algorithm = CombTS(env)
    
    total_reward = 0
    total_regret = 0
    
    print("Running algorithm with memory-mapped environment...")
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


def test_file_persistence():
    """Test file persistence and loading."""
    print("\n" + "=" * 50)
    print("File Persistence Test")
    print("=" * 50)
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.npy', delete=False)
    temp_file.close()
    
    try:
        # Create environment with specific file
        print(f"Creating environment with file: {temp_file.name}")
        env = SimpleEnvironmentMemmap(num_arms=10, num_rounds=100, seed=42, 
                                    matrix_file=temp_file.name, chunk_size=50)
        
        # Get some data
        available_arms_round_0 = env.get_available_arms_for_round(0)
        available_arms_round_50 = env.get_available_arms_for_round(50)
        
        print(f"  Round 0: {available_arms_round_0}")
        print(f"  Round 50: {available_arms_round_50}")
        
        # Check file exists and has correct size
        file_size = os.path.getsize(temp_file.name)
        expected_size = 10 * 100  # num_arms * num_rounds (uint8 = 1 byte)
        print(f"  File size: {file_size} bytes (expected: {expected_size})")
        print(f"  File size correct: {file_size == expected_size}")
        
    finally:
        # Clean up
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)


def test_large_scale_simulation():
    """Test large-scale simulation with memory mapping."""
    print("\n" + "=" * 50)
    print("Large-Scale Simulation Test")
    print("=" * 50)
    
    # Test with large matrix
    num_arms, num_rounds = 500, 20000  # 10M elements
    
    print(f"Testing large-scale simulation: {num_arms} arms × {num_rounds} rounds")
    print(f"Total elements: {num_arms * num_rounds:,}")
    
    start_time = time.time()
    env = SimpleEnvironmentMemmap(num_arms=num_arms, num_rounds=num_rounds, 
                                seed=42, chunk_size=5000)
    generation_time = time.time() - start_time
    
    matrix_info = env.get_availability_matrix_info()
    
    print(f"  Generation time: {generation_time:.2f}s")
    print(f"  File size: {matrix_info['file_size_mb']:.2f} MB")
    print(f"  Matrix shape: {matrix_info['shape']}")
    
    # Test random access performance
    print(f"\nTesting random access performance...")
    test_rounds = [0, 1000, 5000, 10000, 15000, 19999]
    
    start_time = time.time()
    for round_idx in test_rounds:
        available_arms = env.get_available_arms_for_round(round_idx)
        print(f"  Round {round_idx}: {len(available_arms)} arms available")
    access_time = time.time() - start_time
    
    print(f"  Random access time: {access_time:.4f}s")
    print(f"  Average access time per round: {access_time / len(test_rounds):.6f}s")


def test_memory_usage():
    """Test memory usage comparison."""
    print("\n" + "=" * 50)
    print("Memory Usage Comparison")
    print("=" * 50)
    
    import psutil
    import gc
    
    def get_memory_usage():
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    
    # Test memory-mapped approach
    print("Testing memory-mapped approach...")
    gc.collect()
    memory_before = get_memory_usage()
    
    env_memmap = SimpleEnvironmentMemmap(num_arms=100, num_rounds=10000, 
                                        seed=42, chunk_size=1000)
    memory_after = get_memory_usage()
    memory_increase_memmap = memory_after - memory_before
    
    print(f"  Memory increase: {memory_increase_memmap:.2f} MB")
    
    # Test memory approach
    print("Testing memory approach...")
    gc.collect()
    memory_before = get_memory_usage()
    
    from src.environments.simple_environment import SimpleEnvironment
    env_memory = SimpleEnvironment(num_arms=100, num_rounds=10000, seed=42)
    memory_after = get_memory_usage()
    memory_increase_memory = memory_after - memory_before
    
    print(f"  Memory increase: {memory_increase_memory:.2f} MB")
    
    print(f"\nMemory efficiency:")
    print(f"  Memory-mapped: {memory_increase_memmap:.2f} MB")
    print(f"  Memory: {memory_increase_memory:.2f} MB")
    print(f"  Ratio: {memory_increase_memory / memory_increase_memmap:.2f}x")


if __name__ == "__main__":
    test_memmap_generation()
    test_performance_comparison()
    test_algorithm_integration()
    test_file_persistence()
    test_large_scale_simulation()
    test_memory_usage()
    
    print("\n" + "=" * 50)
    print("✓ All memory-mapped tests completed successfully!") 