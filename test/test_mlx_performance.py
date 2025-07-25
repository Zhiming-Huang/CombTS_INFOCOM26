#!/usr/bin/env python3
"""
Test MLX vs NumPy performance for bandit algorithm operations
============================================================

This script compares the performance of MLX and NumPy for operations
commonly used in bandit algorithms.
"""

import time
import numpy as np
import mlx.core as mx
import mlx.random as mx_random
from typing import List, Tuple
import statistics

def benchmark_operation(operation_name: str, numpy_func, mlx_func, 
                       input_sizes: List[int], num_runs: int = 100):
    """Benchmark a specific operation between NumPy and MLX."""
    print(f"\nBenchmarking: {operation_name}")
    print("-" * 50)
    
    results = {}
    
    for size in input_sizes:
        print(f"\nInput size: {size}")
        
        # Generate test data
        if "random" in operation_name.lower():
            np_data = np.random.randn(size, size).astype(np.float32)
            mx_data = mx.random.normal((size, size))
        else:
            np_data = np.arange(size * size, dtype=np.float32).reshape(size, size)
            mx_data = mx.arange(size * size, dtype=mx.float32).reshape(size, size)
        
        # NumPy timing
        np_times = []
        for _ in range(num_runs):
            start = time.time()
            np_result = numpy_func(np_data)
            np_times.append(time.time() - start)
        
        # MLX timing
        mx_times = []
        for _ in range(num_runs):
            start = time.time()
            mx_result = mlx_func(mx_data)
            mx_times.append(time.time() - start)
        
        # Calculate statistics
        np_mean = statistics.mean(np_times) * 1000  # Convert to ms
        np_std = statistics.stdev(np_times) * 1000
        mx_mean = statistics.mean(mx_times) * 1000
        mx_std = statistics.stdev(mx_times) * 1000
        
        speedup = np_mean / mx_mean if mx_mean > 0 else float('inf')
        
        print(f"  NumPy:  {np_mean:.2f} ± {np_std:.2f} ms")
        print(f"  MLX:    {mx_mean:.2f} ± {mx_std:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        results[size] = {
            'numpy_mean': np_mean,
            'numpy_std': np_std,
            'mlx_mean': mx_mean,
            'mlx_std': mx_std,
            'speedup': speedup
        }
    
    return results

def test_bandit_operations():
    """Test operations commonly used in bandit algorithms."""
    
    # Define operations to test
    operations = [
        ("Matrix Multiplication", 
         lambda x: np.dot(x, x),
         lambda x: mx.matmul(x, x)),
        
        ("Element-wise Addition",
         lambda x: x + x,
         lambda x: x + x),
        
        ("Element-wise Multiplication",
         lambda x: x * x,
         lambda x: x * x),
        
        ("Sum Reduction",
         lambda x: np.sum(x),
         lambda x: mx.sum(x)),
        
        ("Mean Calculation",
         lambda x: np.mean(x),
         lambda x: mx.mean(x)),
        
        ("Standard Deviation",
         lambda x: np.std(x),
         lambda x: mx.std(x)),
        
        ("Random Normal Generation",
         lambda x: np.random.normal(0, 1, x.shape),
         lambda x: mx.random.normal(x.shape)),
        
        ("Exponential",
         lambda x: np.exp(x),
         lambda x: mx.exp(x)),
        
        ("Logarithm",
         lambda x: np.log(np.abs(x) + 1e-8),
         lambda x: mx.log(mx.abs(x) + 1e-8)),
        
        ("Square Root",
         lambda x: np.sqrt(np.abs(x)),
         lambda x: mx.sqrt(mx.abs(x)))
    ]
    
    # Test different input sizes
    input_sizes = [10, 50, 100, 200, 500]
    
    all_results = {}
    
    for op_name, np_func, mx_func in operations:
        results = benchmark_operation(op_name, np_func, mx_func, input_sizes, num_runs=50)
        all_results[op_name] = results
    
    # Summary
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    
    for op_name, results in all_results.items():
        print(f"\n{op_name}:")
        for size, data in results.items():
            speedup = data['speedup']
            if speedup > 1.5:
                print(f"  Size {size}: MLX {speedup:.2f}x faster")
            elif speedup < 0.7:
                print(f"  Size {size}: NumPy {1/speedup:.2f}x faster")
            else:
                print(f"  Size {size}: Similar performance")

def test_bandit_specific_operations():
    """Test operations specific to bandit algorithms."""
    print(f"\n{'='*80}")
    print("BANDIT-SPECIFIC OPERATIONS")
    print(f"{'='*80}")
    
    # Simulate bandit algorithm operations
    num_arms = 100
    num_rounds = 1000
    
    print(f"\nSimulating bandit algorithm with {num_arms} arms, {num_rounds} rounds")
    
    # NumPy version
    start = time.time()
    np_rewards = np.random.normal(0, 1, (num_rounds, num_arms))
    np_counts = np.zeros(num_arms)
    np_sums = np.zeros(num_arms)
    
    for round_idx in range(num_rounds):
        # UCB calculation
        if round_idx < num_arms:
            arm = round_idx
        else:
            ucb_values = np_sums / (np_counts + 1e-8) + np.sqrt(2 * np.log(round_idx) / (np_counts + 1e-8))
            arm = np.argmax(ucb_values)
        
        # Update statistics
        reward = np_rewards[round_idx, arm]
        np_counts[arm] += 1
        np_sums[arm] += reward
    
    np_time = time.time() - start
    print(f"NumPy time: {np_time:.4f} seconds")
    
    # MLX version
    start = time.time()
    mx_rewards = mx.random.normal((num_rounds, num_arms))
    mx_counts = mx.zeros(num_arms)
    mx_sums = mx.zeros(num_arms)
    
    for round_idx in range(num_rounds):
        # UCB calculation
        if round_idx < num_arms:
            arm = round_idx
        else:
            ucb_values = mx_sums / (mx_counts + 1e-8) + mx.sqrt(2 * mx.log(round_idx) / (mx_counts + 1e-8))
            arm = mx.argmax(ucb_values).item()
        
        # Update statistics
        reward = mx_rewards[round_idx, arm]
        mx_counts = mx_counts.at[arm].add(1)
        mx_sums = mx_sums.at[arm].add(reward)
    
    mx_time = time.time() - start
    print(f"MLX time: {mx_time:.4f} seconds")
    print(f"Speedup: {np_time / mx_time:.2f}x")

if __name__ == "__main__":
    print("MLX vs NumPy Performance Benchmark")
    print("=" * 80)
    
    # Test general operations
    test_bandit_operations()
    
    # Test bandit-specific operations
    test_bandit_specific_operations()
    
    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")
    print("This benchmark shows the relative performance of MLX vs NumPy")
    print("for operations commonly used in bandit algorithms.")
    print("\nKey considerations:")
    print("1. MLX may be faster for large matrix operations")
    print("2. NumPy may be faster for small operations due to overhead")
    print("3. MLX requires Apple Silicon for best performance")
    print("4. Consider the trade-off between speed and compatibility") 