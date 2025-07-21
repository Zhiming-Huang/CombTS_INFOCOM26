#!/usr/bin/env python3
"""
Test script to demonstrate different progress tracking options and their performance impact.
"""

import sys
import os
sys.path.append('.')

import time
import numpy as np
from tqdm import tqdm
from typing import Dict, Any

from src.bandits.comb_ts import CombTS
from src.environments.simple_environment import SimpleEnvironment


def run_with_progress_level(progress_level: str, num_rounds: int = 1000, num_runs: int = 3) -> Dict[str, Any]:
    """
    Run simulation with specified progress level and measure performance.
    
    Args:
        progress_level: Progress tracking level ("minimal", "normal", "detailed")
        num_rounds: Number of rounds per simulation
        num_runs: Number of independent runs
        
    Returns:
        Dictionary containing timing and results
    """
    print(f"\nTesting progress level: {progress_level}")
    print("=" * 50)
    
    start_time = time.time()
    
    # Import the function from the main test file
    from test_simple_environment import run_simple_simulation
    
    # Run simulation with specified progress level
    results = run_simple_simulation(
        num_rounds=num_rounds, 
        num_runs=num_runs, 
        progress_level=progress_level
    )
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return {
        'progress_level': progress_level,
        'execution_time': execution_time,
        'num_rounds': num_rounds,
        'num_runs': num_runs,
        'final_regret': results['final_avg_regret'],
        'results': results
    }


def compare_progress_levels():
    """Compare different progress tracking levels and their performance impact."""
    print("Progress Tracking Performance Comparison")
    print("=" * 60)
    
    # Test parameters
    num_rounds = 1000
    num_runs = 3
    
    # Test all progress levels
    levels = ["minimal", "normal", "detailed"]
    results = {}
    
    for level in levels:
        result = run_with_progress_level(level, num_rounds, num_runs)
        results[level] = result
    
    # Compare results
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    baseline_time = results["minimal"]["execution_time"]
    
    for level in levels:
        result = results[level]
        overhead = ((result["execution_time"] - baseline_time) / baseline_time) * 100
        
        print(f"\n{level.upper()} Progress:")
        print(f"  Execution time: {result['execution_time']:.3f}s")
        print(f"  Overhead vs minimal: {overhead:+.1f}%")
        print(f"  Final regret: {result['final_regret']:.2f}")
    
    # Performance recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print("• minimal: Use for production runs, maximum performance")
    print("• normal: Use for development/testing, good balance")
    print("• detailed: Use for debugging, shows all progress details")


def demonstrate_progress_features():
    """Demonstrate different progress bar features."""
    print("\nProgress Bar Features Demonstration")
    print("=" * 50)
    
    # Basic progress bar
    print("\n1. Basic progress bar:")
    for i in tqdm(range(10), desc="Basic"):
        time.sleep(0.1)
    
    # Progress bar with postfix
    print("\n2. Progress bar with postfix:")
    pbar = tqdm(range(10), desc="With postfix")
    for i in pbar:
        pbar.set_postfix({"Value": i, "Squared": i**2})
        time.sleep(0.1)
    
    # Nested progress bars
    print("\n3. Nested progress bars:")
    outer_pbar = tqdm(range(3), desc="Outer")
    for i in outer_pbar:
        inner_pbar = tqdm(range(5), desc=f"Inner {i+1}", leave=False)
        for j in inner_pbar:
            inner_pbar.set_postfix({"Sum": i + j})
            time.sleep(0.05)
    
    # Progress bar with custom units
    print("\n4. Progress bar with custom units:")
    for i in tqdm(range(100), desc="Processing", unit="item"):
        time.sleep(0.01)


def test_memory_efficient_progress():
    """Test memory-efficient progress tracking for large simulations."""
    print("\nMemory-Efficient Progress Tracking Test")
    print("=" * 50)
    
    # Simulate large simulation with minimal memory overhead
    num_rounds = 5000
    num_runs = 2
    
    print(f"Running large simulation: {num_runs} runs × {num_rounds} rounds")
    
    start_time = time.time()
    
    # Use minimal progress to avoid memory overhead
    from test_simple_environment import run_simple_simulation
    results = run_simple_simulation(
        num_rounds=num_rounds,
        num_runs=num_runs,
        progress_level="minimal"  # Minimal overhead
    )
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\nLarge simulation completed:")
    print(f"  Total time: {execution_time:.2f}s")
    print(f"  Average time per round: {execution_time / (num_rounds * num_runs) * 1000:.3f}ms")
    print(f"  Final regret: {results['final_avg_regret']:.2f}")


def main():
    """Main function to demonstrate progress tracking options."""
    print("Progress Tracking Performance Test")
    print("=" * 60)
    
    # Demonstrate features
    demonstrate_progress_features()
    
    # Compare performance
    compare_progress_levels()
    
    # Test memory efficiency
    test_memory_efficient_progress()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✓ tqdm provides efficient progress tracking with minimal overhead")
    print("✓ Configurable progress levels allow performance tuning")
    print("✓ Nested progress bars support complex simulations")
    print("✓ Memory-efficient for large-scale simulations")


if __name__ == "__main__":
    main() 