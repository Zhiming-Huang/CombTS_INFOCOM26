#!/usr/bin/env python3
"""
Test confidence interval calculation for randomness.
"""

import numpy as np
from scipy import stats

def test_confidence_interval_reproducibility():
    """Test if confidence interval calculation is deterministic."""
    print("Testing Confidence Interval Reproducibility")
    print("=" * 60)
    
    # Create the same test data multiple times
    np.random.seed(42)  # Use numpy's old seed system for this test
    
    for test_run in range(5):
        print(f"\nTest Run {test_run + 1}:")
        
        # Simulate CL-SG results like in our test
        regrets = np.array([23.392190, 41.095013, 23.112327])
        
        # Calculate statistics exactly as in the main script
        avg_cumulative_regrets = np.mean(regrets)
        std_cumulative_regrets = np.std(regrets)
        num_runs = len(regrets)
        confidence_level = 0.95
        
        # Calculate confidence intervals exactly as in main script
        confidence_interval = stats.t.interval(
            confidence_level,
            num_runs - 1,
            loc=avg_cumulative_regrets,
            scale=std_cumulative_regrets / np.sqrt(num_runs)
        )
        
        final_regret = avg_cumulative_regrets
        final_regret_std = std_cumulative_regrets
        final_regret_ci = (confidence_interval[0], confidence_interval[1])
        
        print(f"  Final regret: {final_regret:.2f} ± {final_regret_std:.2f}")
        print(f"  95% CI: [{final_regret_ci[0]:.2f}, {final_regret_ci[1]:.2f}]")
        print(f"  Exact values: regret={final_regret:.10f}, std={final_regret_std:.10f}")
        print(f"  Exact CI: [{final_regret_ci[0]:.10f}, {final_regret_ci[1]:.10f}]")


def test_numpy_array_reproducibility():
    """Test if numpy array operations are deterministic."""
    print("\n\nTesting NumPy Array Reproducibility")
    print("=" * 60)
    
    # Test the exact operations used in main script
    for test_run in range(5):
        print(f"\nTest Run {test_run + 1}:")
        
        # Create a regrets array like in main script
        regrets_array = np.zeros((3, 1000))
        regrets_array[0, -1] = 23.392190
        regrets_array[1, -1] = 41.095013
        regrets_array[2, -1] = 23.112327
        
        # Calculate statistics exactly as in main script
        avg_cumulative_regrets = np.mean(regrets_array, axis=0)
        std_cumulative_regrets = np.std(regrets_array, axis=0)
        
        final_regret = avg_cumulative_regrets[-1]
        final_regret_std = std_cumulative_regrets[-1]
        
        print(f"  Final regret: {final_regret:.10f}")
        print(f"  Final std: {final_regret_std:.10f}")


if __name__ == "__main__":
    test_confidence_interval_reproducibility()
    test_numpy_array_reproducibility() 