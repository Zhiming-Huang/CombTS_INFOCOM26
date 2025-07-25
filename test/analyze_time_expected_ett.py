#!/usr/bin/env python3
"""
Analyze time-expected ETT results.
"""

import sys
import os
import numpy as np

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.utils.plotting import calculate_time_expected_ett


def analyze_ett_results():
    """Analyze time-expected ETT results from a simulation."""
    print("Time-Expected ETT Analysis")
    print("=" * 60)
    
    # Example results (you can replace with actual results)
    # This is a simplified example - in practice, you would load actual results
    
    # Simulate some results
    num_rounds = 5000
    num_runs = 5
    
    # Example cumulative regrets (these would be actual results from simulation)
    results = {
        'num_rounds': num_rounds,
        'num_runs': num_runs,
        'CTSB': {
            'avg_cumulative_regrets': np.linspace(0, 50, num_rounds)  # Example data
        },
        'CombUCB': {
            'avg_cumulative_regrets': np.linspace(0, 190, num_rounds)  # Example data
        },
        'BG-CTS': {
            'avg_cumulative_regrets': np.linspace(0, 255, num_rounds)  # Example data
        },
        'CL-SG': {
            'avg_cumulative_regrets': np.linspace(0, 69, num_rounds)  # Example data
        }
    }
    
    # Example ETT configuration
    env_config = {
        'ett_min': 1.0,  # Minimum ETT value
        'ett_max': 10.0  # Maximum ETT value
    }
    
    # Calculate time-expected ETT
    ett_results = calculate_time_expected_ett(results, env_config)
    
    print("Time-Expected ETT Results:")
    print("-" * 40)
    
    for alg_name, ett_data in ett_results.items():
        final_ett = ett_data['final_time_expected_ett']
        print(f"{alg_name}: {final_ett:.4f}")
    
    print("\n" + "=" * 60)
    print("Analysis:")
    print("- Lower values indicate better performance")
    print("- Time-expected ETT represents the average ETT per round")
    print("- This metric directly relates to network performance")


def explain_time_expected_ett():
    """Explain the time-expected ETT metric."""
    print("Time-Expected ETT Metric Explanation")
    print("=" * 60)
    
    print("1. What is Time-Expected ETT?")
    print("   - Maps cumulative reward back to cumulative ETT")
    print("   - Divides by total rounds to get average ETT per round")
    print("   - Lower values indicate better network performance")
    print()
    
    print("2. Calculation Process:")
    print("   a) Start with cumulative reward (normalized to [0,1])")
    print("   b) Map back to ETT: ETT = ETT_min + (1 - reward) * (ETT_max - ETT_min)")
    print("   c) Calculate average: Time-Expected ETT = Cumulative ETT / rounds")
    print()
    
    print("3. Interpretation:")
    print("   - Shows the average Expected Transmission Time per round")
    print("   - Direct measure of network efficiency")
    print("   - More intuitive than regret for network applications")
    print()
    
    print("4. Advantages:")
    print("   - Directly relates to network performance")
    print("   - Easier to interpret than regret")
    print("   - Can be compared across different scenarios")
    print("   - Useful for network optimization")


if __name__ == "__main__":
    explain_time_expected_ett()
    print("\n")
    analyze_ett_results() 