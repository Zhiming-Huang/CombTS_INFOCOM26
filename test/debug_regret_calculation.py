#!/usr/bin/env python3
"""
Debug script to test regret calculation.
"""

import sys
import os
import numpy as np

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from example.example_ucsb_comprehensive import setup_ucsb_environment, run_ucsb_simulation

def debug_regret_calculation():
    """Debug regret calculation."""
    
    # Create environment
    env = setup_ucsb_environment(num_rounds=100, main_seed=42)
    
    # Run a short simulation
    results = run_ucsb_simulation(
        env=env,
        num_rounds=100,
        num_runs=1,
        progress_level="normal",
        main_seed=42,
        gamma_values=[0.1]
    )
    
    # Check the regret data for CTSB
    if 'CTSB' in results:
        ctsb_data = results['CTSB']
        avg_regrets = ctsb_data['avg_cumulative_regrets']
        
        print("CTSB Cumulative Regret Analysis:")
        print(f"Number of rounds: {len(avg_regrets)}")
        print(f"Final regret: {avg_regrets[-1]:.3f}")
        
        # Check for zero or decreasing values
        zero_count = np.sum(avg_regrets == 0)
        decreasing_count = np.sum(np.diff(avg_regrets) < 0)
        
        print(f"Zero regret values: {zero_count}")
        print(f"Decreasing regret values: {decreasing_count}")
        
        # Show first 20 values
        print("\nFirst 20 cumulative regret values:")
        for i, regret in enumerate(avg_regrets[:20]):
            print(f"Round {i+1}: {regret:.3f}")
        
        # Show any zero values
        zero_indices = np.where(avg_regrets == 0)[0]
        if len(zero_indices) > 0:
            print(f"\nZero regret at rounds: {zero_indices + 1}")
    
    # Clean up
    env.cleanup()

if __name__ == "__main__":
    debug_regret_calculation() 