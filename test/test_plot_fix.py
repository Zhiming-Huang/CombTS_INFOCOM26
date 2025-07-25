#!/usr/bin/env python3
"""
Test script to fix the plotting issue by correcting the data structure access.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import matplotlib.pyplot as plt
from src.utils.plotting import plot_algorithm_comparison

def test_plot_fix():
    """Test and fix the plotting issue."""
    
    print("=" * 80)
    print("TESTING PLOT FIX")
    print("=" * 80)
    
    # Load results
    try:
        with open("output/data/ucsb_all_algorithms_3hop_results.json", 'r') as f:
            results = json.load(f)
        print("✓ Loaded results")
    except FileNotFoundError:
        print("✗ Results file not found")
        return
    
    # Check the data structure
    print(f"\nData structure:")
    print(f"  Keys in results: {list(results.keys())}")
    print(f"  Algorithms: {list(results['algorithms'].keys())}")
    
    # Create corrected data structure for plotting
    corrected_results = {
        'num_rounds': results['num_rounds']
    }
    
    # Add each algorithm directly to the root level
    for alg_name, alg_data in results['algorithms'].items():
        corrected_results[alg_name] = {
            'avg_cumulative_regrets': alg_data['regrets'],
            'confidence_interval': (alg_data['regrets'], alg_data['regrets'])  # Single run
        }
    
    print(f"\nCorrected data structure:")
    print(f"  Keys in corrected_results: {list(corrected_results.keys())}")
    
    # Test plotting with corrected structure
    algorithms = list(results['algorithms'].keys())
    
    print(f"\nTesting plotting with corrected structure...")
    
    # Use utils plotting function
    plot_algorithm_comparison(
        corrected_results,
        algorithms=algorithms,
        output_path="output/images/ucsb_all_algorithms_3hop_comparison_fixed.pdf",
        title="UCSB Environment: All Algorithms Comparison (3-Hop Paths) - Fixed"
    )
    
    print("Fixed plot saved to: output/images/ucsb_all_algorithms_3hop_comparison_fixed.pdf")
    
    # Also create a simple test plot to verify
    create_simple_test_plot(results, algorithms)

def create_simple_test_plot(results, algorithms):
    """Create a simple test plot to verify plotting works."""
    
    print(f"\nCreating simple test plot...")
    
    # Setup plot
    plt.figure(figsize=(10, 6))
    
    num_rounds = results['num_rounds']
    rounds = np.arange(1, num_rounds + 1)
    
    # Define colors
    colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))
    
    # Plot each algorithm
    for i, alg in enumerate(algorithms):
        regrets = results['algorithms'][alg]['regrets']
        color = colors[i]
        
        # Create display name
        if '_gamma_' in alg:
            parts = alg.split('_gamma_')
            if len(parts) == 2:
                alg_name = parts[0].upper()
                gamma = parts[1]
                display_name = f"{alg_name} (γ={gamma})"
            else:
                display_name = alg
        else:
            display_name = alg
        
        plt.plot(rounds, regrets, label=display_name, color=color, linewidth=1.5)
    
    plt.xlabel('Rounds')
    plt.ylabel('Cumulative Regret')
    plt.title('UCSB Environment: All Algorithms Comparison (3-Hop Paths) - Simple Test')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('output/images/ucsb_all_algorithms_3hop_simple_test.pdf', 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    print("Simple test plot saved to: output/images/ucsb_all_algorithms_3hop_simple_test.pdf")

if __name__ == "__main__":
    test_plot_fix() 