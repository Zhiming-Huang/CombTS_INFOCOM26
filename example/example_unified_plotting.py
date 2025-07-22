#!/usr/bin/env python3
"""
Example: Unified Plotting Logic Demonstration
============================================

This example demonstrates how to use the unified plotting logic
to create different types of plots with flexible algorithm selection.

Features:
- Unified plotting function that accepts regret data and algorithm lists
- Flexible algorithm selection for different plot types
- Support for both base algorithms and gamma algorithms
- Customizable plot generation

Usage:
    python example_unified_plotting.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.utils.plotting import plot_regret_comparison, plot_gamma_comparison, plot_all_results


def create_sample_results():
    """
    Create sample results data for demonstration.
    This simulates the structure of real simulation results.
    """
    num_rounds = 1000
    num_runs = 5
    
    # Create sample data
    results = {
        'num_rounds': num_rounds,
        'num_runs': num_runs
    }
    
    # Base algorithms
    base_algorithms = ['CTSB', 'CombUCB', 'BG-CTS']
    
    for alg in base_algorithms:
        # Generate sample cumulative regret data
        # Simulate different performance characteristics
        if alg == 'CTSB':
            base_regret = 0.1
            growth_rate = 0.8
        elif alg == 'CombUCB':
            base_regret = 0.05
            growth_rate = 0.6
        else:  # BG-CTS
            base_regret = 0.15
            growth_rate = 1.2
        
        # Generate cumulative regrets for multiple runs
        all_regrets = []
        for run in range(num_runs):
            # Add some randomness to each run
            noise = np.random.normal(0, 0.02, num_rounds)
            cumulative_regret = np.cumsum(base_regret * np.power(np.arange(1, num_rounds + 1), growth_rate) + noise)
            all_regrets.append(cumulative_regret)
        
        all_regrets = np.array(all_regrets)
        avg_regrets = np.mean(all_regrets, axis=0)
        std_regrets = np.std(all_regrets, axis=0)
        
        # Calculate confidence intervals
        confidence_level = 0.95
        degrees_of_freedom = num_runs - 1
        t_value = 2.776  # For 5 runs, 95% confidence
        margin_of_error = t_value * std_regrets / np.sqrt(num_runs)
        ci_lower = avg_regrets - margin_of_error
        ci_upper = avg_regrets + margin_of_error
        
        results[alg] = {
            'all_regrets': all_regrets,
            'avg_cumulative_regrets': avg_regrets,
            'std_cumulative_regrets': std_regrets,
            'confidence_interval': (ci_lower, ci_upper),
            'final_regret': avg_regrets[-1],
            'final_regret_std': std_regrets[-1]
        }
    
    # Gamma algorithms
    gamma_algorithms = ['CTS-G', 'CL-SG']
    gamma_values = [0.01, 0.1, 0.5, 1.0]
    
    for alg in gamma_algorithms:
        for gamma in gamma_values:
            alg_key = f"{alg.lower()}_gamma_{gamma}"
            
            # Generate sample data with gamma-dependent performance
            if alg == 'CTS-G':
                base_regret = 0.08 + gamma * 0.1
                growth_rate = 0.7 + gamma * 0.3
            else:  # CL-SG
                base_regret = 0.06 + gamma * 0.05
                growth_rate = 0.5 + gamma * 0.2
            
            # Generate cumulative regrets for multiple runs
            all_regrets = []
            for run in range(num_runs):
                noise = np.random.normal(0, 0.02, num_rounds)
                cumulative_regret = np.cumsum(base_regret * np.power(np.arange(1, num_rounds + 1), growth_rate) + noise)
                all_regrets.append(cumulative_regret)
            
            all_regrets = np.array(all_regrets)
            avg_regrets = np.mean(all_regrets, axis=0)
            std_regrets = np.std(all_regrets, axis=0)
            
            # Calculate confidence intervals
            margin_of_error = t_value * std_regrets / np.sqrt(num_runs)
            ci_lower = avg_regrets - margin_of_error
            ci_upper = avg_regrets + margin_of_error
            
            results[alg_key] = {
                'all_regrets': all_regrets,
                'avg_cumulative_regrets': avg_regrets,
                'std_cumulative_regrets': std_regrets,
                'confidence_interval': (ci_lower, ci_upper),
                'final_regret': avg_regrets[-1],
                'final_regret_std': std_regrets[-1]
            }
    
    return results


def demonstrate_unified_plotting():
    """
    Demonstrate different ways to use the unified plotting functions.
    """
    print("Unified Plotting Logic Demonstration")
    print("=" * 50)
    
    # Create sample results
    results = create_sample_results()
    
    # Create output directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, 'output', 'images')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Demonstration 1: Plot only base algorithms
    print("\n1. Plotting only base algorithms...")
    base_algorithms = ['CTSB', 'CombUCB', 'BG-CTS']
    plot_regret_comparison(
        results=results,
        algorithms=base_algorithms,
        output_path=f"{output_dir}/demo_base_algorithms.pdf",
        title="Base Algorithms Comparison"
    )
    
    # Demonstration 2: Plot base algorithms + gamma algorithms with default gamma
    print("\n2. Plotting base algorithms + gamma algorithms (default gamma=0.01)...")
    comparison_algorithms = base_algorithms + ['cts-g_gamma_0.01', 'cl-sg_gamma_0.01']
    plot_regret_comparison(
        results=results,
        algorithms=comparison_algorithms,
        output_path=f"{output_dir}/demo_all_algorithms_default_gamma.pdf",
        title="All Algorithms (Default Gamma)"
    )
    
    # Demonstration 3: Plot gamma comparison for CTS-G
    print("\n3. Plotting CTS-G gamma comparison...")
    plot_gamma_comparison(
        results=results,
        algorithm_name="CTS-G",
        gamma_values=[0.01, 0.1, 0.5, 1.0],
        output_path=f"{output_dir}/demo_ctsg_gamma_comparison.pdf",
        title="CTS-G: Effect of Gamma Parameter"
    )
    
    # Demonstration 4: Plot gamma comparison for CL-SG
    print("\n4. Plotting CL-SG gamma comparison...")
    plot_gamma_comparison(
        results=results,
        algorithm_name="CL-SG",
        gamma_values=[0.01, 0.1, 0.5, 1.0],
        output_path=f"{output_dir}/demo_clsg_gamma_comparison.pdf",
        title="CL-SG: Effect of Gamma Parameter"
    )
    
    # Demonstration 5: Use the unified plot_all_results function
    print("\n5. Using unified plot_all_results function...")
    plot_all_results(
        results=results,
        output_dir=output_dir,
        base_algorithms=base_algorithms,
        gamma_algorithms=['CTS-G', 'CL-SG'],
        gamma_values=[0.01, 0.1, 0.5, 1.0],
        file_prefix="demo_unified",
        default_gamma=0.01
    )
    
    # Demonstration 6: Custom algorithm selection
    print("\n6. Custom algorithm selection...")
    custom_algorithms = ['CTSB', 'cts-g_gamma_0.1', 'cl-sg_gamma_0.5']
    plot_regret_comparison(
        results=results,
        algorithms=custom_algorithms,
        output_path=f"{output_dir}/demo_custom_selection.pdf",
        title="Custom Algorithm Selection"
    )
    
    print("\nDemonstration completed!")
    print("Generated plots:")
    print("- demo_base_algorithms.pdf")
    print("- demo_all_algorithms_default_gamma.pdf")
    print("- demo_ctsg_gamma_comparison.pdf")
    print("- demo_clsg_gamma_comparison.pdf")
    print("- demo_unified_algorithm_comparison.pdf")
    print("- demo_unified_ctsg_gamma_comparison.pdf")
    print("- demo_unified_clsg_gamma_comparison.pdf")
    print("- demo_custom_selection.pdf")


def show_usage_examples():
    """
    Show usage examples for different scenarios.
    """
    print("\nUsage Examples:")
    print("=" * 30)
    
    print("\n1. For routing_4x4_memmap example:")
    print("""
    # Plot 1: Base algorithms + gamma algorithms with default gamma
    base_algorithms = ['CTSB', 'CombUCB', 'BG-CTS']
    gamma_algorithms = ['CTS-G', 'CL-SG']
    comparison_algorithms = base_algorithms + ['cts-g_gamma_0.01', 'cl-sg_gamma_0.01']
    
    plot_regret_comparison(
        results=results,
        algorithms=comparison_algorithms,
        output_path=f"{output_dir}/routing_4x4_algorithm_comparison.pdf"
    )
    
    # Plot 2: CTS-G gamma comparison
    plot_gamma_comparison(
        results=results,
        algorithm_name="CTS-G",
        gamma_values=[0.01, 0.1, 0.5, 1.0],
        output_path=f"{output_dir}/routing_4x4_ctsg_gamma_comparison.pdf"
    )
    
    # Plot 3: CL-SG gamma comparison
    plot_gamma_comparison(
        results=results,
        algorithm_name="CL-SG",
        gamma_values=[0.01, 0.1, 0.5, 1.0],
        output_path=f"{output_dir}/routing_4x4_clsg_gamma_comparison.pdf"
    )
    """)
    
    print("\n2. Using the unified plot_all_results function:")
    print("""
    plot_all_results(
        results=results,
        output_dir=output_dir,
        base_algorithms=['CTSB', 'CombUCB', 'BG-CTS'],
        gamma_algorithms=['CTS-G', 'CL-SG'],
        gamma_values=[0.01, 0.1, 0.5, 1.0],
        file_prefix="routing_4x4",
        default_gamma=0.01
    )
    """)
    
    print("\n3. Custom algorithm selection:")
    print("""
    # Plot only specific algorithms
    custom_algorithms = ['CTSB', 'cts-g_gamma_0.1', 'cl-sg_gamma_0.5']
    plot_regret_comparison(
        results=results,
        algorithms=custom_algorithms,
        output_path=f"{output_dir}/custom_comparison.pdf"
    )
    """)


def main():
    """
    Main function to run the demonstration.
    """
    print("Unified Plotting Logic Example")
    print("=" * 50)
    
    # Run demonstration
    demonstrate_unified_plotting()
    
    # Show usage examples
    show_usage_examples()
    
    print("\nExample completed!")


if __name__ == "__main__":
    main() 