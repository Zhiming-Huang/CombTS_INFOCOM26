#!/usr/bin/env python3
"""
Gamma Parameter Comparison Script
================================

This script tests CL-SG and CTS-G algorithms with different gamma values
and generates comparison plots.

Usage:
    python test_gamma_comparison.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Import the main function from the example
from example_routing_environment_4x4 import main


def run_gamma_comparison(gamma_values: List[float] = [0.01, 0.1, 0.5, 1.0], 
                        availability_rate: float = 0.8,
                        num_rounds: int = 5000,
                        num_runs: int = 3):
    """
    Run comparison tests for different gamma values.
    
    Args:
        gamma_values: List of gamma values to test
        availability_rate: Link availability rate
        num_rounds: Number of rounds per simulation
        num_runs: Number of independent runs
    """
    print("Gamma Parameter Comparison")
    print("=" * 50)
    print(f"Testing gamma values: {gamma_values}")
    print(f"Availability rate: {availability_rate}")
    print(f"Rounds per simulation: {num_rounds}")
    print(f"Independent runs: {num_runs}")
    print()
    
    # Store results for each gamma value
    all_results = {}
    
    # Test each gamma value
    for gamma in gamma_values:
        print(f"\nTesting gamma = {gamma}")
        print("-" * 30)
        
        # Setup parameters
        gamma_params = {
            'CTS-G': gamma,
            'CL-SG': gamma
        }
        
        # Setup plot options (disable display for batch processing)
        plot_options = {
            'figsize': (4, 3),
            'use_latex': True,
            'font_size': 20,
            'tick_font_size': 10,
            'legend_font_size': 10,
            'line_width': 1.5,
            'marker_size': 6,
            'grid': True,
            'show_plot': False,  # Disable display
            'save_plot': True,
            'dpi': 300,
            'format': 'pdf'
        }
        
        # Run simulation
        try:
            # We need to modify the main function call to use our custom parameters
            # For now, we'll use the command line approach
            import subprocess
            
            cmd = [
                sys.executable, os.path.join('example', 'example_routing_environment_4x4.py'),
                '--availability', str(availability_rate),
                '--gamma-ctsg', str(gamma),
                '--gamma-clsg', str(gamma),
                '--rounds', str(num_rounds),
                '--runs', str(num_runs),
                '--no-plot'  # Disable plot display
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
            
            if result.returncode == 0:
                print(f"✓ Successfully completed simulation for gamma = {gamma}")
                # Extract final regret values from output
                output_lines = result.stdout.split('\n')
                ctsg_regret = None
                clsg_regret = None
                
                for line in output_lines:
                    if 'CTS-G:' in line and 'Final regret:' in line:
                        ctsg_regret = float(line.split('Final regret:')[1].split('±')[0].strip())
                        print(f"  CTS-G final regret: {ctsg_regret}")
                    elif 'CL-SG:' in line and 'Final regret:' in line:
                        clsg_regret = float(line.split('Final regret:')[1].split('±')[0].strip())
                        print(f"  CL-SG final regret: {clsg_regret}")
                
                if ctsg_regret is not None and clsg_regret is not None:
                    all_results[gamma] = {
                        'ctsg_regret': ctsg_regret,
                        'clsg_regret': clsg_regret
                    }
                else:
                    print(f"  ✗ Could not extract regret values from output")
            else:
                print(f"✗ Error running simulation for gamma = {gamma}")
                print(f"Error: {result.stderr}")
                
        except Exception as e:
            print(f"✗ Exception for gamma = {gamma}: {e}")
    
    return all_results


def plot_gamma_comparison(results: Dict[float, Dict[str, float]]):
    """
    Plot comparison of different gamma values.
    
    Args:
        results: Results from run_gamma_comparison
    """
    if not results:
        print("No results to plot")
        return
    
    # Set up the plot
    plt.figure(figsize=(8, 6))
    
    # Try to use LaTeX rendering
    try:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.size'] = 16
    except:
        plt.rcParams['text.usetex'] = False
        plt.rcParams['font.size'] = 12
    
    gamma_values = list(results.keys())
    ctsg_regrets = [results[g]['ctsg_regret'] for g in gamma_values]
    clsg_regrets = [results[g]['clsg_regret'] for g in gamma_values]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Regret vs Gamma
    ax1.plot(gamma_values, ctsg_regrets, 'o-', label='CTS-G', linewidth=2, markersize=8)
    ax1.plot(gamma_values, clsg_regrets, 's-', label='CL-SG', linewidth=2, markersize=8)
    ax1.set_xlabel('Gamma (γ)', fontsize=14)
    ax1.set_ylabel('Final Regret', fontsize=14)
    ax1.set_title('Regret vs Gamma Parameter', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Plot 2: Algorithm Comparison
    x = np.arange(len(gamma_values))
    width = 0.35
    
    ax2.bar(x - width/2, ctsg_regrets, width, label='CTS-G', alpha=0.8)
    ax2.bar(x + width/2, clsg_regrets, width, label='CL-SG', alpha=0.8)
    
    ax2.set_xlabel('Gamma (γ)', fontsize=14)
    ax2.set_ylabel('Final Regret', fontsize=14)
    ax2.set_title('Algorithm Comparison by Gamma', fontsize=16)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{g}' for g in gamma_values])
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, 'output', 'images')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'gamma_comparison.pdf')
    plt.savefig(output_path, bbox_inches='tight', format='pdf', dpi=300)
    plt.show()
    
    print(f"\nGamma comparison plot saved to: {output_path}")
    
    # Print summary table
    print("\nSummary Table:")
    print("-" * 50)
    print(f"{'Gamma':<8} {'CTS-G':<12} {'CL-SG':<12} {'Best':<8}")
    print("-" * 50)
    for gamma in gamma_values:
        ctsg_regret = results[gamma]['ctsg_regret']
        clsg_regret = results[gamma]['clsg_regret']
        best = 'CTS-G' if ctsg_regret < clsg_regret else 'CL-SG'
        print(f"{gamma:<8.2f} {ctsg_regret:<12.2f} {clsg_regret:<12.2f} {best:<8}")


def main():
    """
    Main function to run gamma comparison.
    """
    print("Starting Gamma Parameter Comparison")
    print("=" * 50)
    
    # Test different gamma values
    gamma_values = [0.01, 0.1, 0.5, 1.0]
    
    # Run comparison
    results = run_gamma_comparison(
        gamma_values=gamma_values,
        availability_rate=0.8,
        num_rounds=5000,  # Reduced for faster testing
        num_runs=3        # Reduced for faster testing
    )
    
    # Plot results
    if results:
        plot_gamma_comparison(results)
    else:
        print("No successful results to plot")


if __name__ == "__main__":
    main() 