#!/usr/bin/env python3
"""
Analyze feasible set dynamics across different traces in UCSB dataset.
"""

import sys
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from collections import defaultdict

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment

def analyze_feasible_set_dynamics():
    """Analyze how feasible sets change across different traces."""
    
    # Find UCSB data files
    data_dir = os.path.join(project_root, 'data', 'ucsb')
    neighbortable_files = []
    
    # Look for neighbortable files in subdirectories
    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.isdir(subdir_path):
            files = glob.glob(os.path.join(subdir_path, 'neighbortable-*'))
            neighbortable_files.extend(sorted(files))
    
    if not neighbortable_files:
        print("No neighbortable files found!")
        return
    
    # Use first 50 files for analysis (to see dynamics)
    neighbortable_files = sorted(neighbortable_files)[:50]
    print(f"Analyzing {len(neighbortable_files)} neighbortable files")
    
    # Create environment
    env = UCSBMeshnetMemmapEnvironment(
        neighbortable_files=neighbortable_files,
        routes_per_minute=4,
        source="10.2.1.103",  # Use known working source
        destination="10.2.1.109",  # Use known working destination
        pre_generate_rewards=True,
        use_persistent_files=False,
        rng=np.random.default_rng(42),
        max_feasible_combinations=50,
        max_path_length=3,
        max_paths_per_algorithm=25
    )
    
    print(f"Environment created with source: {env.get_source_destination()[0]}")
    print(f"Destination: {env.get_source_destination()[1]}")
    print(f"Number of rounds: {env.num_rounds}")
    
    # Analyze feasible sets across different rounds
    feasible_set_stats = []
    path_length_distributions = []
    
    # Sample rounds for analysis (every 10th round to avoid too much data)
    sample_rounds = list(range(0, min(env.num_rounds, 200), 10))
    
    print(f"\nAnalyzing {len(sample_rounds)} sample rounds...")
    
    for round_idx in sample_rounds:
        # Get available arms for this round
        available_arms = env.get_available_arms_for_round(round_idx)
        
        # Get feasible combinations
        feasible_combinations = env.get_feasible_combinations(available_arms)
        
        if feasible_combinations:
            # Calculate statistics
            num_feasible = len(feasible_combinations)
            
            # Analyze path lengths
            path_lengths = []
            for combination in feasible_combinations:
                # Convert combination to path length (simplified)
                path_length = len(combination)  # This is approximate
                path_lengths.append(path_length)
            
            # Calculate path length distribution
            length_counts = defaultdict(int)
            for length in path_lengths:
                length_counts[length] += 1
            
            feasible_set_stats.append({
                'round': round_idx,
                'num_feasible': num_feasible,
                'num_available_arms': len(available_arms),
                'avg_path_length': np.mean(path_lengths) if path_lengths else 0,
                'min_path_length': min(path_lengths) if path_lengths else 0,
                'max_path_length': max(path_lengths) if path_lengths else 0
            })
            
            path_length_distributions.append({
                'round': round_idx,
                'length_counts': dict(length_counts)
            })
        else:
            feasible_set_stats.append({
                'round': round_idx,
                'num_feasible': 0,
                'num_available_arms': len(available_arms),
                'avg_path_length': 0,
                'min_path_length': 0,
                'max_path_length': 0
            })
    
    # Print summary statistics
    print(f"\nFeasible Set Dynamics Summary:")
    print(f"=" * 60)
    
    num_feasible_list = [stat['num_feasible'] for stat in feasible_set_stats]
    print(f"Total rounds analyzed: {len(feasible_set_stats)}")
    print(f"Rounds with feasible paths: {sum(1 for n in num_feasible_list if n > 0)}")
    print(f"Rounds without feasible paths: {sum(1 for n in num_feasible_list if n == 0)}")
    print(f"Average feasible paths per round: {np.mean(num_feasible_list):.2f}")
    print(f"Min feasible paths: {min(num_feasible_list)}")
    print(f"Max feasible paths: {max(num_feasible_list)}")
    print(f"Std feasible paths: {np.std(num_feasible_list):.2f}")
    
    # Analyze path length dynamics
    avg_lengths = [stat['avg_path_length'] for stat in feasible_set_stats if stat['num_feasible'] > 0]
    if avg_lengths:
        print(f"\nPath Length Analysis:")
        print(f"Average path length: {np.mean(avg_lengths):.2f}")
        print(f"Min path length: {min(avg_lengths):.2f}")
        print(f"Max path length: {max(avg_lengths):.2f}")
    
    # Show some detailed examples
    print(f"\nDetailed Examples:")
    print(f"=" * 60)
    
    # Show first few rounds with feasible paths
    feasible_rounds = [stat for stat in feasible_set_stats if stat['num_feasible'] > 0]
    for i, stat in enumerate(feasible_rounds[:5]):
        print(f"Round {stat['round']}:")
        print(f"  Feasible paths: {stat['num_feasible']}")
        print(f"  Available arms: {stat['num_available_arms']}")
        print(f"  Avg path length: {stat['avg_path_length']:.2f}")
        print(f"  Path length range: {stat['min_path_length']}-{stat['max_path_length']}")
        
        # Show path length distribution for this round
        if i < len(path_length_distributions):
            dist = path_length_distributions[i]
            if dist['round'] == stat['round']:
                print(f"  Path length distribution: {dict(dist['length_counts'])}")
        print()
    
    # Analyze temporal patterns
    print(f"Temporal Analysis:")
    print(f"=" * 60)
    
    # Check if there are patterns in feasible set availability
    consecutive_zeros = 0
    max_consecutive_zeros = 0
    for stat in feasible_set_stats:
        if stat['num_feasible'] == 0:
            consecutive_zeros += 1
            max_consecutive_zeros = max(max_consecutive_zeros, consecutive_zeros)
        else:
            consecutive_zeros = 0
    
    print(f"Maximum consecutive rounds without feasible paths: {max_consecutive_zeros}")
    
    # Check for stability
    changes = []
    for i in range(1, len(feasible_set_stats)):
        prev = feasible_set_stats[i-1]['num_feasible']
        curr = feasible_set_stats[i]['num_feasible']
        changes.append(abs(curr - prev))
    
    if changes:
        print(f"Average change in feasible paths between consecutive rounds: {np.mean(changes):.2f}")
        print(f"Max change in feasible paths: {max(changes)}")
    
    # Clean up
    env.cleanup()
    
    return feasible_set_stats, path_length_distributions

def plot_feasible_set_dynamics(feasible_set_stats):
    """Plot feasible set dynamics."""
    
    if not feasible_set_stats:
        return
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    rounds = [stat['round'] for stat in feasible_set_stats]
    num_feasible = [stat['num_feasible'] for stat in feasible_set_stats]
    num_available = [stat['num_available_arms'] for stat in feasible_set_stats]
    avg_lengths = [stat['avg_path_length'] for stat in feasible_set_stats]
    
    # Plot 1: Number of feasible paths over time
    ax1.plot(rounds, num_feasible, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Number of Feasible Paths')
    ax1.set_title('Feasible Paths Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Available arms over time
    ax2.plot(rounds, num_available, 'g-', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Number of Available Arms')
    ax2.set_title('Available Arms Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Average path length over time
    ax3.plot(rounds, avg_lengths, 'r-', linewidth=2, marker='^', markersize=4)
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Average Path Length')
    ax3.set_title('Average Path Length Over Time')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Scatter plot of available arms vs feasible paths
    ax4.scatter(num_available, num_feasible, alpha=0.6, s=30)
    ax4.set_xlabel('Number of Available Arms')
    ax4.set_ylabel('Number of Feasible Paths')
    ax4.set_title('Available Arms vs Feasible Paths')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = os.path.join(project_root, 'output', 'images')
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'ucsb_feasible_set_dynamics.pdf')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Feasible set dynamics plot saved to: {plot_path}")
    
    plt.show()

if __name__ == "__main__":
    feasible_set_stats, path_length_distributions = analyze_feasible_set_dynamics()
    plot_feasible_set_dynamics(feasible_set_stats) 