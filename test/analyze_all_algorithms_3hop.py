#!/usr/bin/env python3
"""
Detailed analysis of all bandit algorithms with 3-hop paths in UCSB environment.
Provides insights into algorithm performance and path selection behavior.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import matplotlib.pyplot as plt

def analyze_all_algorithms_3hop():
    """Analyze the performance of all algorithms with 3-hop paths."""
    
    print("=" * 80)
    print("DETAILED ANALYSIS: ALL ALGORITHMS WITH 3-HOP PATHS")
    print("=" * 80)
    
    # Load results
    try:
        with open("output/data/ucsb_all_algorithms_3hop_results.json", 'r') as f:
            results = json.load(f)
        print("✓ Loaded comprehensive test results")
    except FileNotFoundError:
        print("✗ Results file not found. Please run test_all_algorithms_3hop.py first.")
        return
    
    # Extract data
    algorithms = list(results['algorithms'].keys())
    final_regrets = [results['algorithms'][alg]['final_regret'] for alg in algorithms]
    simulation_times = [results['algorithms'][alg]['simulation_time'] for alg in algorithms]
    avg_combinations = [results['algorithms'][alg]['avg_combinations'] for alg in algorithms]
    
    # Calculate average path lengths
    avg_path_lengths = []
    path_distributions = []
    
    for alg in algorithms:
        dist = results['algorithms'][alg]['path_distribution']
        if dist:
            avg_length = sum(int(hop) * dist[hop]['percentage'] / 100 for hop in dist.keys())
            avg_path_lengths.append(avg_length)
            path_distributions.append(dist)
        else:
            avg_path_lengths.append(0)
            path_distributions.append({})
    
    # Print detailed analysis
    print(f"\n{'='*80}")
    print("PERFORMANCE ANALYSIS")
    print(f"{'='*80}")
    
    # Sort algorithms by final regret
    sorted_indices = np.argsort(final_regrets)
    
    print(f"{'Rank':<4} {'Algorithm':<20} {'Final Regret':<12} {'Avg Path Length':<15} {'Sim Time (s)':<12} {'Path Distribution'}")
    print("-" * 100)
    
    for i, idx in enumerate(sorted_indices):
        alg = algorithms[idx]
        regret = final_regrets[idx]
        avg_length = avg_path_lengths[idx]
        sim_time = simulation_times[idx]
        
        # Get path distribution string
        dist = path_distributions[idx]
        path_str = ""
        for hop in sorted(dist.keys()):
            pct = dist[hop]['percentage']
            path_str += f"{hop}hop:{pct:.0f}% "
        
        print(f"{i+1:<4} {alg:<20} {regret:<12.2f} {avg_length:<15.2f} {sim_time:<12.2f} {path_str}")
    
    # Algorithm family analysis
    print(f"\n{'='*80}")
    print("ALGORITHM FAMILY ANALYSIS")
    print(f"{'='*80}")
    
    # Group algorithms by family
    ctsg_algorithms = [alg for alg in algorithms if alg.startswith('cts-g_gamma_')]
    clsg_algorithms = [alg for alg in algorithms if alg.startswith('cl-sg_gamma_')]
    other_algorithms = [alg for alg in algorithms if not (alg.startswith('cts-g_gamma_') or alg.startswith('cl-sg_gamma_'))]
    
    print("CTS-G Family (Thompson Sampling with Gaussian):")
    for alg in ctsg_algorithms:
        regret = results['algorithms'][alg]['final_regret']
        gamma = alg.split('_')[-1]
        print(f"  gamma={gamma}: {regret:.2f}")
    
    print("\nCL-SG Family (Combinatorial Learning with Single Gaussian):")
    for alg in clsg_algorithms:
        regret = results['algorithms'][alg]['final_regret']
        gamma = alg.split('_')[-1]
        print(f"  gamma={gamma}: {regret:.2f}")
    
    print("\nOther Algorithms:")
    for alg in other_algorithms:
        regret = results['algorithms'][alg]['final_regret']
        print(f"  {alg}: {regret:.2f}")
    
    # Path length analysis
    print(f"\n{'='*80}")
    print("PATH LENGTH ANALYSIS")
    print(f"{'='*80}")
    
    # Calculate statistics for each hop count
    hop_stats = {}
    for alg in algorithms:
        dist = path_distributions[algorithms.index(alg)]
        for hop in dist.keys():
            if hop not in hop_stats:
                hop_stats[hop] = []
            hop_stats[hop].append(dist[hop]['percentage'])
    
    for hop in sorted(hop_stats.keys()):
        percentages = hop_stats[hop]
        print(f"{hop}-hop paths:")
        print(f"  Average usage: {np.mean(percentages):.1f}%")
        print(f"  Min usage: {np.min(percentages):.1f}%")
        print(f"  Max usage: {np.max(percentages):.1f}%")
        print(f"  Std dev: {np.std(percentages):.1f}%")
    
    # Create detailed visualization
    create_detailed_plots(algorithms, final_regrets, avg_path_lengths, path_distributions, simulation_times)
    
    # Performance insights
    print(f"\n{'='*80}")
    print("PERFORMANCE INSIGHTS")
    print(f"{'='*80}")
    
    best_algorithm = algorithms[sorted_indices[0]]
    worst_algorithm = algorithms[sorted_indices[-1]]
    
    print(f"Best performing algorithm: {best_algorithm} (regret: {final_regrets[sorted_indices[0]]:.2f})")
    print(f"Worst performing algorithm: {worst_algorithm} (regret: {final_regrets[sorted_indices[-1]]:.2f})")
    print(f"Performance ratio: {final_regrets[sorted_indices[-1]] / final_regrets[sorted_indices[0]]:.2f}x")
    
    # Gamma parameter analysis
    print(f"\nGamma parameter impact:")
    ctsg_gammas = []
    ctsg_regrets = []
    for alg in ctsg_algorithms:
        gamma = float(alg.split('_')[-1])
        regret = results['algorithms'][alg]['final_regret']
        ctsg_gammas.append(gamma)
        ctsg_regrets.append(regret)
    
    if len(ctsg_gammas) > 1:
        ctsg_corr = np.corrcoef(ctsg_gammas, ctsg_regrets)[0, 1]
        print(f"  CTS-G: gamma vs regret correlation: {ctsg_corr:.3f}")
    
    clsg_gammas = []
    clsg_regrets = []
    for alg in clsg_algorithms:
        gamma = float(alg.split('_')[-1])
        regret = results['algorithms'][alg]['final_regret']
        clsg_gammas.append(gamma)
        clsg_regrets.append(regret)
    
    if len(clsg_gammas) > 1:
        clsg_corr = np.corrcoef(clsg_gammas, clsg_regrets)[0, 1]
        print(f"  CL-SG: gamma vs regret correlation: {clsg_corr:.3f}")

def create_detailed_plots(algorithms, final_regrets, avg_path_lengths, path_distributions, simulation_times):
    """Create detailed visualization plots."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Final regret comparison
    colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))
    bars = ax1.bar(range(len(algorithms)), final_regrets, color=colors)
    ax1.set_xlabel('Algorithms')
    ax1.set_ylabel('Final Cumulative Regret')
    ax1.set_title('Final Regret Comparison')
    ax1.set_xticks(range(len(algorithms)))
    ax1.set_xticklabels(algorithms, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, regret in zip(bars, final_regrets):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(final_regrets)*0.01,
                f'{regret:.0f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Average path length vs regret
    scatter = ax2.scatter(avg_path_lengths, final_regrets, c=range(len(algorithms)), 
                         cmap='viridis', s=100, alpha=0.7)
    ax2.set_xlabel('Average Path Length (hops)')
    ax2.set_ylabel('Final Cumulative Regret')
    ax2.set_title('Path Length vs Regret')
    ax2.grid(True, alpha=0.3)
    
    # Add algorithm labels
    for i, alg in enumerate(algorithms):
        ax2.annotate(alg, (avg_path_lengths[i], final_regrets[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 3: Path length distribution heatmap
    hop_counts = set()
    for dist in path_distributions:
        hop_counts.update(dist.keys())
    hop_counts = sorted(hop_counts)
    
    if hop_counts:
        heatmap_data = []
        for dist in path_distributions:
            row = []
            for hop in hop_counts:
                row.append(dist.get(hop, {}).get('percentage', 0))
            heatmap_data.append(row)
        
        im = ax3.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        ax3.set_xlabel('Path Length (hops)')
        ax3.set_ylabel('Algorithms')
        ax3.set_title('Path Length Distribution Heatmap')
        ax3.set_xticks(range(len(hop_counts)))
        ax3.set_xticklabels([f'{h}-hop' for h in hop_counts])
        ax3.set_yticks(range(len(algorithms)))
        ax3.set_yticklabels(algorithms)
        
        # Add text annotations
        for i in range(len(algorithms)):
            for j in range(len(hop_counts)):
                text = ax3.text(j, i, f'{heatmap_data[i][j]:.0f}%',
                               ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax3, label='Percentage (%)')
    
    # Plot 4: Simulation time vs regret
    ax4.scatter(simulation_times, final_regrets, c=range(len(algorithms)), 
               cmap='plasma', s=100, alpha=0.7)
    ax4.set_xlabel('Simulation Time (seconds)')
    ax4.set_ylabel('Final Cumulative Regret')
    ax4.set_title('Simulation Time vs Regret')
    ax4.grid(True, alpha=0.3)
    
    # Add algorithm labels
    for i, alg in enumerate(algorithms):
        ax4.annotate(alg, (simulation_times[i], final_regrets[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('output/images/ucsb_all_algorithms_3hop_detailed_analysis.pdf', 
                dpi=300, bbox_inches='tight')
    print("Detailed analysis plots saved to: output/images/ucsb_all_algorithms_3hop_detailed_analysis.pdf")

if __name__ == "__main__":
    analyze_all_algorithms_3hop() 