#!/usr/bin/env python3
"""
Summary of 3-hop path improvement in UCSB environment.
Shows the difference between original (2-hop only) and improved (mixed 1-3 hop) tests.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import matplotlib.pyplot as plt
import numpy as np

def create_summary_plot():
    """Create a summary plot showing the improvement."""
    
    print("=" * 80)
    print("3-HOP PATH IMPROVEMENT SUMMARY")
    print("=" * 80)
    
    # Load results from both tests
    try:
        with open("output/data/ucsb_10000_rounds_3hop_results.json", 'r') as f:
            original_results = json.load(f)
        print("✓ Loaded original results (2-hop only)")
    except FileNotFoundError:
        print("✗ Original results not found")
        original_results = None
    
    try:
        with open("output/data/ucsb_10000_rounds_3hop_improved_results.json", 'r') as f:
            improved_results = json.load(f)
        print("✓ Loaded improved results (mixed 1-3 hop)")
    except FileNotFoundError:
        print("✗ Improved results not found")
        improved_results = None
    
    if not original_results or not improved_results:
        print("Cannot create summary without both result files")
        return
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Regret comparison
    algorithms = list(original_results['algorithms'].keys())
    original_regrets = [original_results['algorithms'][alg]['final_regret'] for alg in algorithms]
    improved_regrets = [improved_results['algorithms'][alg]['final_regret'] for alg in algorithms]
    
    x = np.arange(len(algorithms))
    width = 0.35
    
    ax1.bar(x - width/2, original_regrets, width, label='Original (2-hop only)', alpha=0.8)
    ax1.bar(x + width/2, improved_regrets, width, label='Improved (1-3 hop)', alpha=0.8)
    
    ax1.set_xlabel('Algorithms')
    ax1.set_ylabel('Final Cumulative Regret')
    ax1.set_title('Regret Comparison: Original vs Improved')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Path length distribution
    path_distributions = []
    labels = []
    
    for alg in algorithms:
        if 'path_distribution' in improved_results['algorithms'][alg]:
            dist = improved_results['algorithms'][alg]['path_distribution']
            if dist:
                # Extract percentages for each hop count
                hop_counts = sorted(dist.keys())
                percentages = [dist[hop]['percentage'] for hop in hop_counts]
                path_distributions.append(percentages)
                labels.append(f"{alg}\n" + "\n".join([f"{hop}hop: {pct:.1f}%" for hop, pct in zip(hop_counts, percentages)]))
    
    if path_distributions:
        # Create stacked bar chart
        hop_counts = sorted(improved_results['algorithms'][algorithms[0]]['path_distribution'].keys())
        bottom = np.zeros(len(path_distributions))
        
        colors = ['#ff7f0e', '#2ca02c', '#d62728']  # Orange, Green, Red for 1, 2, 3 hops
        
        for i, hop_count in enumerate(hop_counts):
            values = [dist[i] for dist in path_distributions]
            ax2.bar(range(len(path_distributions)), values, bottom=bottom, 
                   label=f'{hop_count}-hop paths', color=colors[i % len(colors)], alpha=0.8)
            bottom += values
        
        ax2.set_xlabel('Algorithms')
        ax2.set_ylabel('Path Distribution (%)')
        ax2.set_title('Path Length Distribution (Improved Test)')
        ax2.set_xticks(range(len(path_distributions)))
        ax2.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/images/ucsb_3hop_improvement_summary.pdf', dpi=300, bbox_inches='tight')
    print("Summary plot saved to: output/images/ucsb_3hop_improvement_summary.pdf")
    
    # Print detailed comparison
    print(f"\n{'='*80}")
    print("DETAILED COMPARISON")
    print(f"{'='*80}")
    print(f"{'Algorithm':<15} {'Original Regret':<15} {'Improved Regret':<15} {'Change':<10} {'Path Distribution'}")
    print("-" * 80)
    
    for alg in algorithms:
        orig_regret = original_results['algorithms'][alg]['final_regret']
        impr_regret = improved_results['algorithms'][alg]['final_regret']
        change = impr_regret - orig_regret
        change_str = f"{change:+.1f}"
        
        # Get path distribution
        if 'path_distribution' in improved_results['algorithms'][alg]:
            dist = improved_results['algorithms'][alg]['path_distribution']
            path_str = ""
            for hop in sorted(dist.keys()):
                pct = dist[hop]['percentage']
                path_str += f"{hop}hop:{pct:.0f}% "
        else:
            path_str = "N/A"
        
        print(f"{alg:<15} {orig_regret:<15.2f} {impr_regret:<15.2f} {change_str:<10} {path_str}")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    # Calculate average path length for each algorithm
    print("Average path length per algorithm (improved test):")
    for alg in algorithms:
        if 'path_distribution' in improved_results['algorithms'][alg]:
            dist = improved_results['algorithms'][alg]['path_distribution']
            avg_length = sum(int(hop) * dist[hop]['percentage'] / 100 for hop in dist.keys())
            print(f"  {alg}: {avg_length:.2f} hops")
    
    # Overall improvement
    orig_avg = np.mean(original_regrets)
    impr_avg = np.mean(improved_regrets)
    improvement = ((orig_avg - impr_avg) / orig_avg) * 100
    
    print(f"\nOverall performance:")
    print(f"  Original average regret: {orig_avg:.2f}")
    print(f"  Improved average regret: {impr_avg:.2f}")
    print(f"  Change: {impr_avg - orig_avg:+.2f} ({improvement:+.1f}%)")
    
    if improvement > 0:
        print(f"  ✓ Improved performance with mixed path lengths!")
    else:
        print(f"  ✗ Performance degraded with mixed path lengths")

if __name__ == "__main__":
    create_summary_plot() 