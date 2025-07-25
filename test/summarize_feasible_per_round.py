#!/usr/bin/env python3
"""
Summarize feasible combinations per round with statistics
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment

def summarize_feasible_per_round():
    """Summarize feasible combinations per round with statistics"""
    
    print("Summarizing Feasible Combinations per Round/Minute")
    print("=" * 60)
    
    # Node pair
    source = "10.1.1.109"
    destination = "10.1.1.5"
    
    # Get the data directory
    data_dir = Path(__file__).parent.parent / "data" / "ucsb"
    target_folder = data_dir / "1144393236-1144450070"
    
    # Get all neighbortable files
    neighbortable_files = sorted([f for f in target_folder.glob("neighbortable-*")])
    
    print(f"Node pair: {source} → {destination}")
    print(f"Total files: {len(neighbortable_files)}")
    print()
    
    # Create environment
    env = UCSBMeshnetMemmapEnvironment(
        neighbortable_files=[str(f) for f in neighbortable_files],
        source=source,
        destination=destination,
        routes_per_minute=1
    )
    
    print(f"Environment created with {env.num_rounds} rounds")
    print()
    
    # Collect data for all rounds
    feasible_counts = []
    available_arms_counts = []
    size_distributions = []
    
    for round_idx in range(env.num_rounds):
        # Get available arms for this round
        available_arms = set(env.get_available_arms_for_round(round_idx))
        available_arms_counts.append(len(available_arms))
        
        if available_arms:
            # Get feasible combinations for this round
            feasible_combinations = env.get_feasible_combinations(available_arms)
            feasible_counts.append(len(feasible_combinations))
            
            # Count by size
            size_counts = {}
            for combo in feasible_combinations:
                size = len(combo)
                size_counts[size] = size_counts.get(size, 0) + 1
            size_distributions.append(size_counts)
        else:
            feasible_counts.append(0)
            size_distributions.append({})
    
    # Calculate statistics
    feasible_counts = np.array(feasible_counts)
    available_arms_counts = np.array(available_arms_counts)
    
    print("FEASIBLE COMBINATIONS PER ROUND STATISTICS:")
    print("-" * 50)
    print(f"Total rounds: {len(feasible_counts)}")
    print(f"Rounds with feasible combinations: {np.sum(feasible_counts > 0)}")
    print(f"Rounds without feasible combinations: {np.sum(feasible_counts == 0)}")
    print()
    
    print("Feasible Combinations Count:")
    print(f"  Total feasible combinations: {np.sum(feasible_counts)}")
    print(f"  Average per round: {np.mean(feasible_counts):.2f}")
    print(f"  Median per round: {np.median(feasible_counts):.2f}")
    print(f"  Standard deviation: {np.std(feasible_counts):.2f}")
    print(f"  Minimum per round: {np.min(feasible_counts)}")
    print(f"  Maximum per round: {np.max(feasible_counts)}")
    print()
    
    print("Available Arms Count:")
    print(f"  Average available arms per round: {np.mean(available_arms_counts):.2f}")
    print(f"  Minimum available arms per round: {np.min(available_arms_counts)}")
    print(f"  Maximum available arms per round: {np.max(available_arms_counts)}")
    print()
    
    # Analyze size distribution
    print("SIZE DISTRIBUTION ANALYSIS:")
    print("-" * 50)
    
    total_by_size = {}
    for size_dist in size_distributions:
        for size, count in size_dist.items():
            total_by_size[size] = total_by_size.get(size, 0) + count
    
    for size in sorted(total_by_size.keys()):
        print(f"  {size}-hop paths: {total_by_size[size]} total combinations")
    
    print()
    
    # Show sample rounds
    print("SAMPLE ROUNDS (first 20):")
    print("-" * 50)
    print("Round | Available Arms | Feasible Combinations | Size Breakdown")
    print("-" * 70)
    
    for round_idx in range(min(20, len(feasible_counts))):
        file_name = neighbortable_files[round_idx].name
        size_breakdown = ", ".join([f"{size}-hop:{count}" for size, count in sorted(size_distributions[round_idx].items())])
        if not size_breakdown:
            size_breakdown = "No feasible combinations"
        
        print(f"{round_idx:5d} | {available_arms_counts[round_idx]:14d} | {feasible_counts[round_idx]:20d} | {size_breakdown}")
    
    print()
    
    # Show rounds with different counts
    unique_counts = np.unique(feasible_counts)
    print("ROUNDS BY FEASIBLE COMBINATION COUNT:")
    print("-" * 50)
    for count in sorted(unique_counts):
        num_rounds = np.sum(feasible_counts == count)
        percentage = (num_rounds / len(feasible_counts)) * 100
        print(f"  {count} combinations: {num_rounds} rounds ({percentage:.1f}%)")

if __name__ == "__main__":
    summarize_feasible_per_round() 