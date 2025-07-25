#!/usr/bin/env python3
"""
List feasible combinations count for each round/minute
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment

def list_feasible_per_round():
    """List feasible combinations count for each round/minute"""
    
    print("Listing Feasible Combinations per Round/Minute")
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
    
    # List feasible combinations for each round
    print("Round | File Name | Available Arms | Feasible Combinations | Size Breakdown")
    print("-" * 80)
    
    total_feasible = 0
    rounds_with_feasible = 0
    
    for round_idx in range(env.num_rounds):
        # Get available arms for this round
        available_arms = set(env.get_available_arms_for_round(round_idx))
        
        if available_arms:
            # Get feasible combinations for this round
            feasible_combinations = env.get_feasible_combinations(available_arms)
            
            # Count by size
            size_counts = {}
            for combo in feasible_combinations:
                size = len(combo)
                size_counts[size] = size_counts.get(size, 0) + 1
            
            # Get file name for this round
            file_name = neighbortable_files[round_idx].name
            
            # Format size breakdown
            size_breakdown = ", ".join([f"{size}-hop:{count}" for size, count in sorted(size_counts.items())])
            
            print(f"{round_idx:5d} | {file_name:20s} | {len(available_arms):14d} | {len(feasible_combinations):20d} | {size_breakdown}")
            
            total_feasible += len(feasible_combinations)
            if feasible_combinations:
                rounds_with_feasible += 1
        else:
            file_name = neighbortable_files[round_idx].name
            print(f"{round_idx:5d} | {file_name:20s} | {0:14d} | {0:20d} | No available arms")
    
    print("-" * 80)
    print(f"Summary:")
    print(f"  Total rounds: {env.num_rounds}")
    print(f"  Rounds with feasible combinations: {rounds_with_feasible}")
    print(f"  Total feasible combinations: {total_feasible}")
    print(f"  Average feasible combinations per round: {total_feasible / env.num_rounds:.1f}")

if __name__ == "__main__":
    list_feasible_per_round() 