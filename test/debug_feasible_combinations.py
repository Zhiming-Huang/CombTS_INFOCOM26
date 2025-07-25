#!/usr/bin/env python3
"""
Debug feasible combinations for UCSB environment
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment

def debug_feasible_combinations():
    """Debug why no 3-hop feasible combinations are found"""
    
    print("Debugging Feasible Combinations")
    print("=" * 60)
    
    # Current best node pair
    source = "10.1.1.109"
    destination = "10.1.1.5"
    
    # Get the data directory
    data_dir = Path(__file__).parent.parent / "data" / "ucsb"
    target_folder = data_dir / "1144393236-1144450070"
    
    # Use first 5 files for testing
    test_files = sorted([f for f in target_folder.glob("neighbortable-*")])[:5]
    
    print(f"Using first {len(test_files)} files for testing")
    
    try:
        env = UCSBMeshnetMemmapEnvironment(
            neighbortable_files=[str(f) for f in test_files],
            source=source,
            destination=destination,
            routes_per_minute=1
        )
        
        print(f"Environment created successfully")
        print(f"Total rounds: {env.num_rounds}")
        print(f"Number of arms: {env.num_arms}")
        print(f"Max combination size: {env.max_combination_size}")
        
        # Test first round
        round_idx = 0
        print(f"\nTesting round {round_idx}:")
        
        # Get available arms
        available_arms = set(env.get_available_arms_for_round(round_idx))
        print(f"Available arms: {len(available_arms)}")
        
        if available_arms:
            print(f"Sample available arms: {list(available_arms)[:10]}")
            
            # Get all feasible combinations
            all_feasible_combinations = env.get_feasible_combinations(available_arms)
            print(f"All feasible combinations: {len(all_feasible_combinations)}")
            
            if all_feasible_combinations:
                print(f"Sample all feasible combinations:")
                for i, combo in enumerate(all_feasible_combinations[:5]):
                    print(f"  {i+1}: {combo} (size: {len(combo)})")
                
                # Check size distribution
                size_counts = {}
                for combo in all_feasible_combinations:
                    size = len(combo)
                    size_counts[size] = size_counts.get(size, 0) + 1
                
                print(f"Size distribution of feasible combinations:")
                for size in sorted(size_counts.keys()):
                    print(f"  {size}-hop paths: {size_counts[size]}")
                
                # Filter for 3-hop paths
                three_hop_combinations = [combo for combo in all_feasible_combinations if len(combo) == 3]
                print(f"3-hop feasible combinations: {len(three_hop_combinations)}")
                
                if three_hop_combinations:
                    print(f"Sample 3-hop combinations:")
                    for i, combo in enumerate(three_hop_combinations[:5]):
                        print(f"  {i+1}: {combo}")
            else:
                print("No feasible combinations found!")
                
                # Check if source and destination are connected
                print(f"\nChecking connectivity...")
                print(f"Source: {source}")
                print(f"Destination: {destination}")
                
                # Check if source and destination are in available arms
                source_arms = []
                dest_arms = []
                for arm in available_arms:
                    u, v = env.arm_to_link[arm]
                    if u == source:
                        source_arms.append(arm)
                    if v == destination:
                        dest_arms.append(arm)
                
                print(f"Arms from source: {source_arms}")
                print(f"Arms to destination: {dest_arms}")
        else:
            print("No available arms!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_feasible_combinations() 