#!/usr/bin/env python3
"""
Debug script to test UCSB environment pathfinding.
"""

import sys
import os
import numpy as np
import glob

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment

def debug_ucsb_pathfinding():
    """Debug UCSB environment pathfinding."""
    
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
    
    # Use first 10 files for testing
    neighbortable_files = sorted(neighbortable_files)[:10]
    print(f"Using {len(neighbortable_files)} neighbortable files")
    
    # Create environment
    env = UCSBMeshnetMemmapEnvironment(
        neighbortable_files=neighbortable_files,
        routes_per_minute=4,
        pre_generate_rewards=True,
        use_persistent_files=False,
        rng=np.random.default_rng(42),
        max_feasible_combinations=50,
        max_path_length=3,
        max_paths_per_algorithm=25
    )
    
    print(f"Environment created:")
    print(f"  Source: {env.get_source_destination()[0]}")
    print(f"  Destination: {env.get_source_destination()[1]}")
    print(f"  Number of nodes: {len(env.get_nodes())}")
    print(f"  Number of rounds: {env.num_rounds}")
    print(f"  Max path length: {env.max_path_length}")
    
    # Test pathfinding for first few rounds
    for round_idx in range(min(5, env.num_rounds)):
        print(f"\nRound {round_idx}:")
        
        # Get available arms
        available_arms = env.get_available_arms_for_round(round_idx)
        print(f"  Available arms: {len(available_arms)}")
        
        if len(available_arms) > 0:
            # Get feasible combinations
            feasible_combinations = env.get_feasible_combinations(available_arms)
            print(f"  Feasible combinations: {len(feasible_combinations)}")
            
            if len(feasible_combinations) > 0:
                print(f"  First combination: {list(feasible_combinations[0])}")
                
                # Test optimal path
                optimal_reward = env.get_optimal_path_expected_reward(round_idx)
                print(f"  Optimal reward: {optimal_reward:.3f}")
            else:
                print("  No feasible combinations found!")
        else:
            print("  No available arms!")
    
    # Clean up
    env.cleanup()

if __name__ == "__main__":
    debug_ucsb_pathfinding() 