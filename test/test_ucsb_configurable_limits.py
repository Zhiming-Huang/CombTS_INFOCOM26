#!/usr/bin/env python3
"""
Test script to demonstrate configurable feasible combination limits in UCSB environment.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment

def test_configurable_limits():
    """Test different feasible combination limits."""
    
    # Get the largest dataset folder
    data_dir = "data/ucsb/1144393236-1144450070"
    neighbortable_files = []
    for f in os.listdir(data_dir):
        if f.startswith("neighbortable-"):
            neighbortable_files.append(os.path.join(data_dir, f))
    neighbortable_files.sort()
    
    print(f"Found {len(neighbortable_files)} neighbortable files")
    
    # Test different limit configurations
    test_configs = [
        {"max_feasible_combinations": 10, "max_path_length": 3, "max_paths_per_algorithm": 5, "name": "Very Limited"},
        {"max_feasible_combinations": 25, "max_path_length": 5, "max_paths_per_algorithm": 15, "name": "Moderate"},
        {"max_feasible_combinations": 50, "max_path_length": 8, "max_paths_per_algorithm": 25, "name": "Default"},
        {"max_feasible_combinations": 100, "max_path_length": 10, "max_paths_per_algorithm": 50, "name": "High"},
    ]
    
    # Use the best node pair we found earlier
    source = "10.1.1.109"
    destination = "10.1.1.5"
    
    print(f"\nTesting with source: {source}, destination: {destination}")
    print("=" * 80)
    
    for config in test_configs:
        print(f"\nConfiguration: {config['name']}")
        print(f"  max_feasible_combinations: {config['max_feasible_combinations']}")
        print(f"  max_path_length: {config['max_path_length']}")
        print(f"  max_paths_per_algorithm: {config['max_paths_per_algorithm']}")
        
        # Create environment with custom limits
        env = UCSBMeshnetMemmapEnvironment(
            neighbortable_files=neighbortable_files[:10],  # Use first 10 files for quick test
            routes_per_minute=4,
            source=source,
            destination=destination,
            max_feasible_combinations=config['max_feasible_combinations'],
            max_path_length=config['max_path_length'],
            max_paths_per_algorithm=config['max_paths_per_algorithm']
        )
        
        # Test a few rounds
        total_combinations = 0
        rounds_with_combinations = 0
        
        for round_idx in range(min(20, env.num_rounds)):
            available_arms = env.get_available_arms_for_round(round_idx)
            if available_arms:
                feasible_combinations = env.get_feasible_combinations(set(available_arms))
                total_combinations += len(feasible_combinations)
                if feasible_combinations:
                    rounds_with_combinations += 1
                    
                    # Check path lengths
                    path_lengths = []
                    for combo in feasible_combinations:
                        # Convert arm indices back to path
                        path = []
                        arm_links = [env.arm_to_link[arm] for arm in combo]
                        # Reconstruct path from links
                        current = source
                        path = [current]
                        while current != destination and len(path) < 20:  # Safety limit
                            for u, v in arm_links:
                                if u == current:
                                    current = v
                                    path.append(current)
                                    break
                            else:
                                break
                        path_lengths.append(len(path) - 1)  # Number of hops
                    
                    max_path_len = max(path_lengths) if path_lengths else 0
                    avg_path_len = np.mean(path_lengths) if path_lengths else 0
                    
                    print(f"    Round {round_idx}: {len(feasible_combinations)} combinations, "
                          f"max path length: {max_path_len}, avg path length: {avg_path_len:.1f}")
        
        avg_combinations = total_combinations / max(1, rounds_with_combinations)
        print(f"  Average combinations per round: {avg_combinations:.1f}")
        print(f"  Rounds with combinations: {rounds_with_combinations}/20")
        
        env.cleanup()
    
    print("\n" + "=" * 80)
    print("Test completed!")

if __name__ == "__main__":
    test_configurable_limits() 