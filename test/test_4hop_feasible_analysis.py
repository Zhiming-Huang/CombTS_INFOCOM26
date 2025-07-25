#!/usr/bin/env python3
"""
Analyze feasible combinations count for 4-hop paths.
"""

import sys
import os
import numpy as np

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment


def analyze_feasible_combinations():
    """Analyze feasible combinations for different hop limits."""
    print("Analyzing Feasible Combinations Count")
    print("=" * 60)
    
    # Setup environment
    data_dir = os.path.join(project_root, 'data', 'ucsb')
    trace_period = "1144393236-1144450070"
    trace_path = os.path.join(data_dir, trace_period)
    
    neighbortable_files = sorted([os.path.join(trace_path, f) for f in os.listdir(trace_path) 
                                 if f.startswith('neighbortable-')])
    
    base_config = {
        'neighbortable_files': neighbortable_files,
        'routes_per_minute': 15,
        'source': '10.1.1.100',
        'destination': '10.1.1.102',
        'pre_generate_rewards': True,
        'use_persistent_files': False,
        'max_feasible_combinations': 10000,  # Remove practical limit
        'max_paths_per_algorithm': 10000     # Remove practical limit
    }
    
    # Test different hop limits
    hop_limits = [2, 3, 4]
    
    for max_hops in hop_limits:
        print(f"\n{'='*20} {max_hops}-Hop Analysis {'='*20}")
        
        env_config = base_config.copy()
        env_config['max_path_length'] = max_hops
        
        # Create environment
        rng = np.random.default_rng(2023)
        env = UCSBMeshnetMemmapEnvironment(rng=rng, **env_config)
        
        print(f"Total rounds: {env.num_rounds}")
        print(f"Total arms: {env.num_arms}")
        print()
        
        # Analyze first 100 rounds
        feasible_counts = []
        available_arms_counts = []
        
        rounds_to_check = min(100, env.num_rounds)
        
        for round_idx in range(rounds_to_check):
            available_arms = env.get_available_arms_for_round(round_idx)
            feasible_combinations = env.get_feasible_combinations(available_arms)
            
            available_arms_counts.append(len(available_arms))
            feasible_counts.append(len(feasible_combinations))
            
            if round_idx < 10:  # Show details for first 10 rounds
                print(f"Round {round_idx:2d}: {len(available_arms):3d} available arms → {len(feasible_combinations):3d} feasible combinations")
        
        # Statistics
        print(f"\nStatistics for first {rounds_to_check} rounds:")
        print(f"Available arms - Mean: {np.mean(available_arms_counts):.1f}, "
              f"Min: {np.min(available_arms_counts)}, Max: {np.max(available_arms_counts)}")
        print(f"Feasible combinations - Mean: {np.mean(feasible_counts):.1f}, "
              f"Min: {np.min(feasible_counts)}, Max: {np.max(feasible_counts)}")
        print(f"Standard deviation: {np.std(feasible_counts):.1f}")
        
        # Show distribution
        print(f"\nFeasible combinations distribution:")
        unique_counts, count_frequencies = np.unique(feasible_counts, return_counts=True)
        for count, freq in zip(unique_counts[:10], count_frequencies[:10]):  # Show top 10
            print(f"  {count:2d} combinations: {freq:2d} rounds ({freq/rounds_to_check*100:.1f}%)")
        
        if len(unique_counts) > 10:
            print(f"  ... and {len(unique_counts) - 10} more values")
        
        # Cleanup
        env.cleanup()
    
    print(f"\n{'='*60}")
    print("Analysis complete!")


if __name__ == "__main__":
    analyze_feasible_combinations() 