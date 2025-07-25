#!/usr/bin/env python3
"""
Quick test for 4-hop feasible combinations count.
"""

import sys
import os
import numpy as np

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment


def quick_4hop_test():
    """Quick test of 4-hop feasible combinations."""
    print("Quick 4-Hop Feasible Combinations Test")
    print("=" * 50)
    
    # Setup environment
    data_dir = os.path.join(project_root, 'data', 'ucsb')
    trace_period = "1144393236-1144450070"
    trace_path = os.path.join(data_dir, trace_period)
    
    neighbortable_files = sorted([os.path.join(trace_path, f) for f in os.listdir(trace_path) 
                                 if f.startswith('neighbortable-')])[:20]  # Use fewer files for speed
    
    env_config = {
        'neighbortable_files': neighbortable_files,
        'routes_per_minute': 15,
        'source': '10.1.1.100',
        'destination': '10.1.1.102',
        'pre_generate_rewards': True,
        'use_persistent_files': False,
        'max_feasible_combinations': 5000,  # Reasonable limit
        'max_paths_per_algorithm': 5000,
        'max_path_length': 4
    }
    
    # Create environment
    rng = np.random.default_rng(2023)
    env = UCSBMeshnetMemmapEnvironment(rng=rng, **env_config)
    
    print(f"Environment created with {env.num_rounds} rounds")
    print("Testing first 10 rounds...")
    
    # Test first 10 rounds only
    for round_idx in range(min(10, env.num_rounds)):
        print(f"Round {round_idx}: ", end="", flush=True)
        
        available_arms = env.get_available_arms_for_round(round_idx)
        print(f"{len(available_arms)} available arms → ", end="", flush=True)
        
        feasible_combinations = env.get_feasible_combinations(available_arms)
        print(f"{len(feasible_combinations)} feasible combinations")
        
        if round_idx == 0 and len(feasible_combinations) > 10:
            print(f"  First few combinations:")
            for i, combo in enumerate(list(feasible_combinations)[:5]):
                print(f"    {i+1}: {sorted(list(combo))}")
    
    # Cleanup
    env.cleanup()
    print("\nTest complete!")


if __name__ == "__main__":
    quick_4hop_test() 