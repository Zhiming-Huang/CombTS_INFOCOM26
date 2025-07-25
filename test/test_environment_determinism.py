#!/usr/bin/env python3
"""
Test to check if environment creation is truly deterministic.
"""

import sys
import os
import numpy as np
import hashlib

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment


def hash_environment_state(env, rounds_to_check=10):
    """Create a hash of environment state for comparison."""
    state_data = []
    
    # Basic properties
    state_data.append(str(env.num_rounds))
    state_data.append(str(env.num_arms))
    state_data.append(str(env.source))
    state_data.append(str(env.destination))
    
    # Check first few rounds
    for round_idx in range(min(rounds_to_check, env.num_rounds)):
        available_arms = env.get_available_arms_for_round(round_idx)
        feasible_combinations = env.get_feasible_combinations(available_arms)
        
        state_data.append(str(sorted(available_arms)))
        state_data.append(str(len(feasible_combinations)))
        
        if feasible_combinations:
            first_combo = sorted(list(next(iter(feasible_combinations))))
            optimal_reward = env.get_optimal_path_expected_reward(round_idx)
            state_data.append(str(first_combo))
            state_data.append(f"{optimal_reward:.10f}")
    
    # Create hash
    combined_data = "|".join(state_data)
    return hashlib.md5(combined_data.encode()).hexdigest()


def test_environment_determinism():
    """Test if environment creation is deterministic across multiple runs."""
    print("Testing Environment Determinism")
    print("=" * 60)
    
    # Setup environment config
    data_dir = os.path.join(project_root, 'data', 'ucsb')
    trace_period = "1144393236-1144450070"
    trace_path = os.path.join(data_dir, trace_period)
    
    neighbortable_files = sorted([os.path.join(trace_path, f) for f in os.listdir(trace_path) 
                                 if f.startswith('neighbortable-')])
    
    env_config = {
        'neighbortable_files': neighbortable_files,
        'routes_per_minute': 15,
        'source': '10.1.1.100',
        'destination': '10.1.1.102',
        'pre_generate_rewards': True,
        'use_persistent_files': False,
        'max_feasible_combinations': 50,
        'max_path_length': 3,
        'max_paths_per_algorithm': 25
    }
    
    # Test multiple environment creations with same seed
    seed = 777
    hashes = []
    
    for i in range(5):
        print(f"Creating environment {i+1} with seed {seed}...")
        
        rng = np.random.default_rng(seed)
        env = UCSBMeshnetMemmapEnvironment(rng=rng, **env_config)
        
        env_hash = hash_environment_state(env)
        hashes.append(env_hash)
        
        print(f"  Environment hash: {env_hash}")
        
        # Cleanup
        env.cleanup()
    
    # Check if all hashes are identical
    print(f"\n{'='*60}")
    if len(set(hashes)) == 1:
        print("✅ All environment instances are identical!")
    else:
        print("❌ Environment instances are different!")
        print("Unique hashes:")
        for i, h in enumerate(set(hashes)):
            count = hashes.count(h)
            print(f"  Hash {i+1}: {h} (appears {count} times)")


def test_numpy_generator_consistency():
    """Test NumPy generator consistency."""
    print("\n\nTesting NumPy Generator Consistency")
    print("=" * 60)
    
    seed = 777
    
    for test_run in range(3):
        print(f"\nTest run {test_run + 1}:")
        
        # Create main generator
        main_rng = np.random.default_rng(seed)
        
        # Spawn generators like in main script
        total_algorithms = 11
        num_runs = 3
        total_generators_needed = total_algorithms * (1 + num_runs)
        all_generators = main_rng.spawn(total_generators_needed)
        
        # Test first few generators
        test_values = []
        for i in range(5):
            gen = all_generators[i]
            val = gen.normal(0, 1)
            test_values.append(val)
        
        print(f"  First 5 generator values: {[f'{v:.10f}' for v in test_values]}")


if __name__ == "__main__":
    test_environment_determinism()
    test_numpy_generator_consistency() 