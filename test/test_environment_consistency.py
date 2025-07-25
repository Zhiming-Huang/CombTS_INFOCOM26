#!/usr/bin/env python3
"""
Test to check if each algorithm uses consistent environment state.
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


def hash_array(arr):
    """Create hash of numpy array for comparison."""
    return hashlib.md5(arr.tobytes()).hexdigest()


def test_environment_consistency():
    """Test if multiple environment instances with same seed produce identical state."""
    print("Testing Environment Consistency")
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
        'max_path_length': 2,
        'max_paths_per_algorithm': 25
    }
    
    # Create 5 environment instances with the same seed
    seed = 2023
    environments = []
    environment_hashes = []
    
    for i in range(5):
        print(f"Creating environment {i+1} with seed {seed}...")
        
        # Use same seed for all environments
        rng = np.random.default_rng(seed)
        env = UCSBMeshnetMemmapEnvironment(rng=rng, **env_config)
        
        # Test some basic properties
        env_info = {
            'num_rounds': env.num_rounds,
            'num_arms': env.num_arms,
            'source': env.source,
            'destination': env.destination,
            'nodes_count': len(env.nodes),
            'arms_count': len(env.arms)
        }
        
        # Test first few rounds of topology and rewards
        topology_hashes = []
        rewards_hashes = []
        for round_idx in range(min(10, env.num_rounds)):
            # Test available arms
            available_arms = env.get_available_arms_for_round(round_idx)
            available_arms_hash = hash_array(np.array(sorted(available_arms)))
            
            # Test topology matrix for this round
            if hasattr(env, 'topology_memmap'):
                topology_slice = env.topology_memmap[round_idx]
                topology_hashes.append(hash_array(topology_slice))
            
            # Test rewards matrix for this round  
            if hasattr(env, 'rewards_memmap'):
                rewards_slice = env.rewards_memmap[round_idx]
                rewards_hashes.append(hash_array(rewards_slice))
            
            # Test feasible combinations
            feasible_combinations = env.get_feasible_combinations(available_arms)
            if feasible_combinations:
                first_combination = sorted(list(feasible_combinations[0]))
                optimal_reward = env.get_optimal_path_expected_reward(round_idx)
                env_info[f'round_{round_idx}_arms'] = available_arms_hash
                env_info[f'round_{round_idx}_feasible_count'] = len(feasible_combinations)
                env_info[f'round_{round_idx}_first_combination'] = str(first_combination)
                env_info[f'round_{round_idx}_optimal_reward'] = f'{optimal_reward:.10f}'
        
        env_info['topology_hashes'] = topology_hashes
        env_info['rewards_hashes'] = rewards_hashes
        
        environments.append(env)
        environment_hashes.append(env_info)
        
        print(f"  Environment {i+1} basic info:")
        print(f"    Rounds: {env_info['num_rounds']}, Arms: {env_info['num_arms']}")
        print(f"    Nodes: {env_info['nodes_count']}, Arms list: {env_info['arms_count']}")
        
        # Cleanup
        env.cleanup()
    
    # Compare all environments
    print("\n" + "=" * 60)
    print("Comparing environments...")
    
    reference = environment_hashes[0]
    all_identical = True
    
    for i, env_info in enumerate(environment_hashes[1:], 1):
        print(f"\nComparing environment {i+1} with reference:")
        
        # Compare basic properties
        for key in ['num_rounds', 'num_arms', 'source', 'destination', 'nodes_count', 'arms_count']:
            if env_info[key] != reference[key]:
                print(f"  ❌ {key}: {env_info[key]} != {reference[key]}")
                all_identical = False
            else:
                print(f"  ✅ {key}: {env_info[key]}")
        
        # Compare round-specific data
        for round_idx in range(min(10, reference['num_rounds'])):
            round_keys = [f'round_{round_idx}_arms', f'round_{round_idx}_feasible_count', 
                         f'round_{round_idx}_first_combination', f'round_{round_idx}_optimal_reward']
            
            for key in round_keys:
                if key in env_info and key in reference:
                    if env_info[key] != reference[key]:
                        print(f"  ❌ {key}: {env_info[key]} != {reference[key]}")
                        all_identical = False
        
        # Compare topology and rewards hashes
        if env_info['topology_hashes'] != reference['topology_hashes']:
            print(f"  ❌ Topology hashes differ")
            all_identical = False
        else:
            print(f"  ✅ Topology hashes identical")
            
        if env_info['rewards_hashes'] != reference['rewards_hashes']:
            print(f"  ❌ Rewards hashes differ")
            all_identical = False
        else:
            print(f"  ✅ Rewards hashes identical")
    
    print("\n" + "=" * 60)
    if all_identical:
        print("✅ All environments are identical!")
    else:
        print("❌ Environments have differences!")
    
    return all_identical


if __name__ == "__main__":
    test_environment_consistency() 