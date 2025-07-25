#!/usr/bin/env python3
"""
Detailed test to track all sources of randomness.
"""

import sys
import os
import numpy as np

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.bandits.cl_sg import CLSG
from src.bandits.cts_b import CTSB
from src.bandits.cts_g import CTSG
from src.environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment


def test_clsg_detailed():
    """Test CL-SG algorithm in detail to find any remaining randomness."""
    print("Testing CL-SG Detailed Randomness")
    print("=" * 60)
    
    # Setup environment
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
    
    # Test parameters
    main_seed = 2023
    num_rounds = 10
    
    print(f"Testing with seed: {main_seed}")
    print()
    
    # Run multiple times with same seed
    for test_run in range(3):
        print(f"Test Run {test_run + 1}:")
        
        # Create main RNG
        main_rng = np.random.default_rng(main_seed)
        
        # Create environment
        env_rng = main_rng.spawn(1)[0]
        env = UCSBMeshnetMemmapEnvironment(rng=env_rng, **env_config)
        
        # Create CL-SG generator
        clsg_generator = main_rng.spawn(1)[0]
        
        # Create CL-SG instance
        clsg = CLSG(environment=env, rng=clsg_generator, gamma=0.1, optimistic_init=True)
        
        # Track all random numbers used
        random_numbers = []
        selections = []
        rewards = []
        
        for round_idx in range(num_rounds):
            # Get available arms
            available_arms = env.get_available_arms_for_round(round_idx)
            feasible_combinations = env.get_feasible_combinations(available_arms)
            
            if feasible_combinations:
                # Before selection, record the next random number that will be used
                test_rng_state = clsg.rng.bit_generator.state
                test_random = clsg.rng.normal(0, 1)
                # Restore state
                clsg.rng.bit_generator.state = test_rng_state
                random_numbers.append(test_random)
                
                # Select action
                selected = clsg.select_combination(round_idx)
                selections.append(sorted(list(selected)) if selected else [])
                
                # Get rewards
                total_reward = env.get_expected_reward_for_path(selected, round_idx)
                rewards.append(total_reward)
                
                # Update algorithm
                reward_dict = {}
                for arm in selected:
                    reward_dict[arm] = env.get_reward_for_round(arm, round_idx)
                clsg.update_posterior(selected, reward_dict, round_idx)
            else:
                random_numbers.append(None)
                selections.append([])
                rewards.append(0.0)
        
        print(f"  Random numbers: {[f'{x:.6f}' if x is not None else 'None' for x in random_numbers[:5]]}")
        print(f"  Selections: {selections[:3]}")
        print(f"  Rewards: {[f'{x:.6f}' for x in rewards[:3]]}")
        print()
        
        # Cleanup
        env.cleanup()
    
    print("=" * 60)
    print("Analysis: Check if all runs produce identical results")


def test_environment_randomness():
    """Test if environment creation itself has randomness."""
    print("\nTesting Environment Randomness")
    print("=" * 60)
    
    # Setup environment
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
    
    main_seed = 2023
    
    for test_run in range(3):
        print(f"Environment Test Run {test_run + 1}:")
        
        # Create main RNG
        main_rng = np.random.default_rng(main_seed)
        
        # Create environment
        env_rng = main_rng.spawn(1)[0]
        env = UCSBMeshnetMemmapEnvironment(rng=env_rng, **env_config)
        
        # Test first few rounds
        round_data = []
        for round_idx in range(3):
            available_arms = env.get_available_arms_for_round(round_idx)
            feasible_combinations = env.get_feasible_combinations(available_arms)
            if feasible_combinations:
                first_combination = sorted(list(feasible_combinations[0]))
                optimal_reward = env.get_optimal_path_expected_reward(round_idx)
                round_data.append((len(available_arms), len(feasible_combinations), first_combination, f'{optimal_reward:.6f}'))
            else:
                round_data.append((len(available_arms), 0, [], '0.000000'))
        
        print(f"  Round data: {round_data}")
        
        # Cleanup
        env.cleanup()
    
    print("=" * 60)
    print("Analysis: Check if environment produces identical results")


if __name__ == "__main__":
    test_clsg_detailed()
    test_environment_randomness() 