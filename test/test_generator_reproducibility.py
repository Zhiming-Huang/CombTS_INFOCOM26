#!/usr/bin/env python3
"""
Test generator reproducibility with the new pre-generation approach.
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
from src.bandits.comb_ucb import CombUCB
from src.bandits.bg_cts import BGCTS
from src.bandits.cts_g import CTSG
from src.environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment


def test_generator_reproducibility():
    """Test if generators are reproducible with the new approach."""
    print("Testing generator reproducibility...")
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
    
    # Test with seed 777
    main_seed = 777
    main_rng = np.random.default_rng(main_seed)
    
    print(f"Testing with seed: {main_seed}")
    print()
    
    # Create environment
    env_rng = main_rng.spawn(1)[0]
    env = UCSBMeshnetMemmapEnvironment(rng=env_rng, **env_config)
    
    # Define algorithms (same as in the main script)
    base_algorithms = {
        'CTSB': CTSB,
        'CombUCB': CombUCB,
        'BG-CTS': BGCTS
    }
    
    gamma_algorithms = {
        'CTS-G': CTSG,
        'CL-SG': CLSG
    }
    
    gamma_values = [0.01, 0.1, 0.5, 1.0]
    
    # Calculate total number of algorithms
    total_algorithms = len(base_algorithms) + len(gamma_algorithms) * len(gamma_values)
    print(f"Total algorithms: {total_algorithms}")
    
    # Pre-generate all algorithm generators
    print("Pre-generating all algorithm generators...")
    all_alg_generators = main_rng.spawn(total_algorithms)
    
    # Create algorithm configurations
    alg_configs = []
    alg_idx = 0
    
    # Add base algorithms
    for alg_name, alg_class in base_algorithms.items():
        alg_generator = all_alg_generators[alg_idx]
        alg_configs.append((alg_name, alg_class, alg_generator, alg_idx))
        alg_idx += 1
    
    # Add gamma algorithms
    for alg_name, alg_class in gamma_algorithms.items():
        for gamma in gamma_values:
            alg_key = f"{alg_name.lower()}_gamma_{gamma}"
            alg_generator = all_alg_generators[alg_idx]
            alg_configs.append((alg_key, alg_class, alg_generator, alg_idx))
            alg_idx += 1
    
    print(f"Generated {len(alg_configs)} algorithm configurations")
    print()
    
    # Test each algorithm and show their first few random numbers
    print("Testing each algorithm's random number generation:")
    print("-" * 60)
    
    for alg_name, alg_class, alg_generator, alg_idx in alg_configs:
        # Create algorithm instance
        if alg_name == 'BG-CTS':
            alg = alg_class(environment=env, rnd_generator=alg_generator)
        else:
            alg = alg_class(environment=env, rng=alg_generator)
        
        # Get first few random numbers
        random_numbers = []
        for i in range(5):
            available_arms = env.get_available_arms_for_round(i)
            feasible_combinations = env.get_feasible_combinations(available_arms)
            if feasible_combinations:
                selected = alg.select_combination(i)
                # Get the random number used in this selection
                if hasattr(alg, 'rng'):
                    random_numbers.append(alg.rng.normal(0, 1))
                else:
                    random_numbers.append(alg.rnd_generator.normal(0, 1))
        
        print(f"Position {alg_idx}: {alg_name}")
        print(f"  First 5 random numbers: {[f'{x:.4f}' for x in random_numbers]}")
        print()
    
    # Cleanup
    env.cleanup()
    
    print("=" * 60)
    print("Analysis:")
    print("If the random numbers are consistent across runs, then")
    print("the generator assignment is reproducible.")


def test_multiple_runs():
    """Test multiple runs to verify reproducibility."""
    print("\nTesting multiple runs for reproducibility...")
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
    
    # Test with seed 777
    main_seed = 777
    
    print(f"Testing multiple runs with seed: {main_seed}")
    print()
    
    for run in range(3):
        print(f"Run {run + 1}:")
        
        # Create main RNG
        main_rng = np.random.default_rng(main_seed)
        
        # Create environment
        env_rng = main_rng.spawn(1)[0]
        env = UCSBMeshnetMemmapEnvironment(rng=env_rng, **env_config)
        
        # Pre-generate all algorithm generators
        all_alg_generators = main_rng.spawn(11)  # 11 total algorithms
        
        # Test CL-SG at position 7 (γ=0.01)
        clsg = CLSG(environment=env, rng=all_alg_generators[7], gamma=0.01, optimistic_init=True)
        
        # Get first few random numbers
        random_numbers = []
        for i in range(5):
            available_arms = env.get_available_arms_for_round(i)
            feasible_combinations = env.get_feasible_combinations(available_arms)
            if feasible_combinations:
                selected = clsg.select_combination(i)
                random_numbers.append(clsg.rng.normal(0, 1))
        
        print(f"  CL-SG (position 7) first 5 random numbers: {[f'{x:.4f}' for x in random_numbers]}")
        
        # Cleanup
        env.cleanup()
    
    print("\n" + "=" * 60)
    print("Analysis:")
    print("If all runs show the same random numbers, then")
    print("the generator assignment is fully reproducible.")


if __name__ == "__main__":
    test_generator_reproducibility()
    test_multiple_runs() 