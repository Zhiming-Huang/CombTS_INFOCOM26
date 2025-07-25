#!/usr/bin/env python3
"""
Test random number generation consistency across different runs.
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


def test_generator_consistency():
    """Test if generators are consistent across runs."""
    print("Testing random number generator consistency...")
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
    
    # Test different generator positions
    test_positions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    for pos in test_positions:
        # Create generator at specific position
        test_rng = main_rng.spawn(1)[0]
        
        # Create CL-SG with this generator
        clsg = CLSG(environment=env, rng=test_rng, gamma=0.01, optimistic_init=True)
        
        # Get first few random numbers
        random_numbers = []
        for i in range(5):
            available_arms = env.get_available_arms_for_round(i)
            feasible_combinations = env.get_feasible_combinations(available_arms)
            if feasible_combinations:
                selected = clsg.select_combination(i)
                # Get the random number used in this selection
                random_numbers.append(clsg.rng.normal(0, 1))
        
        print(f"Position {pos}: First 5 random numbers: {[f'{x:.4f}' for x in random_numbers]}")
    
    # Cleanup
    env.cleanup()
    
    print("\n" + "=" * 60)
    print("Analysis:")
    print("If the random numbers are different at each position, then")
    print("the generator assignment is working correctly.")
    print("If they are the same, there might be an issue.")


def test_algorithm_order():
    """Test the order of algorithm creation."""
    print("\nTesting algorithm creation order...")
    print("=" * 60)
    
    main_seed = 777
    main_rng = np.random.default_rng(main_seed)
    
    # Simulate the algorithm creation process
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
    
    print("Algorithm creation order:")
    alg_count = 0
    
    # Base algorithms
    for alg_name, alg_class in base_algorithms.items():
        alg_generator = main_rng.spawn(1)[0]
        print(f"{alg_count}: {alg_name} - Generator {alg_count}")
        alg_count += 1
    
    # Gamma algorithms
    for alg_name, alg_class in gamma_algorithms.items():
        for gamma in gamma_values:
            alg_key = f"{alg_name.lower()}_gamma_{gamma}"
            alg_generator = main_rng.spawn(1)[0]
            print(f"{alg_count}: {alg_key} - Generator {alg_count}")
            alg_count += 1
    
    print(f"\nTotal algorithms: {alg_count}")
    print(f"CL-SG (γ=0.01) should be at position: {3 + 0} = 3")
    print(f"CL-SG (γ=0.1) should be at position: {3 + 1} = 4")
    print(f"CL-SG (γ=0.5) should be at position: {3 + 2} = 5")
    print(f"CL-SG (γ=1.0) should be at position: {3 + 3} = 6")


if __name__ == "__main__":
    test_generator_consistency()
    test_algorithm_order() 