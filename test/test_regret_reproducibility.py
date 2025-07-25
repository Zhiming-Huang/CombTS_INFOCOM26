#!/usr/bin/env python3
"""
Test regret reproducibility with the new pre-generation approach.
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


def test_regret_reproducibility():
    """Test if regret is reproducible with the new approach."""
    print("Testing regret reproducibility...")
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
    num_rounds = 100
    num_runs = 3
    
    print(f"Testing with seed: {main_seed}")
    print(f"Rounds: {num_rounds}, Runs: {num_runs}")
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
        
        # Test CTSB at position 0
        ctsb = CTSB(environment=env, rng=all_alg_generators[0])
        
        # Run a few rounds and collect regrets
        regrets = []
        for i in range(10):
            available_arms = env.get_available_arms_for_round(i)
            feasible_combinations = env.get_feasible_combinations(available_arms)
            if feasible_combinations:
                selected = ctsb.select_combination(i)
                total_reward = env.get_expected_reward_for_path(selected, i)
                optimal_reward = env.get_optimal_path_expected_reward(i)
                regret = optimal_reward - total_reward
                regrets.append(regret)
                
                # Update algorithm
                reward_dict = {}
                for arm in selected:
                    reward_dict[arm] = env.get_reward_for_round(arm, i)
                ctsb.update_posterior(selected, reward_dict, i)
        
        print(f"  CTSB first 10 regrets: {[f'{x:.4f}' for x in regrets]}")
        
        # Cleanup
        env.cleanup()
    
    print("\n" + "=" * 60)
    print("Analysis:")
    print("If all runs show the same regrets, then")
    print("the regret calculation is fully reproducible.")


def test_algorithm_consistency():
    """Test if algorithms behave consistently across runs."""
    print("\nTesting algorithm consistency...")
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
    
    print(f"Testing algorithm consistency with seed: {main_seed}")
    print()
    
    # Create main RNG
    main_rng = np.random.default_rng(main_seed)
    
    # Create environment
    env_rng = main_rng.spawn(1)[0]
    env = UCSBMeshnetMemmapEnvironment(rng=env_rng, **env_config)
    
    # Pre-generate all algorithm generators
    all_alg_generators = main_rng.spawn(11)
    
    # Test different algorithms
    algorithms = [
        ('CTSB', CTSB, all_alg_generators[0]),
        ('CombUCB', CombUCB, all_alg_generators[1]),
        ('BG-CTS', BGCTS, all_alg_generators[2]),
        ('CL-SG', CLSG, all_alg_generators[7])  # γ=0.01
    ]
    
    for alg_name, alg_class, alg_generator in algorithms:
        print(f"Testing {alg_name}:")
        
        # Create algorithm instance
        if alg_name == 'BG-CTS':
            alg = alg_class(environment=env, rnd_generator=alg_generator)
        elif alg_name == 'CL-SG':
            alg = alg_class(environment=env, rng=alg_generator, gamma=0.01, optimistic_init=True)
        else:
            alg = alg_class(environment=env, rng=alg_generator)
        
        # Get first few selections
        selections = []
        for i in range(5):
            available_arms = env.get_available_arms_for_round(i)
            feasible_combinations = env.get_feasible_combinations(available_arms)
            if feasible_combinations:
                selected = alg.select_combination(i)
                selections.append(sorted(list(selected)))
        
        print(f"  First 5 selections: {selections}")
        print()
    
    # Cleanup
    env.cleanup()


if __name__ == "__main__":
    test_regret_reproducibility()
    test_algorithm_consistency() 