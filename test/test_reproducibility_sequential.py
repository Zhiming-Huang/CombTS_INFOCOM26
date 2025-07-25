#!/usr/bin/env python3
"""
Test reproducibility with sequential execution (no multiprocessing).
"""

import sys
import os
import numpy as np
import time
from typing import List, Dict, Any, Tuple

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.bandits.cts_b import CTSB
from src.bandits.comb_ucb import CombUCB
from src.bandits.cts_g import CTSG
from src.bandits.cl_sg import CLSG
from src.bandits.bg_cts import BGCTS
from src.environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment


def run_single_algorithm_sequential(alg_config: Tuple[str, Any, Any, int, int, np.random.Generator]) -> Tuple[str, np.ndarray, np.ndarray]:
    """
    Run a single algorithm sequentially (same as parallel version but no multiprocessing).
    """
    alg_name, alg_class, env, num_rounds, num_runs, alg_generator = alg_config
    
    # Initialize arrays
    regrets_array = np.zeros((num_runs, num_rounds))
    rewards_array = np.zeros((num_runs, num_rounds))
    
    # Run the algorithm
    for run_idx in range(num_runs):
        # Create run-specific generator from the algorithm generator
        run_generator = alg_generator.spawn(1)[0]
        
        # Create algorithm instance with the run-specific generator directly
        if alg_name == 'BG-CTS':
            alg = alg_class(environment=env, rnd_generator=run_generator)
        else:
            alg = alg_class(environment=env, rng=run_generator)
        
        cumulative_regret = 0.0
        
        for round_idx in range(num_rounds):
            # Get available arms for this round
            available_arms = env.get_available_arms_for_round(round_idx)
            
            # Get feasible combinations
            feasible_combinations = env.get_feasible_combinations(available_arms)
            
            if not feasible_combinations:
                # No feasible paths available - regret is 0 for this round
                regrets_array[run_idx, round_idx] = cumulative_regret
                rewards_array[run_idx, round_idx] = 0.0
                continue
            
            # Select action
            selected_combination = alg.select_combination(round_idx)
            
            # Calculate total reward for the path
            total_reward = env.get_expected_reward_for_path(selected_combination, round_idx)
            
            # Get optimal reward for regret calculation
            optimal_reward = env.get_optimal_path_expected_reward(round_idx)
            
            # Update regret
            regret = optimal_reward - total_reward
            cumulative_regret += regret
            
            # Store results
            regrets_array[run_idx, round_idx] = cumulative_regret
            rewards_array[run_idx, round_idx] = total_reward
            
            # Update algorithm
            reward_dict = {}
            for arm in selected_combination:
                reward_dict[arm] = env.get_reward_for_round(arm, round_idx)
            alg.update_posterior(selected_combination, reward_dict, round_idx)
    
    return alg_name, regrets_array, rewards_array


def test_sequential_reproducibility():
    """Test reproducibility with sequential execution."""
    print("Testing Sequential Reproducibility")
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
    
    # Test parameters
    main_seed = 2023
    num_rounds = 100  # Shorter for testing
    num_runs = 3
    
    print(f"Testing with:")
    print(f"  Seed: {main_seed}")
    print(f"  Rounds: {num_rounds}")
    print(f"  Runs: {num_runs}")
    print()
    
    # Run multiple times with same seed
    for test_run in range(3):
        print(f"Test Run {test_run + 1}:")
        
        # Create main RNG
        main_rng = np.random.default_rng(main_seed)
        
        # Create environment
        env_rng = main_rng.spawn(1)[0]
        env = UCSBMeshnetMemmapEnvironment(rng=env_rng, **env_config)
        
        # Pre-generate all algorithm generators
        all_alg_generators = main_rng.spawn(3)  # Just test 3 algorithms
        
        # Test algorithms
        algorithms = [
            ('CTSB', CTSB, all_alg_generators[0]),
            ('CombUCB', CombUCB, all_alg_generators[1]),
            ('CL-SG', CLSG, all_alg_generators[2])
        ]
        
        results = {}
        for i, (alg_name, alg_class, alg_generator) in enumerate(algorithms):
            # Run algorithm sequentially
            if alg_name == 'CL-SG':
                # Create a proper class wrapper for CL-SG
                class CLSGWrapper:
                    def __init__(self, environment, rng):
                        self.alg = CLSG(environment=environment, rng=rng, gamma=0.1, optimistic_init=True)
                    def __getattr__(self, name):
                        return getattr(self.alg, name)
                
                alg_config = (f"{alg_name}_gamma_0.1", CLSGWrapper, env, num_rounds, num_runs, alg_generator)
            else:
                alg_config = (alg_name, alg_class, env, num_rounds, num_runs, alg_generator)
            
            result_name, regrets_array, rewards_array = run_single_algorithm_sequential(alg_config)
            
            # Calculate final regret
            final_regrets = regrets_array[:, -1]
            avg_final_regret = np.mean(final_regrets)
            results[alg_name] = avg_final_regret
        
        print(f"  CTSB: {results['CTSB']:.4f}")
        print(f"  CombUCB: {results['CombUCB']:.4f}")
        print(f"  CL-SG: {results['CL-SG']:.4f}")
        print()
        
        # Cleanup
        env.cleanup()
    
    print("=" * 60)
    print("Analysis:")
    print("If all test runs show identical results, then")
    print("sequential execution is fully reproducible.")


if __name__ == "__main__":
    test_sequential_reproducibility() 