#!/usr/bin/env python3
"""
Test CL-SG at position 7 with seed 777 to verify the previous result.
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
from src.environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment


def test_clsg_position_7():
    """Test CL-SG at position 7 with seed 777."""
    print("Testing CL-SG at position 7 with seed 777...")
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
    
    # Create generators for positions 0-7
    generators = []
    for i in range(8):
        generators.append(main_rng.spawn(1)[0])
    
    # Test CTSB at position 0
    print("Testing CTSB at position 0...")
    ctsb = CTSB(environment=env, rng=generators[0])
    
    # Test CL-SG at position 7 (γ=0.01)
    print("Testing CL-SG at position 7 (γ=0.01)...")
    clsg = CLSG(environment=env, rng=generators[7], gamma=0.01, optimistic_init=True)
    
    # Run a short test
    num_rounds = 100
    ctsb_regrets = []
    clsg_regrets = []
    
    for round_idx in range(num_rounds):
        # Test CTSB
        available_arms = env.get_available_arms_for_round(round_idx)
        feasible_combinations = env.get_feasible_combinations(available_arms)
        
        if feasible_combinations:
            # CTSB
            ctsb_selected = ctsb.select_combination(round_idx)
            ctsb_reward = env.get_expected_reward_for_path(ctsb_selected, round_idx)
            optimal_reward = env.get_optimal_path_expected_reward(round_idx)
            ctsb_regret = optimal_reward - ctsb_reward
            ctsb_regrets.append(ctsb_regret)
            
            # Update CTSB
            reward_dict = {}
            for arm in ctsb_selected:
                reward_dict[arm] = env.get_reward_for_round(arm, round_idx)
            ctsb.update_posterior(ctsb_selected, reward_dict, round_idx)
            
            # CL-SG
            clsg_selected = clsg.select_combination(round_idx)
            clsg_reward = env.get_expected_reward_for_path(clsg_selected, round_idx)
            clsg_regret = optimal_reward - clsg_reward
            clsg_regrets.append(clsg_regret)
            
            # Update CL-SG
            reward_dict = {}
            for arm in clsg_selected:
                reward_dict[arm] = env.get_reward_for_round(arm, round_idx)
            clsg.update_posterior(clsg_selected, reward_dict, round_idx)
    
    # Calculate cumulative regrets
    ctsb_cumulative = np.cumsum(ctsb_regrets)
    clsg_cumulative = np.cumsum(clsg_regrets)
    
    print(f"CTSB final cumulative regret: {ctsb_cumulative[-1]:.2f}")
    print(f"CL-SG final cumulative regret: {clsg_cumulative[-1]:.2f}")
    
    if clsg_cumulative[-1] < ctsb_cumulative[-1]:
        print("✅ CL-SG performs better than CTSB!")
        improvement = (ctsb_cumulative[-1] - clsg_cumulative[-1]) / ctsb_cumulative[-1] * 100
        print(f"Improvement: {improvement:.1f}%")
    else:
        print("❌ CL-SG does not perform better than CTSB")
        difference = (clsg_cumulative[-1] - ctsb_cumulative[-1]) / ctsb_cumulative[-1] * 100
        print(f"Difference: {difference:.1f}%")
    
    # Cleanup
    env.cleanup()


if __name__ == "__main__":
    test_clsg_position_7() 