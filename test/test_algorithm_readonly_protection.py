#!/usr/bin/env python3
"""
Test to verify that algorithms cannot modify read-only environment state.
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
from src.environments.readonly_environment_wrapper import ReadOnlyEnvironmentWrapper


def test_algorithm_readonly_protection():
    """Test that algorithms cannot modify read-only environment."""
    print("Testing Algorithm Read-Only Protection")
    print("=" * 60)
    
    # Setup environment
    data_dir = os.path.join(project_root, 'data', 'ucsb')
    trace_period = "1144393236-1144450070"
    trace_path = os.path.join(data_dir, trace_period)
    
    neighbortable_files = sorted([os.path.join(trace_path, f) for f in os.listdir(trace_path) 
                                 if f.startswith('neighbortable-')])[:5]  # Use fewer files for test
    
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
    
    # Create environment and wrap it
    rng = np.random.default_rng(2023)
    original_env = UCSBMeshnetMemmapEnvironment(rng=rng, **env_config)
    readonly_env = ReadOnlyEnvironmentWrapper(original_env)
    
    print("✅ Created read-only environment")
    
    # Test CL-SG algorithm with read-only environment
    print("\nTesting CL-SG with read-only environment:")
    
    try:
        # Create algorithm with read-only environment
        alg_rng = np.random.default_rng(42)
        clsg = CLSG(environment=readonly_env, rng=alg_rng, gamma=0.1, optimistic_init=True)
        
        print("✅ CL-SG algorithm created successfully")
        
        # Test normal algorithm operations
        available_arms = readonly_env.get_available_arms_for_round(0)
        feasible_combinations = readonly_env.get_feasible_combinations(available_arms)
        
        if feasible_combinations:
            # Select combination
            selected_combination = clsg.select_combination(0)
            print(f"✅ CL-SG selected combination: {sorted(list(selected_combination))}")
            
            # Test reward calculation
            reward = readonly_env.get_expected_reward_for_path(selected_combination, 0)
            print(f"✅ Reward calculation: {reward:.6f}")
            
            # Test update (this should work with read-only environment)
            reward_dict = {}
            for arm in selected_combination:
                reward_dict[arm] = readonly_env.get_reward_for_round(arm, 0)
            
            clsg.update_posterior(selected_combination, reward_dict, 0)
            print("✅ Algorithm update successful")
        
    except Exception as e:
        print(f"❌ Algorithm operation failed: {e}")
    
    # Test that environment state cannot be modified
    print("\nTesting environment modification prevention:")
    
    # Test direct attribute modification
    try:
        readonly_env.num_rounds = 999
        print("❌ Environment attribute modification should have failed!")
    except AttributeError as e:
        print(f"✅ Environment attribute modification prevented: {e}")
    
    # Test array modification
    try:
        if hasattr(readonly_env, 'topology_memmap'):
            topology = readonly_env.topology_memmap
            original_value = topology[0, 0, 0]
            topology[0, 0, 0] = not original_value  # Try to flip the value
            print("❌ Environment array modification should have failed!")
    except (ValueError, RuntimeError) as e:
        print(f"✅ Environment array modification prevented: {type(e).__name__}")
    
    # Test that returned objects can be modified without affecting original
    try:
        available_arms_copy = readonly_env.get_available_arms_for_round(0)
        original_length = len(available_arms_copy)
        
        # Modify the copy
        available_arms_copy.append(999)
        
        # Check that original is unchanged
        available_arms_original = readonly_env.get_available_arms_for_round(0)
        if len(available_arms_original) == original_length:
            print("✅ Modifying returned objects doesn't affect original environment")
        else:
            print("❌ Modifying returned objects affected original environment!")
    
    except Exception as e:
        print(f"⚠️  Object modification test: {e}")
    
    # Compare environment state before and after algorithm usage
    print("\nTesting environment state preservation:")
    
    # Get environment state fingerprint
    state_before = {
        'num_rounds': readonly_env.num_rounds,
        'num_arms': readonly_env.num_arms,
        'source': readonly_env.source,
        'destination': readonly_env.destination
    }
    
    # Run algorithm for a few rounds
    for round_idx in range(5):
        available_arms = readonly_env.get_available_arms_for_round(round_idx)
        feasible_combinations = readonly_env.get_feasible_combinations(available_arms)
        
        if feasible_combinations:
            selected_combination = clsg.select_combination(round_idx)
            reward_dict = {}
            for arm in selected_combination:
                reward_dict[arm] = readonly_env.get_reward_for_round(arm, round_idx)
            clsg.update_posterior(selected_combination, reward_dict, round_idx)
    
    # Get environment state after
    state_after = {
        'num_rounds': readonly_env.num_rounds,
        'num_arms': readonly_env.num_arms,
        'source': readonly_env.source,
        'destination': readonly_env.destination
    }
    
    # Compare states
    if state_before == state_after:
        print("✅ Environment state unchanged after algorithm execution")
    else:
        print("❌ Environment state was modified!")
        for key in state_before:
            if state_before[key] != state_after[key]:
                print(f"  {key}: {state_before[key]} → {state_after[key]}")
    
    # Cleanup
    original_env.cleanup()
    print("\n✅ Test completed")


if __name__ == "__main__":
    test_algorithm_readonly_protection() 