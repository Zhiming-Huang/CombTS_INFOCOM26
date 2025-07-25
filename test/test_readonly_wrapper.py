#!/usr/bin/env python3
"""
Test the read-only environment wrapper.
"""

import sys
import os
import numpy as np

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.environments.readonly_environment_wrapper import ReadOnlyEnvironmentWrapper
from src.environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment


def test_readonly_wrapper():
    """Test the read-only wrapper functionality."""
    print("Testing ReadOnly Environment Wrapper")
    print("=" * 50)
    
    # Create a test environment
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
    
    rng = np.random.default_rng(2023)
    original_env = UCSBMeshnetMemmapEnvironment(rng=rng, **env_config)
    readonly_env = ReadOnlyEnvironmentWrapper(original_env)
    
    print("✅ Created read-only wrapper")
    
    # Test read operations (should work)
    try:
        available_arms = readonly_env.get_available_arms_for_round(0)
        print(f"✅ Read operation successful: {len(available_arms)} available arms")
        
        feasible_combinations = readonly_env.get_feasible_combinations(available_arms)
        print(f"✅ Read operation successful: {len(feasible_combinations)} feasible combinations")
        
        if feasible_combinations:
            first_combination = next(iter(feasible_combinations))
            reward = readonly_env.get_expected_reward_for_path(first_combination, 0)
            print(f"✅ Read operation successful: reward = {reward:.6f}")
        
    except Exception as e:
        print(f"❌ Read operation failed: {e}")
    
    # Test modification attempts (should fail)
    print("\nTesting modification prevention:")
    
    # Try to modify attribute
    try:
        readonly_env.num_rounds = 999
        print("❌ Attribute modification should have failed!")
    except AttributeError as e:
        print(f"✅ Attribute modification prevented: {e}")
    
    # Try to modify returned list
    try:
        available_arms_copy = readonly_env.get_available_arms_for_round(0)
        original_length = len(available_arms_copy)
        available_arms_copy.append(999)  # This should not affect the original
        
        # Check if original is unchanged
        available_arms_again = readonly_env.get_available_arms_for_round(0)
        if len(available_arms_again) == original_length:
            print("✅ List modification did not affect original environment")
        else:
            print("❌ List modification affected original environment!")
            
    except Exception as e:
        print(f"⚠️ List modification test: {e}")
    
    # Test numpy array read-only protection
    try:
        if hasattr(readonly_env, 'topology_memmap'):
            topology = readonly_env.topology_memmap
            print(f"✅ Got topology array: shape {topology.shape}")
            print(f"✅ Array is read-only: {not topology.flags.writeable}")
            
            # Try to modify (should fail)
            try:
                topology[0, 0, 0] = True
                print("❌ Array modification should have failed!")
            except (ValueError, RuntimeError) as e:
                print(f"✅ Array modification prevented: {type(e).__name__}")
        
    except Exception as e:
        print(f"⚠️ Array test: {e}")
    
    # Test accessing basic properties
    try:
        print(f"✅ Environment properties: rounds={readonly_env.num_rounds}, arms={readonly_env.num_arms}")
        print(f"✅ Source: {readonly_env.source}, Destination: {readonly_env.destination}")
    except Exception as e:
        print(f"❌ Property access failed: {e}")
    
    # Cleanup
    original_env.cleanup()
    print("\n✅ Test completed")


if __name__ == "__main__":
    test_readonly_wrapper() 