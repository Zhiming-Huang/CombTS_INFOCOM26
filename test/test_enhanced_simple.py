#!/usr/bin/env python3
"""
Simple test script for the enhanced QurinetEnvironment initialization.
"""

import sys
import os
import numpy as np

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.environments.qurinet_environment_enhanced import QurinetEnvironmentEnhanced


def test_enhanced_init():
    """Test enhanced environment initialization."""
    print("Testing Enhanced QurinetEnvironment Initialization")
    print("=" * 60)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "qurinet")
    
    try:
        print("Creating enhanced environment...")
        
        # Create enhanced environment with minimal settings
        env = QurinetEnvironmentEnhanced(
            data_dir=data_dir,
            date="29-April",
            num_rounds=10,  # Very small number
            pre_generate_availability=False,  # Don't pre-generate
            pre_generate_rewards=False,  # Don't pre-generate
            rng=np.random.default_rng(42),
            use_all_nodes=True
        )
        
        print("Environment created successfully!")
        
        # Get basic info
        info = env.get_environment_info()
        
        print(f"\nEnvironment Info:")
        print(f"  Environment type: {info['environment_type']}")
        print(f"  Sites data size: {info['sites_data_size']}")
        print(f"  Number of nodes: {info['num_nodes']}")
        print(f"  Number of edges: {info['num_edges']}")
        print(f"  Number of arms: {info['num_arms']}")
        print(f"  Source: {info['source']}, Destination: {info['destination']}")
        
        # Test getting available arms
        print(f"\nTesting available arms...")
        available_arms = env.get_available_arms_for_round(0)
        print(f"  Available arms: {len(available_arms)}")
        
        # Test getting feasible combinations (this is where it might be slow)
        print(f"\nTesting feasible combinations...")
        feasible_combinations = env.get_feasible_combinations(available_arms)
        print(f"  Feasible combinations: {len(feasible_combinations)}")
        
        if len(feasible_combinations) > 0:
            print(f"  First combination: {list(feasible_combinations[0])}")
            print(f"  Expected reward: {sum(env.arm_means[arm] for arm in feasible_combinations[0]):.3f}")
        
        print(f"\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_enhanced_init() 