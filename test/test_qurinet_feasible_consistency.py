#!/usr/bin/env python3
"""
Test script to check if CTS-G observes the same feasible combinations as other algorithms
in the Qurinet environment.
"""

import sys
import os
import numpy as np
from typing import Set, List, Dict, Any

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.bandits.cts_b import CTSB
from src.bandits.comb_ucb import CombUCB
from src.bandits.cts_g import CTSG
from src.bandits.cl_sg import CLSG
from src.bandits.bg_cts import BGCTS
from src.environments.qurinet_environment import QurinetEnvironment


def test_qurinet_feasible_consistency():
    """
    Test if all algorithms observe the same feasible combinations in each round
    in the Qurinet environment.
    """
    print("Testing Qurinet Feasible Combinations Consistency")
    print("=" * 60)
    
    # Create Qurinet environment
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "qurinet")
    
    env = QurinetEnvironment(
        data_dir=data_dir,
        date="2-May",
        num_rounds=20,
        pre_generate_availability=True,
        pre_generate_rewards=True,
        rng=np.random.default_rng(42)
    )
    
    # Set availability rate manually
    env.availability_rate = 0.9
    
    print(f"Environment: {env.num_arms} arms, {env.num_rounds} rounds")
    print(f"Source: {env.source}, Destination: {env.destination}")
    print()
    
    # Create algorithms
    algorithms = {
        "CTSB": CTSB(env),
        "CombUCB": CombUCB(env),
        "CTS-G": CTSG(env, gamma=0.01),
        "CL-SG": CLSG(env, gamma=0.01),
        "BG-CTS": BGCTS(env)
    }
    
    # Test for 20 rounds
    for round_idx in range(20):
        print(f"Round {round_idx + 1}:")
        print("-" * 30)
        
        # Get environment's available arms and feasible combinations
        env_available_arms = env.get_available_arms_for_round(round_idx)
        env_feasible_combinations = env.get_feasible_combinations(env_available_arms)
        
        print(f"Environment available arms: {sorted(env_available_arms)}")
        print(f"Environment feasible combinations count: {len(env_feasible_combinations)}")
        
        # Check each algorithm's observed feasible combinations
        algorithm_feasible_combinations = {}
        
        for alg_name, alg in algorithms.items():
            try:
                # Get algorithm's selection (this should use the same feasible combinations)
                selection = alg.select_combination(round_idx)
                
                # For CTS-G specifically, let's check if it's using a different method
                if alg_name == "CTS-G":
                    print(f"\n{alg_name} selection: {selection}")
                    print(f"{alg_name} selection size: {len(selection) if selection else 0}")
                
                # Store for comparison
                algorithm_feasible_combinations[alg_name] = selection
                
            except Exception as e:
                print(f"Error in {alg_name}: {e}")
                algorithm_feasible_combinations[alg_name] = None
        
        # Check consistency
        all_selections = [s for s in algorithm_feasible_combinations.values() if s is not None]
        if all_selections:
            first_selection = all_selections[0]
            consistent = all(s == first_selection for s in all_selections)
            
            if consistent:
                print(f"✓ All algorithms selected the same combination: {first_selection}")
            else:
                print("✗ Algorithms selected different combinations:")
                for alg_name, selection in algorithm_feasible_combinations.items():
                    print(f"  {alg_name}: {selection}")
        
        print()
        
        # Update algorithms with rewards
        for alg_name, alg in algorithms.items():
            if algorithm_feasible_combinations[alg_name] is not None:
                selection = algorithm_feasible_combinations[alg_name]
                rewards = env.get_reward_for_round(selection, round_idx)
                alg.update_posterior(selection, rewards, round_idx)


def test_cts_g_specific_behavior():
    """
    Test CTS-G's specific behavior in detail.
    """
    print("\n" + "=" * 60)
    print("Testing CTS-G Specific Behavior")
    print("=" * 60)
    
    # Create Qurinet environment
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "qurinet")
    
    env = QurinetEnvironment(
        data_dir=data_dir,
        date="2-May",
        num_rounds=10,
        pre_generate_availability=True,
        pre_generate_rewards=True,
        rng=np.random.default_rng(42)
    )
    
    # Set availability rate manually
    env.availability_rate = 0.9
    
    # Create CTS-G algorithm
    cts_g = CTSG(env, gamma=0.01)
    
    print(f"CTS-G gamma: {cts_g.gamma}")
    print(f"Environment arms: {env.num_arms}")
    print()
    
    for round_idx in range(10):
        print(f"Round {round_idx + 1}:")
        print("-" * 20)
        
        # Get environment's available arms
        env_available_arms = env.get_available_arms_for_round(round_idx)
        print(f"Environment available arms: {sorted(env_available_arms)}")
        
        # Get CTS-G's selection
        selection = cts_g.select_combination(round_idx)
        print(f"CTS-G selection: {selection}")
        print(f"CTS-G selection size: {len(selection) if selection else 0}")
        
        # Check if selection is valid
        if selection:
            valid = all(arm in env_available_arms for arm in selection)
            print(f"Selection valid: {valid}")
            
            # Check if it's a feasible path
            if hasattr(env, 'get_path_info'):
                try:
                    path_info = env.get_path_info(selection)
                    print(f"Path info: {path_info}")
                except:
                    print("Could not get path info")
        
        print()
        
        # Update algorithm
        if selection:
            rewards = env.get_reward_for_round(selection, round_idx)
            cts_g.update_posterior(selection, rewards, round_idx)


if __name__ == "__main__":
    # Test feasible combinations consistency
    test_qurinet_feasible_consistency()
    
    # Test CTS-G specific behavior
    test_cts_g_specific_behavior() 