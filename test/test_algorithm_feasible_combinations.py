#!/usr/bin/env python3
"""
Test algorithm feasible combinations handling to verify consistency.
"""

import sys
import os
import numpy as np
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from environments.qurinet_environment import QurinetEnvironment
from bandits.bg_cts import BGCTS
from bandits.cts_b import CTSB
from bandits.comb_ucb import CombUCB
from bandits.cts_g import CTSG
from bandits.cl_sg import CLSG

def test_algorithm_feasible_combinations():
    """Test if all algorithms handle feasible combinations consistently."""
    
    print("Testing Algorithm Feasible Combinations Consistency")
    print("=" * 70)
    
    # Test with different availability rates
    availability_rates = [1.0, 0.9, 0.85, 0.75]
    
    for availability_rate in availability_rates:
        print(f"\nTesting with availability rate: {availability_rate}")
        print("-" * 60)
        
        # Create environment with specific availability rate
        env = QurinetEnvironment(
            data_dir="data/qurinet",
            date="2-May",
            num_rounds=100,
            pre_generate_availability=True,
            pre_generate_rewards=True,
            availability_rate=availability_rate
        )
        
        # Create algorithms
        bg_cts = BGCTS(env)
        cts_b = CTSB(env)
        comb_ucb = CombUCB(env)
        cts_g = CTSG(env, gamma=0.01)
        cl_sg = CLSG(env, gamma=0.01)
        
        algorithms = {
            'BG-CTS': bg_cts,
            'CTS-B': cts_b,
            'CombUCB': comb_ucb,
            'CTS-G': cts_g,
            'CL-SG': cl_sg
        }
        
        # Test for several rounds
        for round_idx in range(5):
            print(f"\nRound {round_idx}:")
            
            # Get available arms and feasible combinations
            available_arms = env.get_available_arms_for_round(round_idx)
            feasible_combinations = env.get_feasible_combinations(available_arms)
            
            print(f"  Available arms: {len(available_arms)}/{env.num_arms}")
            print(f"  Feasible combinations: {len(feasible_combinations)}")
            
            # Test each algorithm's selection
            algorithm_selections = {}
            for alg_name, alg in algorithms.items():
                try:
                    selection = alg.select_combination(round_idx)
                    algorithm_selections[alg_name] = selection
                    
                    # Check if selection is valid
                    is_valid = selection in feasible_combinations
                    print(f"  {alg_name}: {selection} (valid: {is_valid})")
                    
                    if not is_valid:
                        print(f"    WARNING: {alg_name} selection is INVALID!")
                        print(f"    Selection: {selection}")
                        print(f"    Feasible combinations: {feasible_combinations}")
                        
                except Exception as e:
                    print(f"  {alg_name}: ERROR - {e}")
            
            # Check if all algorithms see the same feasible combinations
            all_valid = all(selection in feasible_combinations for selection in algorithm_selections.values())
            if not all_valid:
                print("  WARNING: Inconsistent feasible combinations detected!")
                break

def test_algorithm_selection_consistency():
    """Test if algorithms make consistent selections across rounds."""
    
    print("\n\nTesting Algorithm Selection Consistency")
    print("=" * 70)
    
    # Create environment with 100% availability
    env = QurinetEnvironment(
        data_dir="data/qurinet",
        date="2-May",
        num_rounds=100,
        pre_generate_availability=True,
        pre_generate_rewards=True,
        availability_rate=1.0
    )
    
    # Create algorithms
    bg_cts = BGCTS(env)
    cts_b = CTSB(env)
    comb_ucb = CombUCB(env)
    cts_g = CTSG(env, gamma=0.01)
    cl_sg = CLSG(env, gamma=0.01)
    
    algorithms = {
        'BG-CTS': bg_cts,
        'CTS-B': cts_b,
        'CombUCB': comb_ucb,
        'CTS-G': cts_g,
        'CL-SG': cl_sg
    }
    
    # Track selections over multiple rounds
    selection_history = {alg_name: [] for alg_name in algorithms.keys()}
    
    print("Tracking algorithm selections over 10 rounds:")
    print("-" * 60)
    
    for round_idx in range(10):
        print(f"\nRound {round_idx}:")
        
        available_arms = env.get_available_arms_for_round(round_idx)
        feasible_combinations = env.get_feasible_combinations(available_arms)
        
        print(f"  Available arms: {len(available_arms)}")
        print(f"  Feasible combinations: {len(feasible_combinations)}")
        
        for alg_name, alg in algorithms.items():
            try:
                selection = alg.select_combination(round_idx)
                selection_history[alg_name].append(selection)
                
                # Get optimal combination for comparison
                optimal_combination = env.get_optimal_combination(available_arms)
                optimal_reward = sum(env.arm_means[arm_id] for arm_id in optimal_combination) if optimal_combination else 0
                selected_reward = sum(env.arm_means[arm_id] for arm_id in selection) if selection else 0
                
                print(f"  {alg_name}: {selection} (reward: {selected_reward:.3f}, optimal: {optimal_reward:.3f})")
                
                # Update algorithm with rewards
                rewards = env.get_reward_for_round(selection, round_idx)
                alg.update_posterior(selection, rewards, round_idx)
                
            except Exception as e:
                print(f"  {alg_name}: ERROR - {e}")
    
    # Analyze selection patterns
    print(f"\nSelection Pattern Analysis:")
    print("-" * 60)
    for alg_name, selections in selection_history.items():
        unique_selections = set(tuple(sorted(sel)) for sel in selections if sel)
        print(f"  {alg_name}: {len(unique_selections)} unique selections out of {len(selections)} rounds")
        if len(unique_selections) <= 3:
            print(f"    Selections: {list(unique_selections)}")

def test_cts_g_specific():
    """Test CTS-G algorithm specifically to understand its behavior."""
    
    print("\n\nTesting CTS-G Algorithm Specifically")
    print("=" * 70)
    
    # Create environment with 100% availability
    env = QurinetEnvironment(
        data_dir="data/qurinet",
        date="2-May",
        num_rounds=100,
        pre_generate_availability=True,
        pre_generate_rewards=True,
        availability_rate=1.0
    )
    
    cts_g = CTSG(env, gamma=0.01)
    
    print("CTS-G Algorithm Details:")
    print(f"  Environment: {env.num_arms} arms")
    print(f"  Gamma: 0.01")
    print(f"  Max combination size: {env.max_combination_size}")
    
    # Test for several rounds
    for round_idx in range(5):
        print(f"\nRound {round_idx}:")
        
        available_arms = env.get_available_arms_for_round(round_idx)
        feasible_combinations = env.get_feasible_combinations(available_arms)
        
        print(f"  Available arms: {len(available_arms)}")
        print(f"  Feasible combinations: {len(feasible_combinations)}")
        
        # Show some feasible combinations
        print(f"  Sample feasible combinations:")
        for i, comb in enumerate(feasible_combinations[:3]):
            reward = sum(env.arm_means[arm_id] for arm_id in comb)
            print(f"    {i}: {comb} (reward: {reward:.3f})")
        
        # Get CTS-G selection
        selection = cts_g.select_combination(round_idx)
        selected_reward = sum(env.arm_means[arm_id] for arm_id in selection) if selection else 0
        
        print(f"  CTS-G selection: {selection} (reward: {selected_reward:.3f})")
        
        # Check if it's optimal
        optimal_combination = env.get_optimal_combination(available_arms)
        optimal_reward = sum(env.arm_means[arm_id] for arm_id in optimal_combination) if optimal_combination else 0
        
        print(f"  Optimal combination: {optimal_combination} (reward: {optimal_reward:.3f})")
        print(f"  Is optimal: {selection == optimal_combination}")
        
        # Update CTS-G
        rewards = env.get_reward_for_round(selection, round_idx)
        cts_g.update_posterior(selection, rewards, round_idx)

if __name__ == "__main__":
    test_algorithm_feasible_combinations()
    test_algorithm_selection_consistency()
    test_cts_g_specific() 