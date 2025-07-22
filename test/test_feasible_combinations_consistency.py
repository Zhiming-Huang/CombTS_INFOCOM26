#!/usr/bin/env python3
"""
Test if all algorithms receive the same feasible combinations.
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from environments.qurinet_environment import QurinetEnvironment
from bandits.bg_cts import BGCTS
from bandits.cts_b import CTSB
from bandits.comb_ucb import CombUCB
from bandits.cts_g import CTSG
from bandits.cl_sg import CLSG

def test_feasible_combinations_consistency():
    """Test if all algorithms receive the same feasible combinations."""
    
    print("Testing Feasible Combinations Consistency Across Algorithms")
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
            pre_generate_rewards=True
        )
        
        # Set availability rate manually
        for edge in env.graph.edges():
            arm_id = env.edge_to_arm[edge]
            env.arm_availability_rates[arm_id] = availability_rate
        
        # Update availability matrix if it exists
        if hasattr(env, 'availability_matrix'):
            env.availability_matrix = env.rng.random((env.num_rounds, env.num_arms)) < availability_rate
        
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
        for round_idx in range(3):  # Reduced to 3 rounds for clarity
            print(f"\nRound {round_idx}:")
            
            # Get available arms and feasible combinations from environment
            available_arms = env.get_available_arms_for_round(round_idx)
            feasible_combinations = env.get_feasible_combinations(available_arms)
            
            print(f"  Environment: {len(available_arms)} available arms, {len(feasible_combinations)} feasible combinations")
            print(f"  Available arms: {sorted(available_arms)}")
            print(f"  Feasible combinations:")
            for i, comb in enumerate(feasible_combinations):
                reward = sum(env.arm_means[arm_id] for arm_id in comb)
                print(f"    {i}: {comb} (reward: {reward:.3f})")
            
            # Test each algorithm's view of feasible combinations
            algorithm_feasible_combinations = {}
            for alg_name, alg in algorithms.items():
                try:
                    # Get the algorithm's view of available arms and feasible combinations
                    alg_available_arms = alg.environment.get_available_arms_for_round(round_idx)
                    alg_feasible_combinations = alg.environment.get_feasible_combinations(alg_available_arms)
                    
                    algorithm_feasible_combinations[alg_name] = {
                        'available_arms': alg_available_arms,
                        'feasible_combinations': alg_feasible_combinations
                    }
                    
                    print(f"  {alg_name}: {len(alg_available_arms)} available arms, {len(alg_feasible_combinations)} feasible combinations")
                    print(f"    Available arms: {sorted(alg_available_arms)}")
                    print(f"    Feasible combinations:")
                    for i, comb in enumerate(alg_feasible_combinations):
                        reward = sum(env.arm_means[arm_id] for arm_id in comb)
                        print(f"      {i}: {comb} (reward: {reward:.3f})")
                    
                    # Check if available arms match
                    if alg_available_arms != available_arms:
                        print(f"    WARNING: {alg_name} available arms don't match!")
                        print(f"    Expected: {available_arms}")
                        print(f"    Got: {alg_available_arms}")
                    
                    # Check if feasible combinations match
                    if alg_feasible_combinations != feasible_combinations:
                        print(f"    WARNING: {alg_name} feasible combinations don't match!")
                        print(f"    Expected: {feasible_combinations}")
                        print(f"    Got: {alg_feasible_combinations}")
                    
                except Exception as e:
                    print(f"  {alg_name}: ERROR - {e}")
            
            # Check if all algorithms see the same feasible combinations
            all_consistent = True
            for alg_name, data in algorithm_feasible_combinations.items():
                if data['available_arms'] != available_arms or data['feasible_combinations'] != feasible_combinations:
                    all_consistent = False
                    break
            
            if not all_consistent:
                print("  WARNING: Inconsistent feasible combinations detected!")
                break
            else:
                print("  ✓ All algorithms see the same feasible combinations")
            
            # Only test first round for each availability rate to avoid too much output
            break

def test_algorithm_selections_with_same_feasible_combinations():
    """Test if algorithms make different selections when given the same feasible combinations."""
    
    print("\n\nTesting Algorithm Selections with Same Feasible Combinations")
    print("=" * 70)
    
    # Create environment with 100% availability
    env = QurinetEnvironment(
        data_dir="data/qurinet",
        date="2-May",
        num_rounds=100,
        pre_generate_availability=True,
        pre_generate_rewards=True
    )
    
    # Set availability rate to 1.0
    for edge in env.graph.edges():
        arm_id = env.edge_to_arm[edge]
        env.arm_availability_rates[arm_id] = 1.0
    
    # Update availability matrix if it exists
    if hasattr(env, 'availability_matrix'):
        env.availability_matrix = env.rng.random((env.num_rounds, env.num_arms)) < 1.0
    
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
    
    print("Testing algorithm selections over 10 rounds:")
    print("-" * 60)
    
    for round_idx in range(10):
        print(f"\nRound {round_idx}:")
        
        # Get available arms and feasible combinations
        available_arms = env.get_available_arms_for_round(round_idx)
        feasible_combinations = env.get_feasible_combinations(available_arms)
        
        print(f"  Available arms: {len(available_arms)}")
        print(f"  Feasible combinations: {len(feasible_combinations)}")
        
        # Show all feasible combinations
        print(f"  All feasible combinations:")
        for i, comb in enumerate(feasible_combinations):
            reward = sum(env.arm_means[arm_id] for arm_id in comb)
            print(f"    {i}: {comb} (reward: {reward:.3f})")
        
        # Test each algorithm's selection
        algorithm_selections = {}
        for alg_name, alg in algorithms.items():
            try:
                selection = alg.select_combination(round_idx)
                algorithm_selections[alg_name] = selection
                
                # Check if selection is valid
                is_valid = selection in feasible_combinations
                selected_reward = sum(env.arm_means[arm_id] for arm_id in selection) if selection else 0
                
                print(f"  {alg_name}: {selection} (reward: {selected_reward:.3f}, valid: {is_valid})")
                
                if not is_valid:
                    print(f"    WARNING: {alg_name} selection is INVALID!")
                
                # Update algorithm with rewards
                rewards = env.get_reward_for_round(selection, round_idx)
                alg.update_posterior(selection, rewards, round_idx)
                
            except Exception as e:
                print(f"  {alg_name}: ERROR - {e}")
        
        # Check if all algorithms made different selections
        unique_selections = set(tuple(sorted(sel)) for sel in algorithm_selections.values() if sel)
        print(f"  Unique selections: {len(unique_selections)} out of {len(algorithm_selections)}")
        
        if len(unique_selections) == 1:
            print(f"  WARNING: All algorithms selected the same combination!")
        elif len(unique_selections) == len(algorithm_selections):
            print(f"  ✓ All algorithms selected different combinations")

def test_cts_g_specific_behavior():
    """Test CTS-G specific behavior to understand why it performs well."""
    
    print("\n\nTesting CTS-G Specific Behavior")
    print("=" * 70)
    
    # Create environment with 100% availability
    env = QurinetEnvironment(
        data_dir="data/qurinet",
        date="2-May",
        num_rounds=100,
        pre_generate_availability=True,
        pre_generate_rewards=True
    )
    
    # Set availability rate to 1.0
    for edge in env.graph.edges():
        arm_id = env.edge_to_arm[edge]
        env.arm_availability_rates[arm_id] = 1.0
    
    # Update availability matrix if it exists
    if hasattr(env, 'availability_matrix'):
        env.availability_matrix = env.rng.random((env.num_rounds, env.num_arms)) < 1.0
    
    cts_g = CTSG(env, gamma=0.01)
    
    print("CTS-G Algorithm Details:")
    print(f"  Environment: {env.num_arms} arms")
    print(f"  Gamma: {cts_g.gamma}")
    
    # Track CTS-G selections over multiple rounds
    selection_history = []
    reward_history = []
    
    for round_idx in range(10):
        print(f"\nRound {round_idx}:")
        
        available_arms = env.get_available_arms_for_round(round_idx)
        feasible_combinations = env.get_feasible_combinations(available_arms)
        
        # Get optimal combination
        optimal_combination = env.get_optimal_combination(available_arms)
        optimal_reward = sum(env.arm_means[arm_id] for arm_id in optimal_combination) if optimal_combination else 0
        
        # Get CTS-G selection
        selection = cts_g.select_combination(round_idx)
        selected_reward = sum(env.arm_means[arm_id] for arm_id in selection) if selection else 0
        
        selection_history.append(selection)
        reward_history.append(selected_reward)
        
        print(f"  Available arms: {len(available_arms)}")
        print(f"  Feasible combinations: {len(feasible_combinations)}")
        print(f"  Optimal combination: {optimal_combination} (reward: {optimal_reward:.3f})")
        print(f"  CTS-G selection: {selection} (reward: {selected_reward:.3f})")
        print(f"  Is optimal: {selection == optimal_combination}")
        
        # Update CTS-G
        rewards = env.get_reward_for_round(selection, round_idx)
        cts_g.update_posterior(selection, rewards, round_idx)
    
    # Analyze CTS-G behavior
    print(f"\nCTS-G Behavior Analysis:")
    print(f"  Total rounds: {len(selection_history)}")
    print(f"  Optimal selections: {sum(1 for sel in selection_history if sel == optimal_combination)}")
    print(f"  Average reward: {np.mean(reward_history):.3f}")
    print(f"  Optimal reward: {optimal_reward:.3f}")
    print(f"  Regret per round: {optimal_reward - np.mean(reward_history):.3f}")

if __name__ == "__main__":
    test_feasible_combinations_consistency()
    test_algorithm_selections_with_same_feasible_combinations()
    test_cts_g_specific_behavior() 