#!/usr/bin/env python3
"""
Test if all algorithms observe the same feasible combinations in each round in Qurinet environment.
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

def test_qurinet_feasible_consistency():
    """Test if all algorithms observe the same feasible combinations in each round."""
    
    print("Testing Feasible Combinations Consistency in Qurinet Environment")
    print("=" * 80)
    
    # Create environment with different availability rates
    availability_rates = [1.0, 0.9, 0.85, 0.75]
    
    for availability_rate in availability_rates:
        print(f"\n{'='*20} Testing Availability Rate: {availability_rate} {'='*20}")
        
        # Create environment
        env = QurinetEnvironment(
            data_dir="data/qurinet",
            date="2-May",
            num_rounds=50,  # Test more rounds
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
        
        print(f"Environment: {env.num_arms} arms, {env.num_rounds} rounds")
        print(f"Network: {env.graph.number_of_nodes()} nodes, {env.graph.number_of_edges()} edges")
        
        # Test consistency across rounds
        inconsistent_rounds = []
        
        for round_idx in range(20):  # Test first 20 rounds
            print(f"\nRound {round_idx}:")
            
            # Get environment's view
            env_available_arms = env.get_available_arms_for_round(round_idx)
            env_feasible_combinations = env.get_feasible_combinations(env_available_arms)
            
            print(f"  Environment: {len(env_available_arms)} available, {len(env_feasible_combinations)} feasible")
            
            # Check each algorithm's view
            round_consistent = True
            algorithm_views = {}
            
            for alg_name, alg in algorithms.items():
                try:
                    # Get algorithm's view
                    alg_available_arms = alg.environment.get_available_arms_for_round(round_idx)
                    alg_feasible_combinations = alg.environment.get_feasible_combinations(alg_available_arms)
                    
                    algorithm_views[alg_name] = {
                        'available_arms': alg_available_arms,
                        'feasible_combinations': alg_feasible_combinations
                    }
                    
                    # Check if consistent with environment
                    if (alg_available_arms != env_available_arms or 
                        alg_feasible_combinations != env_feasible_combinations):
                        round_consistent = False
                        print(f"    ❌ {alg_name}: INCONSISTENT!")
                        print(f"      Expected available: {env_available_arms}")
                        print(f"      Got available: {alg_available_arms}")
                        print(f"      Expected feasible: {env_feasible_combinations}")
                        print(f"      Got feasible: {alg_feasible_combinations}")
                    else:
                        print(f"    ✅ {alg_name}: {len(alg_available_arms)} available, {len(alg_feasible_combinations)} feasible")
                        
                except Exception as e:
                    print(f"    ❌ {alg_name}: ERROR - {e}")
                    round_consistent = False
            
            if not round_consistent:
                inconsistent_rounds.append(round_idx)
                print(f"  ❌ Round {round_idx} has inconsistencies!")
            else:
                print(f"  ✅ Round {round_idx} is consistent")
            
            # Show detailed feasible combinations for first few rounds
            if round_idx < 3:
                print(f"    Feasible combinations:")
                for i, comb in enumerate(env_feasible_combinations):
                    reward = sum(env.arm_means[arm_id] for arm_id in comb)
                    print(f"      {i}: {comb} (reward: {reward:.3f})")
        
        # Summary
        print(f"\n{'='*20} Summary for Availability Rate {availability_rate} {'='*20}")
        if inconsistent_rounds:
            print(f"❌ Found inconsistencies in rounds: {inconsistent_rounds}")
        else:
            print(f"✅ All rounds are consistent - all algorithms see the same feasible combinations")
        
        print(f"Tested {20} rounds")
        print(f"Consistent rounds: {20 - len(inconsistent_rounds)}")
        print(f"Inconsistent rounds: {len(inconsistent_rounds)}")

def test_algorithm_selections_with_consistent_feasible():
    """Test algorithm selections when they all see the same feasible combinations."""
    
    print(f"\n{'='*20} Testing Algorithm Selections with Consistent Feasible Combinations {'='*20}")
    
    # Create environment with 100% availability
    env = QurinetEnvironment(
        data_dir="data/qurinet",
        date="2-May",
        num_rounds=50,
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
    
    print("Testing algorithm selections over 15 rounds:")
    print("-" * 60)
    
    # Track selections and rewards
    selection_history = {alg_name: [] for alg_name in algorithms.keys()}
    reward_history = {alg_name: [] for alg_name in algorithms.keys()}
    
    for round_idx in range(15):
        print(f"\nRound {round_idx}:")
        
        # Get feasible combinations
        available_arms = env.get_available_arms_for_round(round_idx)
        feasible_combinations = env.get_feasible_combinations(available_arms)
        
        # Get optimal combination
        optimal_combination = env.get_optimal_combination(available_arms)
        optimal_reward = sum(env.arm_means[arm_id] for arm_id in optimal_combination) if optimal_combination else 0
        
        print(f"  Available arms: {len(available_arms)}")
        print(f"  Feasible combinations: {len(feasible_combinations)}")
        print(f"  Optimal combination: {optimal_combination} (reward: {optimal_reward:.3f})")
        
        # Test each algorithm's selection
        round_selections = {}
        for alg_name, alg in algorithms.items():
            try:
                selection = alg.select_combination(round_idx)
                round_selections[alg_name] = selection
                
                # Check if selection is valid
                is_valid = selection in feasible_combinations
                selected_reward = sum(env.arm_means[arm_id] for arm_id in selection) if selection else 0
                
                selection_history[alg_name].append(selection)
                reward_history[alg_name].append(selected_reward)
                
                print(f"  {alg_name}: {selection} (reward: {selected_reward:.3f}, valid: {is_valid}, optimal: {selection == optimal_combination})")
                
                if not is_valid:
                    print(f"    ❌ WARNING: {alg_name} selection is INVALID!")
                
                # Update algorithm with rewards
                rewards = env.get_reward_for_round(selection, round_idx)
                alg.update_posterior(selection, rewards, round_idx)
                
            except Exception as e:
                print(f"  {alg_name}: ERROR - {e}")
        
        # Check if all algorithms made different selections
        unique_selections = set(tuple(sorted(sel)) for sel in round_selections.values() if sel)
        print(f"  Unique selections: {len(unique_selections)} out of {len(round_selections)}")
        
        if len(unique_selections) == 1:
            print(f"  ⚠️  All algorithms selected the same combination!")
        elif len(unique_selections) == len(round_selections):
            print(f"  ✅ All algorithms selected different combinations")
    
    # Final analysis
    print(f"\n{'='*20} Final Analysis {'='*20}")
    for alg_name in algorithms.keys():
        if selection_history[alg_name]:
            optimal_count = sum(1 for sel in selection_history[alg_name] 
                              if sel == optimal_combination)
            avg_reward = np.mean(reward_history[alg_name])
            print(f"{alg_name}:")
            print(f"  Optimal selections: {optimal_count}/{len(selection_history[alg_name])} ({optimal_count/len(selection_history[alg_name])*100:.1f}%)")
            print(f"  Average reward: {avg_reward:.3f}")
            print(f"  Regret per round: {optimal_reward - avg_reward:.3f}")

if __name__ == "__main__":
    test_qurinet_feasible_consistency()
    test_algorithm_selections_with_consistent_feasible() 