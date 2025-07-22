#!/usr/bin/env python3
"""
Test BG-CTS feasible combinations handling compared to other algorithms.
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

def test_feasible_combinations_consistency():
    """Test if BG-CTS handles feasible combinations consistently with other algorithms."""
    
    print("Testing Feasible Combinations Consistency")
    print("=" * 60)
    
    # Test with different availability rates
    availability_rates = [1.0, 0.9, 0.85, 0.75]
    
    for availability_rate in availability_rates:
        print(f"\nTesting with availability rate: {availability_rate}")
        print("-" * 50)
        
        # Create environment with specific availability rate
        env = QurinetEnvironment(
            data_dir="data/qurinet",
            date="2-May",
            num_rounds=100,
            pre_generate_availability=True,
            pre_generate_rewards=True
        )
        
        # Temporarily set availability rate
        for edge in env.graph.edges():
            arm_id = env.edge_to_arm[edge]
            env.arm_availability_rates[arm_id] = availability_rate
        
        # Create algorithms
        bg_cts = BGCTS(env)
        cts_b = CTSB(env)
        comb_ucb = CombUCB(env)
        
        # Test for several rounds
        for round_idx in range(10):
            print(f"\nRound {round_idx}:")
            
            # Get available arms
            available_arms = env.get_available_arms_for_round(round_idx)
            feasible_combinations = env.get_feasible_combinations(available_arms)
            
            print(f"  Available arms: {len(available_arms)}/{env.num_arms}")
            print(f"  Feasible combinations: {len(feasible_combinations)}")
            
            # Test algorithm selections
            bg_cts_selection = bg_cts.select_combination(round_idx)
            cts_b_selection = cts_b.select_combination(round_idx)
            comb_ucb_selection = comb_ucb.select_combination(round_idx)
            
            print(f"  BG-CTS selection: {bg_cts_selection}")
            print(f"  CTS-B selection: {cts_b_selection}")
            print(f"  CombUCB selection: {comb_ucb_selection}")
            
            # Check if selections are valid
            bg_cts_valid = bg_cts_selection in feasible_combinations
            cts_b_valid = cts_b_selection in feasible_combinations
            comb_ucb_valid = comb_ucb_selection in feasible_combinations
            
            print(f"  BG-CTS valid: {bg_cts_valid}")
            print(f"  CTS-B valid: {cts_b_valid}")
            print(f"  CombUCB valid: {comb_ucb_valid}")
            
            # Check if all algorithms see the same feasible combinations
            if not (bg_cts_valid and cts_b_valid and comb_ucb_valid):
                print("  WARNING: Inconsistent feasible combinations detected!")
                break

def test_bg_cts_selection_logic():
    """Test BG-CTS selection logic in detail."""
    
    print("\n\nTesting BG-CTS Selection Logic")
    print("=" * 60)
    
    # Create environment with 100% availability
    env = QurinetEnvironment(
        data_dir="data/qurinet",
        date="2-May",
        num_rounds=100,
        pre_generate_availability=True,
        pre_generate_rewards=True
    )
    
    # Set all arms available
    for edge in env.graph.edges():
        arm_id = env.edge_to_arm[edge]
        env.arm_availability_rates[arm_id] = 1.0
    
    bg_cts = BGCTS(env)
    
    print(f"Environment: {env.num_arms} arms, max combination size: {env.max_combination_size}")
    print(f"BG-CTS m parameter: {bg_cts.m}")
    
    # Test for several rounds
    for round_idx in range(5):
        print(f"\nRound {round_idx}:")
        
        # Get available arms and feasible combinations
        available_arms = list(env.get_available_arms_for_round(round_idx))
        feasible_combinations = env.get_feasible_combinations(set(available_arms))
        
        print(f"  Available arms: {len(available_arms)}")
        print(f"  Feasible combinations: {len(feasible_combinations)}")
        
        # Check BG-CTS logic
        m = min(bg_cts.m, len(available_arms))
        print(f"  BG-CTS m: {m}")
        print(f"  len(available_arms) > m: {len(available_arms) > m}")
        
        # Show some feasible combinations
        print(f"  Sample feasible combinations:")
        for i, comb in enumerate(feasible_combinations[:3]):
            print(f"    {i}: {comb} (size: {len(comb)})")
        
        # Get BG-CTS selection
        bg_cts_selection = bg_cts.select_combination(round_idx)
        print(f"  BG-CTS selection: {bg_cts_selection} (size: {len(bg_cts_selection)})")
        
        # Check if selection is valid
        if bg_cts_selection in feasible_combinations:
            print(f"  Selection is valid ✓")
        else:
            print(f"  Selection is INVALID ✗")
            print(f"  This might explain poor performance!")

def test_availability_impact_on_bg_cts():
    """Test how availability affects BG-CTS performance."""
    
    print("\n\nTesting Availability Impact on BG-CTS")
    print("=" * 60)
    
    availability_rates = [1.0, 0.9, 0.85, 0.75]
    
    for availability_rate in availability_rates:
        print(f"\nAvailability rate: {availability_rate}")
        print("-" * 40)
        
        # Create environment
        env = QurinetEnvironment(
            data_dir="data/qurinet",
            date="2-May",
            num_rounds=1000,
            pre_generate_availability=True,
            pre_generate_rewards=True
        )
        
        # Set availability rate
        for edge in env.graph.edges():
            arm_id = env.edge_to_arm[edge]
            env.arm_availability_rates[arm_id] = availability_rate
        
        bg_cts = BGCTS(env)
        
        # Run simulation for a few rounds
        total_regret = 0
        valid_selections = 0
        
        for round_idx in range(100):
            available_arms = env.get_available_arms_for_round(round_idx)
            feasible_combinations = env.get_feasible_combinations(available_arms)
            
            # Get optimal combination
            optimal_combination = env.get_optimal_combination(available_arms)
            if optimal_combination:
                optimal_reward = sum(env.arm_means[arm_id] for arm_id in optimal_combination)
            else:
                optimal_reward = 0
            
            # Get BG-CTS selection
            bg_cts_selection = bg_cts.select_combination(round_idx)
            
            if bg_cts_selection in feasible_combinations:
                valid_selections += 1
                selected_reward = sum(env.arm_means[arm_id] for arm_id in bg_cts_selection)
                regret = optimal_reward - selected_reward
                total_regret += regret
                
                # Update BG-CTS
                rewards = env.get_reward_for_round(bg_cts_selection, round_idx)
                bg_cts.update_posterior(bg_cts_selection, rewards, round_idx)
        
        print(f"  Valid selections: {valid_selections}/100 ({valid_selections/100:.1%})")
        print(f"  Average regret: {total_regret/100:.3f}")

if __name__ == "__main__":
    test_feasible_combinations_consistency()
    test_bg_cts_selection_logic()
    test_availability_impact_on_bg_cts() 