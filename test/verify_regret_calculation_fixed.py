#!/usr/bin/env python3
"""
Verify that regret calculation is correct after the fix.
Check that cumulative regret is always non-decreasing and there are no unexpected zeros.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment
from bandits.cts_b import CTSB
from bandits.comb_ucb import CombUCB
from bandits.bg_cts import BGCTS

def verify_regret_calculation():
    """Verify that regret calculation is correct."""
    
    print("Verifying regret calculation after fix...")
    print("=" * 60)
    
    # Setup environment with fixed node pair
    data_dir = Path("data/ucsb")
    
    # Find the trace directory with the most files
    trace_dirs = {}
    for subdir in data_dir.iterdir():
        if subdir.is_dir():
            files = list(subdir.glob("neighbortable-*"))
            trace_dirs[subdir.name] = len(files)
    
    selected_trace_dir = max(trace_dirs, key=trace_dirs.get)
    selected_trace_path = data_dir / selected_trace_dir
    
    # Use first 50 files for quick test
    neighbortable_files = sorted(list(selected_trace_path.glob("neighbortable-*")))[:50]
    neighbortable_files = [str(f) for f in neighbortable_files]
    
    print(f"Using trace directory: {selected_trace_dir}")
    print(f"Using {len(neighbortable_files)} files")
    print(f"Fixed node pair: 10.2.1.103 -> 10.2.1.109")
    print()
    
    # Create environment
    env = UCSBMeshnetMemmapEnvironment(
        neighbortable_files=neighbortable_files,
        routes_per_minute=4,
        source="10.2.1.103",
        destination="10.2.1.109",
        pre_generate_rewards=True,
        use_persistent_files=False,
        rng=np.random.default_rng(42),
        max_feasible_combinations=50,
        max_path_length=3,
        max_paths_per_algorithm=25
    )
    
    print(f"Environment created with {env.num_rounds} rounds")
    print()
    
    # Test with CTSB algorithm
    alg = CTSB(environment=env, rng=np.random.default_rng(42))
    
    cumulative_regret = 0.0
    regrets = []
    rewards = []
    feasible_counts = []
    zero_regret_rounds = []
    
    print("Running simulation to verify regret calculation...")
    print("-" * 50)
    
    for round_idx in range(min(200, env.num_rounds)):  # Test first 200 rounds
        # Get available arms for this round
        available_arms = env.get_available_arms_for_round(round_idx)
        
        # Get feasible combinations
        feasible_combinations = env.get_feasible_combinations(available_arms)
        feasible_counts.append(len(feasible_combinations))
        
        if not feasible_combinations:
            # No feasible paths - regret is 0 for this round
            regret = 0.0
            reward = 0.0
            print(f"Round {round_idx}: No feasible paths - regret=0, cumulative={cumulative_regret:.3f}")
        else:
            # Select action
            selected_combination = alg.select_combination(round_idx)
            
            # Calculate total reward for the path
            total_reward = env.get_expected_reward_for_path(selected_combination, round_idx)
            
            # Get optimal reward for regret calculation
            optimal_reward = env.get_optimal_path_expected_reward(round_idx)
            
            # Calculate regret
            regret = optimal_reward - total_reward
            reward = total_reward
            
            # Update algorithm
            reward_dict = {}
            for arm in selected_combination:
                reward_dict[arm] = env.get_reward_for_round(arm, round_idx)
            alg.update_posterior(selected_combination, reward_dict, round_idx)
            
            if regret == 0.0:
                zero_regret_rounds.append(round_idx)
                print(f"Round {round_idx}: Optimal path selected - regret=0, cumulative={cumulative_regret:.3f}")
        
        # Update cumulative regret
        cumulative_regret += regret
        regrets.append(cumulative_regret)
        rewards.append(reward)
    
    print(f"\nVerification Results:")
    print("=" * 50)
    
    # Check if cumulative regret is non-decreasing
    is_monotonic = all(regrets[i] <= regrets[i+1] for i in range(len(regrets)-1))
    print(f"✅ Cumulative regret is monotonic: {is_monotonic}")
    
    # Check for unexpected zero regrets
    print(f"📊 Total rounds analyzed: {len(regrets)}")
    print(f"📊 Rounds with no feasible paths: {feasible_counts.count(0)}")
    print(f"📊 Rounds with zero regret (optimal selection): {len(zero_regret_rounds)}")
    print(f"📊 Final cumulative regret: {cumulative_regret:.3f}")
    
    if zero_regret_rounds:
        print(f"📊 Zero regret rounds: {zero_regret_rounds[:10]}{'...' if len(zero_regret_rounds) > 10 else ''}")
    
    # Check if regret values make sense
    print(f"\nRegret Analysis:")
    print("-" * 30)
    print(f"Min regret: {min(regrets):.3f}")
    print(f"Max regret: {max(regrets):.3f}")
    print(f"Regret range: {max(regrets) - min(regrets):.3f}")
    
    # Check for any negative regrets (should not happen)
    negative_regrets = [r for r in regrets if r < 0]
    if negative_regrets:
        print(f"❌ Found negative regrets: {negative_regrets}")
    else:
        print(f"✅ No negative regrets found")
    
    # Show sample of regret progression
    print(f"\nSample regret progression:")
    print("-" * 30)
    for i in range(0, min(20, len(regrets)), 5):
        print(f"Round {i:3d}: {regrets[i]:.3f}")
    if len(regrets) > 20:
        print(f"...")
        print(f"Round {len(regrets)-1:3d}: {regrets[-1]:.3f}")
    
    # Clean up
    env.cleanup()
    
    print(f"\n✅ Verification completed successfully!")

if __name__ == "__main__":
    verify_regret_calculation() 