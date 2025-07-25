#!/usr/bin/env python3
"""
Analyze regret patterns to understand when regret becomes zero or negative.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment
from bandits.cts_b import CTSB

def analyze_regret_patterns():
    """Analyze regret patterns over multiple rounds."""
    
    print("Analyzing regret patterns...")
    print("=" * 60)
    
    # Setup environment
    data_dir = Path("data/ucsb")
    
    # Find the trace directory with the most files
    trace_dirs = {}
    for subdir in data_dir.iterdir():
        if subdir.is_dir():
            files = list(subdir.glob("neighbortable-*"))
            trace_dirs[subdir.name] = len(files)
    
    selected_trace_dir = max(trace_dirs, key=trace_dirs.get)
    selected_trace_path = data_dir / selected_trace_dir
    
    # Use first 50 files for analysis
    neighbortable_files = sorted(list(selected_trace_path.glob("neighbortable-*")))[:50]
    neighbortable_files = [str(f) for f in neighbortable_files]
    
    print(f"Using trace directory: {selected_trace_dir}")
    print(f"Using {len(neighbortable_files)} files for analysis")
    print()
    
    # Create environment
    env = UCSBMeshnetMemmapEnvironment(
        neighbortable_files=neighbortable_files,
        source="10.2.1.100",
        destination="10.2.1.20",
        max_path_length=3,
        max_feasible_combinations=50
    )
    
    print(f"Environment created with {env.num_rounds} rounds")
    print()
    
    # Create algorithm
    alg = CTSB(environment=env, rng=np.random.default_rng(42))
    
    # Track regret patterns
    instant_regrets = []
    cumulative_regrets = []
    optimal_rewards = []
    selected_rewards = []
    rounds_with_paths = []
    rounds_without_paths = 0
    
    cumulative_regret = 0.0
    
    print("Running simulation to analyze regret patterns...")
    
    for round_idx in range(env.num_rounds):
        # Get available arms
        available_arms = env.get_available_arms_for_round(round_idx)
        feasible_combinations = env.get_feasible_combinations(available_arms)
        
        if not feasible_combinations:
            rounds_without_paths += 1
            continue
        
        rounds_with_paths.append(round_idx)
        
        # Select action
        selected_combination = alg.select_combination(round_idx)
        
        # Calculate rewards
        total_reward = env.get_expected_reward_for_path(selected_combination, round_idx)
        optimal_reward = env.get_optimal_path_expected_reward(round_idx)
        
        # Calculate regret
        regret = optimal_reward - total_reward
        cumulative_regret += regret
        
        # Store data
        instant_regrets.append(regret)
        cumulative_regrets.append(cumulative_regret)
        optimal_rewards.append(optimal_reward)
        selected_rewards.append(total_reward)
        
        # Update algorithm
        reward_dict = {}
        for arm in selected_combination:
            reward_dict[arm] = env.get_reward_for_round(arm, round_idx)
        alg.update_posterior(selected_combination, reward_dict, round_idx)
    
    print(f"Analysis completed!")
    print(f"Total rounds: {env.num_rounds}")
    print(f"Rounds with paths: {len(rounds_with_paths)}")
    print(f"Rounds without paths: {rounds_without_paths}")
    print()
    
    # Analyze regret patterns
    instant_regrets = np.array(instant_regrets)
    cumulative_regrets = np.array(cumulative_regrets)
    optimal_rewards = np.array(optimal_rewards)
    selected_rewards = np.array(selected_rewards)
    
    print("Regret Analysis:")
    print("-" * 40)
    print(f"Instant regret statistics:")
    print(f"  Min: {np.min(instant_regrets):.6f}")
    print(f"  Max: {np.max(instant_regrets):.6f}")
    print(f"  Mean: {np.mean(instant_regrets):.6f}")
    print(f"  Std: {np.std(instant_regrets):.6f}")
    
    print(f"\nCumulative regret statistics:")
    print(f"  Final: {cumulative_regrets[-1]:.6f}")
    print(f"  Min: {np.min(cumulative_regrets):.6f}")
    print(f"  Max: {np.max(cumulative_regrets):.6f}")
    print(f"  Mean: {np.mean(cumulative_regrets):.6f}")
    
    # Count zero and negative regrets
    zero_instant_regrets = np.sum(instant_regrets == 0)
    negative_instant_regrets = np.sum(instant_regrets < 0)
    zero_cumulative_regrets = np.sum(cumulative_regrets == 0)
    negative_cumulative_regrets = np.sum(cumulative_regrets < 0)
    
    print(f"\nZero/negative regret counts:")
    print(f"  Zero instant regrets: {zero_instant_regrets} ({100*zero_instant_regrets/len(instant_regrets):.1f}%)")
    print(f"  Negative instant regrets: {negative_instant_regrets} ({100*negative_instant_regrets/len(instant_regrets):.1f}%)")
    print(f"  Zero cumulative regrets: {zero_cumulative_regrets} ({100*zero_cumulative_regrets/len(cumulative_regrets):.1f}%)")
    print(f"  Negative cumulative regrets: {negative_cumulative_regrets} ({100*negative_cumulative_regrets/len(cumulative_regrets):.1f}%)")
    
    # Analyze reward patterns
    print(f"\nReward Analysis:")
    print("-" * 40)
    print(f"Optimal reward statistics:")
    print(f"  Min: {np.min(optimal_rewards):.6f}")
    print(f"  Max: {np.max(optimal_rewards):.6f}")
    print(f"  Mean: {np.mean(optimal_rewards):.6f}")
    print(f"  Std: {np.std(optimal_rewards):.6f}")
    
    print(f"\nSelected reward statistics:")
    print(f"  Min: {np.min(selected_rewards):.6f}")
    print(f"  Max: {np.max(selected_rewards):.6f}")
    print(f"  Mean: {np.mean(selected_rewards):.6f}")
    print(f"  Std: {np.std(selected_rewards):.6f}")
    
    # Check if rewards are identical
    if len(np.unique(optimal_rewards)) == 1:
        print(f"  ⚠️  WARNING: All optimal rewards are identical ({optimal_rewards[0]:.6f})")
    if len(np.unique(selected_rewards)) == 1:
        print(f"  ⚠️  WARNING: All selected rewards are identical ({selected_rewards[0]:.6f})")
    
    # Find rounds with zero regret
    if zero_instant_regrets > 0:
        zero_regret_rounds = np.where(instant_regrets == 0)[0]
        print(f"\nRounds with zero instant regret:")
        for i, round_idx in enumerate(zero_regret_rounds[:10]):  # Show first 10
            actual_round = rounds_with_paths[round_idx]
            print(f"  Round {actual_round}: optimal={optimal_rewards[round_idx]:.6f}, selected={selected_rewards[round_idx]:.6f}")
        if len(zero_regret_rounds) > 10:
            print(f"  ... and {len(zero_regret_rounds) - 10} more")
    
    # Find rounds with negative regret
    if negative_instant_regrets > 0:
        negative_regret_rounds = np.where(instant_regrets < 0)[0]
        print(f"\nRounds with negative instant regret:")
        for i, round_idx in enumerate(negative_regret_rounds[:10]):  # Show first 10
            actual_round = rounds_with_paths[round_idx]
            print(f"  Round {actual_round}: regret={instant_regrets[round_idx]:.6f}, optimal={optimal_rewards[round_idx]:.6f}, selected={selected_rewards[round_idx]:.6f}")
        if len(negative_regret_rounds) > 10:
            print(f"  ... and {len(negative_regret_rounds) - 10} more")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Instant regret over time
    axes[0, 0].plot(instant_regrets)
    axes[0, 0].set_title('Instant Regret Over Time')
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Instant Regret')
    axes[0, 0].grid(True)
    
    # Plot 2: Cumulative regret over time
    axes[0, 1].plot(cumulative_regrets)
    axes[0, 1].set_title('Cumulative Regret Over Time')
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Cumulative Regret')
    axes[0, 1].grid(True)
    
    # Plot 3: Rewards comparison
    axes[1, 0].plot(optimal_rewards, label='Optimal', alpha=0.7)
    axes[1, 0].plot(selected_rewards, label='Selected', alpha=0.7)
    axes[1, 0].set_title('Rewards Comparison')
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 4: Regret distribution
    axes[1, 1].hist(instant_regrets, bins=50, alpha=0.7)
    axes[1, 1].set_title('Instant Regret Distribution')
    axes[1, 1].set_xlabel('Instant Regret')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('regret_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to: regret_analysis.png")

if __name__ == "__main__":
    analyze_regret_patterns() 