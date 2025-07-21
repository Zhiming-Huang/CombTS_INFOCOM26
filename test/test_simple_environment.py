#!/usr/bin/env python3
"""
Test script for simple environment to verify sublinear regret of CombTS.
"""

import sys
import os
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
from tqdm import tqdm

from src.bandits.comb_ts import CombTS
from src.environments.simple_environment import SimpleEnvironment


def run_simple_simulation(num_rounds: int = 10000, num_runs: int = 5, 
                         progress_level: str = "normal") -> Dict[str, Any]:
    """
    Run multiple simulations to verify sublinear regret.
    
    Args:
        num_rounds: Number of rounds per simulation
        num_runs: Number of independent runs
        progress_level: Progress tracking level ("minimal", "normal", "detailed")
        
    Returns:
        Dictionary containing aggregated results
    """
    print(f"Running {num_runs} simulations with {num_rounds} rounds each...")
    print(f"Progress level: {progress_level}")
    
    all_cumulative_regrets = []
    all_cumulative_rewards = []
    
    # Configure progress bars based on level
    show_run_progress = progress_level in ["normal", "detailed"]
    show_round_progress = progress_level == "detailed"
    show_regret_updates = progress_level == "detailed"
    
    # Create progress bar for runs
    if show_run_progress:
        run_pbar = tqdm(range(num_runs), desc="Simulation runs", unit="run")
    else:
        run_pbar = range(num_runs)
    
    for run in run_pbar:
        if show_run_progress:
            run_pbar.set_postfix({"Run": f"{run + 1}/{num_runs}"})
        
        # Create environment and algorithm with pre-generated matrices
        env = SimpleEnvironment(
            num_arms=10,
            num_optimal=3,
            optimal_mean=0.9,
            suboptimal_mean=0.1,
            availability_rate=0.5,
            max_combination_size=3,
            num_rounds=num_rounds,  # Pre-generate availability matrix
            pre_generate_rewards=True,  # Pre-generate rewards matrix
            seed=42 + run  # Different seed for each run
        )
        
        algorithm = CombTS(environment=env, alpha=1.0, beta=1.0)
        
        # Run simulation
        cumulative_rewards = []
        cumulative_regrets = []
        total_reward = 0
        total_regret = 0
        
        # Create progress bar for rounds
        if show_round_progress:
            round_pbar = tqdm(range(num_rounds), desc=f"Run {run + 1} rounds", 
                             unit="round", leave=False)
        else:
            round_pbar = range(num_rounds)
        
        for round_num in round_pbar:
            # Get available arms for this round directly from matrix
            available_arms = env.get_available_arms_for_round(round_num)
            
            # Select combination using the available arms
            selected_combination = algorithm.select_combination()
            
            # Generate rewards for selected combination
            rewards = {}
            total_round_reward = 0
            if selected_combination:
                # Get rewards from pre-generated matrix if available
                if hasattr(env, 'rewards_matrix'):
                    for arm in selected_combination:
                        rewards[arm] = env.get_reward_for_round(arm, round_num)
                else:
                    rewards = env.generate_combination_reward(selected_combination)
                total_round_reward = sum(rewards.values())
            
            # Update algorithm
            algorithm.update_posterior(selected_combination, rewards)
            
            # Calculate regret using the same available arms
            optimal_combination = env.get_optimal_combination(available_arms)
            
            # Calculate expected rewards for fair comparison
            optimal_expected_reward = sum(env.arm_means[arm] for arm in optimal_combination) if optimal_combination else 0
            selected_expected_reward = sum(env.arm_means[arm] for arm in selected_combination) if selected_combination else 0
            
            # Calculate regret as difference in expected rewards
            # This ensures we compare expected rewards (mean values) rather than 
            # mixing expected rewards (optimal) with instantaneous rewards (selected)
            regret = optimal_expected_reward - selected_expected_reward
            
            # Update cumulative values
            total_reward += total_round_reward
            total_regret += regret
            
            cumulative_rewards.append(total_reward)
            cumulative_regrets.append(total_regret)
            
            # Update progress bar with current regret
            if show_regret_updates and isinstance(round_pbar, tqdm):
                round_pbar.set_postfix({
                    "Regret": f"{total_regret:.2f}",
                    "Reward": f"{total_reward:.2f}"
                })
        
        all_cumulative_regrets.append(cumulative_regrets)
        all_cumulative_rewards.append(cumulative_rewards)
    
    # Aggregate results
    avg_cumulative_regrets = np.mean(all_cumulative_regrets, axis=0)
    std_cumulative_regrets = np.std(all_cumulative_regrets, axis=0)
    
    avg_cumulative_rewards = np.mean(all_cumulative_rewards, axis=0)
    
    return {
        'num_rounds': num_rounds,
        'num_runs': num_runs,
        'avg_cumulative_regrets': avg_cumulative_regrets,
        'std_cumulative_regrets': std_cumulative_regrets,
        'avg_cumulative_rewards': avg_cumulative_rewards,
        'final_avg_regret': avg_cumulative_regrets[-1],
        'final_std_regret': std_cumulative_regrets[-1]
    }


def analyze_regret_growth(results: Dict[str, Any]):
    """
    Analyze the growth rate of regret to verify sublinear behavior.
    
    Args:
        results: Results from run_simple_simulation
    """
    num_rounds = results['num_rounds']
    avg_regrets = results['avg_cumulative_regrets']
    
    # Calculate growth rates at different points
    points = [100, 500, 1000, 2000, 5000, 10000]
    points = [p for p in points if p <= num_rounds]
    
    print("\nRegret Growth Analysis:")
    print("=" * 50)
    
    for i, point in enumerate(points):
        if point <= len(avg_regrets):
            regret = avg_regrets[point - 1]
            # Calculate growth rate (regret / sqrt(rounds))
            growth_rate = regret / np.sqrt(point)
            print(f"Round {point:5d}: Regret = {regret:8.2f}, Growth Rate = {growth_rate:.4f}")
    
    # Check if regret is sublinear (growth rate should be bounded)
    final_growth_rate = avg_regrets[-1] / np.sqrt(num_rounds)
    print(f"\nFinal growth rate: {final_growth_rate:.4f}")
    
    if final_growth_rate < 10:  # Arbitrary threshold
        print("✓ Regret appears to be sublinear (bounded growth rate)")
    else:
        print("✗ Regret may not be sublinear (unbounded growth rate)")


def plot_results(results: Dict[str, Any]):
    """
    Plot the simulation results.
    
    Args:
        results: Results from run_simple_simulation
    """
    num_rounds = results['num_rounds']
    rounds = np.arange(1, num_rounds + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Cumulative regrets
    axes[0, 0].plot(rounds, results['avg_cumulative_regrets'], 'b-', linewidth=2)
    axes[0, 0].fill_between(rounds, 
                           results['avg_cumulative_regrets'] - results['std_cumulative_regrets'],
                           results['avg_cumulative_regrets'] + results['std_cumulative_regrets'],
                           alpha=0.3, color='blue')
    axes[0, 0].set_title('Cumulative Regret')
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Cumulative Regret')
    axes[0, 0].grid(True)
    
    # Plot 2: Regret growth rate
    growth_rates = results['avg_cumulative_regrets'] / np.sqrt(rounds)
    axes[0, 1].plot(rounds, growth_rates, 'r-', linewidth=2)
    axes[0, 1].set_title('Regret Growth Rate (Regret / √T)')
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Growth Rate')
    axes[0, 1].grid(True)
    
    # Plot 3: Cumulative rewards
    axes[1, 0].plot(rounds, results['avg_cumulative_rewards'], 'g-', linewidth=2)
    axes[1, 0].set_title('Cumulative Rewards')
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('Cumulative Reward')
    axes[1, 0].grid(True)
    
    # Plot 4: Log-log plot of regret (to check sublinearity)
    axes[1, 1].loglog(rounds, results['avg_cumulative_regrets'], 'purple', linewidth=2)
    # Add reference lines for different growth rates
    axes[1, 1].loglog(rounds, np.sqrt(rounds), '--', alpha=0.5, label='√T')
    axes[1, 1].loglog(rounds, rounds, '--', alpha=0.5, label='T')
    axes[1, 1].set_title('Log-Log Plot of Cumulative Regret')
    axes[1, 1].set_xlabel('Round (log scale)')
    axes[1, 1].set_ylabel('Cumulative Regret (log scale)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('output/simple_environment_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function to run the simple environment test."""
    print("Simple Environment CombTS Test")
    print("=" * 40)
    
    # Run simulation with configurable progress level
    # Options: "minimal" (no progress bars), "normal" (run progress), "detailed" (full progress)
    results = run_simple_simulation(num_rounds=10000, num_runs=5, progress_level="normal")
    
    # Analyze results
    print(f"\nSimulation completed!")
    print(f"Final average cumulative regret: {results['final_avg_regret']:.2f} ± {results['final_std_regret']:.2f}")
    
    # Analyze regret growth
    analyze_regret_growth(results)
    
    # Plot results
    print("\nGenerating plots...")
    plot_results(results)
    
    print("\nTest completed! Check output/simple_environment_results.png for plots.")


if __name__ == "__main__":
    main() 