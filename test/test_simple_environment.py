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

from src.bandits.comb_ts import CombTS
from src.environments.simple_environment import SimpleEnvironment


def run_simple_simulation(num_rounds: int = 10000, num_runs: int = 5) -> Dict[str, Any]:
    """
    Run multiple simulations to verify sublinear regret.
    
    Args:
        num_rounds: Number of rounds per simulation
        num_runs: Number of independent runs
        
    Returns:
        Dictionary containing aggregated results
    """
    print(f"Running {num_runs} simulations with {num_rounds} rounds each...")
    
    all_cumulative_regrets = []
    all_cumulative_rewards = []
    
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        
        # Create environment and algorithm
        env = SimpleEnvironment(
            num_arms=10,
            num_optimal=3,
            optimal_mean=0.9,
            suboptimal_mean=0.1,
            availability_rate=0.5,
            max_combination_size=3
        )
        
        algorithm = CombTS(environment=env, alpha=1.0, beta=1.0)
        
        # Run simulation
        cumulative_rewards = []
        cumulative_regrets = []
        total_reward = 0
        total_regret = 0
        
        for round_num in range(num_rounds):
            # Reset available arms for this round
            env.reset_available_arms()
            
            # Select combination (this will use the consistent available arms)
            selected_combination = algorithm.select_combination()
            
            # Generate rewards
            rewards = {}
            total_round_reward = 0
            if selected_combination:
                rewards = env.generate_combination_reward(selected_combination)
                total_round_reward = sum(rewards.values())
            
            # Update algorithm
            algorithm.update_posterior(selected_combination, rewards)
            
            # Calculate regret using the same available arms that were used by the algorithm
            available_arms = env.sample_available_arms_once()
            optimal_combination = env.get_optimal_combination(available_arms)
            optimal_reward = sum(env.arm_means[arm] for arm in optimal_combination)
            regret = optimal_reward - total_round_reward
            
            # Update cumulative values
            total_reward += total_round_reward
            total_regret += regret
            
            cumulative_rewards.append(total_reward)
            cumulative_regrets.append(total_regret)
            
            if round_num % 1000 == 0 and round_num > 0:
                print(f"  Round {round_num}: Cumulative regret = {total_regret:.2f}")
        
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
    
    # Run simulation
    results = run_simple_simulation(num_rounds=10000, num_runs=5)
    
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