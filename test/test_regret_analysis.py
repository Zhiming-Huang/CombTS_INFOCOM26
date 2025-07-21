#!/usr/bin/env python3
"""
Test script to analyze regret growth and verify sublinear behavior.
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


def run_regret_analysis(num_rounds: int = 5000, num_runs: int = 10) -> Dict[str, Any]:
    """
    Run regret analysis to verify sublinear growth.
    
    Args:
        num_rounds: Number of rounds per simulation
        num_runs: Number of independent runs
        
    Returns:
        Dictionary containing analysis results
    """
    print(f"Running regret analysis: {num_runs} runs with {num_rounds} rounds each...")
    
    all_cumulative_regrets = []
    all_instantaneous_regrets = []
    
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
        cumulative_regrets = []
        instantaneous_regrets = []
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
            
            # Update cumulative regret
            total_regret += regret
            cumulative_regrets.append(total_regret)
            instantaneous_regrets.append(regret)
            
            if round_num % 1000 == 0 and round_num > 0:
                print(f"  Round {round_num}: Cumulative regret = {total_regret:.2f}")
        
        all_cumulative_regrets.append(cumulative_regrets)
        all_instantaneous_regrets.append(instantaneous_regrets)
    
    # Aggregate results
    avg_cumulative_regrets = np.mean(all_cumulative_regrets, axis=0)
    std_cumulative_regrets = np.std(all_cumulative_regrets, axis=0)
    
    avg_instantaneous_regrets = np.mean(all_instantaneous_regrets, axis=0)
    
    return {
        'num_rounds': num_rounds,
        'num_runs': num_runs,
        'avg_cumulative_regrets': avg_cumulative_regrets,
        'std_cumulative_regrets': std_cumulative_regrets,
        'avg_instantaneous_regrets': avg_instantaneous_regrets,
        'final_avg_regret': avg_cumulative_regrets[-1],
        'final_std_regret': std_cumulative_regrets[-1]
    }


def analyze_growth_rates(results: Dict[str, Any]):
    """
    Analyze different growth rates to determine if regret is sublinear.
    
    Args:
        results: Results from run_regret_analysis
    """
    num_rounds = results['num_rounds']
    avg_regrets = results['avg_cumulative_regrets']
    rounds = np.arange(1, num_rounds + 1)
    
    # Calculate different growth rates
    sqrt_growth = avg_regrets / np.sqrt(rounds)
    log_growth = avg_regrets / np.log(rounds + 1)
    linear_growth = avg_regrets / rounds
    
    # Check if growth rates are bounded
    print("\nGrowth Rate Analysis:")
    print("=" * 60)
    
    # Test points
    test_points = [100, 500, 1000, 2000, 5000]
    test_points = [p for p in test_points if p <= num_rounds]
    
    print(f"{'Round':>6} {'Regret':>8} {'Regret/√T':>10} {'Regret/log(T)':>12} {'Regret/T':>10}")
    print("-" * 60)
    
    for point in test_points:
        idx = point - 1
        regret = avg_regrets[idx]
        sqrt_rate = sqrt_growth[idx]
        log_rate = log_growth[idx]
        linear_rate = linear_growth[idx]
        
        print(f"{point:6d} {regret:8.2f} {sqrt_rate:10.4f} {log_rate:12.4f} {linear_rate:10.4f}")
    
    # Final analysis
    final_sqrt_rate = sqrt_growth[-1]
    final_log_rate = log_growth[-1]
    final_linear_rate = linear_growth[-1]
    
    print("\nFinal Growth Rates:")
    print(f"  Regret/√T: {final_sqrt_rate:.4f}")
    print(f"  Regret/log(T): {final_log_rate:.4f}")
    print(f"  Regret/T: {final_linear_rate:.4f}")
    
    # Determine if regret is sublinear
    if final_linear_rate < 0.1:  # Linear growth should be very small
        print("✓ Regret appears to be sublinear (linear growth rate is small)")
    else:
        print("✗ Regret may be linear or worse")
    
    if final_sqrt_rate < 5:  # Square root growth should be bounded
        print("✓ Regret appears to be O(√T) or better")
    else:
        print("✗ Regret may be worse than O(√T)")


def plot_regret_analysis(results: Dict[str, Any]):
    """
    Plot comprehensive regret analysis.
    
    Args:
        results: Results from run_regret_analysis
    """
    num_rounds = results['num_rounds']
    rounds = np.arange(1, num_rounds + 1)
    avg_regrets = results['avg_cumulative_regrets']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Cumulative regret
    axes[0, 0].plot(rounds, avg_regrets, 'b-', linewidth=2)
    axes[0, 0].fill_between(rounds, 
                           avg_regrets - results['std_cumulative_regrets'],
                           avg_regrets + results['std_cumulative_regrets'],
                           alpha=0.3, color='blue')
    axes[0, 0].set_title('Cumulative Regret')
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Cumulative Regret')
    axes[0, 0].grid(True)
    
    # Plot 2: Instantaneous regret
    axes[0, 1].plot(rounds, results['avg_instantaneous_regrets'], 'r-', linewidth=1, alpha=0.7)
    axes[0, 1].set_title('Instantaneous Regret')
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Instantaneous Regret')
    axes[0, 1].grid(True)
    
    # Plot 3: Regret growth rates
    sqrt_growth = avg_regrets / np.sqrt(rounds)
    log_growth = avg_regrets / np.log(rounds + 1)
    linear_growth = avg_regrets / rounds
    
    axes[0, 2].plot(rounds, sqrt_growth, 'g-', linewidth=2, label='Regret/√T')
    axes[0, 2].plot(rounds, log_growth, 'orange', linewidth=2, label='Regret/log(T)')
    axes[0, 2].plot(rounds, linear_growth, 'red', linewidth=2, label='Regret/T')
    axes[0, 2].set_title('Growth Rates')
    axes[0, 2].set_xlabel('Round')
    axes[0, 2].set_ylabel('Growth Rate')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Plot 4: Log-log plot
    axes[1, 0].loglog(rounds, avg_regrets, 'purple', linewidth=2, label='Cumulative Regret')
    axes[1, 0].loglog(rounds, np.sqrt(rounds), '--', alpha=0.5, label='√T')
    axes[1, 0].loglog(rounds, rounds, '--', alpha=0.5, label='T')
    axes[1, 0].set_title('Log-Log Plot')
    axes[1, 0].set_xlabel('Round (log scale)')
    axes[1, 0].set_ylabel('Cumulative Regret (log scale)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 5: Moving average of instantaneous regret
    window_size = min(100, num_rounds // 10)
    if window_size > 1:
        moving_avg = np.convolve(results['avg_instantaneous_regrets'], 
                                np.ones(window_size)/window_size, mode='valid')
        axes[1, 1].plot(rounds[window_size-1:], moving_avg, 'b-', linewidth=2)
        axes[1, 1].set_title(f'Moving Average of Instantaneous Regret (window={window_size})')
        axes[1, 1].set_xlabel('Round')
        axes[1, 1].set_ylabel('Moving Average Regret')
        axes[1, 1].grid(True)
    
    # Plot 6: Regret per round (smoothed)
    axes[1, 2].plot(rounds, avg_regrets / rounds, 'orange', linewidth=2)
    axes[1, 2].set_title('Average Regret per Round')
    axes[1, 2].set_xlabel('Round')
    axes[1, 2].set_ylabel('Average Regret per Round')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig('output/regret_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function to run regret analysis."""
    print("CombTS Regret Analysis")
    print("=" * 30)
    
    # Run analysis
    results = run_regret_analysis(num_rounds=5000, num_runs=10)
    
    # Analyze results
    print(f"\nAnalysis completed!")
    print(f"Final average cumulative regret: {results['final_avg_regret']:.2f} ± {results['final_std_regret']:.2f}")
    
    # Analyze growth rates
    analyze_growth_rates(results)
    
    # Plot results
    print("\nGenerating plots...")
    plot_regret_analysis(results)
    
    print("\nAnalysis completed! Check output/regret_analysis.png for plots.")


if __name__ == "__main__":
    main() 