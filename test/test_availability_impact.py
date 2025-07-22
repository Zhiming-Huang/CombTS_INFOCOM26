#!/usr/bin/env python3
"""
Test the impact of 70% link unavailability on algorithm performance.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from environments.qurinet_environment import QurinetEnvironment

def analyze_availability_impact():
    """Analyze how 70% unavailability affects the environment."""
    
    print("Link Availability Impact Analysis")
    print("=" * 50)
    
    try:
        # Create environment with 70% unavailability
        env = QurinetEnvironment(
            data_dir="data/qurinet",
            date="2-May",
            num_rounds=1000,
            pre_generate_availability=True,
            pre_generate_rewards=True
        )
        
        print(f"Network Statistics:")
        print(f"  Total edges: {len(env.graph.edges())}")
        print(f"  Total arms: {env.num_arms}")
        print(f"  Link availability rate: {env.arm_availability_rates[0]:.1f}")
        print()
        
        # Analyze availability patterns
        print("Availability Analysis:")
        print(f"  Average availability: {np.mean(env.arm_availability_rates):.3f}")
        print(f"  Min availability: {np.min(env.arm_availability_rates):.3f}")
        print(f"  Max availability: {np.max(env.arm_availability_rates):.3f}")
        print()
        
        # Analyze availability matrix
        if hasattr(env, 'availability_matrix'):
            total_available = np.sum(env.availability_matrix)
            total_possible = env.availability_matrix.size
            actual_availability_rate = total_available / total_possible
            
            print(f"Availability Matrix Analysis:")
            print(f"  Total available slots: {total_available}")
            print(f"  Total possible slots: {total_possible}")
            print(f"  Actual availability rate: {actual_availability_rate:.3f}")
            print(f"  Expected availability rate: 0.300")
            print()
        
        # Analyze feasible combinations
        print("Feasible Combinations Analysis:")
        
        # Sample a few rounds to see availability patterns
        sample_rounds = [0, 100, 500, 999]
        
        for round_num in sample_rounds:
            available_arms = env.get_available_arms_for_round(round_num)
            feasible_combinations = env.get_feasible_combinations(available_arms)
            
            print(f"  Round {round_num}:")
            print(f"    Available arms: {len(available_arms)}/{env.num_arms} ({len(available_arms)/env.num_arms:.1%})")
            print(f"    Feasible combinations: {len(feasible_combinations)}")
            
            if feasible_combinations:
                # Find optimal combination for this round
                optimal_combination = env.get_optimal_combination(available_arms)
                if optimal_combination:
                    optimal_reward = sum(env.arm_means[arm_id] for arm_id in optimal_combination)
                    path_info = env.get_path_info(optimal_combination)
                    print(f"    Optimal path: {path_info.get('path', 'N/A')}")
                    print(f"    Optimal reward: {optimal_reward:.3f}")
            print()
        
        # Analyze reward distribution
        print("Reward Distribution:")
        rewards = env.arm_means
        print(f"  Average reward: {np.mean(rewards):.3f}")
        print(f"  Min reward: {np.min(rewards):.3f}")
        print(f"  Max reward: {np.max(rewards):.3f}")
        print(f"  Std reward: {np.std(rewards):.3f}")
        print()
        
        # Plot availability patterns
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Availability over time for first 100 rounds
        plt.subplot(2, 2, 1)
        rounds_to_plot = min(100, env.num_rounds)
        availability_counts = []
        for round_num in range(rounds_to_plot):
            available_arms = env.get_available_arms_for_round(round_num)
            availability_counts.append(len(available_arms))
        
        plt.plot(range(rounds_to_plot), availability_counts, 'b-', alpha=0.7)
        plt.axhline(y=env.num_arms * 0.3, color='r', linestyle='--', alpha=0.7, label='Expected (30%)')
        plt.xlabel('Round')
        plt.ylabel('Available Arms')
        plt.title('Available Arms Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Availability distribution
        plt.subplot(2, 2, 2)
        plt.hist(availability_counts, bins=20, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(x=env.num_arms * 0.3, color='r', linestyle='--', alpha=0.7, label='Expected (30%)')
        plt.xlabel('Number of Available Arms')
        plt.ylabel('Frequency')
        plt.title('Distribution of Available Arms')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Reward vs Availability
        plt.subplot(2, 2, 3)
        edge_rewards = []
        edge_availabilities = []
        
        for edge in env.graph.edges():
            arm_id = env.edge_to_arm[edge]
            edge_rewards.append(env.arm_means[arm_id])
            edge_availabilities.append(env.arm_availability_rates[arm_id])
        
        plt.scatter(edge_availabilities, edge_rewards, alpha=0.7, s=50)
        plt.xlabel('Availability Rate')
        plt.ylabel('Reward')
        plt.title('Reward vs Availability')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Optimal path availability
        plt.subplot(2, 2, 4)
        optimal_path_availability = []
        
        for round_num in range(rounds_to_plot):
            available_arms = env.get_available_arms_for_round(round_num)
            optimal_combination = env.get_optimal_combination(available_arms)
            if optimal_combination:
                # Check if optimal path is available
                path_available = all(arm_id in available_arms for arm_id in optimal_combination)
                optimal_path_availability.append(1 if path_available else 0)
            else:
                optimal_path_availability.append(0)
        
        plt.plot(range(len(optimal_path_availability)), optimal_path_availability, 'g-', alpha=0.7)
        plt.xlabel('Round')
        plt.ylabel('Optimal Path Available')
        plt.title('Optimal Path Availability')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'images')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'availability_impact_analysis.pdf'), 
                    bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Analysis plots saved to: {os.path.join(output_dir, 'availability_impact_analysis.pdf')}")
        
        # Summary statistics
        print("\nSummary:")
        print(f"  Expected available arms per round: {env.num_arms * 0.3:.1f}")
        print(f"  Actual average available arms: {np.mean(availability_counts):.1f}")
        print(f"  Optimal path availability rate: {np.mean(optimal_path_availability):.1%}")
        
    except Exception as e:
        print(f"Error analyzing availability impact: {e}")

if __name__ == "__main__":
    analyze_availability_impact() 