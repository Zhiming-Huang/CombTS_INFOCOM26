#!/usr/bin/env python3
"""
Test different distance-based reward methods for the Qurinet environment.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from environments.qurinet_environment import QurinetEnvironment

def test_reward_methods():
    """Test different reward calculation methods."""
    
    # Sample distances (in km) from typical wireless network
    sample_distances = np.array([0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
    
    print("Distance-based Reward Methods Comparison")
    print("=" * 50)
    print(f"Sample distances (km): {sample_distances}")
    print()
    
    # Method 1: Exponential decay
    scale_factor = 2.0
    exp_rewards = np.exp(-sample_distances / scale_factor)
    
    # Method 2: Gaussian decay
    gaussian_rewards = np.exp(-(sample_distances / scale_factor) ** 2)
    
    # Method 3: Power function decay
    power = 2.0
    power_rewards = 1.0 / (1.0 + (sample_distances / scale_factor) ** power)
    
    # Method 4: Linear decay
    max_dist = max(sample_distances)
    linear_rewards = np.maximum(0.1, 1.0 - sample_distances / max_dist)
    
    # Method 5: Original inverse distance
    inv_rewards = 1.0 / (sample_distances + 0.01)
    
    # Print results
    methods = {
        "Exponential": exp_rewards,
        "Gaussian": gaussian_rewards,
        "Power (1/(1+x²))": power_rewards,
        "Linear": linear_rewards,
        "Inverse (1/x)": inv_rewards
    }
    
    for method_name, rewards in methods.items():
        print(f"{method_name:15} rewards: {rewards}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    for method_name, rewards in methods.items():
        plt.plot(sample_distances, rewards, 'o-', label=method_name, linewidth=2, markersize=6)
    
    plt.xlabel('Distance (km)', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title('Distance-based Reward Methods Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max(sample_distances))
    plt.ylim(0, 1.1)
    
    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'images')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'reward_methods_comparison.pdf'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"\nPlot saved to: {os.path.join(output_dir, 'reward_methods_comparison.pdf')}")
    
    # Test with actual Qurinet environment
    print("\n" + "=" * 50)
    print("Testing with actual Qurinet environment...")
    
    try:
        env = QurinetEnvironment(
            data_dir="data/qurinet",
            date="2-May",
            num_rounds=1000,
            pre_generate_availability=True,
            pre_generate_rewards=True
        )
        
        # Get actual distances and rewards from environment
        actual_distances = []
        actual_rewards = []
        
        for edge in env.graph.edges():
            # Estimate distance from signal strength data
            distances = []
            for node in edge:
                for interface in [0, 1]:
                    scan_file = os.path.join(env.data_dir, f"{node}-{interface}.csv")
                    if os.path.exists(scan_file):
                        try:
                            import pandas as pd
                            scan_data = pd.read_csv(scan_file)
                            for _, row in scan_data.iterrows():
                                essid = row['ESSID']
                                if essid != 'qurinet' and '-' in essid:
                                    parts = essid.split('-')
                                    if len(parts) == 2:
                                        target_node = int(parts[0])
                                        if target_node in [edge[0], edge[1]]:
                                            signal_dbm = row['Signal(dBm)']
                                            distance = env._signal_to_distance_estimate(signal_dbm, 16)
                                            distances.append(distance)
                        except:
                            pass
            
            if distances:
                avg_dist = np.mean(distances)
                actual_distances.append(avg_dist)
                # Get reward for this edge
                arm_id = env.edge_to_arm[edge]
                actual_rewards.append(env.arm_means[arm_id])
        
        if actual_distances:
            print(f"Actual network distances: {len(actual_distances)} links")
            print(f"Distance range: {min(actual_distances):.3f} - {max(actual_distances):.3f} km")
            print(f"Reward range: {min(actual_rewards):.3f} - {max(actual_rewards):.3f}")
            
            # Plot actual vs theoretical
            plt.figure(figsize=(10, 6))
            
            # Plot theoretical curves
            x_theoretical = np.linspace(0, max(actual_distances), 100)
            plt.plot(x_theoretical, np.exp(-x_theoretical / scale_factor), 
                    'b-', label='Exponential (theoretical)', alpha=0.7)
            plt.plot(x_theoretical, np.exp(-(x_theoretical / scale_factor) ** 2), 
                    'g-', label='Gaussian (theoretical)', alpha=0.7)
            
            # Plot actual data
            plt.scatter(actual_distances, actual_rewards, 
                       c='red', s=50, alpha=0.8, label='Actual network data')
            
            plt.xlabel('Distance (km)', fontsize=12)
            plt.ylabel('Reward', fontsize=12)
            plt.title('Actual vs Theoretical Reward Methods', fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            
            plt.savefig(os.path.join(output_dir, 'actual_vs_theoretical_rewards.pdf'), 
                        bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"Actual vs theoretical plot saved to: {os.path.join(output_dir, 'actual_vs_theoretical_rewards.pdf')}")
        
    except Exception as e:
        print(f"Error testing with actual environment: {e}")

if __name__ == "__main__":
    test_reward_methods() 