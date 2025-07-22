#!/usr/bin/env python3
"""
Test ETX-based reward calculation for the Qurinet environment.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from environments.qurinet_environment import QurinetEnvironment

def test_etx_calculation():
    """Test ETX calculation with sample data."""
    
    print("ETX Calculation Test")
    print("=" * 50)
    
    # Sample signal strength and quality values
    test_cases = [
        (-30, 100),  # Excellent signal and quality
        (-50, 80),   # Good signal and quality
        (-70, 60),   # Moderate signal and quality
        (-80, 40),   # Poor signal and quality
        (-90, 20),   # Very poor signal and quality
    ]
    
    print("Signal(dBm) | Quality | ETX | 1/ETX")
    print("-" * 40)
    
    for signal_dbm, quality in test_cases:
        # Create temporary environment to test ETX calculation
        env = QurinetEnvironment(
            data_dir="data/qurinet",
            date="2-May",
            num_rounds=100,
            pre_generate_availability=True,
            pre_generate_rewards=True
        )
        
        etx = env._estimate_etx(signal_dbm, quality)
        inv_etx = 1.0 / etx
        
        print(f"{signal_dbm:10} | {quality:7} | {etx:.2f} | {inv_etx:.3f}")
    
    print()

def test_etx_vs_distance():
    """Compare ETX-based rewards with distance-based rewards."""
    
    print("ETX vs Distance-based Rewards Comparison")
    print("=" * 60)
    
    try:
        # Create environment with ETX-based rewards
        env_etx = QurinetEnvironment(
            data_dir="data/qurinet",
            date="2-May",
            num_rounds=1000,
            pre_generate_availability=True,
            pre_generate_rewards=True
        )
        
        # Collect actual data
        edge_data = []
        
        for edge in env_etx.graph.edges():
            arm_id = env_etx.edge_to_arm[edge]
            reward = env_etx.arm_means[arm_id]
            
            # Get signal and quality data for this edge
            signal_values = []
            quality_values = []
            
            for node in edge:
                for interface in [0, 1]:
                    scan_file = os.path.join(env_etx.data_dir, f"{node}-{interface}.csv")
                    if os.path.exists(scan_file):
                        try:
                            scan_data = pd.read_csv(scan_file)
                            for _, row in scan_data.iterrows():
                                essid = row['ESSID']
                                if essid != 'qurinet' and '-' in essid:
                                    parts = essid.split('-')
                                    if len(parts) == 2:
                                        target_node = int(parts[0])
                                        if target_node in [edge[0], edge[1]]:
                                            signal_values.append(row['Signal(dBm)'])
                                            quality_values.append(row['Quality'])
                        except:
                            pass
            
            if signal_values and quality_values:
                avg_signal = np.mean(signal_values)
                avg_quality = np.mean(quality_values)
                estimated_etx = env_etx._estimate_etx(avg_signal, avg_quality)
                
                edge_data.append({
                    'edge': edge,
                    'signal_dbm': avg_signal,
                    'quality': avg_quality,
                    'etx': estimated_etx,
                    'reward': reward,
                    'inv_etx': 1.0 / estimated_etx
                })
        
        if edge_data:
            print(f"Analyzed {len(edge_data)} edges")
            print()
            
            # Print detailed results
            print("Edge | Signal(dBm) | Quality | ETX | 1/ETX | Reward")
            print("-" * 60)
            
            for data in sorted(edge_data, key=lambda x: x['etx']):
                edge_str = f"{data['edge'][0]}-{data['edge'][1]}"
                print(f"{edge_str:8} | {data['signal_dbm']:10.1f} | {data['quality']:7.1f} | "
                      f"{data['etx']:4.2f} | {data['inv_etx']:5.3f} | {data['reward']:6.3f}")
            
            # Statistics
            etx_values = [d['etx'] for d in edge_data]
            reward_values = [d['reward'] for d in edge_data]
            inv_etx_values = [d['inv_etx'] for d in edge_data]
            
            print()
            print("Statistics:")
            print(f"ETX range: {min(etx_values):.2f} - {max(etx_values):.2f}")
            print(f"1/ETX range: {min(inv_etx_values):.3f} - {max(inv_etx_values):.3f}")
            print(f"Reward range: {min(reward_values):.3f} - {max(reward_values):.3f}")
            print(f"Average ETX: {np.mean(etx_values):.2f}")
            print(f"Average reward: {np.mean(reward_values):.3f}")
            
            # Correlation analysis
            correlation = np.corrcoef(etx_values, reward_values)[0, 1]
            print(f"ETX-Reward correlation: {correlation:.3f}")
            
            # Plot ETX vs Reward
            plt.figure(figsize=(10, 6))
            
            plt.subplot(1, 2, 1)
            plt.scatter(etx_values, reward_values, alpha=0.7, s=50)
            plt.xlabel('ETX', fontsize=12)
            plt.ylabel('Reward', fontsize=12)
            plt.title('ETX vs Reward', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.scatter(inv_etx_values, reward_values, alpha=0.7, s=50, color='red')
            plt.xlabel('1/ETX', fontsize=12)
            plt.ylabel('Reward', fontsize=12)
            plt.title('1/ETX vs Reward', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'images')
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'etx_reward_analysis.pdf'), 
                        bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"\nPlot saved to: {os.path.join(output_dir, 'etx_reward_analysis.pdf')}")
        
    except Exception as e:
        print(f"Error testing ETX rewards: {e}")

if __name__ == "__main__":
    test_etx_calculation()
    test_etx_vs_distance() 