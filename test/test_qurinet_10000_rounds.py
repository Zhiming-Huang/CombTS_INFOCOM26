#!/usr/bin/env python3
"""
Test algorithms for 10000 rounds in Qurinet environment and plot regret curves.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from environments.qurinet_environment import QurinetEnvironment
from bandits.bg_cts import BGCTS
from bandits.cts_b import CTSB
from bandits.comb_ucb import CombUCB
from bandits.cts_g import CTSG
from bandits.cl_sg import CLSG

# Set up plotting style
matplotlib.rcParams.update({
    'figure.figsize': (4, 3),
    'font.size': 20,
    'axes.labelsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'text.usetex': True,
    'lines.linewidth': 1.5,
    'axes.grid': True
})

def test_algorithms_10000_rounds():
    """Test all algorithms for 10000 rounds and track regret."""
    
    print("Testing algorithms for 10000 rounds in Qurinet environment")
    print("=" * 60)
    
    # Create environment
    env = QurinetEnvironment(
        data_dir="data/qurinet",
        date="2-May",
        num_rounds=10000,
        pre_generate_availability=True,
        pre_generate_rewards=True
    )
    
    # Set availability rate to 1.0 (all links available)
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
    
    print(f"Environment: {env.num_arms} arms, {env.num_rounds} rounds")
    print(f"Network: {env.graph.number_of_nodes()} nodes, {env.graph.number_of_edges()} edges")
    
    # Track cumulative regret for each algorithm
    cumulative_regret = {alg_name: [] for alg_name in algorithms.keys()}
    current_regret = {alg_name: 0.0 for alg_name in algorithms.keys()}
    
    # Track selections and rewards for analysis
    selection_history = {alg_name: [] for alg_name in algorithms.keys()}
    reward_history = {alg_name: [] for alg_name in algorithms.keys()}
    
    # Progress tracking
    progress_interval = 1000
    
    for round_idx in range(env.num_rounds):
        # Get available arms and feasible combinations
        available_arms = env.get_available_arms_for_round(round_idx)
        feasible_combinations = env.get_feasible_combinations(available_arms)
        
        # Get optimal combination and its reward
        optimal_combination = env.get_optimal_combination(available_arms)
        optimal_reward = sum(env.arm_means[arm_id] for arm_id in optimal_combination) if optimal_combination else 0
        
        # Test each algorithm
        for alg_name, alg in algorithms.items():
            try:
                # Get algorithm's selection
                selection = alg.select_combination(round_idx)
                
                # Calculate selected reward
                selected_reward = sum(env.arm_means[arm_id] for arm_id in selection) if selection else 0
                
                # Calculate regret for this round
                round_regret = optimal_reward - selected_reward
                current_regret[alg_name] += round_regret
                
                # Store data
                selection_history[alg_name].append(selection)
                reward_history[alg_name].append(selected_reward)
                cumulative_regret[alg_name].append(current_regret[alg_name])
                
                # Update algorithm with rewards
                rewards = env.get_reward_for_round(selection, round_idx)
                alg.update_posterior(selection, rewards, round_idx)
                
            except Exception as e:
                print(f"Error in {alg_name} at round {round_idx}: {e}")
                # Continue with other algorithms
        
        # Progress report
        if (round_idx + 1) % progress_interval == 0:
            print(f"Completed {round_idx + 1} rounds")
            for alg_name in algorithms.keys():
                if cumulative_regret[alg_name]:
                    avg_reward = np.mean(reward_history[alg_name][-progress_interval:])
                    print(f"  {alg_name}: avg reward = {avg_reward:.3f}, cumulative regret = {current_regret[alg_name]:.3f}")
    
    # Final analysis
    print(f"\n{'='*20} Final Analysis {'='*20}")
    for alg_name in algorithms.keys():
        if cumulative_regret[alg_name]:
            final_regret = cumulative_regret[alg_name][-1]
            avg_reward = np.mean(reward_history[alg_name])
            optimal_count = sum(1 for sel in selection_history[alg_name] 
                              if sel == optimal_combination)
            print(f"{alg_name}:")
            print(f"  Final cumulative regret: {final_regret:.3f}")
            print(f"  Average reward: {avg_reward:.3f}")
            print(f"  Optimal selections: {optimal_count}/{len(selection_history[alg_name])} ({optimal_count/len(selection_history[alg_name])*100:.1f}%)")
    
    return cumulative_regret, algorithms.keys()

def plot_regret_curves(cumulative_regret, algorithm_names):
    """Plot cumulative regret curves for all algorithms."""
    
    print("\nPlotting regret curves...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(4, 3))
    
    # Colors for different algorithms
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    
    # Plot each algorithm's regret curve
    for i, alg_name in enumerate(algorithm_names):
        if cumulative_regret[alg_name]:
            # Sample points for plotting (every 100 rounds to avoid too many points)
            x = np.arange(0, len(cumulative_regret[alg_name]), 100)
            y = [cumulative_regret[alg_name][j] for j in x]
            
            # Plot with marker every T/10 points
            markevery = max(1, len(x) // 10)
            ax.plot(x, y, color=colors[i], marker=markers[i], markevery=markevery, 
                   linewidth=1.5, markersize=4, label=alg_name)
    
    # Customize plot
    ax.set_xlabel('t')
    ax.set_ylabel('Regret')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Use ScalarFormatter with useMathText=True
    from matplotlib.ticker import ScalarFormatter
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    
    plt.tight_layout()
    
    # Save plot
    output_dir = "output/images"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "qurinet_10000_rounds_regret.pdf")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Regret plot saved to: {plot_path}")
    
    plt.show()

def analyze_algorithm_performance(cumulative_regret, algorithm_names):
    """Analyze algorithm performance in detail."""
    
    print(f"\n{'='*20} Detailed Performance Analysis {'='*20}")
    
    # Analyze different time periods
    periods = [
        (0, 1000, "Early (1-1000)"),
        (1000, 5000, "Middle (1001-5000)"),
        (5000, 10000, "Late (5001-10000)")
    ]
    
    for start, end, period_name in periods:
        print(f"\n{period_name} period:")
        print("-" * 40)
        
        for alg_name in algorithm_names:
            if cumulative_regret[alg_name] and len(cumulative_regret[alg_name]) > end:
                # Calculate regret growth rate in this period
                start_regret = cumulative_regret[alg_name][start]
                end_regret = cumulative_regret[alg_name][end]
                regret_growth = end_regret - start_regret
                rounds = end - start
                growth_rate = regret_growth / rounds if rounds > 0 else 0
                
                print(f"  {alg_name}:")
                print(f"    Regret growth: {regret_growth:.3f}")
                print(f"    Growth rate per round: {growth_rate:.6f}")
                print(f"    Final regret in period: {end_regret:.3f}")

if __name__ == "__main__":
    # Run the test
    cumulative_regret, algorithm_names = test_algorithms_10000_rounds()
    
    # Plot the results
    plot_regret_curves(cumulative_regret, algorithm_names)
    
    # Analyze performance
    analyze_algorithm_performance(cumulative_regret, algorithm_names) 