#!/usr/bin/env python3
"""
Check UCSB regret data and print statistics for each algorithm.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Any

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment
from src.bandits.cts_b import CTSB
from src.bandits.comb_ucb import CombUCB
from src.bandits.cts_g import CTSG
from src.bandits.cl_sg import CLSG
from src.bandits.bg_cts import BGCTS

def setup_ucsb_meshnet_environment(routes_per_minute: int = 4, pre_generate_rewards: bool = True):
    """Setup the UCSB MeshNet environment with memory-mapped data."""
    data_dir = os.path.join(project_root, "data", "ucsb", "1143927049-1143953729")
    neighbortable_files = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.startswith('neighbortable-')
    ]
    neighbortable_files = sorted(neighbortable_files, key=lambda x: int(x.split('-')[-1]))
    
    env = UCSBMeshnetMemmapEnvironment(
        neighbortable_files=neighbortable_files,
        routes_per_minute=routes_per_minute,
        pre_generate_rewards=pre_generate_rewards,
        use_persistent_files=False,
        seed=42
    )
    return env

def run_ucsb_with_regret_analysis(num_runs: int = 3, gamma_values: List[float] = [0.01]):
    """Run UCSB simulation and analyze regret data."""
    print("UCSB MeshNet Regret Analysis")
    print("=" * 50)
    
    # Setup environment
    env = setup_ucsb_meshnet_environment(routes_per_minute=2)
    # Use connected node pair for testing
    env.source = "10.1.1.109"
    env.destination = "10.1.1.5"
    print(f"Environment: {env.num_rounds} rounds, {len(env.get_nodes())} nodes")
    print(f"Source: {env.get_source_destination()[0]}")
    print(f"Destination: {env.get_source_destination()[1]}")
    
    # Setup algorithms
    base_algorithms = ["CTSB", "CombUCB", "BG-CTS"]
    all_algorithms = base_algorithms + [f"CTS-G_gamma_{g}" for g in gamma_values] + [f"CL-SG_gamma_{g}" for g in gamma_values]
    
    num_rounds = env.num_rounds
    results = {alg: np.zeros((num_runs, num_rounds)) for alg in all_algorithms}
    rewards = {alg: np.zeros((num_runs, num_rounds)) for alg in all_algorithms}
    
    nodes = env.get_nodes()
    source, destination = env.get_source_destination()
    
    print(f"\nRunning {num_runs} simulations...")
    
    for run in range(num_runs):
        print(f"\n--- Run {run + 1} ---")
        
        # Setup algorithms for this run
        alg_instances = {
            "CTSB": CTSB(environment=env, rng=np.random.default_rng(run*100+1)),
            "CombUCB": CombUCB(environment=env, rng=np.random.default_rng(run*100+2)),
            "BG-CTS": BGCTS(environment=env, rnd_generator=np.random.default_rng(run*100+3)),
        }
        for gamma in gamma_values:
            alg_instances[f"CTS-G_gamma_{gamma}"] = CTSG(environment=env, rng=np.random.default_rng(run*100+10+int(gamma*100)), gamma=gamma)
            alg_instances[f"CL-SG_gamma_{gamma}"] = CLSG(environment=env, rng=np.random.default_rng(run*100+20+int(gamma*100)), gamma=gamma)
        
        # Calculate optimal rewards for this run
        optimal_rewards = np.zeros(num_rounds)
        for round_idx in range(num_rounds):
            adj = env.get_available_links_for_round(round_idx)
            G = nx.Graph()
            for i, u in enumerate(nodes):
                for j, v in enumerate(nodes):
                    if adj[i, j]:
                        G.add_edge(u, v, weight=1.0)
            try:
                path = nx.shortest_path(G, source=source, target=destination)
                opt_reward = 0.0
                for k in range(len(path)-1):
                    opt_reward += env.get_reward_for_link(path[k], path[k+1], round_idx)
                optimal_rewards[round_idx] = opt_reward
            except nx.NetworkXNoPath:
                optimal_rewards[round_idx] = 0.0
        
        # Run algorithms
        for alg_name, alg in alg_instances.items():
            print(f"  {alg_name}: ", end="")
            for round_idx in range(num_rounds):
                adj = env.get_available_links_for_round(round_idx)
                G = nx.Graph()
                for i, u in enumerate(nodes):
                    for j, v in enumerate(nodes):
                        if adj[i, j]:
                            G.add_edge(u, v, weight=1.0)
                try:
                    selected_path = alg.select_combination(round_idx)
                    if not selected_path or len(selected_path) < 2:
                        total_reward = 0.0
                    else:
                        total_reward = 0.0
                        for k in range(len(selected_path)-1):
                            total_reward += env.get_reward_for_link(selected_path[k], selected_path[k+1], round_idx)
                    alg.update_posterior(selected_path, {i: env.get_reward_for_link(selected_path[i], selected_path[i+1], round_idx) for i in range(len(selected_path)-1)} if selected_path and len(selected_path) > 1 else {}, round_idx)
                except Exception:
                    total_reward = 0.0
                
                rewards[alg_name][run, round_idx] = total_reward
                results[alg_name][run, round_idx] = optimal_rewards[round_idx] - total_reward
                
                # Print some sample regret values
                if round_idx in [0, 100, 200, 300, 400, 500, 600, 700, 800, 889]:
                    print(f"R{round_idx}:{results[alg_name][run, round_idx]:.3f} ", end="")
            
            print()  # New line for next algorithm
    
    # Analyze results
    print(f"\n=== Regret Analysis ===")
    for alg_name in all_algorithms:
        alg_regrets = results[alg_name]
        avg_regrets = np.mean(alg_regrets, axis=0)
        final_regret = avg_regrets[-1]
        cumulative_regret = np.sum(avg_regrets)
        
        print(f"{alg_name}:")
        print(f"  Final regret: {final_regret:.3f}")
        print(f"  Cumulative regret: {cumulative_regret:.3f}")
        print(f"  Mean regret per round: {np.mean(avg_regrets):.3f}")
        print(f"  Std regret per round: {np.std(avg_regrets):.3f}")
        print()
    
    return {
        "regrets": results,
        "rewards": rewards,
        "num_rounds": num_rounds,
        "num_runs": num_runs,
        "gamma_values": gamma_values,
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Check UCSB regret data")
    parser.add_argument('--runs', type=int, default=2, help='Number of runs')
    parser.add_argument('--gamma', type=float, nargs='+', default=[0.01], help='Gamma values')
    args = parser.parse_args()
    
    results = run_ucsb_with_regret_analysis(num_runs=args.runs, gamma_values=args.gamma)

if __name__ == "__main__":
    main() 