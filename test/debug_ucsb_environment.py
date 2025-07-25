#!/usr/bin/env python3
"""
Debug script for UCSB MeshNet environment to identify performance bottlenecks.
"""

import sys
import os
import numpy as np
import time
from typing import List, Dict, Any
import networkx as nx

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
    print(f"Found {len(neighbortable_files)} neighbortable files")
    print(f"First few files: {neighbortable_files[:3]}")
    
    env = UCSBMeshnetMemmapEnvironment(
        neighbortable_files=neighbortable_files,
        routes_per_minute=routes_per_minute,
        pre_generate_rewards=pre_generate_rewards,
        use_persistent_files=False,
        seed=42
    )
    return env

def debug_single_round(env, round_idx, alg_instances):
    """Debug a single round execution."""
    print(f"\n=== Debugging Round {round_idx} ===")
    
    # Time environment operations
    start_time = time.time()
    adj = env.get_available_links_for_round(round_idx)
    env_time = time.time() - start_time
    print(f"  Environment get_available_links: {env_time:.4f}s")
    
    # Time graph construction
    start_time = time.time()
    nodes = env.get_nodes()
    G = nx.Graph()
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if adj[i, j]:
                G.add_edge(u, v, weight=1.0)
    graph_time = time.time() - start_time
    print(f"  Graph construction: {graph_time:.4f}s")
    
    # Time optimal path calculation
    start_time = time.time()
    source, destination = env.get_source_destination()
    try:
        path = nx.shortest_path(G, source=source, target=destination)
        opt_reward = 0.0
        for k in range(len(path)-1):
            opt_reward += env.get_reward_for_link(path[k], path[k+1], round_idx)
        optimal_rewards = opt_reward
    except nx.NetworkXNoPath:
        optimal_rewards = 0.0
    opt_time = time.time() - start_time
    print(f"  Optimal path calculation: {opt_time:.4f}s")
    
    # Time algorithm execution for each algorithm
    for alg_name, alg in alg_instances.items():
        print(f"  --- {alg_name} ---")
        
        # Time algorithm selection
        start_time = time.time()
        try:
            selected_path = alg.select_combination(round_idx)
            select_time = time.time() - start_time
            print(f"    select_combination: {select_time:.4f}s")
            
            # Time reward calculation
            start_time = time.time()
            if not selected_path or len(selected_path) < 2:
                total_reward = 0.0
            else:
                total_reward = 0.0
                for k in range(len(selected_path)-1):
                    total_reward += env.get_reward_for_link(selected_path[k], selected_path[k+1], round_idx)
            reward_time = time.time() - start_time
            print(f"    reward calculation: {reward_time:.4f}s")
            
            # Time posterior update
            start_time = time.time()
            reward_dict = {i: env.get_reward_for_link(selected_path[i], selected_path[i+1], round_idx) 
                          for i in range(len(selected_path)-1)} if selected_path and len(selected_path) > 1 else {}
            alg.update_posterior(selected_path, reward_dict, round_idx)
            update_time = time.time() - start_time
            print(f"    update_posterior: {update_time:.4f}s")
            
        except Exception as e:
            print(f"    Error in {alg_name}: {e}")
            continue
    
    total_round_time = env_time + graph_time + opt_time
    print(f"  Total round time: {total_round_time:.4f}s")

def debug_ucsb_simulation(num_rounds: int = 10, num_runs: int = 1, gamma_values: List[float] = [0.01]):
    """Debug UCSB simulation with detailed timing."""
    print("UCSB MeshNet Environment Debug")
    print("=" * 50)
    
    # Setup environment
    print("Setting up environment...")
    start_time = time.time()
    env = setup_ucsb_meshnet_environment(routes_per_minute=2)
    setup_time = time.time() - start_time
    print(f"Environment setup time: {setup_time:.4f}s")
    
    print(f"Environment info:")
    print(f"  Total rounds: {env.num_rounds}")
    print(f"  Number of nodes: {len(env.get_nodes())}")
    print(f"  Source: {env.get_source_destination()[0]}")
    print(f"  Destination: {env.get_source_destination()[1]}")
    
    # Setup algorithms
    print("\nSetting up algorithms...")
    start_time = time.time()
    alg_instances = {
        "CTSB": CTSB(environment=env, rng=np.random.default_rng(1)),
        "CombUCB": CombUCB(environment=env, rng=np.random.default_rng(2)),
        "BG-CTS": BGCTS(environment=env, rnd_generator=np.random.default_rng(3)),
    }
    for gamma in gamma_values:
        alg_instances[f"CTS-G_gamma_{gamma}"] = CTSG(environment=env, rng=np.random.default_rng(10+int(gamma*100)), gamma=gamma)
        alg_instances[f"CL-SG_gamma_{gamma}"] = CLSG(environment=env, rng=np.random.default_rng(20+int(gamma*100)), gamma=gamma)
    alg_setup_time = time.time() - start_time
    print(f"Algorithm setup time: {alg_setup_time:.4f}s")
    
    # Debug first few rounds
    print(f"\nDebugging first {num_rounds} rounds...")
    for round_idx in range(min(num_rounds, env.num_rounds)):
        debug_single_round(env, round_idx, alg_instances)
        
        # Add a small delay to see progress
        if round_idx < num_rounds - 1:
            time.sleep(0.1)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Debug UCSB MeshNet environment")
    parser.add_argument('--rounds', type=int, default=5, help='Number of rounds to debug')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--gamma', type=float, nargs='+', default=[0.01], help='Gamma values')
    args = parser.parse_args()
    
    debug_ucsb_simulation(num_rounds=args.rounds, num_runs=args.runs, gamma_values=args.gamma)

if __name__ == "__main__":
    main() 