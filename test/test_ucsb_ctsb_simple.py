#!/usr/bin/env python3
"""
Simple UCSB test with only CTS-B algorithm for 100 rounds.
"""

import sys
import os
import numpy as np
import networkx as nx
from typing import List, Dict, Any

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment
from src.bandits.cts_b import CTSB

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
    # Use connected node pair for testing
    env.source = "10.1.1.109"
    env.destination = "10.1.1.5"
    return env

def run_ctsb_test(num_rounds: int = 100):
    """Run CTS-B test on UCSB environment."""
    print("UCSB CTS-B Algorithm Test")
    print("=" * 40)
    
    # Setup environment
    env = setup_ucsb_meshnet_environment(routes_per_minute=2)
    print(f"Environment: {env.num_rounds} total rounds, {len(env.get_nodes())} nodes")
    print(f"Source: {env.get_source_destination()[0]}")
    print(f"Destination: {env.get_source_destination()[1]}")
    print(f"Testing for {num_rounds} rounds")
    print()
    
    # Setup CTS-B algorithm
    alg = CTSB(environment=env, rng=np.random.default_rng(42))
    
    # Arrays to store results
    regrets = np.zeros(num_rounds)
    rewards = np.zeros(num_rounds)
    optimal_rewards = np.zeros(num_rounds)
    
    nodes = env.get_nodes()
    source, destination = env.get_source_destination()
    
    print("Running CTS-B algorithm...")
    print("Round | Algorithm Reward | Optimal Reward | Regret | Path Length")
    print("-" * 60)
    
    for round_idx in range(num_rounds):
        # Calculate optimal reward for this round
        adj = env.get_available_links_for_round(round_idx)
        G = nx.Graph()
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if adj[i, j]:
                    G.add_edge(u, v, weight=1.0)
        
        try:
            opt_path = nx.shortest_path(G, source=source, target=destination)
            opt_reward = 0.0
            for k in range(len(opt_path)-1):
                opt_reward += env.get_reward_for_link(opt_path[k], opt_path[k+1], round_idx)
            optimal_rewards[round_idx] = opt_reward
        except nx.NetworkXNoPath:
            optimal_rewards[round_idx] = 0.0
        
        # Run CTS-B algorithm
        try:
            selected_arms = alg.select_combination(round_idx)
            if not selected_arms:
                alg_reward = 0.0
                path_length = 0
            else:
                # Convert selected arms back to path
                selected_edges = [env.arm_to_link[arm] for arm in selected_arms]
                # Build path from edges
                path_nodes = []
                edge_dict = {edge: True for edge in selected_edges}
                
                # Find path from source to destination using selected edges
                current = source
                path_nodes = [current]
                visited = {current}
                
                while current != destination and len(path_nodes) < 20:  # Prevent infinite loop
                    found_next = False
                    for edge in selected_edges:
                        if edge[0] == current and edge[1] not in visited:
                            current = edge[1]
                            path_nodes.append(current)
                            visited.add(current)
                            found_next = True
                            break
                        elif edge[1] == current and edge[0] not in visited:
                            current = edge[0]
                            path_nodes.append(current)
                            visited.add(current)
                            found_next = True
                            break
                    if not found_next:
                        break
                
                if current == destination and len(path_nodes) > 1:
                    alg_reward = 0.0
                    for k in range(len(path_nodes)-1):
                        alg_reward += env.get_reward_for_link(path_nodes[k], path_nodes[k+1], round_idx)
                    path_length = len(path_nodes) - 1
                else:
                    alg_reward = 0.0
                    path_length = 0
            
            # Update algorithm with arm-based rewards
            reward_dict = {}
            for arm in selected_arms:
                edge = env.arm_to_link[arm]
                reward_dict[arm] = env.get_reward_for_link(edge[0], edge[1], round_idx)
            alg.update_posterior(selected_arms, reward_dict, round_idx)
            
        except Exception as e:
            print(f"Error in round {round_idx}: {e}")
            alg_reward = 0.0
            path_length = 0
        
        rewards[round_idx] = alg_reward
        regrets[round_idx] = optimal_rewards[round_idx] - alg_reward
        
        # Print progress every 5 rounds for more detail
        if (round_idx + 1) % 5 == 0 or round_idx < 10:
            print(f"{round_idx+1:5d} | {alg_reward:14.3f} | {optimal_rewards[round_idx]:13.3f} | {regrets[round_idx]:6.3f} | {path_length:11d}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total rounds: {num_rounds}")
    print(f"Average algorithm reward: {np.mean(rewards):.3f}")
    print(f"Average optimal reward: {np.mean(optimal_rewards):.3f}")
    print(f"Average regret: {np.mean(regrets):.3f}")
    print(f"Final regret: {regrets[-1]:.3f}")
    print(f"Cumulative regret: {np.sum(regrets):.3f}")
    print(f"Regret standard deviation: {np.std(regrets):.3f}")
    
    # Check if algorithm found any valid paths
    valid_paths = np.sum(rewards > 0)
    print(f"Rounds with valid paths: {valid_paths}/{num_rounds} ({100*valid_paths/num_rounds:.1f}%)")
    
    return {
        "regrets": regrets,
        "rewards": rewards,
        "optimal_rewards": optimal_rewards,
        "num_rounds": num_rounds
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test CTS-B on UCSB environment")
    parser.add_argument('--rounds', type=int, default=100, help='Number of rounds to test')
    args = parser.parse_args()
    
    results = run_ctsb_test(num_rounds=args.rounds)

if __name__ == "__main__":
    main() 