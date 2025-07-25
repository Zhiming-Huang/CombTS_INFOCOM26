#!/usr/bin/env python3
"""
Compare algorithm performance with different node pairs.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment
from bandits.cts_b import CTSB
from bandits.comb_ucb import CombUCB
from bandits.bg_cts import BGCTS

def compare_node_pairs():
    """Compare algorithm performance with different node pairs."""
    
    print("Comparing algorithm performance with different node pairs...")
    print("=" * 70)
    
    # Setup environment
    data_dir = Path("data/ucsb")
    
    # Find the trace directory with the most files
    trace_dirs = {}
    for subdir in data_dir.iterdir():
        if subdir.is_dir():
            files = list(subdir.glob("neighbortable-*"))
            trace_dirs[subdir.name] = len(files)
    
    selected_trace_dir = max(trace_dirs, key=trace_dirs.get)
    selected_trace_path = data_dir / selected_trace_dir
    
    # Use first 25 files for comparison
    neighbortable_files = sorted(list(selected_trace_path.glob("neighbortable-*")))[:25]
    neighbortable_files = [str(f) for f in neighbortable_files]
    
    print(f"Using trace directory: {selected_trace_dir}")
    print(f"Using {len(neighbortable_files)} files for comparison")
    print()
    
    # Test different node pairs
    node_pairs = [
        ("10.2.1.100", "10.2.1.20"),  # Current pair
        ("10.2.1.8", "10.2.1.100"),   # Previous pair
        ("10.2.1.4", "10.2.1.2"),     # Another pair
        ("10.2.1.103", "10.2.1.109"), # Known good pair
    ]
    
    results = {}
    
    for source, destination in node_pairs:
        print(f"Testing node pair: {source} -> {destination}")
        print("-" * 50)
        
        try:
            # Create environment
            env = UCSBMeshnetMemmapEnvironment(
                neighbortable_files=neighbortable_files,
                source=source,
                destination=destination,
                max_path_length=3,
                max_feasible_combinations=50
            )
            
            print(f"  Environment created with {env.num_rounds} rounds")
            
            # Analyze path availability
            total_paths = 0
            rounds_with_paths = 0
            
            for round_idx in range(min(50, env.num_rounds)):
                available_arms = env.get_available_arms_for_round(round_idx)
                feasible_combinations = env.get_feasible_combinations(available_arms)
                
                if feasible_combinations:
                    rounds_with_paths += 1
                    total_paths += len(feasible_combinations)
            
            print(f"  Rounds with paths: {rounds_with_paths}/50")
            print(f"  Average paths per round: {total_paths/rounds_with_paths:.1f}" if rounds_with_paths > 0 else "  No paths available")
            
            if rounds_with_paths == 0:
                print(f"  ❌ No feasible paths for this node pair")
                results[(source, destination)] = None
                continue
            
            # Test algorithms
            algorithms = {
                'CTSB': CTSB,
                'CombUCB': CombUCB,
                'BG-CTS': BGCTS
            }
            
            algorithm_results = {}
            
            for alg_name, alg_class in algorithms.items():
                print(f"  Testing {alg_name}...")
                
                # Create algorithm
                if alg_name == 'BG-CTS':
                    alg = alg_class(environment=env, rnd_generator=np.random.default_rng(42))
                else:
                    alg = alg_class(environment=env, rng=np.random.default_rng(42))
                
                # Run simulation
                cumulative_regret = 0.0
                for round_idx in range(min(100, env.num_rounds)):
                    available_arms = env.get_available_arms_for_round(round_idx)
                    feasible_combinations = env.get_feasible_combinations(available_arms)
                    
                    if not feasible_combinations:
                        continue
                    
                    # Select action
                    selected_combination = alg.select_combination(round_idx)
                    
                    # Calculate rewards
                    total_reward = env.get_expected_reward_for_path(selected_combination, round_idx)
                    optimal_reward = env.get_optimal_path_expected_reward(round_idx)
                    
                    # Calculate regret
                    regret = optimal_reward - total_reward
                    cumulative_regret += regret
                    
                    # Update algorithm
                    reward_dict = {}
                    for arm in selected_combination:
                        reward_dict[arm] = env.get_reward_for_round(arm, round_idx)
                    alg.update_posterior(selected_combination, reward_dict, round_idx)
                
                algorithm_results[alg_name] = cumulative_regret
                print(f"    Final regret: {cumulative_regret:.4f}")
            
            results[(source, destination)] = algorithm_results
            
            # Find best algorithm
            if algorithm_results:
                best_alg = min(algorithm_results, key=algorithm_results.get)
                best_regret = algorithm_results[best_alg]
                print(f"  Best algorithm: {best_alg} (regret: {best_regret:.4f})")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[(source, destination)] = None
        
        print()
    
    # Summary comparison
    print("Summary Comparison:")
    print("=" * 70)
    
    for (source, destination), result in results.items():
        print(f"\nNode pair: {source} -> {destination}")
        if result is None:
            print("  No results (no feasible paths or error)")
        else:
            for alg_name, regret in sorted(result.items(), key=lambda x: x[1]):
                print(f"  {alg_name}: {regret:.4f}")
    
    # Analyze why performance changed
    print(f"\nAnalysis:")
    print("=" * 70)
    
    # Find pairs with results
    valid_pairs = [(pair, result) for pair, result in results.items() if result is not None]
    
    if len(valid_pairs) >= 2:
        print("Algorithm ranking changes across node pairs:")
        
        for i, ((source1, dest1), result1) in enumerate(valid_pairs):
            for j, ((source2, dest2), result2) in enumerate(valid_pairs[i+1:], i+1):
                print(f"\nComparing {source1}->{dest1} vs {source2}->{dest2}:")
                
                # Sort algorithms by regret for each pair
                sorted1 = sorted(result1.items(), key=lambda x: x[1])
                sorted2 = sorted(result2.items(), key=lambda x: x[1])
                
                print(f"  {source1}->{dest1} ranking: {[alg for alg, _ in sorted1]}")
                print(f"  {source2}->{dest2} ranking: {[alg for alg, _ in sorted2]}")
                
                # Check if rankings changed
                if [alg for alg, _ in sorted1] != [alg for alg, _ in sorted2]:
                    print(f"  ⚠️  Ranking changed!")

if __name__ == "__main__":
    compare_node_pairs() 