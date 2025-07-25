# %%

#!/usr/bin/env python3
"""
Example: UCSB Mesh Network Comprehensive Algorithm Comparison
============================================================

This example demonstrates comprehensive algorithm comparison using the UCSB mesh network
environment with real network trace data. It ensures all algorithms run on the same
environment by creating a single environment instance and using NumPy generators with
spawn function for reproducible results.

Features:
- Single environment instance shared across all algorithms
- NumPy generator with spawn function for reproducible results
- All algorithm comparison (CTSB, CombUCB, CTS-G, CL-SG, BG-CTS)
- Multiple gamma values (0.01, 0.1, 0.5, 1.0) for gamma algorithms
- Customizable number of rounds and runs
- Three plots: algorithm comparison, CTS-G gamma comparison, CL-SG gamma comparison
- Memory-mapped data storage for efficient large-scale simulations

Usage:
    python example_ucsb_comprehensive.py --rounds 10000 --runs 5 --default-gamma 0.1
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from scipy import stats
import tempfile
import glob

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.bandits.cts_b import CTSB
from src.bandits.comb_ucb import CombUCB
from src.bandits.cts_g import CTSG
from src.bandits.cl_sg import CLSG
from src.bandits.bg_cts import BGCTS
from src.environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment
from src.utils.plotting import plot_all_results


def setup_ucsb_environment(num_rounds: int = 10000, fixed_source: str = None, fixed_destination: str = None):
    """
    Setup the UCSB mesh network environment.
    
    Args:
        num_rounds: Number of rounds for the environment
        fixed_source: Fixed source node (if provided, overrides automatic selection)
        fixed_destination: Fixed destination node (if provided, overrides automatic selection)
        
    Returns:
        Environment configuration dictionary
    """
    # Find UCSB data files
    data_dir = os.path.join(project_root, 'data', 'ucsb')
    
    # Find the trace directory with the most neighbortable files
    trace_dirs = {}
    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.isdir(subdir_path):
            files = glob.glob(os.path.join(subdir_path, 'neighbortable-*'))
            trace_dirs[subdir] = len(files)
    
    if not trace_dirs:
        raise FileNotFoundError("No neighbortable files found in UCSB data directory")
    
    # Select the trace directory with the most files
    selected_trace_dir = max(trace_dirs, key=trace_dirs.get)
    selected_trace_path = os.path.join(data_dir, selected_trace_dir)
    
    print(f"Available trace directories:")
    for trace_dir, file_count in sorted(trace_dirs.items()):
        print(f"  {trace_dir}: {file_count} files")
    print(f"Selected trace directory: {selected_trace_dir} ({trace_dirs[selected_trace_dir]} files)")
    print()
    
    # Get all neighbortable files from the selected trace directory
    neighbortable_files = glob.glob(os.path.join(selected_trace_path, 'neighbortable-*'))
    neighbortable_files = sorted(neighbortable_files)
    
    if not neighbortable_files:
        raise FileNotFoundError(f"No neighbortable files found in {selected_trace_dir}")
    
    # Calculate routes_per_minute to achieve more rounds than requested
    # We want to generate 15000 rounds for the environment, but only use num_rounds for simulation
    target_rounds = 15000  # Generate more rounds than requested
    routes_per_minute = max(1, target_rounds // len(neighbortable_files))
    actual_rounds = len(neighbortable_files) * routes_per_minute
    
    print(f"Found {len(neighbortable_files)} neighbortable files in {selected_trace_dir}")
    print(f"Calculated routes_per_minute: {routes_per_minute}")
    print(f"Environment will have {actual_rounds} rounds (using first {num_rounds} for simulation)")
    print()
    
    # Find a valid source-destination pair
    if fixed_source and fixed_destination:
        print(f"Using fixed node pair: {fixed_source} -> {fixed_destination}")
        source, destination = fixed_source, fixed_destination
    else:
        # Find a valid source-destination pair by analyzing the first neighbortable file
        source, destination = find_valid_source_destination(neighbortable_files[0], neighbortable_files)
        print(f"Selected source: {source}")
        print(f"Selected destination: {destination}")
    
    # Create environment (will be created with proper seed in run_ucsb_simulation)
    env_config = {
        'neighbortable_files': neighbortable_files,
        'routes_per_minute': routes_per_minute,
        'source': source,
        'destination': destination,
        'pre_generate_rewards': True,
        'use_persistent_files': False,
        'max_feasible_combinations': 50,
        'max_path_length': 3,  # 3-hop limit (1-hop, 2-hop, 3-hop paths)
        'max_paths_per_algorithm': 25
    }
    
    return env_config


def find_valid_source_destination(neighbortable_file: str, all_neighbortable_files: list) -> tuple:
    """
    Find a valid source-destination pair that are in the same connected component.
    Also analyze path availability across all timestamps.
    
    Args:
        neighbortable_file: Path to the first neighbortable file
        all_neighbortable_files: List of all neighbortable files to analyze
        
    Returns:
        Tuple of (source, destination) node IDs
    """
    import networkx as nx
    
    print("Analyzing network topology and path availability...")
    
    # Parse the first neighbortable file
    nodes = set()
    edges = []
    
    with open(neighbortable_file, 'r') as fin:
        for line in fin:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            src = parts[0]
            nodes.add(src)
            
            for i in range(1, len(parts), 2):
                dst = parts[i]
                try:
                    ett = float(parts[i+1])
                    if ett < 1000:  # Valid ETT
                        nodes.add(dst)
                        edges.append((src, dst))
                except Exception:
                    continue
    
    # Create NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    print(f"Network topology from first file:")
    print(f"  Total nodes: {len(nodes)}")
    print(f"  Total edges: {len(edges)}")
    
    # Find connected components
    components = list(nx.connected_components(G))
    print(f"  Connected components: {len(components)}")
    for i, component in enumerate(components):
        print(f"    Component {i+1}: {len(component)} nodes")
    
    # Find the largest connected component
    largest_cc = max(components, key=len)
    largest_cc_nodes = sorted(list(largest_cc))  # Ensure deterministic sorting
    
    if len(largest_cc_nodes) < 2:
        raise ValueError("No connected component with at least 2 nodes found")
    
    print(f"  Largest component: {len(largest_cc_nodes)} nodes")
    print(f"  Sample nodes: {largest_cc_nodes[:5]}")
    
    # Select source and destination from the largest component
    # Try to find nodes that are not too close (at least 2 hops apart)
    # and have moderate path diversity
    source = largest_cc_nodes[0]
    
    # Find destinations that are at least 2 hops away
    candidates = []
    for node in largest_cc_nodes[1:]:
        try:
            path_length = nx.shortest_path_length(G, source, node)
            if path_length >= 2:
                candidates.append((node, path_length))
        except nx.NetworkXNoPath:
            continue
    
    # Sort candidates by path length (prefer 2-hop paths)
    candidates.sort(key=lambda x: x[1])
    
    # Select destination (prefer 2-hop paths for moderate complexity)
    destination = None
    for node, length in candidates:
        destination = node
        break
    
    # If no distant node found, just use any other node
    if destination is None:
        destination = largest_cc_nodes[1]
    
    print(f"  Selected source: {source}")
    print(f"  Selected destination: {destination}")
    
    # Analyze path availability across all timestamps
    print(f"\nAnalyzing path availability across {len(all_neighbortable_files)} timestamps...")
    
    path_counts = []
    path_lengths = []
    sample_timestamps = []
    
    # Sample some timestamps for detailed analysis
    sample_indices = [0, len(all_neighbortable_files)//4, len(all_neighbortable_files)//2, 
                     3*len(all_neighbortable_files)//4, len(all_neighbortable_files)-1]
    
    for i, file_path in enumerate(all_neighbortable_files):
        # Build graph for this timestamp
        G_timestamp = nx.Graph()
        with open(file_path, 'r') as fin:
            for line in fin:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                
                src = parts[0]
                G_timestamp.add_node(src)
                
                for j in range(1, len(parts), 2):
                    if j + 1 < len(parts):
                        dst = parts[j]
                        try:
                            ett = float(parts[j+1])
                            if ett < 1000:  # Valid ETT
                                G_timestamp.add_edge(src, dst)
                        except ValueError:
                            continue
        
        # Check if both nodes exist and are connected
        if G_timestamp.has_node(source) and G_timestamp.has_node(destination):
            try:
                # Find all simple paths
                all_paths = list(nx.all_simple_paths(G_timestamp, source, destination, cutoff=3))
                path_counts.append(len(all_paths))
                
                if all_paths:
                    # Calculate path lengths
                    lengths = [len(path) - 1 for path in all_paths]  # Convert to hop count
                    path_lengths.extend(lengths)
                    
                    # Store sample for detailed analysis
                    if i in sample_indices:
                        sample_timestamps.append({
                            'index': i,
                            'file': os.path.basename(file_path),
                            'paths': len(all_paths),
                            'lengths': lengths
                        })
                else:
                    path_counts.append(0)
                    
            except nx.NetworkXNoPath:
                path_counts.append(0)
        else:
            path_counts.append(0)
    
    # Print summary statistics
    print(f"Path availability summary:")
    print(f"  Total timestamps analyzed: {len(path_counts)}")
    print(f"  Timestamps with paths: {sum(1 for count in path_counts if count > 0)}")
    print(f"  Timestamps without paths: {sum(1 for count in path_counts if count == 0)}")
    print(f"  Average paths per timestamp: {np.mean(path_counts):.1f}")
    print(f"  Min paths: {min(path_counts)}, Max paths: {max(path_counts)}")
    
    if path_lengths:
        print(f"  Total paths found: {len(path_lengths)}")
        print(f"  Average path length: {np.mean(path_lengths):.1f} hops")
        print(f"  Path length distribution: {dict(zip(*np.unique(path_lengths, return_counts=True)))}")
    
    # Print detailed analysis for sample timestamps
    print(f"\nDetailed analysis for sample timestamps:")
    for sample in sample_timestamps:
        print(f"  Timestamp {sample['index']} ({sample['file']}):")
        print(f"    Paths: {sample['paths']}")
        if sample['lengths']:
            print(f"    Path lengths: {sample['lengths']}")
        else:
            print(f"    No paths available")
    
    return source, destination


def run_ucsb_simulation(env_config: dict,
                       num_rounds: int = 10000,
                       num_runs: int = 5,
                       progress_level: str = "normal",
                       main_seed: int = 42,
                       gamma_values: List[float] = [0.01, 0.1, 0.5, 1.0]) -> Dict[str, Any]:
    """
    Run comprehensive UCSB simulation with all algorithms.
    
    Args:
        env: UCSB environment instance
        num_rounds: Number of rounds per simulation
        num_runs: Number of independent runs
        progress_level: Progress display level
        main_seed: Main random seed
        gamma_values: List of gamma values for gamma algorithms
        
    Returns:
        Dictionary containing all simulation results
    """
    print(f"Running UCSB simulation with {num_rounds} rounds, {num_runs} runs")
    print(f"Gamma values: {gamma_values}")
    
    # Create main generator for all randomness
    main_rng = np.random.default_rng(main_seed)
    
    # Create environment with proper seed from main generator
    env_rng = main_rng.spawn(1)[0]
    env = UCSBMeshnetMemmapEnvironment(
        rng=env_rng,
        **env_config
    )
    
    # Define algorithms
    base_algorithms = {
        'CTSB': CTSB,
        'CombUCB': CombUCB,
        'BG-CTS': BGCTS
    }
    
    gamma_algorithms = {
        'CTS-G': CTSG,
        'CL-SG': CLSG
    }
    
    # Setup memory-mapped files for each algorithm
    algorithm_files = {}
    results = {
        'num_rounds': num_rounds,
        'num_runs': num_runs,
        'gamma_values': gamma_values,
        'confidence_level': 0.95,
        'algorithm_files': algorithm_files
    }
    
    # Create temporary files for each algorithm
    for alg_name in list(base_algorithms.keys()) + [f"{alg.lower()}_gamma_{gamma}" for alg in gamma_algorithms.keys() for gamma in gamma_values]:
        regrets_file = tempfile.NamedTemporaryFile(delete=False, suffix='_regrets.dat').name
        rewards_file = tempfile.NamedTemporaryFile(delete=False, suffix='_rewards.dat').name
        algorithm_files[alg_name] = (regrets_file, rewards_file)
        
        # Create memory-mapped arrays
        regrets_memmap = np.memmap(regrets_file, dtype=float, mode='w+', shape=(num_runs, num_rounds))
        rewards_memmap = np.memmap(rewards_file, dtype=float, mode='w+', shape=(num_runs, num_rounds))
        
        # Initialize arrays
        regrets_memmap[:] = 0.0
        rewards_memmap[:] = 0.0
        
        # Flush to disk
        regrets_memmap.flush()
        rewards_memmap.flush()
        del regrets_memmap, rewards_memmap
    
    # Run simulations with progress bars
    print("Starting simulation runs...")
    
    # Create progress bar for runs
    run_pbar = tqdm(range(num_runs), desc="Runs", position=0, leave=True)
    
    for run_idx in run_pbar:
        run_pbar.set_description(f"Run {run_idx + 1}/{num_runs}")
        
        # Create run-specific generator using spawn
        run_rng = main_rng.spawn(1)[0]
        
        # Run base algorithms
        for alg_name, alg_class in base_algorithms.items():
            regrets_file, rewards_file = algorithm_files[alg_name]
            regrets_memmap = np.memmap(regrets_file, dtype=float, mode='r+', shape=(num_runs, num_rounds))
            rewards_memmap = np.memmap(rewards_file, dtype=float, mode='r+', shape=(num_runs, num_rounds))
            
            # Create algorithm instance with environment and run-specific generator
            if alg_name == 'BG-CTS':
                alg = alg_class(environment=env, rnd_generator=run_rng.spawn(1)[0])
            else:
                alg = alg_class(environment=env, rng=run_rng.spawn(1)[0])
            
            cumulative_regret = 0.0
            
            # Create progress bar for rounds
            round_pbar = tqdm(range(num_rounds), desc=f"  {alg_name}", position=1, leave=False)
            
            for round_idx in round_pbar:
                # Get available arms for this round
                available_arms = env.get_available_arms_for_round(round_idx)
                
                # Get feasible combinations
                feasible_combinations = env.get_feasible_combinations(available_arms)
                
                if not feasible_combinations:
                    # No feasible paths available - regret is 0 for this round
                    # Update progress bar and continue with regret tracking
                    round_pbar.update(1)
                    regrets_memmap[run_idx, round_idx] = cumulative_regret
                    rewards_memmap[run_idx, round_idx] = 0.0
                    continue
                
                # Select action
                selected_combination = alg.select_combination(round_idx)
                
                # Calculate total reward for the path using the same method as optimal calculation
                total_reward = env.get_expected_reward_for_path(selected_combination, round_idx)
                
                # Get optimal reward for regret calculation
                optimal_reward = env.get_optimal_path_expected_reward(round_idx)
                
                # Update regret
                regret = optimal_reward - total_reward
                cumulative_regret += regret
                
                # Store results
                regrets_memmap[run_idx, round_idx] = cumulative_regret
                rewards_memmap[run_idx, round_idx] = total_reward
                
                # Update algorithm with arm-based rewards (still use individual rewards for algorithm update)
                reward_dict = {}
                for arm in selected_combination:
                    reward_dict[arm] = env.get_reward_for_round(arm, round_idx)
                alg.update_posterior(selected_combination, reward_dict, round_idx)
            
            # Flush to disk
            regrets_memmap.flush()
            rewards_memmap.flush()
            del regrets_memmap, rewards_memmap
        
        # Run gamma algorithms
        for alg_name, alg_class in gamma_algorithms.items():
            for gamma in gamma_values:
                alg_key = f"{alg_name.lower()}_gamma_{gamma}"
                regrets_file, rewards_file = algorithm_files[alg_key]
                regrets_memmap = np.memmap(regrets_file, dtype=float, mode='r+', shape=(num_runs, num_rounds))
                rewards_memmap = np.memmap(rewards_file, dtype=float, mode='r+', shape=(num_runs, num_rounds))
                
                # Create algorithm instance with environment, gamma parameter and run-specific generator
                alg = alg_class(environment=env, gamma=gamma, rng=run_rng.spawn(1)[0])
                
                cumulative_regret = 0.0
                
                # Create progress bar for gamma algorithm rounds
                gamma_round_pbar = tqdm(range(num_rounds), desc=f"  {alg_name}(γ={gamma})", position=1, leave=False)
                
                for round_idx in gamma_round_pbar:
                    # Get available arms for this round
                    available_arms = env.get_available_arms_for_round(round_idx)
                    
                    # Get feasible combinations
                    feasible_combinations = env.get_feasible_combinations(available_arms)
                    
                    if not feasible_combinations:
                        # No feasible paths available - regret is 0 for this round
                        # Update progress bar and continue with regret tracking
                        gamma_round_pbar.update(1)
                        regrets_memmap[run_idx, round_idx] = cumulative_regret
                        rewards_memmap[run_idx, round_idx] = 0.0
                        continue
                    
                    # Select action
                    selected_combination = alg.select_combination(round_idx)
                    
                    # Calculate total reward for the path using the same method as optimal calculation
                    total_reward = env.get_expected_reward_for_path(selected_combination, round_idx)
                    
                    # Get optimal reward for regret calculation
                    optimal_reward = env.get_optimal_path_expected_reward(round_idx)
                    
                    # Update regret
                    regret = optimal_reward - total_reward
                    cumulative_regret += regret
                    
                    # Store results
                    regrets_memmap[run_idx, round_idx] = cumulative_regret
                    rewards_memmap[run_idx, round_idx] = total_reward
                    
                    # Update algorithm with arm-based rewards (still use individual rewards for algorithm update)
                    reward_dict = {}
                    for arm in selected_combination:
                        reward_dict[arm] = env.get_reward_for_round(arm, round_idx)
                    alg.update_posterior(selected_combination, reward_dict, round_idx)
                
                # Flush to disk
                regrets_memmap.flush()
                rewards_memmap.flush()
                del regrets_memmap, rewards_memmap
    
    # Process results
    for alg_name in algorithm_files.keys():
        regrets_file, rewards_file = algorithm_files[alg_name]
        regrets_memmap = np.memmap(regrets_file, dtype=float, mode='r', shape=(num_runs, num_rounds))
        rewards_memmap = np.memmap(rewards_file, dtype=float, mode='r', shape=(num_runs, num_rounds))
        
        # Calculate statistics
        avg_cumulative_regrets = np.mean(regrets_memmap, axis=0)
        std_cumulative_regrets = np.std(regrets_memmap, axis=0)
        
        # Calculate confidence intervals
        confidence_interval = stats.t.interval(
            results['confidence_level'],
            num_runs - 1,
            loc=avg_cumulative_regrets,
            scale=std_cumulative_regrets / np.sqrt(num_runs)
        )
        
        results[alg_name] = {
            'avg_cumulative_regrets': avg_cumulative_regrets,
            'std_cumulative_regrets': std_cumulative_regrets,
            'confidence_interval': confidence_interval,
            'final_regret': avg_cumulative_regrets[-1],
            'final_regret_std': std_cumulative_regrets[-1],
            'final_regret_ci': (confidence_interval[0][-1], confidence_interval[1][-1])
        }
        
        del regrets_memmap, rewards_memmap
    
    # Clean up environment
    env.cleanup()
    
    return results


def analyze_ucsb_results(results: Dict[str, Any], default_gamma: float = 0.1):
    """
    Analyze UCSB simulation results.
    
    Args:
        results: Results from run_ucsb_simulation
        default_gamma: Default gamma value for comparison
    """
    print("\nUCSB Mesh Network Simulation Results:")
    print("=" * 80)
    
    # Get base algorithms and gamma values
    base_algorithms = ["CTSB", "CombUCB", "BG-CTS"]
    gamma_values = results.get('gamma_values', [0.01, 0.1, 0.5, 1.0])
    gamma_algorithms = ["CTS-G", "CL-SG"]
    
    num_runs = results['num_runs']
    confidence_level = results['confidence_level']
    
    # Print results for base algorithms
    print("\nBase Algorithms:")
    print("-" * 40)
    for alg in base_algorithms:
        if alg in results:
            alg_data = results[alg]
            final_regret = alg_data['final_regret']
            final_regret_std = alg_data['final_regret_std']
            final_regret_ci = alg_data['final_regret_ci']
            
            print(f"{alg}:")
            print(f"  Final regret: {final_regret:.2f} ± {final_regret_std:.2f}")
            print(f"  {confidence_level*100:.0f}% CI: [{final_regret_ci[0]:.2f}, {final_regret_ci[1]:.2f}]")
    
    # Print results for gamma algorithms (using default gamma)
    print(f"\nGamma Algorithms (γ={default_gamma}):")
    print("-" * 40)
    for alg in gamma_algorithms:
        alg_key = f"{alg.lower()}_gamma_{default_gamma}"
        if alg_key in results:
            alg_data = results[alg_key]
            final_regret = alg_data['final_regret']
            final_regret_std = alg_data['final_regret_std']
            final_regret_ci = alg_data['final_regret_ci']
            
            print(f"{alg} (γ={default_gamma}):")
            print(f"  Final regret: {final_regret:.2f} ± {final_regret_std:.2f}")
            print(f"  {confidence_level*100:.0f}% CI: [{final_regret_ci[0]:.2f}, {final_regret_ci[1]:.2f}]")
    
    # Find best algorithm among base algorithms
    if len(base_algorithms) > 1:
        print(f"\nAlgorithm Comparison:")
        print("-" * 40)
        
        base_regrets = {}
        for alg in base_algorithms:
            if alg in results:
                base_regrets[alg] = results[alg]['final_regret']
        
        if base_regrets:
            best_alg = min(base_regrets, key=base_regrets.get)
            best_regret = base_regrets[best_alg]
            
            for alg in base_regrets:
                if alg != best_alg:
                    regret = base_regrets[alg]
                    improvement = (regret - best_regret) / regret * 100
                    print(f"{alg} vs {best_alg}: {improvement:.1f}% improvement")


def cleanup_temp_files(results: Dict[str, Any], keep_files: bool = False):
    """
    Clean up temporary memory-mapped files.
    
    Args:
        results: Results from run_ucsb_simulation
        keep_files: Whether to keep the memory-mapped files for later analysis
    """
    if keep_files:
        print("Memory-mapped files preserved for later analysis")
        return
        
    algorithm_files = results.get('algorithm_files', {})
    for alg, (regrets_file, rewards_file) in algorithm_files.items():
        try:
            os.remove(regrets_file)
            os.remove(rewards_file)
        except OSError:
            pass  # File might not exist or already be removed


def main(num_rounds=10000, num_runs=5, keep_memmap_files=False, default_gamma=0.1, fixed_source=None, fixed_destination=None):
    """
    Main function to run the UCSB comprehensive example.
    
    Args:
        num_rounds: Number of rounds per simulation
        num_runs: Number of independent runs
        keep_memmap_files: Whether to keep memory-mapped files for later analysis
        default_gamma: Default gamma value for the first comparison plot
        fixed_source: Fixed source node for reproducibility
        fixed_destination: Fixed destination node for reproducibility
    """
    print("UCSB Mesh Network Comprehensive Algorithm Comparison")
    print("=" * 80)
    
    # Get environment configuration
    env_config = setup_ucsb_environment(num_rounds, fixed_source=fixed_source, fixed_destination=fixed_destination)
    
    # Run simulation (environment will be created inside with proper seed management)
    print(f"\nRunning UCSB simulation...")
    gamma_values = [0.01, 0.1, 0.5, 1.0]
    
    results = run_ucsb_simulation(
        env_config=env_config,
        num_rounds=num_rounds,
        num_runs=num_runs,
        progress_level="normal",
        main_seed=15,
        gamma_values=gamma_values
    )
    
    # Analyze results
    analyze_ucsb_results(results, default_gamma)
    
    # Plot results using utils plotting functions
    print(f"\nGenerating plots...")
    output_dir = os.path.join(project_root, 'output', 'images')
    os.makedirs(output_dir, exist_ok=True)
    
    plot_all_results(
        results=results,
        output_dir=output_dir,
        base_algorithms=['CTSB', 'CombUCB', 'BG-CTS'],
        gamma_algorithms=['CTS-G', 'CL-SG'],
        gamma_values=gamma_values,
        file_prefix="ucsb_comprehensive",
        default_gamma=default_gamma
    )
    
    # Clean up temporary files
    cleanup_temp_files(results, keep_files=keep_memmap_files)
    
    # Clean up environment
    # The environment is managed by the memory-mapped files, so no explicit cleanup here
    # unless the environment class itself has a cleanup method.
    # For now, we assume the memory-mapped files handle the environment state.
    
    print("\nUCSB comprehensive example completed!")
    print(f"Plots saved to: {output_dir}")


# %%
# Run the simulation with default parameters
if __name__ == "__main__":
    # For command line usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Run UCSB comprehensive algorithm comparison.")
    parser.add_argument('--rounds', type=int, default=10000,
                       help='Number of rounds per simulation (default: 10000)')
    parser.add_argument('--runs', type=int, default=5,
                       help='Number of independent runs (default: 5)')
    parser.add_argument('--keep-memmap', action='store_true',
                       help='Keep memory-mapped files for later analysis')
    parser.add_argument('--default-gamma', type=float, default=0.1,
                       help='Default gamma value for the first comparison plot (default: 0.1)')
    parser.add_argument('--fixed-source', type=str, default=None,
                       help='Fixed source node for reproducibility (e.g., "10.2.1.5")')
    parser.add_argument('--fixed-destination', type=str, default=None,
                       help='Fixed destination node for reproducibility (e.g., "10.2.1.27")')
    
    args, unknown = parser.parse_known_args()
    
    print(f"Configuration:")
    print(f"  Number of rounds: {args.rounds}")
    print(f"  Number of runs: {args.runs}")
    print(f"  Keep memory-mapped files: {args.keep_memmap}")
    print(f"  Default gamma for comparison: {args.default_gamma}")
    if args.fixed_source and args.fixed_destination:
        print(f"  Fixed node pair: {args.fixed_source} -> {args.fixed_destination}")
    else:
        print(f"  Node pair: Auto-selected")
    print()
    
    main(num_rounds=args.rounds, num_runs=args.runs, keep_memmap_files=args.keep_memmap, 
         default_gamma=args.default_gamma, fixed_source=args.fixed_source, fixed_destination=args.fixed_destination)

# For # %% execution, run with default parameters
print("Running UCSB comprehensive example with default parameters...")
print("Configuration:")
print("  Number of rounds: 10000")
print("  Number of runs: 5")
print("  Fixed node pair: 10.2.1.103 -> 10.2.1.109")
print("  Main seed: 15")
print()

main(num_rounds=10000, num_runs=5, keep_memmap_files=False, 
     default_gamma=0.1, fixed_source="10.2.1.103", fixed_destination="10.2.1.109") 