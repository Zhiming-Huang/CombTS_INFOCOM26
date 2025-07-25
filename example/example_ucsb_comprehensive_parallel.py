# %%

#!/usr/bin/env python3
"""
Example: UCSB Mesh Network Comprehensive Algorithm Comparison (Parallel Version)
===============================================================================

This example demonstrates comprehensive algorithm comparison using the UCSB mesh network
environment with real network trace data. It runs algorithms in parallel for faster execution.

Features:
- Parallel execution of independent algorithms
- Single environment instance shared across all algorithms
- NumPy generator with spawn function for reproducible results
- All algorithm comparison (CTSB, CombUCB, CTS-G, CL-SG, BG-CTS)
- Multiple gamma values (0.01, 0.1, 0.5, 1.0) for gamma algorithms
- Customizable number of rounds and runs
- Three plots: algorithm comparison, CTS-G gamma comparison, CL-SG gamma comparison
- Memory-mapped data storage for efficient large-scale simulations

Usage:
    python example_ucsb_comprehensive_parallel.py --rounds 10000 --runs 5 --default-gamma 0.1
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from scipy import stats
import tempfile
import glob
import multiprocessing as mp
from functools import partial
import time
import pickle
import json
from pathlib import Path

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
from src.environments.readonly_environment_wrapper import ReadOnlyEnvironmentWrapper
from src.utils.plotting import plot_all_results


def setup_ucsb_environment(num_rounds: int = 10000, fixed_source: str = None, fixed_destination: str = None, trace_period: str = None, max_path_length: int = 3):
    """
    Setup the UCSB mesh network environment.
    
    Args:
        num_rounds: Number of rounds for the environment
        fixed_source: Fixed source node (if provided, overrides automatic selection)
        fixed_destination: Fixed destination node (if provided, overrides automatic selection)
        trace_period: Specific trace period to use (e.g., "1143927049-1143953729")
        max_path_length: Maximum path length in hops (default: 3)
        
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
    
    # Select trace directory
    if trace_period:
        if trace_period not in trace_dirs:
            available_periods = list(trace_dirs.keys())
            raise ValueError(f"Trace period '{trace_period}' not found. Available periods: {available_periods}")
        selected_trace_dir = trace_period
        print(f"Using specified trace period: {selected_trace_dir}")
    else:
        # Select the trace directory with the most files
        selected_trace_dir = max(trace_dirs, key=trace_dirs.get)
        print(f"Using trace period with most files: {selected_trace_dir}")
    
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
        'max_path_length': max_path_length,  # Configurable hop limit
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
    
    return source, destination


def run_single_algorithm(alg_config: Tuple[str, Any, Tuple, int, int, np.random.Generator, List[np.random.Generator]]) -> Tuple[str, np.ndarray, np.ndarray]:
    """
    Run a single algorithm in a separate process.
    
    Args:
        alg_config: Tuple containing (alg_name, alg_class, env_config, num_rounds, num_runs, env_generator, alg_generators)
        
    Returns:
        Tuple of (algorithm_name, regrets_array, rewards_array)
    """
    alg_name, alg_class, env_config, num_rounds, num_runs, env_generator, alg_generators = alg_config
    
    # Create a fresh environment instance for this algorithm to avoid shared state
    # Use the dedicated environment generator
    original_env = UCSBMeshnetMemmapEnvironment(rng=env_generator, **env_config)
    
    # Wrap the environment with read-only protection
    env = ReadOnlyEnvironmentWrapper(original_env)
    
    # Initialize arrays
    regrets_array = np.zeros((num_runs, num_rounds))
    rewards_array = np.zeros((num_runs, num_rounds))
    
    # Run the algorithm
    for run_idx in range(num_runs):
        # Create run-specific generator from the algorithm generator
        run_generator = alg_generators[run_idx]
        
        # Create algorithm instance with the run-specific generator directly
        if alg_name == 'BG-CTS':
            alg = alg_class(environment=env, rnd_generator=run_generator)
        elif '_gamma_' in alg_name:
            # Extract gamma value from algorithm name
            gamma_str = alg_name.split('_gamma_')[1]
            gamma = float(gamma_str)
            
            if alg_name.startswith('cl-sg'):
                # CL-SG needs gamma and optimistic_init parameters
                alg = alg_class(environment=env, rng=run_generator, gamma=gamma, optimistic_init=True)
            elif alg_name.startswith('cts-g'):
                # CTS-G needs gamma parameter
                alg = alg_class(environment=env, rng=run_generator, gamma=gamma)
            else:
                # Default case for other gamma algorithms
                alg = alg_class(environment=env, rng=run_generator, gamma=gamma)
        else:
            alg = alg_class(environment=env, rng=run_generator)
        
        cumulative_regret = 0.0
        cumulative_reward = 0.0
        
        # Run rounds
        for round_idx in range(num_rounds):
            # Get available arms for this round
            available_arms = env.get_available_arms_for_round(round_idx)
            feasible_combinations = env.get_feasible_combinations(available_arms)
            
            if feasible_combinations:
                # Select combination
                selected_combination = alg.select_combination(round_idx)
                
                if selected_combination:
                    # Calculate total reward for the selected combination
                    total_reward = env.get_expected_reward_for_path(selected_combination, round_idx)
                    
                    # Get optimal reward for comparison
                    optimal_reward = env.get_optimal_path_expected_reward(round_idx)
                    
                    # Update regret
                    regret = optimal_reward - total_reward
                    cumulative_regret += regret
                    cumulative_reward += total_reward
                    
                    # Update algorithm with individual arm rewards
                    reward_dict = {}
                    for arm in selected_combination:
                        reward_dict[arm] = env.get_reward_for_round(arm, round_idx)
                    alg.update_posterior(selected_combination, reward_dict, round_idx)
                else:
                    # No valid combination found, use dummy values
                    optimal_reward = env.get_optimal_path_expected_reward(round_idx) if hasattr(env, 'get_optimal_path_expected_reward') else 0
                    cumulative_regret += optimal_reward
            else:
                # No feasible combinations available
                optimal_reward = env.get_optimal_path_expected_reward(round_idx) if hasattr(env, 'get_optimal_path_expected_reward') else 0
                cumulative_regret += optimal_reward
            
            # Store cumulative values
            regrets_array[run_idx, round_idx] = cumulative_regret
            rewards_array[run_idx, round_idx] = cumulative_reward
    
    # Cleanup environment (cleanup the original environment, not the wrapper)
    original_env.cleanup()
    
    return alg_name, regrets_array, rewards_array


def run_ucsb_simulation_parallel(env_config: dict,
                                num_rounds: int = 10000,
                                num_runs: int = 5,
                                main_seed: int = 42,
                                gamma_values: List[float] = [0.01, 0.1, 0.5, 1.0],
                                n_jobs: int = None,
                                use_parallel: bool = True) -> Dict[str, Any]:
    """
    Run comprehensive UCSB simulation with all algorithms in parallel.
    
    Args:
        env_config: Environment configuration
        num_rounds: Number of rounds per simulation
        num_runs: Number of independent runs
        main_seed: Main random seed
        gamma_values: List of gamma values for gamma algorithms
        n_jobs: Number of parallel jobs (None for auto)
        use_parallel: Whether to use parallel processing (if False, runs sequentially for reproducibility)
        
    Returns:
        Dictionary containing all simulation results
    """
    print(f"Running UCSB simulation with {num_rounds} rounds, {num_runs} runs ({'Parallel' if use_parallel else 'Sequential'})")
    print(f"Gamma values: {gamma_values}")
    
    if use_parallel and n_jobs is None:
        n_jobs = min(mp.cpu_count(), 8)  # Limit to 8 cores to avoid memory issues
    
    if use_parallel:
        print(f"Using {n_jobs} parallel processes")
    else:
        print("Using sequential execution for reproducibility")
    
    # Define algorithms in deterministic order
    base_algorithms = [
        ('CTSB', CTSB),
        ('CombUCB', CombUCB),
        ('BG-CTS', BGCTS)
    ]
    
    gamma_algorithms = [
        ('CTS-G', CTSG),
        ('CL-SG', CLSG)
    ]
    
    # Calculate total number of algorithms
    total_algorithms = len(base_algorithms) + len(gamma_algorithms) * len(gamma_values)
    print(f"Total algorithms: {total_algorithms}")
    
    # Pre-generate all algorithm generators for reproducibility
    print("Pre-generating all algorithm generators...")
    main_rng = np.random.default_rng(main_seed)
    
    # Pre-generate generators: 1 for each algorithm's environment + num_runs for each algorithm's runs
    total_generators_needed = total_algorithms * (1 + num_runs)  # 1 for env + num_runs for runs
    all_generators = main_rng.spawn(total_generators_needed)
    
    # Prepare algorithm configurations with pre-generated generators
    alg_configs = []
    generator_idx = 0
    
    # Add base algorithms in deterministic order
    for alg_name, alg_class in base_algorithms:
        # Get one generator for environment creation
        env_generator = all_generators[generator_idx]
        generator_idx += 1
        
        # Get generators for all runs of this algorithm
        alg_generators = []
        for run_idx in range(num_runs):
            alg_generators.append(all_generators[generator_idx])
            generator_idx += 1
        alg_configs.append((alg_name, alg_class, env_config, num_rounds, num_runs, env_generator, alg_generators))
    
    # Add gamma algorithms in deterministic order
    for alg_name, alg_class in gamma_algorithms:
        for gamma in gamma_values:
            alg_key = f"{alg_name.lower()}_gamma_{gamma}"
            # Get one generator for environment creation
            env_generator = all_generators[generator_idx]
            generator_idx += 1
            
            # Get generators for all runs of this algorithm
            alg_generators = []
            for run_idx in range(num_runs):
                alg_generators.append(all_generators[generator_idx])
                generator_idx += 1
            alg_configs.append((alg_key, alg_class, env_config, num_rounds, num_runs, env_generator, alg_generators))
    
    print(f"Generated {len(alg_configs)} algorithm configurations")
    
    # Execute algorithms
    print(f"Starting execution of {len(alg_configs)} algorithms...")
    start_time = time.time()
    if use_parallel:
        # Parallel execution
        with mp.Pool(processes=n_jobs) as pool:
            results_list = list(tqdm(pool.imap(run_single_algorithm, alg_configs), 
                                   total=len(alg_configs), desc="Algorithms"))
    else:
        # Sequential execution
        results_list = []
        for config in tqdm(alg_configs, desc="Algorithms"):
            result = run_single_algorithm(config)
            results_list.append(result)
    
    execution_time = time.time() - start_time
    print(f"Execution completed in {execution_time:.2f} seconds")
    
    # Process results
    results = {
        'num_rounds': num_rounds,
        'num_runs': num_runs,
        'gamma_values': gamma_values,
        'confidence_level': 0.95
    }
    
    for alg_name, regrets_array, rewards_array in results_list:
        # Calculate statistics
        avg_cumulative_regrets = np.mean(regrets_array, axis=0)
        std_cumulative_regrets = np.std(regrets_array, axis=0)
        
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
    
    # No need to clean up shared environment since each algorithm creates its own
    
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


def save_simulation_results(results: Dict[str, Any], save_path: str, metadata: Dict[str, Any] = None):
    """
    Save simulation results to file with metadata.
    
    Args:
        results: Results dictionary from simulation
        save_path: Path to save the results
        metadata: Additional metadata to save with results
    """
    # Create directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for saving
    save_data = {
        'results': results,
        'metadata': metadata or {},
        'save_timestamp': time.time()
    }
    
    # Convert numpy arrays to lists for JSON serialization if saving as JSON
    if save_path.endswith('.json'):
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        serializable_results[key][subkey] = subvalue.tolist()
                    elif isinstance(subvalue, tuple) and len(subvalue) == 2 and isinstance(subvalue[0], np.ndarray):
                        # Handle confidence intervals
                        serializable_results[key][subkey] = (subvalue[0].tolist(), subvalue[1].tolist())
                    else:
                        serializable_results[key][subkey] = subvalue
            else:
                serializable_results[key] = value
        
        save_data['results'] = serializable_results
        
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
    else:
        # Save as pickle (preserves numpy arrays)
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
    
    print(f"Results saved to: {save_path}")


def load_simulation_results(load_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load simulation results from file.
    
    Args:
        load_path: Path to load the results from
        
    Returns:
        Tuple of (results, metadata)
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Results file not found: {load_path}")
    
    if load_path.endswith('.json'):
        with open(load_path, 'r') as f:
            save_data = json.load(f)
        
        # Convert lists back to numpy arrays
        results = save_data['results']
        for key, value in results.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, list):
                        results[key][subkey] = np.array(subvalue)
                    elif isinstance(subvalue, list) and len(subvalue) == 2 and isinstance(subvalue[0], list):
                        # Handle confidence intervals
                        results[key][subkey] = (np.array(subvalue[0]), np.array(subvalue[1]))
    else:
        # Load from pickle
        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)
        results = save_data['results']
    
    metadata = save_data.get('metadata', {})
    print(f"Results loaded from: {load_path}")
    return results, metadata


def generate_save_filename(metadata: Dict[str, Any]) -> str:
    """
    Generate a descriptive filename for saved results.
    
    Args:
        metadata: Metadata containing simulation parameters
        
    Returns:
        Generated filename
    """
    # Extract key parameters
    num_rounds = metadata.get('num_rounds', 'unknown')
    num_runs = metadata.get('num_runs', 'unknown')
    main_seed = metadata.get('main_seed', 'unknown')
    source = metadata.get('fixed_source', 'unknown').replace('.', '_')
    dest = metadata.get('fixed_destination', 'unknown').replace('.', '_')
    trace_period = metadata.get('trace_period', 'unknown')
    
    # Create filename
    filename = f"ucsb_results_r{num_rounds}_n{num_runs}_s{main_seed}_{source}_to_{dest}_{trace_period}"
    return filename


def plot_from_saved_data(load_path: str, output_dir: str = None):
    """
    Load saved results and generate plots without running simulation.
    
    Args:
        load_path: Path to saved results file
        output_dir: Directory to save plots (default: output/images)
    """
    # Load results
    results, metadata = load_simulation_results(load_path)
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(project_root, 'output', 'images')
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract parameters from metadata
    default_gamma = metadata.get('default_gamma', 0.1)
    gamma_values = metadata.get('gamma_values', [0.01, 0.1, 0.5, 1.0])
    
    # Analyze and print results
    analyze_ucsb_results(results, default_gamma)
    
    # Generate plots
    print(f"\nGenerating plots from saved data...")
    plot_all_results(
        results=results,
        output_dir=output_dir,
        base_algorithms=['CTSB', 'CombUCB', 'BG-CTS'],
        gamma_algorithms=['CTS-G', 'CL-SG'],
        gamma_values=gamma_values,
        file_prefix="ucsb_comprehensive_parallel",
        default_gamma=default_gamma
    )
    
    print(f"Plots generated from saved data!")
    print(f"Plots saved to: {output_dir}")


def main(num_rounds=10000, num_runs=5, default_gamma=0.1, fixed_source="10.1.1.102", fixed_destination="10.1.1.25", n_jobs=None, main_seed=42, trace_period="1143927049-1143953729", max_path_length=3, use_parallel=True, save_data=None, load_data=None, plot_only=False):
    """
    Main function to run the UCSB comprehensive example with parallel execution.
    
    Args:
        num_rounds: Number of rounds per simulation
        num_runs: Number of independent runs
        default_gamma: Default gamma value for the first comparison plot
        fixed_source: Fixed source node for reproducibility
        fixed_destination: Fixed destination node for reproducibility
        n_jobs: Number of parallel jobs
        main_seed: Main random seed for reproducibility
        trace_period: Specific trace period to use
        max_path_length: Maximum path length in hops
        use_parallel: Whether to use parallel processing
        save_data: Path to save simulation data (None to skip saving)
        load_data: Path to load saved simulation data (None to run new simulation)
        plot_only: If True with load_data, only generate plots without printing analysis
    """
    
    # Handle load data and plot only mode
    if load_data:
        print("=" * 80)
        print("Loading saved simulation data...")
        print("=" * 80)
        plot_from_saved_data(load_data)
        return
    
    print("UCSB Mesh Network Comprehensive Algorithm Comparison (Parallel)")
    print("=" * 80)
    print(f"Fixed node pair: {fixed_source} -> {fixed_destination}")
    print(f"Trace period: {trace_period}")
    print(f"Max path length: {max_path_length} hops")
    print(f"Execution mode: {'Parallel' if use_parallel else 'Sequential'}")
    print()
    
    # Get environment configuration
    env_config = setup_ucsb_environment(num_rounds, fixed_source=fixed_source, fixed_destination=fixed_destination, trace_period=trace_period, max_path_length=max_path_length)
    
    # Run simulation with parallel execution
    print(f"\nRunning UCSB simulation...")
    gamma_values = [0.01, 0.1, 0.5, 1.0]
    
    results = run_ucsb_simulation_parallel(
        env_config=env_config,
        num_rounds=num_rounds,
        num_runs=num_runs,
        main_seed=main_seed,
        gamma_values=gamma_values,
        n_jobs=n_jobs,
        use_parallel=use_parallel
    )
    
    # Save data if requested
    if save_data:
        metadata = {
            'num_rounds': num_rounds,
            'num_runs': num_runs,
            'default_gamma': default_gamma,
            'fixed_source': fixed_source,
            'fixed_destination': fixed_destination,
            'main_seed': main_seed,
            'trace_period': trace_period,
            'max_path_length': max_path_length,
            'gamma_values': gamma_values,
            'use_parallel': use_parallel
        }
        
        # Auto-generate filename if directory provided
        if os.path.isdir(save_data):
            filename = generate_save_filename(metadata)
            save_path = os.path.join(save_data, filename + '.pkl')
        else:
            save_path = save_data
            
        save_simulation_results(results, save_path, metadata)
    
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
        file_prefix="ucsb_comprehensive_parallel",
        default_gamma=default_gamma
    )
    
    print("\nUCSB comprehensive example completed!")
    print(f"Plots saved to: {output_dir}")


# %%
# Run the simulation with default parameters
if __name__ == "__main__":
    # For command line usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Run UCSB comprehensive algorithm comparison (parallel).")
    parser.add_argument('--rounds', type=int, default=10000,
                       help='Number of rounds per simulation (default: 10000)')
    parser.add_argument('--runs', type=int, default=5,
                       help='Number of independent runs (default: 5)')
    parser.add_argument('--default-gamma', type=float, default=0.1,
                       help='Default gamma value for the first comparison plot (default: 0.1)')
    parser.add_argument('--fixed-source', type=str, default="10.1.1.102",
                       help='Fixed source node for reproducibility (default: 10.1.1.102)')
    parser.add_argument('--fixed-destination', type=str, default="10.1.1.25",
                       help='Fixed destination node for reproducibility (default: 10.1.1.25)')
    parser.add_argument('--n-jobs', type=int, default=None,
                       help='Number of parallel jobs (default: auto)')
    parser.add_argument('--main-seed', type=int, default=42,
                       help='Main random seed for reproducibility (default: 42)')
    parser.add_argument('--trace-period', type=str, default="1143927049-1143953729",
                       help='Specific trace period to use (default: 1143927049-1143953729)')
    parser.add_argument('--max-path-length', type=int, default=3,
                       help='Maximum path length in hops (default: 3)')
    parser.add_argument('--sequential', action='store_true',
                       help='Use sequential execution for full reproducibility (default: parallel)')
    parser.add_argument('--save-data', type=str, default=None,
                       help='Path to save simulation data (file path or directory for auto-naming)')
    parser.add_argument('--load-data', type=str, default=None,
                       help='Path to load saved simulation data and generate plots')
    parser.add_argument('--plot-only', action='store_true',
                       help='When used with --load-data, only generate plots without analysis')
    
    args, unknown = parser.parse_known_args()
    
    # If no command line arguments provided, use default parameters for # %% execution
    if len(sys.argv) == 1:
        print("Running UCSB comprehensive example with parallel execution (default parameters)...")
        print("Configuration:")
        print("  Number of rounds: 10000")
        print("  Number of runs: 5")
        print("  Fixed node pair: 10.1.1.102 -> 10.1.1.25")
        print("  Trace period: 1143927049-1143953729 (Period 1)")
        print("  Max path length: 3 hops")
        print("  Main seed: 10")
        print("  Parallel jobs: auto")
        print()
        
        main(num_rounds=10000, num_runs=5, default_gamma=0.1, 
             fixed_source="10.1.1.102", fixed_destination="10.1.1.25", n_jobs=None, main_seed=10, use_parallel=True)
    else:
        print(f"Configuration:")
        print(f"  Number of rounds: {args.rounds}")
        print(f"  Number of runs: {args.runs}")
        print(f"  Default gamma for comparison: {args.default_gamma}")
        print(f"  Fixed node pair: {args.fixed_source} -> {args.fixed_destination}")
        print(f"  Trace period: {args.trace_period}")
        print(f"  Max path length: {args.max_path_length} hops")
        print(f"  Number of parallel jobs: {args.n_jobs if args.n_jobs else 'auto'}")
        print(f"  Main seed: {args.main_seed}")
        print(f"  Execution mode: {'Sequential' if args.sequential else 'Parallel'}")
        if args.save_data:
            print(f"  Save data to: {args.save_data}")
        if args.load_data:
            print(f"  Load data from: {args.load_data}")
        print()
        
        main(num_rounds=args.rounds, num_runs=args.runs, default_gamma=args.default_gamma,
             fixed_source=args.fixed_source, fixed_destination=args.fixed_destination,
             n_jobs=args.n_jobs, main_seed=args.main_seed, trace_period=args.trace_period, 
             max_path_length=args.max_path_length, use_parallel=not args.sequential,
             save_data=args.save_data, load_data=args.load_data, plot_only=args.plot_only) 