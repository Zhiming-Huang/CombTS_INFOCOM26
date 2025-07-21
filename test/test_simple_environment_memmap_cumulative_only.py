# %%
#!/usr/bin/env python3
"""
Memory-mapped test script for simple environment with cumulative regret only.
Supports both CTS-B and CombUCB algorithms.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import tempfile
from scipy import stats
import inspect

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.bandits.cts_b import CTSB
from src.bandits.comb_ucb import CombUCB
from src.bandits.cts_g import CTSG
from src.bandits.cl_sg import CLSG
from src.bandits.bg_cts import BGCTS
from src.environments.simple_environment import SimpleEnvironment


def setup_test_environment():
    """Setup the test environment and return project root."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.insert(0, project_root)
    return project_root


def check_imports():
    """Check if all required imports work."""
    try:
        from src.bandits.cts_b import CTSB
        from src.bandits.comb_ucb import CombUCB
        from src.bandits.cts_g import CTSG
        from src.environments.simple_environment import SimpleEnvironment
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False


def get_output_path(filename):
    """Get output file path relative to project root."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, 'output', 'images')
    return os.path.join(output_dir, filename)


def ensure_output_directory():
    """Ensure output directory exists."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, 'output', 'images')
    os.makedirs(output_dir, exist_ok=True)


def run_memory_mapped_simulation(num_rounds: int = 10000, num_runs: int = 5, 
                                progress_level: str = "normal", 
                                algorithms: List[str] = ["CTSB"],
                                main_seed: int = 15) -> Dict[str, Any]:
    """
    Run memory-mapped simulation with pre-allocated arrays.
    
    Args:
        num_rounds: Number of rounds per simulation
        num_runs: Number of independent runs
        progress_level: Progress tracking level ("minimal", "normal", "detailed")
        algorithms: List of algorithms to test ("CTSB", "CombUCB")
        main_seed: Seed for the main random number generator
        
    Returns:
        Dictionary containing aggregated results with confidence intervals
    """
    print(f"Running {num_runs} simulations with {num_rounds} rounds each...")
    print(f"Algorithms: {', '.join(algorithms)}")
    print(f"Progress level: {progress_level}")
    
    # Create output data directory for memory-mapped files
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_data_dir = os.path.join(project_root, 'output', 'data')
    os.makedirs(output_data_dir, exist_ok=True)
    
    # Create memory-mapped files for each algorithm
    algorithm_files = {}
    algorithm_mmaps = {}
    
    for alg in algorithms:
        regrets_file = os.path.join(output_data_dir, f"{alg.lower()}_regrets.dat")
        rewards_file = os.path.join(output_data_dir, f"{alg.lower()}_rewards.dat")
        
        regrets_mmap = np.memmap(regrets_file, dtype=np.float64, mode='w+', 
                                 shape=(num_runs, num_rounds))
        rewards_mmap = np.memmap(rewards_file, dtype=np.float64, mode='w+', 
                                 shape=(num_runs, num_rounds))
        
        algorithm_files[alg] = (regrets_file, rewards_file)
        algorithm_mmaps[alg] = (regrets_mmap, rewards_mmap)
    
    print(f"Created memory-mapped arrays: {num_runs} runs × {num_rounds} rounds × {len(algorithms)} algorithms")
    print(f"Storage location: {output_data_dir}")
    
    memory_usage = algorithm_mmaps[algorithms[0]][0].nbytes / (1024 * 1024)
    print(f"Memory usage: {memory_usage:.2f} MB per array")
    
    # Configure progress bars based on level
    show_run_progress = progress_level in ["normal", "detailed"]
    show_round_progress = progress_level == "detailed"
    
    # Create progress bar for runs
    if show_run_progress:
        run_pbar = tqdm(range(num_runs), desc="Simulation runs", unit="run")
    else:
        run_pbar = range(num_runs)
    
    # Setup a main random generator for fairness (only once, outside run loop)
    main_rng = np.random.default_rng(main_seed)
    child_rngs = main_rng.spawn(1 + len(algorithms))
    env_rng = child_rngs[0]
    rng_dict = {alg: child_rngs[i+1] for i, alg in enumerate(algorithms)}
    shared_env = SimpleEnvironment(
        num_arms=10,
        num_optimal=3,
        optimal_mean=0.9,
        suboptimal_mean=0.85,
        availability_rate=0.6,
        max_combination_size=3,
        num_rounds=num_rounds,
        pre_generate_rewards=True,
        rng=env_rng
    )
    for run in run_pbar:
        if show_run_progress:
            run_pbar.set_postfix({"Run": f"{run + 1}/{num_runs}"})
        # Use the same environment for all runs
        temp_env = shared_env
        algorithm_instances = {}
        if "CTSB" in algorithms:
            algorithm_instances["CTSB"] = CTSB(environment=temp_env, rng=rng_dict["CTSB"])
        if "CombUCB" in algorithms:
            algorithm_instances["CombUCB"] = CombUCB(environment=temp_env, rng=rng_dict["CombUCB"])
        if "CTS-G" in algorithms:
            algorithm_instances["CTS-G"] = CTSG(environment=temp_env, rng=rng_dict["CTS-G"], gamma=0.1)
        if "CL-SG" in algorithms:
            algorithm_instances["CL-SG"] = CLSG(environment=temp_env, rng=rng_dict["CL-SG"], gamma=0.1)
        if "BG-CTS" in algorithms:
            algorithm_instances["BG-CTS"] = BGCTS(environment=temp_env, rnd_generator=rng_dict["BG-CTS"])
        
        # Run simulation for each algorithm
        for alg_name in algorithms:
            algorithm = algorithm_instances[alg_name]
            regrets_mmap, rewards_mmap = algorithm_mmaps[alg_name]
            # Remove per-algorithm env creation and assignment
            # env = SimpleEnvironment(...)
            # algorithm.environment = env
            # Use temp_env (shared_env) for all algorithms
            # Run simulation
            total_reward = 0
            total_regret = 0
            # Create progress bar for rounds
            if show_round_progress:
                round_pbar = tqdm(range(num_rounds), desc=f"Run {run + 1} {alg_name} rounds", 
                                 unit="round", leave=False)
            else:
                round_pbar = range(num_rounds)
            for round_num in round_pbar:
                selected_combination = algorithm.select_combination(round_num)
                rewards = {}
                total_round_reward = 0
                if selected_combination:
                    for arm in selected_combination:
                        rewards[arm] = temp_env.get_reward_for_round(arm, round_num)
                    total_round_reward = sum(rewards.values())
                # Update algorithm
                algorithm.update_posterior(selected_combination, rewards, round_num)
                available_arms = temp_env.get_available_arms_for_round(round_num)
                optimal_combination = temp_env.get_optimal_combination(available_arms)
                optimal_expected_reward = sum(temp_env.arm_means[arm] for arm in optimal_combination) if optimal_combination else 0
                selected_expected_reward = sum(temp_env.arm_means[arm] for arm in selected_combination) if selected_combination else 0
                regret = optimal_expected_reward - selected_expected_reward
                total_reward += total_round_reward
                total_regret += regret
                regrets_mmap[run, round_num] = total_regret
                rewards_mmap[run, round_num] = total_reward
            regrets_mmap.flush()
            rewards_mmap.flush()
    
    # Calculate statistics for each algorithm
    results = {
        'num_rounds': num_rounds,
        'num_runs': num_runs,
        'algorithms': algorithms,
        'confidence_level': 0.95
    }
    
    for alg_name in algorithms:
        regrets_mmap, _ = algorithm_mmaps[alg_name]
        
        avg_regrets = np.mean(regrets_mmap, axis=0)
        std_regrets = np.std(regrets_mmap, axis=0)
        
        # Calculate 95% confidence intervals using t-student distribution
        t_critical = stats.t.ppf((1 + 0.95) / 2, num_runs - 1)
        confidence_intervals = t_critical * std_regrets / np.sqrt(num_runs)
        
        results[alg_name.lower()] = {
            'avg_cumulative_regrets': avg_regrets,
            'std_cumulative_regrets': std_regrets,
            'confidence_intervals': confidence_intervals,
            'final_avg_regret': avg_regrets[-1],
            'final_std_regret': std_regrets[-1]
        }
    
    # Clean up memory-mapped files
    for alg_name in algorithms:
        regrets_mmap, rewards_mmap = algorithm_mmaps[alg_name]
        del regrets_mmap
        del rewards_mmap
    
    results['output_data_dir'] = output_data_dir
    return results


def analyze_regret_growth_with_confidence(results: Dict[str, Any]):
    """
    Analyze regret growth with confidence intervals.
    
    Args:
        results: Results from run_memory_mapped_simulation
    """
    num_rounds = results['num_rounds']
    algorithms = results['algorithms']
    confidence_level = results['confidence_level']
    
    # Calculate growth rates at different points
    points = [100, 500, 1000, 2000, 5000, 10000]
    points = [p for p in points if p <= num_rounds]
    
    print(f"\nRegret Growth Analysis (with {confidence_level*100:.0f}% confidence intervals):")
    print("=" * 70)
    
    for point in points:
        if point <= num_rounds:
            print(f"Round {point:5d}:", end=" ")
            for alg in algorithms:
                alg_data = results[alg.lower()]
                regret = alg_data['avg_cumulative_regrets'][point - 1]
                ci = alg_data['confidence_intervals'][point - 1]
                growth_rate = regret / np.sqrt(point)
                print(f"{alg}: {regret:8.2f} ± {ci:6.2f} (rate: {growth_rate:.4f})", end=" ")
            print()
    
    # Check if regret is sublinear for each algorithm
    print(f"\nFinal growth rates:")
    for alg in algorithms:
        alg_data = results[alg.lower()]
        final_growth_rate = alg_data['avg_cumulative_regrets'][-1] / np.sqrt(num_rounds)
        final_ci = alg_data['confidence_intervals'][-1] / np.sqrt(num_rounds)
        print(f"{alg}: {final_growth_rate:.4f} ± {final_ci:.4f}")
        
        if final_growth_rate < 10:
            print(f"✓ {alg} appears to be sublinear (bounded growth rate)")
        else:
            print(f"✗ {alg} may not be sublinear (unbounded growth rate)")


def analyze_algorithm_comparison(results: Dict[str, Any]):
    """
    Analyze comparison results between algorithms.
    
    Args:
        results: Results from run_memory_mapped_simulation
    """
    algorithms = results['algorithms']
    
    if len(algorithms) < 2:
        return
    
    print(f"\nAlgorithm Comparison Results:")
    print("=" * 50)
    
    # Get final regrets for each algorithm
    final_regrets = {}
    for alg in algorithms:
        alg_data = results[alg.lower()]
        final_regrets[alg] = alg_data['final_avg_regret']
        print(f"{alg} final regret: {final_regrets[alg]:.2f} ± {alg_data['final_std_regret']:.2f}")
    
    # Determine winner
    best_alg = min(final_regrets, key=final_regrets.get)
    worst_alg = max(final_regrets, key=final_regrets.get)
    
    if best_alg != worst_alg:
        improvement = ((final_regrets[worst_alg] - final_regrets[best_alg]) / final_regrets[worst_alg]) * 100
        print(f"Winner: {best_alg} (improvement: {improvement:.1f}%)")
    else:
        print("Algorithms perform similarly")


def cleanup_temp_files(results: Dict[str, Any]):
    """Clean up temporary files."""
    output_data_dir = results.get('output_data_dir')
    if output_data_dir:
        algorithms = results['algorithms']
        for alg in algorithms:
            regrets_file = os.path.join(output_data_dir, f"{alg.lower()}_regrets.dat")
            rewards_file = os.path.join(output_data_dir, f"{alg.lower()}_rewards.dat")
            
            # Remove only the .dat files, keep the directory
            if os.path.exists(regrets_file):
                os.remove(regrets_file)
            if os.path.exists(rewards_file):
                os.remove(rewards_file)
        print(f"Cleaned up memory-mapped files in: {output_data_dir}")


def plot_cumulative_regret_only(results: Dict[str, Any]):
    """
    Plot cumulative regret for all algorithms.
    
    Args:
        results: Results from run_memory_mapped_simulation
    """
    # Set seaborn style similar to provided code
    sns.set_theme()
    sns.set_style("whitegrid")
    
    # Set figure size and parameters
    plt.figure(figsize=(4, 3))
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 20
    
    num_rounds = results['num_rounds']
    rounds = np.arange(1, num_rounds + 1)
    algorithms = results['algorithms']
    
    # Define markers and display names for different algorithms
    markers = {'CTSB': 'o', 'CombUCB': 's', 'CTS-G': '^', 'CL-SG': 'D', 'BG-CTS': 'P'}
    display_names = {'CTSB': 'CTS-B', 'CombUCB': 'CombUCB', 'CTS-G': 'CTS-G', 'CL-SG': 'CL-SG', 'BG-CTS': 'BG-CTS'}
    
    # Plot cumulative regret for each algorithm
    markevery = int(num_rounds / 10)  # Mark every T/10 points
    
    for alg in algorithms:
        alg_data = results[alg.lower()]
        avg_regrets = alg_data['avg_cumulative_regrets']
        marker = markers.get(alg, 'o')
        label = display_names.get(alg, alg)
        plt.plot(rounds, avg_regrets, label=label, marker=marker, markevery=markevery, linewidth=1.5)
    
    # Set legend
    plt.legend(fontsize=10)
    
    # Create a ScalarFormatter object for scientific notation
    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    
    # Apply formatter to axes
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().xaxis.set_major_formatter(formatter)
    
    # Set tick font sizes
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # Grid and labels
    plt.grid(True)
    plt.xlabel('t', fontsize=10)
    plt.ylabel('Regret', fontsize=10)
    
    # Save as PDF vector graphics
    output_path = get_output_path('simple_environment_cumulative_regret_only.pdf')
    plt.savefig(output_path, bbox_inches='tight', format='pdf', dpi=300)
    plt.show()
    
    print(f"Cumulative regret plot saved to: {output_path}")


def main(main_seed=15):
    print("Memory-Mapped Simple Environment Algorithm Test (Cumulative Regret Only)")
    print("=" * 70)
    # Setup environment
    project_root = setup_test_environment()
    if not check_imports():
        print("Warning: Some imports failed. Check the Python path setup.")
    ensure_output_directory()
    num_rounds = 100000
    num_runs = 10
    progress_level = "normal"
    algorithms = ["CTSB", "CombUCB", "CTS-G", "CL-SG", "BG-CTS"]
    print("Configuration:")
    print(f"  Rounds: {num_rounds}")
    print(f"  Runs: {num_runs}")
    print(f"  Progress level: {progress_level}")
    print(f"  Algorithms: {', '.join(algorithms)}")
    print(f"  Main seed: {main_seed}")
    print()
    results = run_memory_mapped_simulation(
        num_rounds=num_rounds,
        num_runs=num_runs,
        progress_level=progress_level,
        algorithms=algorithms,
        main_seed=main_seed
    )
    analyze_regret_growth_with_confidence(results)
    if len(algorithms) > 1:
        analyze_algorithm_comparison(results)
    plot_cumulative_regret_only(results)
    cleanup_temp_files(results)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run memory-mapped combinatorial bandit simulation.")
    parser.add_argument('--main_seed', type=int, default=15, help='Main random seed (default: 15)')
    args = parser.parse_args()
    main(main_seed=args.main_seed) 