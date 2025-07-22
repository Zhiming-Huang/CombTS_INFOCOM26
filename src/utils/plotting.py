#!/usr/bin/env python3
"""
Plotting utilities for combinatorial bandit algorithms.
Unified plotting logic that accepts regret data and algorithm lists.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union
from matplotlib.ticker import ScalarFormatter


def setup_plot_style(figsize: tuple = (4, 3), use_latex: bool = True, font_size: int = 20):
    """
    Setup consistent plot style following test file conventions.
    
    Args:
        figsize: Figure size (width, height) in inches
        use_latex: Whether to use LaTeX rendering
        font_size: Main font size
    """
    # Set seaborn style similar to test file
    sns.set_theme()
    sns.set_style("whitegrid")
    
    # Set figure size
    plt.figure(figsize=figsize)
    
    # Try to use LaTeX rendering
    if use_latex:
        try:
            plt.rcParams['text.usetex'] = True
            plt.rcParams['font.size'] = font_size
        except:
            plt.rcParams['text.usetex'] = False
            plt.rcParams['font.size'] = 12
    else:
        plt.rcParams['text.usetex'] = False
        plt.rcParams['font.size'] = font_size


def plot_regret_comparison(results: Dict[str, Any],
                          algorithms: List[str],
                          output_path: str,
                          title: str = "Algorithm Comparison",
                          figsize: tuple = (4, 3),
                          use_latex: bool = True,
                          show_confidence_intervals: bool = True):
    """
    Unified function to plot regret comparison for any set of algorithms.
    
    Args:
        results: Results dictionary with algorithm data
        algorithms: List of algorithm names to plot (keys in results)
        output_path: Path to save the plot
        title: Plot title
        figsize: Figure size
        use_latex: Whether to use LaTeX rendering
        show_confidence_intervals: Whether to show confidence intervals
    """
    setup_plot_style(figsize, use_latex)
    
    num_rounds = results['num_rounds']
    rounds = np.arange(1, num_rounds + 1)
    
    # Define markers and colors
    markers = ['o', 's', '^', 'D', 'P', 'v', '<', '>', 'p', '*']
    colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))
    
    # Plot cumulative regret for each algorithm
    markevery = max(1, int(num_rounds / 10))  # Mark every T/10 points
    
    for i, alg in enumerate(algorithms):
        if alg in results:
            alg_data = results[alg]
            avg_regrets = alg_data['avg_cumulative_regrets']
            marker = markers[i % len(markers)]
            color = colors[i]
            
            # Plot confidence interval if available and requested
            if show_confidence_intervals and 'confidence_interval' in alg_data:
                ci_lower, ci_upper = alg_data['confidence_interval']
                plt.fill_between(rounds, ci_lower, ci_upper, alpha=0.1, color=color)
            
            # Create display name (handle gamma algorithms)
            if '_gamma_' in alg:
                # Extract algorithm name and gamma value
                parts = alg.split('_gamma_')
                if len(parts) == 2:
                    alg_name = parts[0].upper()
                    gamma = parts[1]
                    display_name = f"{alg_name} ($\\gamma={gamma}$)"
                else:
                    display_name = alg
            else:
                # Map algorithm names to display names
                display_names = {
                    'CTSB': 'CTS-B',
                    'CombUCB': 'CombUCB',
                    'CTS-G': 'CTS-G',
                    'CL-SG': 'CL-SG',
                    'BG-CTS': 'BG-CTS'
                }
                display_name = display_names.get(alg, alg)
            
            plt.plot(rounds, avg_regrets, label=display_name, marker=marker, 
                    markevery=markevery, linewidth=1.5, color=color)
    
    # Set legend
    plt.legend(fontsize=10)
    
    # Create a ScalarFormatter object for scientific notation
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
    
    # Save plot without title and with tight layout
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, format='pdf', dpi=300)
    plt.close()  # Close the figure to free memory
    
    print(f"Regret comparison plot saved to: {output_path}")


def plot_gamma_comparison(results: Dict[str, Any],
                         algorithm_name: str,
                         gamma_values: List[float],
                         output_path: str,
                         title: Optional[str] = None,
                         figsize: tuple = (4, 3),
                         use_latex: bool = True,
                         show_confidence_intervals: bool = True):
    """
    Plot gamma comparison for a specific algorithm.
    
    Args:
        results: Results dictionary with algorithm data
        algorithm_name: Name of the algorithm (e.g., 'CTS-G', 'CL-SG')
        gamma_values: List of gamma values to plot
        output_path: Path to save the plot
        title: Plot title (if None, will be auto-generated)
        figsize: Figure size
        use_latex: Whether to use LaTeX rendering
        show_confidence_intervals: Whether to show confidence intervals
    """
    setup_plot_style(figsize, use_latex)
    
    num_rounds = results['num_rounds']
    rounds = np.arange(1, num_rounds + 1)
    
    # Colors for different gamma values
    colors = plt.cm.viridis(np.linspace(0, 1, len(gamma_values)))
    
    # Plot regret curves for different gamma values
    markevery = max(1, int(num_rounds / 10))  # Mark every T/10 points
    
    for i, gamma in enumerate(gamma_values):
        # Find the algorithm data for this gamma
        alg_key = f"{algorithm_name.lower()}_gamma_{gamma}"
        if alg_key in results:
            alg_data = results[alg_key]
            avg_regrets = alg_data['avg_cumulative_regrets']
            
            # Plot confidence interval if available and requested
            if show_confidence_intervals and 'confidence_interval' in alg_data:
                ci_lower, ci_upper = alg_data['confidence_interval']
                plt.fill_between(rounds, ci_lower, ci_upper, alpha=0.1, color=colors[i])
            
            plt.plot(rounds, avg_regrets, color=colors[i], linewidth=1.5, 
                    label=f'{algorithm_name} ($\\gamma={gamma}$)', marker='o', markevery=markevery, alpha=0.8)
    
    # Set legend
    plt.legend(fontsize=10)
    
    # Set scientific notation
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().xaxis.set_major_formatter(formatter)
    
    # Set tick font sizes
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # Grid and labels
    plt.grid(True)
    plt.xlabel('t', fontsize=10)
    plt.ylabel('Regret', fontsize=10)
    
    # Save plot without title and with tight layout
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, format='pdf', dpi=300)
    plt.close()  # Close the figure to free memory
    
    print(f"{algorithm_name} gamma comparison plot saved to: {output_path}")


def plot_all_results(results: Dict[str, Any],
                    output_dir: str,
                    base_algorithms: List[str] = None,
                    gamma_algorithms: List[str] = None,
                    gamma_values: List[float] = None,
                    file_prefix: str = "results",
                    default_gamma: float = 0.01):
    """
    Unified function to plot all results with flexible algorithm selection.
    
    Args:
        results: Results dictionary with all algorithm data
        output_dir: Directory to save plots
        base_algorithms: List of base algorithm names (without gamma)
        gamma_algorithms: List of gamma algorithm names (e.g., ['CTS-G', 'CL-SG'])
        gamma_values: List of gamma values used in experiments
        file_prefix: Prefix for output file names
        default_gamma: Default gamma value for algorithm comparison plot
    """
    # Set default values if not provided
    if base_algorithms is None:
        base_algorithms = ['CTSB', 'CombUCB', 'BG-CTS']
    
    if gamma_algorithms is None:
        gamma_algorithms = ['CTS-G', 'CL-SG']
    
    if gamma_values is None:
        gamma_values = [0.01, 0.1, 0.5, 1.0]
    
    # Plot 1: Algorithm comparison (base algorithms + gamma algorithms with default gamma)
    comparison_algorithms = base_algorithms.copy()
    for alg in gamma_algorithms:
        comparison_algorithms.append(f"{alg.lower()}_gamma_{default_gamma}")
    
    plot_regret_comparison(
        results=results,
        algorithms=comparison_algorithms,
        output_path=f"{output_dir}/{file_prefix}_algorithm_comparison.pdf",
        title="Algorithm Comparison"
    )
    
    # Plot 2 & 3: Gamma comparisons for each gamma algorithm
    for alg in gamma_algorithms:
        plot_gamma_comparison(
            results=results,
            algorithm_name=alg,
            gamma_values=gamma_values,
            output_path=f"{output_dir}/{file_prefix}_{alg.lower().replace('-', '')}_gamma_comparison.pdf",
            title=f"{alg} Algorithm: Regret vs Rounds"
        )


# Legacy functions for backward compatibility
def plot_algorithm_comparison(results: Dict[str, Any], 
                            algorithms: List[str],
                            output_path: str,
                            title: str = "Algorithm Comparison",
                            figsize: tuple = (4, 3),
                            use_latex: bool = True):
    """
    Legacy function for backward compatibility.
    """
    return plot_regret_comparison(results, algorithms, output_path, title, figsize, use_latex)


def plot_routing_algorithm_comparison(results: Dict[str, Any],
                                    output_path: str,
                                    title: str = "Algorithm Comparison (4x4 Mesh Network)",
                                    figsize: tuple = (4, 3),
                                    use_latex: bool = True,
                                    default_gamma: float = 0.01):
    """
    Legacy function for backward compatibility.
    """
    base_algorithms = ['CTSB', 'CombUCB', 'BG-CTS']
    gamma_algorithms = ['CTS-G', 'CL-SG']
    
    comparison_algorithms = base_algorithms.copy()
    comparison_algorithms.append(f"cts-g_gamma_{default_gamma}")
    comparison_algorithms.append(f"cl-sg_gamma_{default_gamma}")
    
    return plot_regret_comparison(results, comparison_algorithms, output_path, title, figsize, use_latex)


def plot_all_routing_results(results: Dict[str, Any],
                           output_dir: str,
                           gamma_values: List[float] = [0.01, 0.1, 0.5, 1.0],
                           file_prefix: str = "routing",
                           default_gamma: float = 0.01):
    """
    Legacy function for backward compatibility.
    """
    return plot_all_results(
        results=results,
        output_dir=output_dir,
        base_algorithms=['CTSB', 'CombUCB', 'BG-CTS'],
        gamma_algorithms=['CTS-G', 'CL-SG'],
        gamma_values=gamma_values,
        file_prefix=file_prefix,
        default_gamma=default_gamma
    ) 