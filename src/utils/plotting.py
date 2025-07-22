#!/usr/bin/env python3
"""
Plotting utilities for combinatorial bandit algorithms.
Follows the style from test files with consistent formatting.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
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


def plot_algorithm_comparison(results: Dict[str, Any], 
                            algorithms: List[str],
                            output_path: str,
                            title: str = "Algorithm Comparison",
                            figsize: tuple = (4, 3),
                            use_latex: bool = True):
    """
    Plot algorithm comparison following test file style.
    
    Args:
        results: Results dictionary with algorithm data
        algorithms: List of algorithm names to plot
        output_path: Path to save the plot
        title: Plot title
        figsize: Figure size
        use_latex: Whether to use LaTeX rendering
    """
    setup_plot_style(figsize, use_latex)
    
    num_rounds = results['num_rounds']
    rounds = np.arange(1, num_rounds + 1)
    
    # Define markers and display names (following test file style)
    markers = {'CTSB': 'o', 'CombUCB': 's', 'CTS-G': '^', 'CL-SG': 'D', 'BG-CTS': 'P'}
    display_names = {'CTSB': 'CTS-B', 'CombUCB': 'CombUCB', 'CTS-G': 'CTS-G', 'CL-SG': 'CL-SG', 'BG-CTS': 'BG-CTS'}
    
    # Plot cumulative regret for each algorithm (following test file style)
    markevery = int(num_rounds / 10)  # Mark every T/10 points
    
    for alg in algorithms:
        if alg in results:
            alg_data = results[alg]
            avg_regrets = alg_data['avg_cumulative_regrets']
            marker = markers.get(alg, 'o')
            label = display_names.get(alg, alg)
            
            # Plot confidence interval if available
            if 'confidence_interval' in alg_data:
                ci_lower, ci_upper = alg_data['confidence_interval']
                plt.fill_between(rounds, ci_lower, ci_upper, alpha=0.1)
            
            plt.plot(rounds, avg_regrets, label=label, marker=marker, markevery=markevery, linewidth=1.5)
    
    # Set legend (following test file style)
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
    plt.show()
    
    print(f"Algorithm comparison plot saved to: {output_path}")


def plot_gamma_comparison(results: Dict[str, Any],
                         algorithm_name: str,
                         gamma_values: List[float],
                         output_path: str,
                         title: Optional[str] = None,
                         figsize: tuple = (4, 3),
                         use_latex: bool = True):
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
    """
    setup_plot_style(figsize, use_latex)
    
    num_rounds = results['num_rounds']
    rounds = np.arange(1, num_rounds + 1)
    
    # Colors for different gamma values
    colors = plt.cm.viridis(np.linspace(0, 1, len(gamma_values)))
    
    # Plot regret curves for different gamma values
    markevery = int(num_rounds / 10)  # Mark every T/10 points
    
    for i, gamma in enumerate(gamma_values):
        # Find the algorithm data for this gamma
        alg_key = f"{algorithm_name.lower()}_gamma_{gamma}"
        if alg_key in results:
            alg_data = results[alg_key]
            avg_regrets = alg_data['avg_cumulative_regrets']
            
            # Plot confidence interval if available
            if 'confidence_interval' in alg_data:
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
    plt.show()
    
    print(f"{algorithm_name} gamma comparison plot saved to: {output_path}")


def plot_routing_algorithm_comparison(results: Dict[str, Any],
                                    output_path: str,
                                    title: str = "Algorithm Comparison (4x4 Mesh Network)",
                                    figsize: tuple = (4, 3),
                                    use_latex: bool = True,
                                    default_gamma: float = 0.1):
    """
    Plot algorithm comparison including gamma algorithms with specific gamma values.
    
    Args:
        results: Results dictionary with all algorithm data
        output_path: Path to save the plot
        title: Plot title
        figsize: Figure size
        use_latex: Whether to use LaTeX rendering
        default_gamma: Default gamma value for CTS-G and CL-SG
    """
    setup_plot_style(figsize, use_latex)
    
    num_rounds = results['num_rounds']
    rounds = np.arange(1, num_rounds + 1)
    
    # Define algorithms and their display names
    algorithms = [
        ("CTSB", "CTS-B"),
        ("CombUCB", "CombUCB"), 
        (f"cts-g_gamma_{default_gamma}", f"CTS-G ($\\gamma={default_gamma}$)"),
        (f"cl-sg_gamma_{default_gamma}", f"CL-SG ($\\gamma={default_gamma}$)"),
        ("BG-CTS", "BG-CTS")
    ]
    
    # Define markers and colors
    markers = ['o', 's', '^', 'D', 'P']
    colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))
    
    # Plot cumulative regret for each algorithm
    markevery = int(num_rounds / 10)  # Mark every T/10 points
    
    for i, (alg_key, display_name) in enumerate(algorithms):
        if alg_key in results:
            alg_data = results[alg_key]
            avg_regrets = alg_data['avg_cumulative_regrets']
            
            # Plot confidence interval if available
            if 'confidence_interval' in alg_data:
                ci_lower, ci_upper = alg_data['confidence_interval']
                plt.fill_between(rounds, ci_lower, ci_upper, alpha=0.1, color=colors[i])
            
            plt.plot(rounds, avg_regrets, label=display_name, marker=markers[i], 
                    markevery=markevery, linewidth=1.5, color=colors[i])
    
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
    plt.show()
    
    print(f"Algorithm comparison plot saved to: {output_path}")


def plot_all_routing_results(results: Dict[str, Any],
                           output_dir: str,
                           gamma_values: List[float] = [0.01, 0.1, 0.5, 1.0]):
    """
    Plot all three figures for routing results.
    
    Args:
        results: Results dictionary with all algorithm data
        output_dir: Directory to save plots
        gamma_values: List of gamma values used in experiments
    """
    # Plot 1: Algorithm comparison (using default gamma values)
    plot_routing_algorithm_comparison(
        results=results,
        output_path=f"{output_dir}/routing_algorithm_comparison.pdf",
        title="Algorithm Comparison (4x4 Mesh Network)"
    )
    
    # Plot 2: CTS-G gamma comparison
    plot_gamma_comparison(
        results=results,
        algorithm_name="CTS-G",
        gamma_values=gamma_values,
        output_path=f"{output_dir}/routing_ctsg_gamma_comparison.pdf",
        title="CTS-G Algorithm: Regret vs Rounds"
    )
    
    # Plot 3: CL-SG gamma comparison
    plot_gamma_comparison(
        results=results,
        algorithm_name="CL-SG",
        gamma_values=gamma_values,
        output_path=f"{output_dir}/routing_clsg_gamma_comparison.pdf",
        title="CL-SG Algorithm: Regret vs Rounds"
    ) 