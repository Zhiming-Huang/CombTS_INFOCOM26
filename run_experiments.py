#!/usr/bin/env python3
"""
CombTS INFOCOM26 - One-Click Experiment Reproduction Script (Python Version)
=============================================================================

This script reproduces all experiments with the exact settings used in the paper.

Features:
- Routing 4x4 mesh network simulation
- UCSB real wireless mesh network simulation  
- Automatic data saving and plot generation
- Log scale visualization with optimized legends
- Memory-mapped efficient computation

Usage:
    python run_experiments.py [quick|full]
    
Arguments:
    quick - Run with reduced rounds/runs for fast testing (default)
    full  - Run with full paper settings (longer computation time)

=============================================================================
"""

import sys
import os
import subprocess
import time
from pathlib import Path
import argparse

# ANSI color codes for pretty output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    BLUE = '\033[0;34m'
    YELLOW = '\033[1;33m'
    NC = '\033[0m'  # No Color

def print_colored(text, color):
    """Print text with color."""
    print(f"{color}{text}{Colors.NC}")

def print_header(text):
    """Print a header with formatting."""
    print_colored("=" * 80, Colors.BLUE)
    print_colored(text, Colors.BLUE)
    print_colored("=" * 80, Colors.BLUE)

def print_section(text):
    """Print a section header."""
    print_colored("=" * 80, Colors.GREEN)
    print_colored(text, Colors.GREEN)
    print_colored("=" * 80, Colors.GREEN)

def run_command(cmd, description):
    """Run a command and handle errors."""
    print_colored(f"Running {description}...", Colors.BLUE)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
        print_colored(f"✓ {description} completed successfully", Colors.GREEN)
        return True
    except subprocess.CalledProcessError as e:
        print_colored(f"✗ {description} failed with exit code {e.returncode}", Colors.RED)
        return False

def check_file_exists(filepath, description):
    """Check if a file exists and report status."""
    if Path(filepath).exists():
        print(f"    ✓ {description}")
        return True
    else:
        print_colored(f"    ✗ {description} (missing)", Colors.RED)
        return False

def main():
    """Main experiment runner."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run CombTS INFOCOM26 experiments")
    parser.add_argument('mode', nargs='?', default='quick', 
                       choices=['quick', 'full'],
                       help='Experiment mode: quick (fast testing) or full (paper settings)')
    args = parser.parse_args()
    
    # Setup paths
    project_root = Path.cwd()
    output_dir = project_root / "output"
    data_dir = output_dir / "data"
    images_dir = output_dir / "images"
    
    # Create output directories
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Print header
    print_header("CombTS INFOCOM26 - Experiment Reproduction Script")
    print_colored(f"Mode: {args.mode}", Colors.YELLOW)
    print_colored(f"Project root: {project_root}", Colors.YELLOW)
    print_colored(f"Output directory: {output_dir}", Colors.YELLOW)
    print()
    
    # Set experiment parameters based on mode
    if args.mode == 'full':
        print_colored("Running FULL experiments (paper settings)", Colors.YELLOW)
        routing_rounds, routing_runs = 10000, 5
        ucsb_rounds, ucsb_runs = 10000, 10  # Changed to 10 runs
    else:
        print_colored("Running QUICK experiments (reduced settings for testing)", Colors.YELLOW)
        routing_rounds, routing_runs = 1000, 3
        ucsb_rounds, ucsb_runs = 3000, 3
    
    print(f"Routing 4x4: {routing_rounds} rounds × {routing_runs} runs")
    print(f"UCSB: {ucsb_rounds} rounds × {ucsb_runs} runs")
    print()
    
    # =============================================================================
    # Experiment 1: Routing 4x4 Mesh Network
    # =============================================================================
    print_section("Experiment 1: Routing 4x4 Mesh Network")
    print("Configuration:")
    print("  - Network: 4×4 mesh topology")
    print("  - Source: node 0, Destination: node 15")
    print("  - Links: 24 arms with 75% availability")
    print("  - Algorithms: CTSB, CombUCB, BG-CTS, CTS-G, CL-SG")
    print("  - Gamma values: [0.01, 0.1, 0.5, 1.0]")
    print()
    
    routing_cmd = (f"python example/example_routing_4x4_memmap.py "
                  f"--rounds {routing_rounds} "
                  f"--runs {routing_runs} "
                  f"--keep-memmap "
                  f"--default-gamma 0.01")
    
    if not run_command(routing_cmd, "routing 4x4 simulation"):
        print_colored("Routing experiment failed. Exiting...", Colors.RED)
        sys.exit(1)
    
    # =============================================================================
    # Experiment 2: UCSB Real Wireless Mesh Network
    # =============================================================================
    print()
    print_section("Experiment 2: UCSB Real Wireless Mesh Network")
    print("Configuration:")
    print("  - Network: Real deployment at UC Santa Barbara")
    print("  - Trace period: 1144393236-1144450070")
    print("  - Node pair: 10.1.1.102 → 10.1.1.25")
    print("  - Max path length: 3 hops")
    print("  - Algorithms: CTSB, CombUCB, BG-CTS, CTS-G, CL-SG")
    print("  - Gamma values: [0.01, 0.1, 0.5, 1.0]")
    print()
    
    ucsb_cmd = (f"python example/example_ucsb_comprehensive_parallel.py "
               f"--fixed-source 10.1.1.100 "
               f"--fixed-destination 10.1.1.102 "
               f"--trace-period 1144393236-1144450070 "
               f"--max-path-length 3 "
               f"--rounds {ucsb_rounds} "
               f"--runs {ucsb_runs} "
               f"--main-seed 123 "
               f"--sequential "
               f"--save-data output/data")
    
    if not run_command(ucsb_cmd, "UCSB mesh network simulation"):
        print_colored("UCSB experiment failed. Exiting...", Colors.RED)
        sys.exit(1)
    
    # =============================================================================
    # Results Summary
    # =============================================================================
    print()
    print_section("Experiment Completion Summary")
    
    print_colored("Generated Data Files:", Colors.BLUE)
    print(f"📁 Data directory: {data_dir}")
    
    # List data files
    pkl_files = list(data_dir.glob("*.pkl"))
    for pkl_file in sorted(pkl_files)[-10:]:  # Show last 10 files
        size = pkl_file.stat().st_size / (1024*1024)  # Size in MB
        print(f"  └── {pkl_file.name} ({size:.1f}MB)")
    
    print()
    print_colored("Generated Plot Files:", Colors.BLUE)
    print(f"📊 Images directory: {images_dir}")
    
    # Check routing plots
    print("  Routing 4x4 plots:")
    routing_plots = ['algorithm_comparison', 'ctsg_gamma_comparison', 
                    'clsg_gamma_comparison', 'combined_gamma_comparison']
    for plot in routing_plots:
        filename = f"routing_4x4_{plot}.pdf"
        check_file_exists(images_dir / filename, filename)
    
    # Check UCSB plots
    print("  UCSB mesh network plots:")
    ucsb_plots = ['algorithm_comparison', 'ctsg_gamma_comparison',
                 'clsg_gamma_comparison', 'combined_gamma_comparison']
    for plot in ucsb_plots:
        filename = f"ucsb_comprehensive_parallel_{plot}.pdf"
        check_file_exists(images_dir / filename, filename)
    
    # =============================================================================
    # Visualization Features Summary
    # =============================================================================
    print()
    print_colored("Visualization Features:", Colors.BLUE)
    features = [
        "✓ Log-scale y-axis for better regret visualization",
        "✓ Legend positioned inside plots (lower right)",
        "✓ Multi-marker system for gamma value distinction",
        "✓ LaTeX rendering for mathematical symbols",
        "✓ PDF output with publication-quality formatting",
        "✓ Confidence intervals with transparent shading"
    ]
    for feature in features:
        print(f"  {feature}")
    
    # =============================================================================
    # Quick Analysis Commands
    # =============================================================================
    print()
    print_colored("Quick Analysis Commands:", Colors.BLUE)
    print("To reload and regenerate plots from saved data:")
    
    # Find latest data files
    routing_files = list(data_dir.glob("routing_4x4_results_*.pkl"))
    ucsb_files = list(data_dir.glob("ucsb_results_*.pkl"))
    
    if routing_files:
        latest_routing = sorted(routing_files)[-1]
        print(f"  # Routing 4x4:")
        print(f"  python example/example_routing_4x4_memmap.py --load-data \"{latest_routing}\"")
    
    if ucsb_files:
        latest_ucsb = sorted(ucsb_files)[-1]
        print(f"  # UCSB network:")
        print(f"  python example/example_ucsb_comprehensive_parallel.py --load-data \"{latest_ucsb}\"")
    
    # =============================================================================
    # Final Summary
    # =============================================================================
    print()
    print_colored("=" * 80, Colors.GREEN)
    print_colored("🎉 All experiments completed successfully!", Colors.GREEN)
    print_colored("=" * 80, Colors.GREEN)
    print(f"Results are saved in: {output_dir}")
    print(f"📊 Plots: {images_dir}/*.pdf")
    print(f"📁 Data: {data_dir}/*.pkl")
    print()
    print_colored("Next steps:", Colors.YELLOW)
    print(f"1. Review generated plots in {images_dir}")
    print("2. Use saved data files for quick plot regeneration")
    print("3. Modify experiment parameters in this script for custom runs")

if __name__ == "__main__":
    main() 