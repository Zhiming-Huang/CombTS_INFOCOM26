#!/usr/bin/env python3
"""
Analyze 2-hop and 3-hop path changes for 10.1.1.100 -> 10.1.1.102 in Period 3.
"""

import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, Counter

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def parse_neighbortable_file(file_path):
    """Parse neighbortable file and return edges."""
    edges = set()
    try:
        with open(file_path, 'r') as fin:
            for line in fin:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                src = parts[0]
                for i in range(1, len(parts), 2):
                    dst = parts[i]
                    try:
                        ett = float(parts[i+1])
                        if ett < 1000:
                            edges.add((src, dst))
                    except Exception:
                        continue
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    return edges

def analyze_hop_paths_period3():
    """Analyze 2-hop and 3-hop path changes for 10.1.1.100 -> 10.1.1.102 in Period 3."""
    
    # Configuration
    source = "10.1.1.100"
    destination = "10.1.1.102"
    
    # Get Period 3 data
    ucsb_dir = Path("data/ucsb")
    period3_folder = ucsb_dir / "1144393236-1144450070"
    
    if not period3_folder.exists():
        print("Period 3 folder not found!")
        return
    
    print(f"Analyzing {source} -> {destination} hop paths in Period 3: {period3_folder}")
    
    # Get all trace files
    trace_files = sorted([f for f in period3_folder.iterdir() if f.name.startswith("neighbortable")])
    print(f"Found {len(trace_files)} trace files")
    
    # Get all unique nodes from all files
    all_nodes = set()
    for file_path in trace_files:
        edges = parse_neighbortable_file(file_path)
        for src, dst in edges:
            all_nodes.add(src)
            all_nodes.add(dst)
    
    all_nodes = sorted(all_nodes)
    print(f"Total unique nodes: {len(all_nodes)}")
    
    # Analyze each timestamp
    hop_analysis = []
    
    for i, file_path in enumerate(trace_files):
        if i % 100 == 0:  # Show progress every 100 timestamps
            print(f"Analyzing timestamp {i+1}/{len(trace_files)}: {file_path.name}")
        
        # Parse edges for this timestamp
        edges = parse_neighbortable_file(file_path)
        
        # Build graph
        G = nx.Graph()
        G.add_nodes_from(all_nodes)
        G.add_edges_from(edges)
        
        # Check if path exists
        if source in G and destination in G and nx.has_path(G, source, destination):
            try:
                # Find all simple paths up to 3 hops
                paths = list(nx.all_simple_paths(G, source, destination, cutoff=3))
                
                if paths:
                    # Analyze paths by hop count
                    hop_counts = defaultdict(int)
                    for path in paths:
                        hop_count = len(path) - 1  # Convert to hop count
                        hop_counts[hop_count] += 1
                    
                    hop_analysis.append({
                        'timestamp': i,
                        'file': file_path.name,
                        'total_paths': len(paths),
                        'hop_2_paths': hop_counts.get(2, 0),
                        'hop_3_paths': hop_counts.get(3, 0),
                        'hop_1_paths': hop_counts.get(1, 0),
                        'hop_2_percentage': hop_counts.get(2, 0) / len(paths) * 100,
                        'hop_3_percentage': hop_counts.get(3, 0) / len(paths) * 100,
                        'hop_1_percentage': hop_counts.get(1, 0) / len(paths) * 100
                    })
                else:
                    hop_analysis.append({
                        'timestamp': i,
                        'file': file_path.name,
                        'total_paths': 0,
                        'hop_2_paths': 0,
                        'hop_3_paths': 0,
                        'hop_1_paths': 0,
                        'hop_2_percentage': 0,
                        'hop_3_percentage': 0,
                        'hop_1_percentage': 0
                    })
                    
            except nx.NetworkXNoPath:
                hop_analysis.append({
                    'timestamp': i,
                    'file': file_path.name,
                    'total_paths': 0,
                    'hop_2_paths': 0,
                    'hop_3_paths': 0,
                    'hop_1_paths': 0,
                    'hop_2_percentage': 0,
                    'hop_3_percentage': 0,
                    'hop_1_percentage': 0
                })
        else:
            hop_analysis.append({
                'timestamp': i,
                'file': file_path.name,
                'total_paths': 0,
                'hop_2_paths': 0,
                'hop_3_paths': 0,
                'hop_1_paths': 0,
                'hop_2_percentage': 0,
                'hop_3_percentage': 0,
                'hop_1_percentage': 0
            })
    
    # Calculate statistics
    timestamps_with_paths = [a for a in hop_analysis if a['total_paths'] > 0]
    timestamps_without_paths = [a for a in hop_analysis if a['total_paths'] == 0]
    
    print(f"\n{'='*80}")
    print(f"HOP PATH ANALYSIS RESULTS FOR {source} -> {destination}")
    print(f"{'='*80}")
    
    print(f"Total timestamps: {len(trace_files)}")
    print(f"Timestamps with paths: {len(timestamps_with_paths)} ({len(timestamps_with_paths)/len(trace_files)*100:.1f}%)")
    print(f"Timestamps without paths: {len(timestamps_without_paths)} ({len(timestamps_without_paths)/len(trace_files)*100:.1f}%)")
    
    if timestamps_with_paths:
        # Extract data for analysis
        hop_2_counts = [a['hop_2_paths'] for a in timestamps_with_paths]
        hop_3_counts = [a['hop_3_paths'] for a in timestamps_with_paths]
        hop_1_counts = [a['hop_1_paths'] for a in timestamps_with_paths]
        total_paths = [a['total_paths'] for a in timestamps_with_paths]
        
        print(f"\n2-hop path statistics:")
        print(f"  Mean: {np.mean(hop_2_counts):.1f}")
        print(f"  Median: {np.median(hop_2_counts):.1f}")
        print(f"  Min: {min(hop_2_counts)}")
        print(f"  Max: {max(hop_2_counts)}")
        print(f"  Std: {np.std(hop_2_counts):.1f}")
        
        print(f"\n3-hop path statistics:")
        print(f"  Mean: {np.mean(hop_3_counts):.1f}")
        print(f"  Median: {np.median(hop_3_counts):.1f}")
        print(f"  Min: {min(hop_3_counts)}")
        print(f"  Max: {max(hop_3_counts)}")
        print(f"  Std: {np.std(hop_3_counts):.1f}")
        
        print(f"\n1-hop path statistics:")
        print(f"  Mean: {np.mean(hop_1_counts):.1f}")
        print(f"  Median: {np.median(hop_1_counts):.1f}")
        print(f"  Min: {min(hop_1_counts)}")
        print(f"  Max: {max(hop_1_counts)}")
        print(f"  Std: {np.std(hop_1_counts):.1f}")
        
        # Calculate percentages
        hop_2_percentages = [a['hop_2_percentage'] for a in timestamps_with_paths]
        hop_3_percentages = [a['hop_3_percentage'] for a in timestamps_with_paths]
        hop_1_percentages = [a['hop_1_percentage'] for a in timestamps_with_paths]
        
        print(f"\nPercentage statistics:")
        print(f"  2-hop paths: {np.mean(hop_2_percentages):.1f}% ± {np.std(hop_2_percentages):.1f}%")
        print(f"  3-hop paths: {np.mean(hop_3_percentages):.1f}% ± {np.std(hop_3_percentages):.1f}%")
        print(f"  1-hop paths: {np.mean(hop_1_percentages):.1f}% ± {np.std(hop_1_percentages):.1f}%")
        
        # Show detailed analysis for first 20 timestamps with paths
        print(f"\n{'='*80}")
        print("DETAILED HOP ANALYSIS (FIRST 20 TIMESTAMPS WITH PATHS)")
        print(f"{'='*80}")
        
        for i, analysis in enumerate(timestamps_with_paths[:20]):
            print(f"\nTimestamp {analysis['timestamp']+1}: {analysis['file']}")
            print(f"  Total paths: {analysis['total_paths']}")
            print(f"  1-hop paths: {analysis['hop_1_paths']} ({analysis['hop_1_percentage']:.1f}%)")
            print(f"  2-hop paths: {analysis['hop_2_paths']} ({analysis['hop_2_percentage']:.1f}%)")
            print(f"  3-hop paths: {analysis['hop_3_paths']} ({analysis['hop_3_percentage']:.1f}%)")
        
        # Create visualization
        create_hop_analysis_plot(hop_analysis, timestamps_with_paths, timestamps_without_paths)
        
    else:
        print(f"No paths found between {source} and {destination} in any timestamp.")
    
    return hop_analysis

def create_hop_analysis_plot(hop_analysis, timestamps_with_paths, timestamps_without_paths):
    """Create plots showing hop path changes over time."""
    
    # Set up the plot
    plt.rcParams.update({
        'figure.figsize': (12, 10),
        'font.size': 12,
        'text.usetex': True,
        'font.family': 'serif'
    })
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract data
    timestamps = [a['timestamp'] for a in hop_analysis]
    hop_2_counts = [a['hop_2_paths'] for a in hop_analysis]
    hop_3_counts = [a['hop_3_paths'] for a in hop_analysis]
    hop_1_counts = [a['hop_1_paths'] for a in hop_analysis]
    total_paths = [a['total_paths'] for a in hop_analysis]
    
    # Plot 1: 2-hop paths over time
    ax1.plot(timestamps, hop_2_counts, 'b-', linewidth=1, alpha=0.7, label='2-hop paths')
    ax1.scatter(timestamps, hop_2_counts, c='blue', s=10, alpha=0.6)
    
    # Mark periods without paths
    if timestamps_without_paths:
        no_path_timestamps = [a['timestamp'] for a in timestamps_without_paths]
        ax1.scatter(no_path_timestamps, [0] * len(no_path_timestamps), 
                   c='red', s=5, alpha=0.5, label='No paths')
    
    ax1.set_xlabel('Timestamp Index')
    ax1.set_ylabel('Number of 2-hop Paths')
    ax1.set_title('2-hop Path Count Changes Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add statistics text
    valid_2hop = [c for c in hop_2_counts if c > 0]
    if valid_2hop:
        stats_text = f'Mean: {np.mean(valid_2hop):.1f}\nMedian: {np.median(valid_2hop):.1f}\nMin: {min(valid_2hop)}\nMax: {max(valid_2hop)}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Plot 2: 3-hop paths over time
    ax2.plot(timestamps, hop_3_counts, 'g-', linewidth=1, alpha=0.7, label='3-hop paths')
    ax2.scatter(timestamps, hop_3_counts, c='green', s=10, alpha=0.6)
    
    # Mark periods without paths
    if timestamps_without_paths:
        ax2.scatter(no_path_timestamps, [0] * len(no_path_timestamps), 
                   c='red', s=5, alpha=0.5, label='No paths')
    
    ax2.set_xlabel('Timestamp Index')
    ax2.set_ylabel('Number of 3-hop Paths')
    ax2.set_title('3-hop Path Count Changes Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add statistics text
    valid_3hop = [c for c in hop_3_counts if c > 0]
    if valid_3hop:
        stats_text = f'Mean: {np.mean(valid_3hop):.1f}\nMedian: {np.median(valid_3hop):.1f}\nMin: {min(valid_3hop)}\nMax: {max(valid_3hop)}'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Plot 3: Comparison of 2-hop vs 3-hop paths
    ax3.plot(timestamps, hop_2_counts, 'b-', linewidth=1, alpha=0.7, label='2-hop paths')
    ax3.plot(timestamps, hop_3_counts, 'g-', linewidth=1, alpha=0.7, label='3-hop paths')
    ax3.scatter(timestamps, hop_2_counts, c='blue', s=8, alpha=0.6)
    ax3.scatter(timestamps, hop_3_counts, c='green', s=8, alpha=0.6)
    
    ax3.set_xlabel('Timestamp Index')
    ax3.set_ylabel('Number of Paths')
    ax3.set_title('2-hop vs 3-hop Path Comparison')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Percentage distribution over time
    hop_2_percentages = [a['hop_2_percentage'] for a in hop_analysis]
    hop_3_percentages = [a['hop_3_percentage'] for a in hop_analysis]
    hop_1_percentages = [a['hop_1_percentage'] for a in hop_analysis]
    
    ax4.plot(timestamps, hop_1_percentages, 'r-', linewidth=1, alpha=0.7, label='1-hop (%)')
    ax4.plot(timestamps, hop_2_percentages, 'b-', linewidth=1, alpha=0.7, label='2-hop (%)')
    ax4.plot(timestamps, hop_3_percentages, 'g-', linewidth=1, alpha=0.7, label='3-hop (%)')
    
    ax4.set_xlabel('Timestamp Index')
    ax4.set_ylabel('Percentage of Total Paths')
    ax4.set_title('Path Length Distribution Over Time')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("output/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_file = output_dir / "period3_100_102_hop_analysis.pdf"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_file}")
    
    plt.show()

if __name__ == "__main__":
    hop_analysis = analyze_hop_paths_period3() 
"""
Analyze 2-hop and 3-hop path changes for 10.1.1.100 -> 10.1.1.102 in Period 3.
"""

import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, Counter

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def parse_neighbortable_file(file_path):
    """Parse neighbortable file and return edges."""
    edges = set()
    try:
        with open(file_path, 'r') as fin:
            for line in fin:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                src = parts[0]
                for i in range(1, len(parts), 2):
                    dst = parts[i]
                    try:
                        ett = float(parts[i+1])
                        if ett < 1000:
                            edges.add((src, dst))
                    except Exception:
                        continue
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    return edges

def analyze_hop_paths_period3():
    """Analyze 2-hop and 3-hop path changes for 10.1.1.100 -> 10.1.1.102 in Period 3."""
    
    # Configuration
    source = "10.1.1.100"
    destination = "10.1.1.102"
    
    # Get Period 3 data
    ucsb_dir = Path("data/ucsb")
    period3_folder = ucsb_dir / "1144393236-1144450070"
    
    if not period3_folder.exists():
        print("Period 3 folder not found!")
        return
    
    print(f"Analyzing {source} -> {destination} hop paths in Period 3: {period3_folder}")
    
    # Get all trace files
    trace_files = sorted([f for f in period3_folder.iterdir() if f.name.startswith("neighbortable")])
    print(f"Found {len(trace_files)} trace files")
    
    # Get all unique nodes from all files
    all_nodes = set()
    for file_path in trace_files:
        edges = parse_neighbortable_file(file_path)
        for src, dst in edges:
            all_nodes.add(src)
            all_nodes.add(dst)
    
    all_nodes = sorted(all_nodes)
    print(f"Total unique nodes: {len(all_nodes)}")
    
    # Analyze each timestamp
    hop_analysis = []
    
    for i, file_path in enumerate(trace_files):
        if i % 100 == 0:  # Show progress every 100 timestamps
            print(f"Analyzing timestamp {i+1}/{len(trace_files)}: {file_path.name}")
        
        # Parse edges for this timestamp
        edges = parse_neighbortable_file(file_path)
        
        # Build graph
        G = nx.Graph()
        G.add_nodes_from(all_nodes)
        G.add_edges_from(edges)
        
        # Check if path exists
        if source in G and destination in G and nx.has_path(G, source, destination):
            try:
                # Find all simple paths up to 3 hops
                paths = list(nx.all_simple_paths(G, source, destination, cutoff=3))
                
                if paths:
                    # Analyze paths by hop count
                    hop_counts = defaultdict(int)
                    for path in paths:
                        hop_count = len(path) - 1  # Convert to hop count
                        hop_counts[hop_count] += 1
                    
                    hop_analysis.append({
                        'timestamp': i,
                        'file': file_path.name,
                        'total_paths': len(paths),
                        'hop_2_paths': hop_counts.get(2, 0),
                        'hop_3_paths': hop_counts.get(3, 0),
                        'hop_1_paths': hop_counts.get(1, 0),
                        'hop_2_percentage': hop_counts.get(2, 0) / len(paths) * 100,
                        'hop_3_percentage': hop_counts.get(3, 0) / len(paths) * 100,
                        'hop_1_percentage': hop_counts.get(1, 0) / len(paths) * 100
                    })
                else:
                    hop_analysis.append({
                        'timestamp': i,
                        'file': file_path.name,
                        'total_paths': 0,
                        'hop_2_paths': 0,
                        'hop_3_paths': 0,
                        'hop_1_paths': 0,
                        'hop_2_percentage': 0,
                        'hop_3_percentage': 0,
                        'hop_1_percentage': 0
                    })
                    
            except nx.NetworkXNoPath:
                hop_analysis.append({
                    'timestamp': i,
                    'file': file_path.name,
                    'total_paths': 0,
                    'hop_2_paths': 0,
                    'hop_3_paths': 0,
                    'hop_1_paths': 0,
                    'hop_2_percentage': 0,
                    'hop_3_percentage': 0,
                    'hop_1_percentage': 0
                })
        else:
            hop_analysis.append({
                'timestamp': i,
                'file': file_path.name,
                'total_paths': 0,
                'hop_2_paths': 0,
                'hop_3_paths': 0,
                'hop_1_paths': 0,
                'hop_2_percentage': 0,
                'hop_3_percentage': 0,
                'hop_1_percentage': 0
            })
    
    # Calculate statistics
    timestamps_with_paths = [a for a in hop_analysis if a['total_paths'] > 0]
    timestamps_without_paths = [a for a in hop_analysis if a['total_paths'] == 0]
    
    print(f"\n{'='*80}")
    print(f"HOP PATH ANALYSIS RESULTS FOR {source} -> {destination}")
    print(f"{'='*80}")
    
    print(f"Total timestamps: {len(trace_files)}")
    print(f"Timestamps with paths: {len(timestamps_with_paths)} ({len(timestamps_with_paths)/len(trace_files)*100:.1f}%)")
    print(f"Timestamps without paths: {len(timestamps_without_paths)} ({len(timestamps_without_paths)/len(trace_files)*100:.1f}%)")
    
    if timestamps_with_paths:
        # Extract data for analysis
        hop_2_counts = [a['hop_2_paths'] for a in timestamps_with_paths]
        hop_3_counts = [a['hop_3_paths'] for a in timestamps_with_paths]
        hop_1_counts = [a['hop_1_paths'] for a in timestamps_with_paths]
        total_paths = [a['total_paths'] for a in timestamps_with_paths]
        
        print(f"\n2-hop path statistics:")
        print(f"  Mean: {np.mean(hop_2_counts):.1f}")
        print(f"  Median: {np.median(hop_2_counts):.1f}")
        print(f"  Min: {min(hop_2_counts)}")
        print(f"  Max: {max(hop_2_counts)}")
        print(f"  Std: {np.std(hop_2_counts):.1f}")
        
        print(f"\n3-hop path statistics:")
        print(f"  Mean: {np.mean(hop_3_counts):.1f}")
        print(f"  Median: {np.median(hop_3_counts):.1f}")
        print(f"  Min: {min(hop_3_counts)}")
        print(f"  Max: {max(hop_3_counts)}")
        print(f"  Std: {np.std(hop_3_counts):.1f}")
        
        print(f"\n1-hop path statistics:")
        print(f"  Mean: {np.mean(hop_1_counts):.1f}")
        print(f"  Median: {np.median(hop_1_counts):.1f}")
        print(f"  Min: {min(hop_1_counts)}")
        print(f"  Max: {max(hop_1_counts)}")
        print(f"  Std: {np.std(hop_1_counts):.1f}")
        
        # Calculate percentages
        hop_2_percentages = [a['hop_2_percentage'] for a in timestamps_with_paths]
        hop_3_percentages = [a['hop_3_percentage'] for a in timestamps_with_paths]
        hop_1_percentages = [a['hop_1_percentage'] for a in timestamps_with_paths]
        
        print(f"\nPercentage statistics:")
        print(f"  2-hop paths: {np.mean(hop_2_percentages):.1f}% ± {np.std(hop_2_percentages):.1f}%")
        print(f"  3-hop paths: {np.mean(hop_3_percentages):.1f}% ± {np.std(hop_3_percentages):.1f}%")
        print(f"  1-hop paths: {np.mean(hop_1_percentages):.1f}% ± {np.std(hop_1_percentages):.1f}%")
        
        # Show detailed analysis for first 20 timestamps with paths
        print(f"\n{'='*80}")
        print("DETAILED HOP ANALYSIS (FIRST 20 TIMESTAMPS WITH PATHS)")
        print(f"{'='*80}")
        
        for i, analysis in enumerate(timestamps_with_paths[:20]):
            print(f"\nTimestamp {analysis['timestamp']+1}: {analysis['file']}")
            print(f"  Total paths: {analysis['total_paths']}")
            print(f"  1-hop paths: {analysis['hop_1_paths']} ({analysis['hop_1_percentage']:.1f}%)")
            print(f"  2-hop paths: {analysis['hop_2_paths']} ({analysis['hop_2_percentage']:.1f}%)")
            print(f"  3-hop paths: {analysis['hop_3_paths']} ({analysis['hop_3_percentage']:.1f}%)")
        
        # Create visualization
        create_hop_analysis_plot(hop_analysis, timestamps_with_paths, timestamps_without_paths)
        
    else:
        print(f"No paths found between {source} and {destination} in any timestamp.")
    
    return hop_analysis

def create_hop_analysis_plot(hop_analysis, timestamps_with_paths, timestamps_without_paths):
    """Create plots showing hop path changes over time."""
    
    # Set up the plot
    plt.rcParams.update({
        'figure.figsize': (12, 10),
        'font.size': 12,
        'text.usetex': True,
        'font.family': 'serif'
    })
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract data
    timestamps = [a['timestamp'] for a in hop_analysis]
    hop_2_counts = [a['hop_2_paths'] for a in hop_analysis]
    hop_3_counts = [a['hop_3_paths'] for a in hop_analysis]
    hop_1_counts = [a['hop_1_paths'] for a in hop_analysis]
    total_paths = [a['total_paths'] for a in hop_analysis]
    
    # Plot 1: 2-hop paths over time
    ax1.plot(timestamps, hop_2_counts, 'b-', linewidth=1, alpha=0.7, label='2-hop paths')
    ax1.scatter(timestamps, hop_2_counts, c='blue', s=10, alpha=0.6)
    
    # Mark periods without paths
    if timestamps_without_paths:
        no_path_timestamps = [a['timestamp'] for a in timestamps_without_paths]
        ax1.scatter(no_path_timestamps, [0] * len(no_path_timestamps), 
                   c='red', s=5, alpha=0.5, label='No paths')
    
    ax1.set_xlabel('Timestamp Index')
    ax1.set_ylabel('Number of 2-hop Paths')
    ax1.set_title('2-hop Path Count Changes Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add statistics text
    valid_2hop = [c for c in hop_2_counts if c > 0]
    if valid_2hop:
        stats_text = f'Mean: {np.mean(valid_2hop):.1f}\nMedian: {np.median(valid_2hop):.1f}\nMin: {min(valid_2hop)}\nMax: {max(valid_2hop)}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Plot 2: 3-hop paths over time
    ax2.plot(timestamps, hop_3_counts, 'g-', linewidth=1, alpha=0.7, label='3-hop paths')
    ax2.scatter(timestamps, hop_3_counts, c='green', s=10, alpha=0.6)
    
    # Mark periods without paths
    if timestamps_without_paths:
        ax2.scatter(no_path_timestamps, [0] * len(no_path_timestamps), 
                   c='red', s=5, alpha=0.5, label='No paths')
    
    ax2.set_xlabel('Timestamp Index')
    ax2.set_ylabel('Number of 3-hop Paths')
    ax2.set_title('3-hop Path Count Changes Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add statistics text
    valid_3hop = [c for c in hop_3_counts if c > 0]
    if valid_3hop:
        stats_text = f'Mean: {np.mean(valid_3hop):.1f}\nMedian: {np.median(valid_3hop):.1f}\nMin: {min(valid_3hop)}\nMax: {max(valid_3hop)}'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Plot 3: Comparison of 2-hop vs 3-hop paths
    ax3.plot(timestamps, hop_2_counts, 'b-', linewidth=1, alpha=0.7, label='2-hop paths')
    ax3.plot(timestamps, hop_3_counts, 'g-', linewidth=1, alpha=0.7, label='3-hop paths')
    ax3.scatter(timestamps, hop_2_counts, c='blue', s=8, alpha=0.6)
    ax3.scatter(timestamps, hop_3_counts, c='green', s=8, alpha=0.6)
    
    ax3.set_xlabel('Timestamp Index')
    ax3.set_ylabel('Number of Paths')
    ax3.set_title('2-hop vs 3-hop Path Comparison')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Percentage distribution over time
    hop_2_percentages = [a['hop_2_percentage'] for a in hop_analysis]
    hop_3_percentages = [a['hop_3_percentage'] for a in hop_analysis]
    hop_1_percentages = [a['hop_1_percentage'] for a in hop_analysis]
    
    ax4.plot(timestamps, hop_1_percentages, 'r-', linewidth=1, alpha=0.7, label='1-hop (%)')
    ax4.plot(timestamps, hop_2_percentages, 'b-', linewidth=1, alpha=0.7, label='2-hop (%)')
    ax4.plot(timestamps, hop_3_percentages, 'g-', linewidth=1, alpha=0.7, label='3-hop (%)')
    
    ax4.set_xlabel('Timestamp Index')
    ax4.set_ylabel('Percentage of Total Paths')
    ax4.set_title('Path Length Distribution Over Time')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("output/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_file = output_dir / "period3_100_102_hop_analysis.pdf"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_file}")
    
    plt.show()

if __name__ == "__main__":
    hop_analysis = analyze_hop_paths_period3() 