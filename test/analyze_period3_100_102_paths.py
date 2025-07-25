#!/usr/bin/env python3
"""
Analyze path changes for 10.1.1.100 -> 10.1.1.102 in Period 3.
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

def analyze_100_102_paths_period3():
    """Analyze path changes for 10.1.1.100 -> 10.1.1.102 in Period 3."""
    
    # Configuration
    source = "10.1.1.100"
    destination = "10.1.1.102"
    
    # Get Period 3 data
    ucsb_dir = Path("data/ucsb")
    period3_folder = ucsb_dir / "1144393236-1144450070"
    
    if not period3_folder.exists():
        print("Period 3 folder not found!")
        return
    
    print(f"Analyzing {source} -> {destination} in Period 3: {period3_folder}")
    
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
    path_counts = []
    path_details = []
    timestamps_with_paths = []
    timestamps_without_paths = []
    
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
                    path_count = len(paths)
                    path_counts.append(path_count)
                    timestamps_with_paths.append(i)
                    
                    # Analyze path lengths
                    path_lengths = [len(path) - 1 for path in paths]  # Convert to hop count
                    path_details.append({
                        'timestamp': i,
                        'file': file_path.name,
                        'path_count': path_count,
                        'path_lengths': path_lengths,
                        'avg_length': np.mean(path_lengths),
                        'min_length': min(path_lengths),
                        'max_length': max(path_lengths)
                    })
                else:
                    timestamps_without_paths.append(i)
                    
            except nx.NetworkXNoPath:
                timestamps_without_paths.append(i)
        else:
            timestamps_without_paths.append(i)
    
    # Calculate statistics
    if path_counts:
        print(f"\n{'='*80}")
        print(f"PATH ANALYSIS RESULTS FOR {source} -> {destination}")
        print(f"{'='*80}")
        
        print(f"Total timestamps: {len(trace_files)}")
        print(f"Timestamps with paths: {len(timestamps_with_paths)} ({len(timestamps_with_paths)/len(trace_files)*100:.1f}%)")
        print(f"Timestamps without paths: {len(timestamps_without_paths)} ({len(timestamps_without_paths)/len(trace_files)*100:.1f}%)")
        
        print(f"\nPath count statistics:")
        print(f"  Mean: {np.mean(path_counts):.1f}")
        print(f"  Median: {np.median(path_counts):.1f}")
        print(f"  Min: {min(path_counts)}")
        print(f"  Max: {max(path_counts)}")
        print(f"  Std: {np.std(path_counts):.1f}")
        
        # Analyze path length distribution
        all_path_lengths = []
        for detail in path_details:
            all_path_lengths.extend(detail['path_lengths'])
        
        length_counter = Counter(all_path_lengths)
        print(f"\nPath length distribution:")
        for length in sorted(length_counter.keys()):
            count = length_counter[length]
            percentage = count / len(all_path_lengths) * 100
            print(f"  {length} hops: {count} paths ({percentage:.1f}%)")
        
        # Show periods of connectivity/disconnectivity
        print(f"\nConnectivity analysis:")
        
        # Find continuous periods with paths
        continuous_with_paths = []
        start = timestamps_with_paths[0]
        prev = start
        
        for ts in timestamps_with_paths[1:]:
            if ts != prev + 1:
                continuous_with_paths.append((start, prev))
                start = ts
            prev = ts
        continuous_with_paths.append((start, prev))
        
        print(f"  Continuous periods with paths: {len(continuous_with_paths)}")
        for i, (start, end) in enumerate(continuous_with_paths[:10]):  # Show first 10
            duration = end - start + 1
            print(f"    Period {i+1}: Timestamps {start+1}-{end+1} ({duration} timestamps)")
        if len(continuous_with_paths) > 10:
            print(f"    ... and {len(continuous_with_paths) - 10} more periods")
        
        # Find continuous periods without paths
        if timestamps_without_paths:
            continuous_without_paths = []
            start = timestamps_without_paths[0]
            prev = start
            
            for ts in timestamps_without_paths[1:]:
                if ts != prev + 1:
                    continuous_without_paths.append((start, prev))
                    start = ts
                prev = ts
            continuous_without_paths.append((start, prev))
            
            print(f"  Continuous periods without paths: {len(continuous_without_paths)}")
            for i, (start, end) in enumerate(continuous_without_paths[:10]):  # Show first 10
                duration = end - start + 1
                print(f"    Period {i+1}: Timestamps {start+1}-{end+1} ({duration} timestamps)")
            if len(continuous_without_paths) > 10:
                print(f"    ... and {len(continuous_without_paths) - 10} more periods")
        
        # Show detailed path analysis for first 20 timestamps with paths
        print(f"\n{'='*80}")
        print("DETAILED PATH ANALYSIS (FIRST 20 TIMESTAMPS WITH PATHS)")
        print(f"{'='*80}")
        
        for i, detail in enumerate(path_details[:20]):
            print(f"\nTimestamp {detail['timestamp']+1}: {detail['file']}")
            print(f"  Path count: {detail['path_count']}")
            print(f"  Path lengths: {detail['path_lengths']}")
            print(f"  Average length: {detail['avg_length']:.1f} hops")
            print(f"  Length range: {detail['min_length']} - {detail['max_length']} hops")
        
        # Create visualization
        create_path_analysis_plot(path_counts, timestamps_with_paths, timestamps_without_paths, trace_files)
        
    else:
        print(f"No paths found between {source} and {destination} in any timestamp.")
    
    return path_details

def create_path_analysis_plot(path_counts, timestamps_with_paths, timestamps_without_paths, trace_files):
    """Create a plot showing path count changes over time."""
    
    # Set up the plot
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 12,
        'text.usetex': True,
        'font.family': 'serif'
    })
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Path count over time
    ax1.plot(timestamps_with_paths, path_counts, 'b-', linewidth=1, alpha=0.7)
    ax1.scatter(timestamps_with_paths, path_counts, c='blue', s=10, alpha=0.6)
    
    # Mark periods without paths
    if timestamps_without_paths:
        ax1.scatter(timestamps_without_paths, [0] * len(timestamps_without_paths), 
                   c='red', s=5, alpha=0.5, label='No paths')
    
    ax1.set_xlabel('Timestamp Index')
    ax1.set_ylabel('Number of Paths')
    ax1.set_title(f'Path Count Changes: 10.1.1.100 → 10.1.1.102 (Period 3)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add statistics text
    stats_text = f'Mean: {np.mean(path_counts):.1f}\nMedian: {np.median(path_counts):.1f}\nMin: {min(path_counts)}\nMax: {max(path_counts)}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Connectivity status over time
    connectivity = np.zeros(len(trace_files))
    connectivity[timestamps_with_paths] = 1
    
    ax2.plot(range(len(trace_files)), connectivity, 'g-', linewidth=2, alpha=0.8)
    ax2.fill_between(range(len(trace_files)), connectivity, alpha=0.3, color='green')
    
    ax2.set_xlabel('Timestamp Index')
    ax2.set_ylabel('Connectivity Status')
    ax2.set_title('Connectivity Status Over Time (1=Connected, 0=Disconnected)')
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, alpha=0.3)
    
    # Add connectivity statistics
    connectivity_stats = f'Connected: {len(timestamps_with_paths)} ({len(timestamps_with_paths)/len(trace_files)*100:.1f}%)\nDisconnected: {len(timestamps_without_paths)} ({len(timestamps_without_paths)/len(trace_files)*100:.1f}%)'
    ax2.text(0.02, 0.98, connectivity_stats, transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("output/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_file = output_dir / "period3_100_102_path_analysis.pdf"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_file}")
    
    plt.show()

if __name__ == "__main__":
    path_details = analyze_100_102_paths_period3() 
"""
Analyze path changes for 10.1.1.100 -> 10.1.1.102 in Period 3.
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

def analyze_100_102_paths_period3():
    """Analyze path changes for 10.1.1.100 -> 10.1.1.102 in Period 3."""
    
    # Configuration
    source = "10.1.1.100"
    destination = "10.1.1.102"
    
    # Get Period 3 data
    ucsb_dir = Path("data/ucsb")
    period3_folder = ucsb_dir / "1144393236-1144450070"
    
    if not period3_folder.exists():
        print("Period 3 folder not found!")
        return
    
    print(f"Analyzing {source} -> {destination} in Period 3: {period3_folder}")
    
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
    path_counts = []
    path_details = []
    timestamps_with_paths = []
    timestamps_without_paths = []
    
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
                    path_count = len(paths)
                    path_counts.append(path_count)
                    timestamps_with_paths.append(i)
                    
                    # Analyze path lengths
                    path_lengths = [len(path) - 1 for path in paths]  # Convert to hop count
                    path_details.append({
                        'timestamp': i,
                        'file': file_path.name,
                        'path_count': path_count,
                        'path_lengths': path_lengths,
                        'avg_length': np.mean(path_lengths),
                        'min_length': min(path_lengths),
                        'max_length': max(path_lengths)
                    })
                else:
                    timestamps_without_paths.append(i)
                    
            except nx.NetworkXNoPath:
                timestamps_without_paths.append(i)
        else:
            timestamps_without_paths.append(i)
    
    # Calculate statistics
    if path_counts:
        print(f"\n{'='*80}")
        print(f"PATH ANALYSIS RESULTS FOR {source} -> {destination}")
        print(f"{'='*80}")
        
        print(f"Total timestamps: {len(trace_files)}")
        print(f"Timestamps with paths: {len(timestamps_with_paths)} ({len(timestamps_with_paths)/len(trace_files)*100:.1f}%)")
        print(f"Timestamps without paths: {len(timestamps_without_paths)} ({len(timestamps_without_paths)/len(trace_files)*100:.1f}%)")
        
        print(f"\nPath count statistics:")
        print(f"  Mean: {np.mean(path_counts):.1f}")
        print(f"  Median: {np.median(path_counts):.1f}")
        print(f"  Min: {min(path_counts)}")
        print(f"  Max: {max(path_counts)}")
        print(f"  Std: {np.std(path_counts):.1f}")
        
        # Analyze path length distribution
        all_path_lengths = []
        for detail in path_details:
            all_path_lengths.extend(detail['path_lengths'])
        
        length_counter = Counter(all_path_lengths)
        print(f"\nPath length distribution:")
        for length in sorted(length_counter.keys()):
            count = length_counter[length]
            percentage = count / len(all_path_lengths) * 100
            print(f"  {length} hops: {count} paths ({percentage:.1f}%)")
        
        # Show periods of connectivity/disconnectivity
        print(f"\nConnectivity analysis:")
        
        # Find continuous periods with paths
        continuous_with_paths = []
        start = timestamps_with_paths[0]
        prev = start
        
        for ts in timestamps_with_paths[1:]:
            if ts != prev + 1:
                continuous_with_paths.append((start, prev))
                start = ts
            prev = ts
        continuous_with_paths.append((start, prev))
        
        print(f"  Continuous periods with paths: {len(continuous_with_paths)}")
        for i, (start, end) in enumerate(continuous_with_paths[:10]):  # Show first 10
            duration = end - start + 1
            print(f"    Period {i+1}: Timestamps {start+1}-{end+1} ({duration} timestamps)")
        if len(continuous_with_paths) > 10:
            print(f"    ... and {len(continuous_with_paths) - 10} more periods")
        
        # Find continuous periods without paths
        if timestamps_without_paths:
            continuous_without_paths = []
            start = timestamps_without_paths[0]
            prev = start
            
            for ts in timestamps_without_paths[1:]:
                if ts != prev + 1:
                    continuous_without_paths.append((start, prev))
                    start = ts
                prev = ts
            continuous_without_paths.append((start, prev))
            
            print(f"  Continuous periods without paths: {len(continuous_without_paths)}")
            for i, (start, end) in enumerate(continuous_without_paths[:10]):  # Show first 10
                duration = end - start + 1
                print(f"    Period {i+1}: Timestamps {start+1}-{end+1} ({duration} timestamps)")
            if len(continuous_without_paths) > 10:
                print(f"    ... and {len(continuous_without_paths) - 10} more periods")
        
        # Show detailed path analysis for first 20 timestamps with paths
        print(f"\n{'='*80}")
        print("DETAILED PATH ANALYSIS (FIRST 20 TIMESTAMPS WITH PATHS)")
        print(f"{'='*80}")
        
        for i, detail in enumerate(path_details[:20]):
            print(f"\nTimestamp {detail['timestamp']+1}: {detail['file']}")
            print(f"  Path count: {detail['path_count']}")
            print(f"  Path lengths: {detail['path_lengths']}")
            print(f"  Average length: {detail['avg_length']:.1f} hops")
            print(f"  Length range: {detail['min_length']} - {detail['max_length']} hops")
        
        # Create visualization
        create_path_analysis_plot(path_counts, timestamps_with_paths, timestamps_without_paths, trace_files)
        
    else:
        print(f"No paths found between {source} and {destination} in any timestamp.")
    
    return path_details

def create_path_analysis_plot(path_counts, timestamps_with_paths, timestamps_without_paths, trace_files):
    """Create a plot showing path count changes over time."""
    
    # Set up the plot
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 12,
        'text.usetex': True,
        'font.family': 'serif'
    })
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Path count over time
    ax1.plot(timestamps_with_paths, path_counts, 'b-', linewidth=1, alpha=0.7)
    ax1.scatter(timestamps_with_paths, path_counts, c='blue', s=10, alpha=0.6)
    
    # Mark periods without paths
    if timestamps_without_paths:
        ax1.scatter(timestamps_without_paths, [0] * len(timestamps_without_paths), 
                   c='red', s=5, alpha=0.5, label='No paths')
    
    ax1.set_xlabel('Timestamp Index')
    ax1.set_ylabel('Number of Paths')
    ax1.set_title(f'Path Count Changes: 10.1.1.100 → 10.1.1.102 (Period 3)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add statistics text
    stats_text = f'Mean: {np.mean(path_counts):.1f}\nMedian: {np.median(path_counts):.1f}\nMin: {min(path_counts)}\nMax: {max(path_counts)}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Connectivity status over time
    connectivity = np.zeros(len(trace_files))
    connectivity[timestamps_with_paths] = 1
    
    ax2.plot(range(len(trace_files)), connectivity, 'g-', linewidth=2, alpha=0.8)
    ax2.fill_between(range(len(trace_files)), connectivity, alpha=0.3, color='green')
    
    ax2.set_xlabel('Timestamp Index')
    ax2.set_ylabel('Connectivity Status')
    ax2.set_title('Connectivity Status Over Time (1=Connected, 0=Disconnected)')
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, alpha=0.3)
    
    # Add connectivity statistics
    connectivity_stats = f'Connected: {len(timestamps_with_paths)} ({len(timestamps_with_paths)/len(trace_files)*100:.1f}%)\nDisconnected: {len(timestamps_without_paths)} ({len(timestamps_without_paths)/len(trace_files)*100:.1f}%)'
    ax2.text(0.02, 0.98, connectivity_stats, transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("output/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_file = output_dir / "period3_100_102_path_analysis.pdf"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_file}")
    
    plt.show()

if __name__ == "__main__":
    path_details = analyze_100_102_paths_period3() 