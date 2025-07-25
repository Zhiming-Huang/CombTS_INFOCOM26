#!/usr/bin/env python3
"""
Verify Timestamp Paths
======================

This script verifies the actual number of valid paths in each neighbortable file
for the specific node pair 10.1.1.102 -> 10.1.1.25.
"""

import os
import glob
import networkx as nx
import numpy as np
from typing import List, Set, Tuple

# Add project root to Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_neighbortable_file(file_path: str) -> Tuple[Set[str], List[Tuple[str, str]]]:
    """Parse a neighbortable file to extract nodes and edges."""
    nodes = set()
    edges = []
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 3:  # Need at least node + neighbor + ETT
                        src = parts[0]
                        nodes.add(src)
                        
                        # Parse neighbor pairs: neighbor1 ETT1 neighbor2 ETT2 ...
                        for i in range(1, len(parts), 2):
                            if i + 1 < len(parts):
                                dst = parts[i]
                                try:
                                    ett = float(parts[i + 1])
                                    if ett < 1000:  # Only include valid links
                                        nodes.add(dst)
                                        edges.append((src, dst))
                                except ValueError:
                                    continue
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    
    return nodes, edges

def verify_all_timestamps():
    """Verify paths for all timestamp files."""
    print("Verifying All Timestamp Paths")
    print("=" * 80)
    
    # Setup
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'ucsb')
    period1_dir = os.path.join(data_dir, '1143927049-1143953729')
    
    # Get all trace files
    neighbortable_files = glob.glob(os.path.join(period1_dir, 'neighbortable-*'))
    neighbortable_files = sorted(neighbortable_files)
    
    print(f"Found {len(neighbortable_files)} trace files")
    
    # Test node pair
    src = '10.1.1.102'
    dst = '10.1.1.25'
    
    print(f"Testing node pair: {src} -> {dst}")
    print(f"Checking all {len(neighbortable_files)} timestamp files...")
    
    # Statistics
    total_files = len(neighbortable_files)
    files_with_paths = 0
    files_with_multiple_paths = 0
    path_counts = []
    all_paths = []
    
    # Check each file
    for i, file_path in enumerate(neighbortable_files):
        filename = os.path.basename(file_path)
        
        # Parse file
        nodes, edges = parse_neighbortable_file(file_path)
        
        # Build graph
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        
        # Check connectivity
        if src in G and dst in G and nx.has_path(G, src, dst):
            # Find all simple paths up to 3 hops
            all_simple_paths = list(nx.all_simple_paths(G, src, dst, cutoff=3))
            path_count = len(all_simple_paths)
            
            path_counts.append(path_count)
            files_with_paths += 1
            
            if path_count >= 2:
                files_with_multiple_paths += 1
                all_paths.append((filename, all_simple_paths))
            
            # Print details for first 20 files or files with multiple paths
            if i < 20 or path_count >= 2:
                print(f"  {filename}: {path_count} paths")
                if path_count >= 2:
                    print(f"    Paths: {all_simple_paths}")
        else:
            print(f"  {filename}: No path found")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total files: {total_files}")
    print(f"Files with any paths: {files_with_paths} ({files_with_paths/total_files*100:.1f}%)")
    print(f"Files with multiple paths: {files_with_multiple_paths} ({files_with_multiple_paths/total_files*100:.1f}%)")
    
    if path_counts:
        print(f"Path count statistics:")
        print(f"  Average: {np.mean(path_counts):.2f}")
        print(f"  Median: {np.median(path_counts):.1f}")
        print(f"  Min: {min(path_counts)}")
        print(f"  Max: {max(path_counts)}")
        print(f"  Std: {np.std(path_counts):.2f}")
    
    # Show all files with multiple paths
    if all_paths:
        print(f"\nFiles with multiple paths ({len(all_paths)}):")
        for filename, paths in all_paths:
            print(f"  {filename}: {len(paths)} paths")
            for i, path in enumerate(paths):
                print(f"    Path {i+1}: {' -> '.join(path)}")
    
    return path_counts, all_paths

def analyze_path_patterns(all_paths):
    """Analyze patterns in the paths found."""
    print(f"\n{'='*80}")
    print("PATH PATTERN ANALYSIS")
    print(f"{'='*80}")
    
    if not all_paths:
        print("No multiple paths found to analyze")
        return
    
    # Collect all unique paths
    unique_paths = set()
    path_frequencies = {}
    
    for filename, paths in all_paths:
        for path in paths:
            path_tuple = tuple(path)
            unique_paths.add(path_tuple)
            if path_tuple in path_frequencies:
                path_frequencies[path_tuple] += 1
            else:
                path_frequencies[path_tuple] = 1
    
    print(f"Unique paths found: {len(unique_paths)}")
    print(f"Path frequencies:")
    for path_tuple, freq in sorted(path_frequencies.items(), key=lambda x: x[1], reverse=True):
        path_str = ' -> '.join(path_tuple)
        print(f"  {path_str}: {freq} times")

def main():
    """Main function."""
    path_counts, all_paths = verify_all_timestamps()
    analyze_path_patterns(all_paths)

if __name__ == "__main__":
    main() 