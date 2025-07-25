#!/usr/bin/env python3
"""
Verify Connectivity Analysis Results
===================================

This script verifies the results from the connectivity analysis
by checking specific node pairs manually.
"""

import os
import glob
import networkx as nx
import numpy as np

# Add project root to Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_neighbortable_file(file_path: str):
    """Parse a neighbortable file to extract nodes and edges."""
    nodes = set()
    edges = []
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        node = parts[0]
                        neighbor = parts[1]
                        nodes.add(node)
                        nodes.add(neighbor)
                        edges.append((node, neighbor))
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    
    return nodes, edges

def verify_node_pair(period_dir: str, src: str, dst: str, sample_size: int = 10):
    """
    Verify the connectivity analysis for a specific node pair.
    
    Args:
        period_dir: Directory containing trace files
        src: Source node
        dst: Destination node
        sample_size: Number of files to sample
    """
    print(f"Verifying {src} -> {dst}")
    print("=" * 50)
    
    # Get trace files
    neighbortable_files = glob.glob(os.path.join(period_dir, 'neighbortable-*'))
    neighbortable_files = sorted(neighbortable_files)
    
    # Sample files
    sample_files = neighbortable_files[:sample_size]
    
    path_counts = []
    path_lengths = []
    
    for i, file_path in enumerate(sample_files):
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
            all_paths = list(nx.all_simple_paths(G, src, dst, cutoff=3))
            
            if len(all_paths) >= 2:
                path_count = len(all_paths)
                path_lengths_list = [len(path) - 1 for path in all_paths]
                avg_length = sum(path_lengths_list) / len(path_lengths_list)
                
                path_counts.append(path_count)
                path_lengths.append(avg_length)
                
                print(f"  {filename}: {path_count} paths, avg length: {avg_length:.2f}")
                print(f"    Paths: {all_paths}")
            else:
                print(f"  {filename}: {len(all_paths)} paths (insufficient)")
        else:
            print(f"  {filename}: No path found")
    
    if path_counts:
        avg_count = np.mean(path_counts)
        avg_length = np.mean(path_lengths)
        print(f"\nSummary:")
        print(f"  Average paths: {avg_count:.1f}")
        print(f"  Average length: {avg_length:.2f}")
        print(f"  Stable periods: {len(path_counts)}/{sample_size}")
    else:
        print(f"\nNo stable periods found in sample")

def main():
    """Main function to verify connectivity results."""
    print("Verifying Connectivity Analysis Results")
    print("=" * 80)
    
    # Define the period 1 directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'ucsb')
    period1_dir = os.path.join(data_dir, '1143927049-1143953729')
    
    if not os.path.exists(period1_dir):
        print(f"Period 1 directory not found: {period1_dir}")
        return
    
    # Test the top node pairs from connectivity analysis
    test_pairs = [
        ('10.1.1.102', '10.1.1.25'),
        ('10.1.1.101', '10.1.1.102'),
        ('10.1.1.101', '10.1.1.25'),
        ('10.1.1.102', '10.1.1.103'),
        ('10.1.1.103', '10.1.1.25'),
    ]
    
    for src, dst in test_pairs:
        verify_node_pair(period1_dir, src, dst, sample_size=20)
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main() 