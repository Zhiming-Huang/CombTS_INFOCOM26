#!/usr/bin/env python3
"""
Debug Nodes Set
===============

This script debugs the nodes set to see if it contains nodes from all trace files.
"""

import os
import sys
import glob
import numpy as np
from typing import Set

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment

def debug_nodes_set():
    """Debug the nodes set."""
    print("Debugging Nodes Set")
    print("=" * 80)
    
    # Setup
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'ucsb')
    period1_dir = os.path.join(data_dir, '1143927049-1143953729')
    
    # Get trace files
    neighbortable_files = glob.glob(os.path.join(period1_dir, 'neighbortable-*'))
    neighbortable_files = sorted(neighbortable_files)
    
    # Test node pair
    src = '10.1.1.102'
    dst = '10.1.1.25'
    
    print(f"Testing node pair: {src} -> {dst}")
    
    # Test with different numbers of files
    test_cases = [
        ("Single file", neighbortable_files[:1]),
        ("First 5 files", neighbortable_files[:5]),
        ("First 10 files", neighbortable_files[:10]),
        ("All files", neighbortable_files)
    ]
    
    for case_name, test_files in test_cases:
        print(f"\n--- {case_name} ---")
        print(f"Number of files: {len(test_files)}")
        
        # Create environment
        env = UCSBMeshnetMemmapEnvironment(
            neighbortable_files=test_files,
            source=src,
            destination=dst,
            routes_per_minute=4,
            max_path_length=3,
            max_paths_per_algorithm=25,
            seed=42
        )
        
        print(f"Total nodes in environment: {len(env.nodes)}")
        print(f"Sample nodes: {sorted(env.nodes)[:10]}")
        
        # Check if source and destination are in nodes
        print(f"Source '{src}' in nodes: {src in env.nodes}")
        print(f"Destination '{dst}' in nodes: {dst in env.nodes}")
        
        # Check adjacency matrix for first round
        adj = env.get_available_links_for_round(0)
        src_idx = env.node_idx[src]
        dst_idx = env.node_idx[dst]
        
        print(f"Adjacency matrix shape: {adj.shape}")
        print(f"Direct link {src} -> {dst}: {adj[src_idx, dst_idx]}")
        
        # Count total True values in adjacency matrix
        total_links = np.sum(adj)
        print(f"Total links in adjacency matrix: {total_links}")
        
        # Manually verify first file
        if test_files:
            first_file = test_files[0]
            nodes, edges = parse_neighbortable_file(first_file)
            print(f"Manual nodes in first file: {len(nodes)}")
            print(f"Manual edges in first file: {len(edges)}")
            
            # Check if manual nodes are subset of environment nodes
            manual_nodes_set = set(nodes)
            env_nodes_set = set(env.nodes)
            extra_nodes = env_nodes_set - manual_nodes_set
            
            print(f"Extra nodes in environment: {len(extra_nodes)}")
            if extra_nodes:
                print(f"Sample extra nodes: {sorted(list(extra_nodes))[:5]}")

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

def main():
    """Main function."""
    debug_nodes_set()

if __name__ == "__main__":
    main() 