#!/usr/bin/env python3
"""
Debug Adjacency Matrix
======================

This script debugs the adjacency matrix to see what links are marked as available.
"""

import os
import sys
import glob
import numpy as np
import networkx as nx
from typing import List, Set, Tuple

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment

def debug_adjacency_matrix():
    """Debug the adjacency matrix."""
    print("Debugging Adjacency Matrix")
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
    
    # Use the first file
    test_files = [neighbortable_files[0]]
    
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
    
    print(f"Testing file: {os.path.basename(test_files[0])}")
    
    # Get adjacency matrix for round 0
    adj = env.get_available_links_for_round(0)
    print(f"Adjacency matrix shape: {adj.shape}")
    
    # Check specific links
    src_idx = env.node_idx[src]
    dst_idx = env.node_idx[dst]
    
    print(f"\nChecking specific links:")
    print(f"Source node '{src}' index: {src_idx}")
    print(f"Destination node '{dst}' index: {dst_idx}")
    print(f"Direct link {src} -> {dst}: {adj[src_idx, dst_idx]}")
    print(f"Direct link {dst} -> {src}: {adj[dst_idx, src_idx]}")
    
    # Check all links involving source and destination
    print(f"\nAll links involving {src}:")
    for i, node in enumerate(env.nodes):
        if adj[src_idx, i]:
            print(f"  {src} -> {node}: True")
    
    print(f"\nAll links involving {dst}:")
    for i, node in enumerate(env.nodes):
        if adj[dst_idx, i]:
            print(f"  {dst} -> {node}: True")
    
    # Manually verify what links should exist
    print(f"\nManual verification:")
    nodes, edges = parse_neighbortable_file(test_files[0])
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    print(f"Manual graph edges involving {src}:")
    for neighbor in G.neighbors(src):
        print(f"  {src} -> {neighbor}")
    
    print(f"Manual graph edges involving {dst}:")
    for neighbor in G.neighbors(dst):
        print(f"  {dst} -> {neighbor}")
    
    # Check if there are any discrepancies
    print(f"\nChecking for discrepancies:")
    for i, node in enumerate(env.nodes):
        if adj[src_idx, i]:
            if not G.has_edge(src, node):
                print(f"  ⚠️  Environment has {src} -> {node} but manual graph doesn't")
    
    for i, node in enumerate(env.nodes):
        if adj[dst_idx, i]:
            if not G.has_edge(dst, node):
                print(f"  ⚠️  Environment has {dst} -> {node} but manual graph doesn't")

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
    debug_adjacency_matrix()

if __name__ == "__main__":
    main() 