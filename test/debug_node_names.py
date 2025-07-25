#!/usr/bin/env python3
"""
Debug Node Names and Path Analysis
==================================

This script checks the actual node names in trace files and verifies
the path analysis results.
"""

import os
import glob
import networkx as nx
from collections import defaultdict

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

def debug_node_analysis():
    """Debug the node analysis to find the discrepancy."""
    print("Debugging Node Analysis")
    print("=" * 50)
    
    # Define the period 1 directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'ucsb')
    period1_dir = os.path.join(data_dir, '1143927049-1143953729')
    
    # Get first few trace files
    neighbortable_files = glob.glob(os.path.join(period1_dir, 'neighbortable-*'))
    neighbortable_files = sorted(neighbortable_files)[:5]  # Check first 5 files
    
    print(f"Checking first {len(neighbortable_files)} trace files...")
    
    all_nodes = set()
    
    for i, file_path in enumerate(neighbortable_files):
        print(f"\nFile {i+1}: {os.path.basename(file_path)}")
        
        nodes, edges = parse_neighbortable_file(file_path)
        all_nodes.update(nodes)
        
        print(f"  Nodes in this file: {len(nodes)}")
        print(f"  Edges in this file: {len(edges)}")
        
        # Build graph
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        
        # Check if our target nodes exist
        target_nodes = ['10.1.1.102', '10.1.1.25', '10.1.1.101', '10.1.1.103']
        for node in target_nodes:
            if node in G:
                print(f"    ✓ {node} found in graph")
            else:
                print(f"    ✗ {node} NOT found in graph")
        
        # Check connectivity
        if '10.1.1.102' in G and '10.1.1.25' in G:
            if nx.has_path(G, '10.1.1.102', '10.1.1.25'):
                paths = list(nx.all_simple_paths(G, '10.1.1.102', '10.1.1.25'))
                print(f"    Paths from 10.1.1.102 to 10.1.1.25: {len(paths)}")
                if paths:
                    print(f"    First path: {' -> '.join(paths[0])}")
            else:
                print(f"    No path from 10.1.1.102 to 10.1.1.25")
    
    print(f"\nAll unique nodes found: {len(all_nodes)}")
    print(f"Sample nodes: {sorted(list(all_nodes))[:10]}")
    
    # Check if our target nodes are in the overall set
    target_nodes = ['10.1.1.102', '10.1.1.25', '10.1.1.101', '10.1.1.103']
    for node in target_nodes:
        if node in all_nodes:
            print(f"✓ {node} found in overall node set")
        else:
            print(f"✗ {node} NOT found in overall node set")

if __name__ == "__main__":
    debug_node_analysis() 