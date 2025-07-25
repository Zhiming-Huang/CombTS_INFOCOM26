#!/usr/bin/env python3
"""
Quick analysis to find the best node pair based on connection statistics.
"""

import sys
import os
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

def quick_node_pair_analysis():
    """Quick analysis based on connection statistics."""
    
    # Use the folder with most traces
    ucsb_dir = "data/ucsb/1144393236-1144450070"
    
    print("Quick Node Pair Analysis")
    print("=" * 40)
    print(f"Analyzing folder: {ucsb_dir}")
    
    # Collect all neighbortable files
    neighbortable_files = []
    for f in os.listdir(ucsb_dir):
        if f.startswith("neighbortable-"):
            neighbortable_files.append(os.path.join(ucsb_dir, f))
    neighbortable_files.sort()
    
    print(f"Found {len(neighbortable_files)} neighbortable files")
    
    # Use a small sample for quick analysis
    sample_files = neighbortable_files[::100]  # Every 100th file
    print(f"Using {len(sample_files)} sample files for quick analysis")
    
    # Collect all unique nodes
    print("Collecting nodes...")
    all_nodes = set()
    for f in neighbortable_files[:3]:  # Just first 3 files
        with open(f, 'r') as fin:
            for line in fin:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                src = parts[0]
                all_nodes.add(src)
                for i in range(1, len(parts), 2):
                    if i + 1 < len(parts):
                        dst = parts[i]
                        try:
                            ett = float(parts[i+1])
                            if ett < 1000:
                                all_nodes.add(dst)
                        except ValueError:
                            continue
    
    all_nodes = sorted(all_nodes)
    print(f"Total unique nodes: {len(all_nodes)}")
    
    # Analyze connectivity patterns
    node_connectivity = defaultdict(int)  # node -> total connections
    node_pair_connectivity = defaultdict(int)  # (node1, node2) -> connection count
    subnet_nodes = defaultdict(list)
    
    # Group nodes by subnet
    for node in all_nodes:
        subnet = '.'.join(node.split('.')[:3])
        subnet_nodes[subnet].append(node)
    
    print(f"Subnets: {list(subnet_nodes.keys())}")
    for subnet, nodes in subnet_nodes.items():
        print(f"  {subnet}: {len(nodes)} nodes")
    
    # Analyze connectivity across sample files
    print("Analyzing connectivity patterns...")
    for f in tqdm(sample_files, desc="Analyzing files"):
        # Build graph
        G = nx.Graph()
        with open(f, 'r') as fin:
            for line in fin:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                src = parts[0]
                for i in range(1, len(parts), 2):
                    if i + 1 < len(parts):
                        dst = parts[i]
                        try:
                            ett = float(parts[i+1])
                            if ett < 1000:
                                G.add_edge(src, dst)
                                node_connectivity[src] += 1
                                node_connectivity[dst] += 1
                        except ValueError:
                            continue
        
        # Find connected components
        components = list(nx.connected_components(G))
        
        # Analyze each component
        for component in components:
            component_nodes = list(component)
            if len(component_nodes) < 2:
                continue
            
            # Count connections between nodes in this component
            for i, node1 in enumerate(component_nodes):
                for j, node2 in enumerate(component_nodes[i+1:], i+1):
                    if G.has_edge(node1, node2):
                        pair = tuple(sorted([node1, node2]))
                        node_pair_connectivity[pair] += 1
    
    print(f"Analysis completed. Found {len(node_pair_connectivity)} connected node pairs.")
    
    # Find the most connected nodes
    print("\nMost Connected Nodes:")
    sorted_nodes = sorted(node_connectivity.items(), key=lambda x: x[1], reverse=True)
    for i, (node, connections) in enumerate(sorted_nodes[:10]):
        print(f"{i+1:2d}. {node}: {connections} connections")
    
    # Find the most connected node pairs
    print("\nMost Connected Node Pairs:")
    sorted_pairs = sorted(node_pair_connectivity.items(), key=lambda x: x[1], reverse=True)
    
    print(f"{'Rank':<4} {'Node Pair':<25} {'Connections':<12} {'Subnet':<10}")
    print("-" * 55)
    
    for i, (pair, connections) in enumerate(sorted_pairs[:15]):
        pair_str = f"{pair[0]} → {pair[1]}"
        subnet1 = '.'.join(pair[0].split('.')[:3])
        subnet2 = '.'.join(pair[1].split('.')[:3])
        subnet_info = f"{subnet1}-{subnet2}" if subnet1 != subnet2 else subnet1
        print(f"{i+1:<4} {pair_str:<25} {connections:<12} {subnet_info:<10}")
    
    # Find the best cross-subnet pair (most connections between different subnets)
    cross_subnet_pairs = []
    for pair, connections in sorted_pairs:
        subnet1 = '.'.join(pair[0].split('.')[:3])
        subnet2 = '.'.join(pair[1].split('.')[:3])
        if subnet1 != subnet2:
            cross_subnet_pairs.append((pair, connections, subnet1, subnet2))
    
    if cross_subnet_pairs:
        print(f"\nBest Cross-Subnet Node Pairs:")
        print(f"{'Rank':<4} {'Node Pair':<25} {'Connections':<12} {'Subnets':<15}")
        print("-" * 60)
        
        for i, (pair, connections, subnet1, subnet2) in enumerate(cross_subnet_pairs[:10]):
            pair_str = f"{pair[0]} → {pair[1]}"
            subnet_info = f"{subnet1}-{subnet2}"
            print(f"{i+1:<4} {pair_str:<25} {connections:<12} {subnet_info:<15}")
        
        # Recommend the best cross-subnet pair
        best_cross_pair = cross_subnet_pairs[0]
        print(f"\n" + "=" * 50)
        print(f"RECOMMENDED CROSS-SUBNET PAIR:")
        print(f"Source: {best_cross_pair[0][0]}")
        print(f"Destination: {best_cross_pair[0][1]}")
        print(f"Connections: {best_cross_pair[1]}")
        print(f"Subnets: {best_cross_pair[2]} → {best_cross_pair[3]}")
        print("=" * 50)
        
        return best_cross_pair[0][0], best_cross_pair[0][1]
    
    # If no cross-subnet pairs, recommend the most connected pair
    if sorted_pairs:
        best_pair = sorted_pairs[0]
        print(f"\n" + "=" * 50)
        print(f"RECOMMENDED NODE PAIR:")
        print(f"Source: {best_pair[0][0]}")
        print(f"Destination: {best_pair[0][1]}")
        print(f"Connections: {best_pair[1]}")
        print("=" * 50)
        
        return best_pair[0][0], best_pair[0][1]
    
    return None, None

def test_recommended_pair():
    """Test the recommended node pair."""
    source, destination = quick_node_pair_analysis()
    
    if source and destination:
        print(f"\nTesting recommended pair: {source} → {destination}")
        
        # Test with a few files
        ucsb_dir = "data/ucsb/1144393236-1144450070"
        neighbortable_files = []
        for f in os.listdir(ucsb_dir):
            if f.startswith("neighbortable-"):
                neighbortable_files.append(os.path.join(ucsb_dir, f))
        neighbortable_files.sort()
        
        # Test first 5 files
        test_files = neighbortable_files[:5]
        
        for i, f in enumerate(tqdm(test_files, desc="Testing files")):
            # Build graph
            G = nx.Graph()
            with open(f, 'r') as fin:
                for line in fin:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue
                    src = parts[0]
                    for j in range(1, len(parts), 2):
                        if j + 1 < len(parts):
                            dst = parts[j]
                            try:
                                ett = float(parts[j+1])
                                if ett < 1000:
                                    G.add_edge(src, dst)
                            except ValueError:
                                continue
            
            try:
                # Check if path exists
                if nx.has_path(G, source, destination):
                    shortest_path = nx.shortest_path(G, source, destination)
                    all_paths = list(nx.all_simple_paths(G, source, destination, cutoff=6))
                    print(f"\nFile {i+1}: {os.path.basename(f)}")
                    print(f"  Paths found: {len(all_paths)}")
                    print(f"  Shortest path: {' → '.join(shortest_path)} ({len(shortest_path)-1} hops)")
                else:
                    print(f"\nFile {i+1}: {os.path.basename(f)} - No path found")
            except Exception as e:
                print(f"\nFile {i+1}: {os.path.basename(f)} - Error: {e}")

if __name__ == "__main__":
    test_recommended_pair() 