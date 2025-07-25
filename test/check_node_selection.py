#!/usr/bin/env python3
"""
Check what node pairs are actually selected by the find_valid_source_destination function.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def check_node_selection():
    """Check what node pairs are selected by the current logic."""
    
    print("Checking node pair selection logic...")
    print("=" * 60)
    
    # Setup environment
    data_dir = Path("data/ucsb")
    
    # Find the trace directory with the most files
    trace_dirs = {}
    for subdir in data_dir.iterdir():
        if subdir.is_dir():
            files = list(subdir.glob("neighbortable-*"))
            trace_dirs[subdir.name] = len(files)
    
    selected_trace_dir = max(trace_dirs, key=trace_dirs.get)
    selected_trace_path = data_dir / selected_trace_dir
    
    # Use first 25 files
    neighbortable_files = sorted(list(selected_trace_path.glob("neighbortable-*")))[:25]
    neighbortable_files = [str(f) for f in neighbortable_files]
    
    print(f"Using trace directory: {selected_trace_dir}")
    print(f"Using {len(neighbortable_files)} files")
    print()
    
    # Simulate the node selection logic
    import networkx as nx
    
    # Parse the first neighbortable file
    first_file = neighbortable_files[0]
    print(f"Analyzing first file: {Path(first_file).name}")
    
    nodes = set()
    edges = []
    
    with open(first_file, 'r') as fin:
        for line in fin:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            src = parts[0]
            nodes.add(src)
            
            for i in range(1, len(parts), 2):
                dst = parts[i]
                try:
                    ett = float(parts[i+1])
                    if ett < 1000:  # Valid ETT
                        nodes.add(dst)
                        edges.append((src, dst))
                except Exception:
                    continue
    
    # Create NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    print(f"Graph: {len(nodes)} nodes, {len(edges)} edges")
    
    # Find connected components
    components = list(nx.connected_components(G))
    print(f"Connected components: {len(components)}")
    
    for i, component in enumerate(components):
        print(f"  Component {i+1}: {len(component)} nodes")
        component_list = sorted(list(component))
        print(f"    Sample nodes: {component_list[:5]}")
    
    # Find the largest connected component
    largest_cc = max(components, key=len)
    largest_cc_nodes = sorted(list(largest_cc))
    
    print(f"\nLargest component: {len(largest_cc_nodes)} nodes")
    print(f"All nodes: {largest_cc_nodes}")
    
    # Test different source nodes
    print(f"\nTesting different source nodes:")
    print("-" * 50)
    
    for source_idx, source in enumerate(largest_cc_nodes[:5]):  # Test first 5 nodes
        print(f"\nSource {source_idx + 1}: {source}")
        
        # Find destinations that are at least 2 hops away
        destinations = []
        for node in largest_cc_nodes:
            if node != source:
                try:
                    path_length = nx.shortest_path_length(G, source, node)
                    if path_length >= 2:
                        destinations.append((node, path_length))
                except nx.NetworkXNoPath:
                    continue
        
        # Sort by path length
        destinations.sort(key=lambda x: x[1])
        
        print(f"  Destinations (≥2 hops):")
        for dest, length in destinations[:5]:  # Show first 5
            print(f"    {dest} ({length} hops)")
        
        # Test the first destination
        if destinations:
            dest, length = destinations[0]
            print(f"  Selected destination: {dest} ({length} hops)")
            
            # Analyze path availability for this pair
            total_paths = 0
            rounds_with_paths = 0
            
            for round_idx in range(min(20, len(neighbortable_files) * 4)):  # 20 rounds
                # Build graph for this round
                G_round = nx.Graph()
                file_idx = round_idx // 4
                if file_idx < len(neighbortable_files):
                    with open(neighbortable_files[file_idx], 'r') as fin:
                        for line in fin:
                            if line.startswith('#') or not line.strip():
                                continue
                            parts = line.strip().split()
                            if len(parts) < 3:
                                continue
                            
                            src = parts[0]
                            G_round.add_node(src)
                            
                            for j in range(1, len(parts), 2):
                                if j + 1 < len(parts):
                                    dst = parts[j]
                                    try:
                                        ett = float(parts[j+1])
                                        if ett < 1000:
                                            G_round.add_edge(src, dst)
                                    except ValueError:
                                        continue
                    
                    # Check if both nodes exist and are connected
                    if G_round.has_node(source) and G_round.has_node(dest):
                        try:
                            all_paths = list(nx.all_simple_paths(G_round, source, dest, cutoff=3))
                            if all_paths:
                                rounds_with_paths += 1
                                total_paths += len(all_paths)
                        except nx.NetworkXNoPath:
                            pass
            
            if rounds_with_paths > 0:
                avg_paths = total_paths / rounds_with_paths
                print(f"  Path analysis: {rounds_with_paths}/20 rounds have paths")
                print(f"  Average paths per round: {avg_paths:.1f}")
            else:
                print(f"  Path analysis: No paths found")

if __name__ == "__main__":
    check_node_selection() 