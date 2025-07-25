#!/usr/bin/env python3
"""
Check if node selection is consistent across multiple runs.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def check_node_selection_consistency():
    """Check if node selection is consistent across multiple runs."""
    
    print("Checking node selection consistency...")
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
    
    # Simulate the node selection logic multiple times
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
    largest_cc_nodes = list(largest_cc)
    
    print(f"\nLargest component: {len(largest_cc_nodes)} nodes")
    print(f"All nodes: {largest_cc_nodes}")
    
    # Test node selection consistency
    print(f"\nTesting node selection consistency:")
    print("-" * 50)
    
    selected_pairs = []
    
    for run in range(5):
        print(f"\nRun {run + 1}:")
        
        # Simulate the current selection logic
        source = largest_cc_nodes[0]
        
        # Find destinations that are at least 2 hops away
        candidates = []
        for node in largest_cc_nodes[1:]:
            try:
                path_length = nx.shortest_path_length(G, source, node)
                if path_length >= 2:
                    candidates.append((node, path_length))
            except nx.NetworkXNoPath:
                continue
        
        # Sort candidates by path length (prefer 2-hop paths)
        candidates.sort(key=lambda x: x[1])
        
        # Select destination (prefer 2-hop paths for moderate complexity)
        destination = None
        for node, length in candidates:
            destination = node
            break
        
        # If no distant node found, just use any other node
        if destination is None:
            destination = largest_cc_nodes[1]
        
        selected_pairs.append((source, destination))
        print(f"  Selected: {source} -> {destination}")
        
        # Show candidates for reference
        print(f"  Candidates: {candidates[:5]}")
    
    # Check consistency
    print(f"\nConsistency check:")
    print("-" * 50)
    
    unique_pairs = set(selected_pairs)
    if len(unique_pairs) == 1:
        print(f"✅ Node selection is CONSISTENT")
        print(f"  Selected pair: {selected_pairs[0]}")
    else:
        print(f"❌ Node selection is INCONSISTENT")
        print(f"  Unique pairs: {unique_pairs}")
        print(f"  All selections: {selected_pairs}")
    
    # Show why it might be inconsistent
    print(f"\nAnalysis:")
    print("-" * 50)
    
    # Check if largest_cc_nodes is sorted consistently
    print(f"Largest component nodes (first 10): {largest_cc_nodes[:10]}")
    
    # Check if the sorting is deterministic
    test_nodes = list(largest_cc_nodes)
    sorted_nodes = sorted(test_nodes)
    print(f"Sorted nodes (first 10): {sorted_nodes[:10]}")
    
    if test_nodes == sorted_nodes:
        print(f"✅ Nodes are already sorted")
    else:
        print(f"❌ Nodes are not sorted consistently")
        print(f"  Original order: {test_nodes[:10]}")
        print(f"  Sorted order: {sorted_nodes[:10]}")

if __name__ == "__main__":
    check_node_selection_consistency() 