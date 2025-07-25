#!/usr/bin/env python3
"""
Analyze network connectivity to understand why there are no feasible paths
between 10.1.1.100 and 10.2.1.9 in the UCSB mesh network.
"""

import os
import sys
import numpy as np
import networkx as nx
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment

def analyze_network_connectivity():
    """Analyze network connectivity and topology."""
    
    # Target node pair
    source_node = "10.1.1.100"
    dest_node = "10.2.1.9"
    
    print(f"Analyzing network connectivity between {source_node} and {dest_node}")
    print("=" * 70)
    
    # UCSB data directory
    ucsb_data_dir = Path("data/ucsb")
    
    # Use the first trace for detailed analysis
    trace_dir = None
    for item in ucsb_data_dir.iterdir():
        if item.is_dir() and item.name.startswith("114"):
            trace_dir = item
            break
    
    if not trace_dir:
        print("No trace directories found")
        return
    
    print(f"Using trace: {trace_dir.name}")
    print()
    
    # Get all neighbortable files
    neighbortable_files = []
    for file in trace_dir.glob("neighbortable-*"):
        neighbortable_files.append(str(file))
    
    if not neighbortable_files:
        print("No neighbortable files found")
        return
    
    neighbortable_files.sort()
    print(f"Found {len(neighbortable_files)} neighbortable files")
    print()
    
    # Create environment
    env = UCSBMeshnetMemmapEnvironment(
        neighbortable_files=neighbortable_files,
        source=source_node,
        destination=dest_node,
        max_path_length=3,
        max_feasible_combinations=50
    )
    
    # Get network information
    nodes = env.get_nodes()
    print(f"Total nodes in network: {len(nodes)}")
    print(f"Source node: {source_node}")
    print(f"Destination node: {dest_node}")
    print()
    
    # Check if nodes exist
    if source_node not in nodes:
        print(f"❌ Source node {source_node} not found in network")
        print(f"   Available nodes starting with '10.1.1': {[n for n in nodes if n.startswith('10.1.1')]}")
        return
    
    if dest_node not in nodes:
        print(f"❌ Destination node {dest_node} not found in network")
        print(f"   Available nodes starting with '10.2.1': {[n for n in nodes if n.startswith('10.2.1')]}")
        return
    
    print(f"✅ Both nodes found in network")
    print()
    
    # Analyze network topology from the first neighbortable file
    print("Analyzing network topology from first neighbortable file...")
    first_file = neighbortable_files[0]
    print(f"File: {os.path.basename(first_file)}")
    
    # Build graph from neighbortable
    G = nx.Graph()
    with open(first_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            
            node = parts[0]
            G.add_node(node)
            
            # Add edges (neighbors)
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    neighbor = parts[i]
                    try:
                        ett = float(parts[i + 1])
                        if ett < 1000:  # Valid ETT
                            G.add_edge(node, neighbor)
                    except ValueError:
                        continue
    
    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print()
    
    # Check connectivity
    if not G.has_node(source_node):
        print(f"❌ Source node {source_node} not in graph")
        return
    
    if not G.has_node(dest_node):
        print(f"❌ Destination node {dest_node} not in graph")
        return
    
    print(f"✅ Both nodes in graph")
    
    # Check if nodes are connected
    try:
        shortest_path = nx.shortest_path(G, source_node, dest_node)
        print(f"✅ Nodes are connected!")
        print(f"   Shortest path: {' -> '.join(shortest_path)}")
        print(f"   Path length: {len(shortest_path) - 1} hops")
    except nx.NetworkXNoPath:
        print(f"❌ Nodes are not connected!")
        
        # Analyze connected components
        components = list(nx.connected_components(G))
        print(f"\nNetwork has {len(components)} connected components:")
        
        source_component = None
        dest_component = None
        
        for i, component in enumerate(components):
            print(f"  Component {i+1}: {len(component)} nodes")
            if source_node in component:
                source_component = i
                print(f"    Contains source node {source_node}")
            if dest_node in component:
                dest_component = i
                print(f"    Contains destination node {dest_node}")
        
        if source_component is not None and dest_component is not None:
            print(f"\nSource node is in component {source_component + 1}")
            print(f"Destination node is in component {dest_component + 1}")
            if source_component != dest_component:
                print(f"❌ Nodes are in different connected components!")
            else:
                print(f"✅ Nodes are in the same component but no path found")
        
        # Show some nodes from each component
        print(f"\nSample nodes from each component:")
        for i, component in enumerate(components[:3]):  # Show first 3 components
            sample_nodes = list(component)[:5]
            print(f"  Component {i+1}: {sample_nodes}")
    
    # Check available arms from environment
    print(f"\nChecking available arms from environment...")
    try:
        available_arms = env.get_available_arms_for_round(0)
        print(f"Available arms for round 0: {len(available_arms)}")
        
        # Get feasible combinations
        feasible_combinations = env.get_feasible_combinations(available_arms)
        print(f"Feasible combinations: {len(feasible_combinations)}")
        
        if feasible_combinations:
            print("✅ Environment found feasible combinations!")
        else:
            print("❌ Environment found no feasible combinations")
            
    except Exception as e:
        print(f"Error getting available arms: {e}")
    
    # Show some sample nodes from the network
    print(f"\nSample nodes from network:")
    sample_nodes = nodes[:10]
    print(f"  {sample_nodes}")
    
    # Show nodes by subnet
    subnets = {}
    for node in nodes:
        if '.' in node:
            subnet = '.'.join(node.split('.')[:3])
            if subnet not in subnets:
                subnets[subnet] = []
            subnets[subnet].append(node)
    
    print(f"\nNodes by subnet:")
    for subnet, subnet_nodes in sorted(subnets.items()):
        print(f"  {subnet}.*: {len(subnet_nodes)} nodes")
        if len(subnet_nodes) <= 5:
            print(f"    {subnet_nodes}")
        else:
            print(f"    {subnet_nodes[:3]}...{subnet_nodes[-2:]}")

if __name__ == "__main__":
    analyze_network_connectivity() 