#!/usr/bin/env python3
"""
Analyze UCSB nodes to understand node selection logic.
"""

import sys
import os
import numpy as np
from collections import defaultdict

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

def analyze_ucsb_nodes():
    """Analyze UCSB nodes to understand the selection of 10.2.1.105 -> 10.2.1.4."""
    
    # Use the folder with most traces
    ucsb_dir = "data/ucsb/1144393236-1144450070"
    
    print("UCSB Node Analysis")
    print("=" * 50)
    print(f"Analyzing folder: {ucsb_dir}")
    
    # Collect all nodes and their connections
    all_nodes = set()
    node_connections = defaultdict(set)
    node_appearances = defaultdict(int)
    
    neighbortable_files = []
    for f in os.listdir(ucsb_dir):
        if f.startswith("neighbortable-"):
            neighbortable_files.append(os.path.join(ucsb_dir, f))
    neighbortable_files.sort()
    
    print(f"Found {len(neighbortable_files)} neighbortable files")
    
    # Analyze first few files to understand node patterns
    for i, f in enumerate(neighbortable_files[:10]):  # Analyze first 10 files
        print(f"\nAnalyzing file {i+1}: {os.path.basename(f)}")
        
        with open(f, 'r') as fin:
            for line in fin:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                    
                src = parts[0]
                all_nodes.add(src)
                node_appearances[src] += 1
                
                for i in range(1, len(parts), 2):
                    if i + 1 < len(parts):
                        dst = parts[i]
                        try:
                            ett = float(parts[i+1])
                            if ett < 1000:  # Valid ETT
                                all_nodes.add(dst)
                                node_appearances[dst] += 1
                                node_connections[src].add(dst)
                                node_connections[dst].add(src)
                        except ValueError:
                            continue
    
    # Sort nodes by appearance count
    sorted_nodes = sorted(all_nodes, key=lambda x: node_appearances[x], reverse=True)
    
    print(f"\nTotal unique nodes found: {len(all_nodes)}")
    print(f"Nodes sorted by appearance count:")
    for i, node in enumerate(sorted_nodes[:20]):  # Show top 20
        print(f"  {i+1:2d}. {node}: {node_appearances[node]} appearances")
    
    # Check if 10.2.1.105 and 10.2.1.4 are in the data
    target_source = "10.2.1.105"
    target_dest = "10.2.1.4"
    
    print(f"\nChecking target nodes:")
    print(f"  {target_source}: {'Found' if target_source in all_nodes else 'NOT FOUND'}")
    print(f"  {target_dest}: {'Found' if target_dest in all_nodes else 'NOT FOUND'}")
    
    if target_source in all_nodes and target_dest in all_nodes:
        print(f"  {target_source} appears {node_appearances[target_source]} times")
        print(f"  {target_dest} appears {node_appearances[target_dest]} times")
        
        # Check if they are connected
        if target_dest in node_connections[target_source]:
            print(f"  Direct connection: {target_source} -> {target_dest}")
        else:
            print(f"  No direct connection between {target_source} and {target_dest}")
            
            # Check if there's a path between them
            print(f"  Checking for path between {target_source} and {target_dest}...")
            
            # Simple BFS to find path
            visited = set()
            queue = [(target_source, [target_source])]
            
            while queue:
                current, path = queue.pop(0)
                if current == target_dest:
                    print(f"  Path found: {' -> '.join(path)}")
                    break
                    
                if current in visited:
                    continue
                visited.add(current)
                
                for neighbor in node_connections[current]:
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))
            else:
                print(f"  No path found between {target_source} and {target_dest}")
    
    # Show the default selection logic
    print(f"\nDefault node selection logic:")
    print(f"  First node (alphabetically): {sorted(all_nodes)[0]}")
    print(f"  Last node (alphabetically): {sorted(all_nodes)[-1]}")
    
    # Check if our target nodes match the default logic
    sorted_all_nodes = sorted(all_nodes)
    if sorted_all_nodes[0] == target_source and sorted_all_nodes[-1] == target_dest:
        print(f"  ✓ Target nodes match default selection (first and last alphabetically)")
    else:
        print(f"  ✗ Target nodes do NOT match default selection")
        print(f"    Default would be: {sorted_all_nodes[0]} -> {sorted_all_nodes[-1]}")

if __name__ == "__main__":
    analyze_ucsb_nodes() 