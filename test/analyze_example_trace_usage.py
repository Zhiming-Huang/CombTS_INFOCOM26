#!/usr/bin/env python3
"""
Analyze which trace files the example program actually uses.
"""

import os
import sys
import glob
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def analyze_example_trace_usage():
    """Analyze which trace files the example program uses."""
    
    print("Analyzing example program trace file usage...")
    print("=" * 60)
    
    # Get project root
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data' / 'ucsb'
    
    print(f"UCSB data directory: {data_dir}")
    print()
    
    # Find all trace directories
    trace_dirs = []
    for item in data_dir.iterdir():
        if item.is_dir() and item.name.startswith("114"):
            trace_dirs.append(item)
    
    print(f"Found {len(trace_dirs)} trace directories:")
    for trace_dir in sorted(trace_dirs):
        print(f"  {trace_dir.name}")
    print()
    
    # Simulate the example program's file selection logic
    neighbortable_files = []
    
    # Look for neighbortable files in subdirectories (same logic as example)
    for subdir in data_dir.iterdir():
        if subdir.is_dir():
            files = glob.glob(str(subdir / 'neighbortable-*'))
            neighbortable_files.extend(sorted(files))
    
    print(f"Total neighbortable files found: {len(neighbortable_files)}")
    
    # Calculate how many files would be used for different round counts
    routes_per_minute = 4
    
    test_rounds = [100, 1000, 10000]
    
    for num_rounds in test_rounds:
        files_needed = max(1, num_rounds // routes_per_minute)
        files_to_use = sorted(neighbortable_files)[:files_needed]
        
        print(f"\nFor {num_rounds} rounds:")
        print(f"  Files needed: {files_needed}")
        print(f"  Files to use: {len(files_to_use)}")
        
        # Analyze which trace directories these files come from
        trace_usage = {}
        for file_path in files_to_use:
            trace_dir = Path(file_path).parent.name
            if trace_dir not in trace_usage:
                trace_usage[trace_dir] = 0
            trace_usage[trace_dir] += 1
        
        print(f"  Trace directory usage:")
        for trace_dir, count in sorted(trace_usage.items()):
            print(f"    {trace_dir}: {count} files")
    
    # Show the first few files that would be used
    print(f"\nFirst 10 neighbortable files that would be used:")
    for i, file_path in enumerate(sorted(neighbortable_files)[:10]):
        trace_dir = Path(file_path).parent.name
        file_name = Path(file_path).name
        print(f"  {i+1:2d}. {trace_dir}/{file_name}")
    
    # Show which trace directory contains the first file (used for source-destination selection)
    if neighbortable_files:
        first_file = sorted(neighbortable_files)[0]
        first_trace_dir = Path(first_file).parent.name
        print(f"\nFirst file used for source-destination selection:")
        print(f"  {first_trace_dir}/{Path(first_file).name}")
        
        # Analyze the first file to see what nodes are available
        print(f"\nAnalyzing first file for node selection...")
        nodes = set()
        with open(first_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) >= 1:
                    nodes.add(parts[0])
                    # Add neighbors too
                    for i in range(1, len(parts), 2):
                        if i < len(parts):
                            nodes.add(parts[i])
        
        nodes = sorted(list(nodes))
        print(f"  Total nodes in first file: {len(nodes)}")
        print(f"  Sample nodes: {nodes[:10]}")
        
        # Show nodes by subnet
        subnets = {}
        for node in nodes:
            if '.' in node:
                subnet = '.'.join(node.split('.')[:3])
                if subnet not in subnets:
                    subnets[subnet] = []
                subnets[subnet].append(node)
        
        print(f"  Nodes by subnet:")
        for subnet, subnet_nodes in sorted(subnets.items()):
            print(f"    {subnet}.*: {len(subnet_nodes)} nodes")
            if len(subnet_nodes) <= 5:
                print(f"      {subnet_nodes}")
            else:
                print(f"      {subnet_nodes[:3]}...{subnet_nodes[-2:]}")

if __name__ == "__main__":
    analyze_example_trace_usage() 