#!/usr/bin/env python3
"""
Test Fixed Environment
======================

This script tests the fixed UCSB environment to verify that it now correctly
returns the right number of feasible paths.
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

def test_fixed_environment():
    """Test the fixed environment."""
    print("Testing Fixed Environment")
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
    print(f"Using first 20 trace files for testing")
    
    # Test first 20 files
    test_files = neighbortable_files[:20]
    
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
    
    print(f"Environment created with {len(test_files)} trace files")
    print(f"Environment rounds: {env.num_rounds}")
    
    # Test each round
    discrepancies = []
    
    for round_idx in range(min(20, env.num_rounds)):
        print(f"\n--- Round {round_idx} ---")
        
        # Get environment's feasible combinations
        available_arms = set(env.get_available_arms_for_round(round_idx))
        env_feasible = env.get_feasible_combinations(available_arms)
        env_path_count = len(env_feasible) if env_feasible else 0
        
        print(f"Environment feasible paths: {env_path_count}")
        print(f"Available arms: {len(available_arms)}")
        
        # Manually verify using the corresponding trace file
        if round_idx < len(test_files):
            trace_file = test_files[round_idx]
            filename = os.path.basename(trace_file)
            
            # Parse trace file manually
            nodes, edges = parse_neighbortable_file(trace_file)
            
            # Build graph
            G = nx.Graph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)
            
            # Check connectivity manually
            if src in G and dst in G and nx.has_path(G, src, dst):
                # Find all simple paths up to 3 hops
                all_paths = list(nx.all_simple_paths(G, src, dst, cutoff=3))
                manual_path_count = len(all_paths)
                
                print(f"Manual verification ({filename}): {manual_path_count} paths")
                
                if manual_path_count >= 2:
                    print(f"  Paths: {all_paths}")
                
                # Check for discrepancy
                if env_path_count != manual_path_count:
                    discrepancy = {
                        'round': round_idx,
                        'filename': filename,
                        'env_paths': env_path_count,
                        'manual_paths': manual_path_count,
                        'env_feasible': env_feasible,
                        'manual_paths_detail': all_paths
                    }
                    discrepancies.append(discrepancy)
                    print(f"  ⚠️  DISCREPANCY FOUND!")
                else:
                    print(f"  ✓ Path counts match")
            else:
                print(f"Manual verification ({filename}): No path found")
                if env_path_count > 0:
                    discrepancy = {
                        'round': round_idx,
                        'filename': filename,
                        'env_paths': env_path_count,
                        'manual_paths': 0,
                        'env_feasible': env_feasible,
                        'manual_paths_detail': []
                    }
                    discrepancies.append(discrepancy)
                    print(f"  ⚠️  DISCREPANCY FOUND!")
                else:
                    print(f"  ✓ No paths (both agree)")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total rounds tested: {min(20, env.num_rounds)}")
    print(f"Discrepancies found: {len(discrepancies)}")
    
    if discrepancies:
        print(f"\nDiscrepancies:")
        for disc in discrepancies:
            print(f"  Round {disc['round']} ({disc['filename']}):")
            print(f"    Environment: {disc['env_paths']} paths")
            print(f"    Manual: {disc['manual_paths']} paths")
    else:
        print(f"✓ All path counts match between environment and manual verification")
        print(f"✓ Environment fix successful!")

def test_specific_rounds():
    """Test specific rounds that should have multiple paths."""
    print(f"\n{'='*80}")
    print("TESTING SPECIFIC ROUNDS WITH MULTIPLE PATHS")
    print(f"{'='*80}")
    
    # Setup
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'ucsb')
    period1_dir = os.path.join(data_dir, '1143927049-1143953729')
    
    # Get trace files
    neighbortable_files = glob.glob(os.path.join(period1_dir, 'neighbortable-*'))
    neighbortable_files = sorted(neighbortable_files)
    
    # Test node pair
    src = '10.1.1.102'
    dst = '10.1.1.25'
    
    # Known files with multiple paths
    multi_path_files = [
        'neighbortable-1143927769',
        'neighbortable-1143930529', 
        'neighbortable-1143940309'
    ]
    
    # Find these files in the list
    test_files = []
    for filename in multi_path_files:
        for file_path in neighbortable_files:
            if os.path.basename(file_path) == filename:
                test_files.append(file_path)
                break
    
    if not test_files:
        print("Could not find multi-path files")
        return
    
    print(f"Testing {len(test_files)} files with known multiple paths")
    
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
    
    # Test each round
    for round_idx in range(min(len(test_files) * 4, env.num_rounds)):
        available_arms = set(env.get_available_arms_for_round(round_idx))
        env_feasible = env.get_feasible_combinations(available_arms)
        env_path_count = len(env_feasible) if env_feasible else 0
        
        print(f"Round {round_idx}: {env_path_count} feasible paths")
        
        if env_path_count >= 2:
            print(f"  ✓ Found multiple paths as expected")
        elif env_path_count == 1:
            print(f"  ⚠️  Only 1 path found")
        else:
            print(f"  ✗ No paths found")

def main():
    """Main function."""
    test_fixed_environment()
    test_specific_rounds()

if __name__ == "__main__":
    main() 