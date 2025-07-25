#!/usr/bin/env python3
"""
Check 3-hop paths without loops for UCSB meshnet
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment

def check_3hop_paths_no_loops():
    """Check 3-hop paths without loops for the current node pair"""
    
    # Current best node pair
    source = "10.1.1.109"
    destination = "10.1.1.5"
    
    print(f"Checking 3-hop paths without loops for {source} → {destination}")
    print("=" * 60)
    
    # Get the data directory
    data_dir = Path(__file__).parent.parent / "data" / "ucsb"
    
    # Find the folder with most files (1144393236-1144450070)
    target_folder = data_dir / "1144393236-1144450070"
    
    if not target_folder.exists():
        print(f"Error: Target folder {target_folder} not found")
        return
    
    # Get all neighbortable files
    neighbortable_files = sorted([f for f in target_folder.glob("neighbortable-*")])
    
    print(f"Found {len(neighbortable_files)} neighbortable files")
    
    # Test on first 10 files for quick analysis
    test_files = neighbortable_files[:10]
    
    total_paths = 0
    total_connections = 0
    
    for i, file_path in enumerate(test_files):
        print(f"\nProcessing file {i+1}/{len(test_files)}: {file_path.name}")
        
        try:
            # Create environment
            env = UCSBMeshnetMemmapEnvironment(
                neighbortable_files=[str(file_path)],
                source=source,
                destination=destination,
                routes_per_minute=1
            )
            
            # Get all feasible paths for round 0
            feasible_paths = env.get_feasible_combinations(set(range(env.num_rounds)))
            
            # Filter for 3-hop paths (arm sets with 3 arms)
            three_hop_paths = []
            for arm_set in feasible_paths:
                if len(arm_set) == 3:  # 3-hop paths have 3 arms
                    three_hop_paths.append(arm_set)
            
            print(f"  Total feasible paths: {len(feasible_paths)}")
            print(f"  3-hop paths: {len(three_hop_paths)}")
            
            if three_hop_paths:
                print(f"  Sample 3-hop paths (arm sets):")
                for j, arm_set in enumerate(three_hop_paths[:5]):  # Show first 5
                    print(f"    {j+1}: {sorted(arm_set)}")
                if len(three_hop_paths) > 5:
                    print(f"    ... and {len(three_hop_paths) - 5} more")
            
            total_paths += len(three_hop_paths)
            total_connections += 1
            
        except Exception as e:
            print(f"  Error processing file: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Files with connections: {total_connections}/{len(test_files)}")
    print(f"Average 3-hop paths without loops per file: {total_paths / total_connections:.1f}")
    print(f"Total 3-hop paths without loops: {total_paths}")
    
    # Estimate for all files
    if total_connections > 0:
        estimated_total = (total_paths / total_connections) * len(neighbortable_files)
        print(f"Estimated total 3-hop paths without loops (all {len(neighbortable_files)} files): {estimated_total:,.0f}")

if __name__ == "__main__":
    check_3hop_paths_no_loops() 