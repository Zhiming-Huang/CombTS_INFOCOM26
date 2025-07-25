#!/usr/bin/env python3
"""
List files corresponding to rounds without available arms
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment

def list_files_without_arms():
    """List all files corresponding to rounds without available arms"""
    
    print("Files Corresponding to Rounds Without Available Arms")
    print("=" * 60)
    
    # Node pair
    source = "10.1.1.109"
    destination = "10.1.1.5"
    
    # Get the data directory
    data_dir = Path(__file__).parent.parent / "data" / "ucsb"
    target_folder = data_dir / "1144393236-1144450070"
    
    # Get all neighbortable files
    neighbortable_files = sorted([f for f in target_folder.glob("neighbortable-*")])
    
    print(f"Node pair: {source} → {destination}")
    print(f"Total files: {len(neighbortable_files)}")
    print()
    
    # Create environment
    env = UCSBMeshnetMemmapEnvironment(
        neighbortable_files=[str(f) for f in neighbortable_files],
        source=source,
        destination=destination,
        routes_per_minute=1
    )
    
    print(f"Environment created with {env.num_rounds} rounds")
    print()
    
    # Find rounds without available arms
    rounds_without_arms = []
    
    print("Analyzing each round...")
    for round_idx in range(env.num_rounds):
        # Get available arms for this round
        available_arms = set(env.get_available_arms_for_round(round_idx))
        
        if not available_arms:
            rounds_without_arms.append(round_idx)
    
    print(f"Analysis completed!")
    print()
    
    # Summary
    print("SUMMARY:")
    print("-" * 40)
    print(f"Total rounds: {env.num_rounds}")
    print(f"Rounds without available arms: {len(rounds_without_arms)}")
    print(f"Percentage: {(len(rounds_without_arms)/env.num_rounds)*100:.1f}%")
    print()
    
    # List all files without available arms
    print("ALL FILES WITHOUT AVAILABLE ARMS:")
    print("-" * 60)
    print("Round | File Name")
    print("-" * 60)
    
    for round_idx in rounds_without_arms:
        file_name = neighbortable_files[round_idx].name
        print(f"{round_idx:5d} | {file_name}")
    
    print("-" * 60)
    print(f"Total: {len(rounds_without_arms)} files")
    print()
    
    # Group consecutive rounds
    print("CONSECUTIVE GROUPS:")
    print("-" * 40)
    
    if rounds_without_arms:
        consecutive_groups = []
        current_group = [rounds_without_arms[0]]
        
        for i in range(1, len(rounds_without_arms)):
            if rounds_without_arms[i] == rounds_without_arms[i-1] + 1:
                current_group.append(rounds_without_arms[i])
            else:
                consecutive_groups.append(current_group)
                current_group = [rounds_without_arms[i]]
        
        consecutive_groups.append(current_group)
        
        for i, group in enumerate(consecutive_groups):
            start_file = neighbortable_files[group[0]].name
            end_file = neighbortable_files[group[-1]].name
            print(f"Group {i+1}: Rounds {group[0]}-{group[-1]} ({len(group)} rounds)")
            print(f"  Files: {start_file} to {end_file}")
            print()
    
    # Time analysis
    print("TIME ANALYSIS:")
    print("-" * 40)
    
    if rounds_without_arms:
        # Convert round indices to file timestamps
        timestamps = []
        for round_idx in rounds_without_arms:
            file_name = neighbortable_files[round_idx].name
            # Extract timestamp from filename (neighbortable-1144410396)
            timestamp = file_name.split('-')[1]
            timestamps.append(int(timestamp))
        
        timestamps = np.array(timestamps)
        
        print(f"First timestamp: {np.min(timestamps)}")
        print(f"Last timestamp: {np.max(timestamps)}")
        print(f"Time span: {np.max(timestamps) - np.min(timestamps)} seconds")
        print(f"Average gap between timestamps: {np.mean(np.diff(timestamps)):.1f} seconds")
        
        # Check if timestamps are consecutive (60 seconds apart)
        time_diffs = np.diff(timestamps)
        consecutive_timestamps = np.sum(time_diffs == 60)
        print(f"Consecutive timestamps (60s apart): {consecutive_timestamps}/{len(time_diffs)}")
    
    # File size analysis (optional)
    print()
    print("FILE SIZE ANALYSIS:")
    print("-" * 40)
    
    file_sizes = []
    for round_idx in rounds_without_arms:
        file_path = neighbortable_files[round_idx]
        file_size = file_path.stat().st_size
        file_sizes.append(file_size)
    
    if file_sizes:
        file_sizes = np.array(file_sizes)
        print(f"Average file size: {np.mean(file_sizes):.0f} bytes")
        print(f"Minimum file size: {np.min(file_sizes)} bytes")
        print(f"Maximum file size: {np.max(file_sizes)} bytes")
        
        # Check if files are empty or very small
        empty_files = np.sum(file_sizes == 0)
        small_files = np.sum(file_sizes < 100)  # Less than 100 bytes
        print(f"Empty files (0 bytes): {empty_files}")
        print(f"Small files (<100 bytes): {small_files}")

if __name__ == "__main__":
    list_files_without_arms() 