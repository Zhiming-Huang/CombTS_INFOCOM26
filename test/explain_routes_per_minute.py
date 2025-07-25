#!/usr/bin/env python3
"""
Explain routes_per_minute parameter with examples
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment

def explain_routes_per_minute():
    """Explain routes_per_minute parameter with examples"""
    
    print("Understanding routes_per_minute Parameter")
    print("=" * 60)
    
    # Node pair
    source = "10.1.1.109"
    destination = "10.1.1.5"
    
    # Get the data directory
    data_dir = Path(__file__).parent.parent / "data" / "ucsb"
    target_folder = data_dir / "1144393236-1144450070"
    
    # Get first 5 files for demonstration
    neighbortable_files = sorted([f for f in target_folder.glob("neighbortable-*")])[:5]
    
    print(f"Node pair: {source} → {destination}")
    print(f"Using first {len(neighbortable_files)} neighbortable files for demonstration")
    print()
    
    # Test different routes_per_minute values
    routes_per_minute_values = [1, 2, 4, 10]
    
    for routes_per_minute in routes_per_minute_values:
        print(f"Testing with routes_per_minute = {routes_per_minute}")
        print("-" * 40)
        
        # Create environment
        env = UCSBMeshnetMemmapEnvironment(
            neighbortable_files=[str(f) for f in neighbortable_files],
            source=source,
            destination=destination,
            routes_per_minute=routes_per_minute
        )
        
        print(f"  Number of neighbortable files: {env.num_timestamps}")
        print(f"  Routes per minute: {env.routes_per_minute}")
        print(f"  Total rounds: {env.num_rounds}")
        print(f"  Calculation: {env.num_timestamps} files × {env.routes_per_minute} routes/min = {env.num_rounds} rounds")
        print()
        
        # Show how rounds are mapped to files
        print(f"  Round to file mapping:")
        for round_idx in range(min(10, env.num_rounds)):
            file_idx = round_idx // routes_per_minute
            sub_round = round_idx % routes_per_minute
            file_name = neighbortable_files[file_idx].name
            print(f"    Round {round_idx:2d} → File {file_idx} ({file_name}) sub-round {sub_round}")
        
        print()
        
        # Show feasible combinations for first few rounds
        print(f"  Feasible combinations for first 3 rounds:")
        for round_idx in range(min(3, env.num_rounds)):
            available_arms = set(env.get_available_arms_for_round(round_idx))
            feasible_combinations = env.get_feasible_combinations(available_arms)
            print(f"    Round {round_idx}: {len(feasible_combinations)} feasible combinations")
        
        print()
        print("=" * 60)
        print()
    
    print("SUMMARY:")
    print("Meaning of routes_per_minute parameter:")
    print("1. Number of rounds per neighbortable file (timestamp)")
    print("2. Total rounds = number of neighbortable files × routes_per_minute")
    print("3. Each file will be reused routes_per_minute times, each time corresponding to a round")
    print("4. This increases the training rounds for algorithms, improving learning effectiveness")
    print("5. Each round has the same topology structure, but rewards may differ (if pre_generate_rewards=True)")

if __name__ == "__main__":
    explain_routes_per_minute() 