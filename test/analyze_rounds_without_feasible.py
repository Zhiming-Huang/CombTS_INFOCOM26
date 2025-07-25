#!/usr/bin/env python3
"""
Analyze rounds without feasible combinations
"""

import os
import sys
import numpy as np
import networkx as nx
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment

def analyze_rounds_without_feasible():
    """Analyze which rounds have no feasible combinations and find reasons"""
    
    print("Analyzing Rounds Without Feasible Combinations")
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
    
    # Find rounds without feasible combinations
    rounds_without_feasible = []
    rounds_without_arms = []
    
    print("Analyzing each round...")
    for round_idx in range(env.num_rounds):
        # Get available arms for this round
        available_arms = set(env.get_available_arms_for_round(round_idx))
        
        if not available_arms:
            rounds_without_arms.append(round_idx)
            continue
        
        # Get feasible combinations for this round
        feasible_combinations = env.get_feasible_combinations(available_arms)
        
        if not feasible_combinations:
            rounds_without_feasible.append(round_idx)
    
    print(f"Analysis completed!")
    print()
    
    # Summary
    print("SUMMARY:")
    print("-" * 40)
    print(f"Total rounds: {env.num_rounds}")
    print(f"Rounds without available arms: {len(rounds_without_arms)}")
    print(f"Rounds with arms but no feasible combinations: {len(rounds_without_feasible)}")
    print(f"Total rounds without feasible combinations: {len(rounds_without_arms) + len(rounds_without_feasible)}")
    print()
    
    # Analyze rounds without available arms
    if rounds_without_arms:
        print("ROUNDS WITHOUT AVAILABLE ARMS:")
        print("-" * 40)
        print(f"Count: {len(rounds_without_arms)} rounds")
        print(f"Percentage: {(len(rounds_without_arms)/env.num_rounds)*100:.1f}%")
        print()
        
        # Show first 10 rounds without arms
        print("First 10 rounds without available arms:")
        for i, round_idx in enumerate(rounds_without_arms[:10]):
            file_name = neighbortable_files[round_idx].name
            print(f"  Round {round_idx}: {file_name}")
        
        if len(rounds_without_arms) > 10:
            print(f"  ... and {len(rounds_without_arms) - 10} more")
        print()
        
        # Check if these rounds are consecutive
        consecutive_groups = []
        current_group = [rounds_without_arms[0]]
        
        for i in range(1, len(rounds_without_arms)):
            if rounds_without_arms[i] == rounds_without_arms[i-1] + 1:
                current_group.append(rounds_without_arms[i])
            else:
                consecutive_groups.append(current_group)
                current_group = [rounds_without_arms[i]]
        
        consecutive_groups.append(current_group)
        
        print("Consecutive groups of rounds without arms:")
        for i, group in enumerate(consecutive_groups[:5]):
            print(f"  Group {i+1}: Rounds {group[0]}-{group[-1]} ({len(group)} rounds)")
        if len(consecutive_groups) > 5:
            print(f"  ... and {len(consecutive_groups) - 5} more groups")
        print()
    
    # Analyze rounds with arms but no feasible combinations
    if rounds_without_feasible:
        print("ROUNDS WITH ARMS BUT NO FEASIBLE COMBINATIONS:")
        print("-" * 50)
        print(f"Count: {len(rounds_without_feasible)} rounds")
        print(f"Percentage: {(len(rounds_without_feasible)/env.num_rounds)*100:.1f}%")
        print()
        
        # Show first 10 rounds without feasible combinations
        print("First 10 rounds with arms but no feasible combinations:")
        for i, round_idx in enumerate(rounds_without_feasible[:10]):
            file_name = neighbortable_files[round_idx].name
            available_arms = set(env.get_available_arms_for_round(round_idx))
            print(f"  Round {round_idx}: {file_name} ({len(available_arms)} available arms)")
        
        if len(rounds_without_feasible) > 10:
            print(f"  ... and {len(rounds_without_feasible) - 10} more")
        print()
        
        # Analyze why these rounds have no feasible combinations
        print("Analyzing why these rounds have no feasible combinations...")
        print()
        
        # Check a few sample rounds
        sample_rounds = rounds_without_feasible[:5]
        for round_idx in sample_rounds:
            print(f"Detailed analysis for Round {round_idx}:")
            file_name = neighbortable_files[round_idx].name
            print(f"  File: {file_name}")
            
            # Get available arms
            available_arms = set(env.get_available_arms_for_round(round_idx))
            print(f"  Available arms: {len(available_arms)}")
            
            # Check if source and destination are in the graph
            available_edges = set(env.arm_to_link[arm] for arm in available_arms)
            
            # Build graph manually to check connectivity
            G = nx.Graph()
            for edge in available_edges:
                G.add_edge(edge[0], edge[1])
            
            print(f"  Source {source} in graph: {source in G}")
            print(f"  Destination {destination} in graph: {destination in G}")
            
            if source in G and destination in G:
                try:
                    # Check if there's a path
                    shortest_path = nx.shortest_path(G, source, destination)
                    print(f"  Shortest path length: {len(shortest_path) - 1} hops")
                    print(f"  Shortest path: {' → '.join(shortest_path)}")
                    
                    # Check why environment didn't find it
                    feasible_combinations = env.get_feasible_combinations(available_arms)
                    print(f"  Environment found: {len(feasible_combinations)} feasible combinations")
                    
                except nx.NetworkXNoPath:
                    print(f"  No path exists between source and destination")
            else:
                print(f"  Source or destination not in graph")
            
            print()
    
    # Overall statistics
    print("OVERALL STATISTICS:")
    print("-" * 40)
    total_without_feasible = len(rounds_without_arms) + len(rounds_without_feasible)
    print(f"Rounds with feasible combinations: {env.num_rounds - total_without_feasible}")
    print(f"Rounds without feasible combinations: {total_without_feasible}")
    print(f"Success rate: {((env.num_rounds - total_without_feasible)/env.num_rounds)*100:.1f}%")
    
    # Time-based analysis
    print()
    print("TIME-BASED ANALYSIS:")
    print("-" * 40)
    
    # Check if rounds without feasible combinations are clustered in time
    all_problematic_rounds = sorted(rounds_without_arms + rounds_without_feasible)
    
    if all_problematic_rounds:
        # Find gaps between problematic rounds
        gaps = []
        for i in range(1, len(all_problematic_rounds)):
            gap = all_problematic_rounds[i] - all_problematic_rounds[i-1]
            gaps.append(gap)
        
        print(f"Average gap between problematic rounds: {np.mean(gaps):.1f} rounds")
        print(f"Maximum gap between problematic rounds: {np.max(gaps)} rounds")
        print(f"Minimum gap between problematic rounds: {np.min(gaps)} rounds")
        
        # Check for temporal clustering
        consecutive_count = 0
        max_consecutive = 0
        current_consecutive = 1
        
        for i in range(1, len(all_problematic_rounds)):
            if all_problematic_rounds[i] == all_problematic_rounds[i-1] + 1:
                current_consecutive += 1
            else:
                if current_consecutive > 1:
                    consecutive_count += 1
                max_consecutive = max(max_consecutive, current_consecutive)
                current_consecutive = 1
        
        if current_consecutive > 1:
            consecutive_count += 1
        max_consecutive = max(max_consecutive, current_consecutive)
        
        print(f"Number of consecutive groups: {consecutive_count}")
        print(f"Maximum consecutive problematic rounds: {max_consecutive}")

if __name__ == "__main__":
    analyze_rounds_without_feasible() 