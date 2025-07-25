#!/usr/bin/env python3
"""
Analyze feasible combinations with up to 3 hops
"""

import os
import sys
import numpy as np
import networkx as nx
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from environments.ucsb_meshnet_memmap import UCSBMeshnetMemmapEnvironment

class UCSBMeshnetMemmapEnvironment3Hop(UCSBMeshnetMemmapEnvironment):
    """Extended UCSB environment that can find up to 3-hop paths"""
    
    def get_feasible_combinations(self, available_arms: set) -> list:
        """
        Get feasible path combinations from available arms (arm indices).
        Modified to find up to 3-hop paths.
        Returns: List of sets of arm indices, each set is a path from source to destination.
        """
        if not available_arms:
            return []
        
        # Convert available arms to (u, v) edges
        available_edges = set(self.arm_to_link[arm] for arm in available_arms)
        
        # Build subgraph
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(available_edges)
        
        # Find all simple paths up to 3 hops
        try:
            paths = []
            
            # Find all simple paths with cutoff=3 (up to 3 hops)
            for path in nx.all_simple_paths(G, self.source, self.destination, cutoff=3):
                if len(path) <= 4:  # 3 hops = 4 nodes
                    paths.append(path)
            
            # Limit to reasonable number of paths to avoid computational explosion
            if len(paths) > 1000:
                print(f"Warning: Found {len(paths)} paths, limiting to 1000")
                paths = paths[:1000]
                    
        except Exception as e:
            print(f"Warning: Error finding paths: {e}")
            return []
        
        # Convert paths to arm index sets
        feasible_combinations = []
        for path in paths:
            if len(path) < 2:
                continue
            path_arms = set()
            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1])
                if edge in self.link_to_arm:
                    path_arms.add(self.link_to_arm[edge])
                elif (edge[1], edge[0]) in self.link_to_arm:
                    path_arms.add(self.link_to_arm[(edge[1], edge[0])])
            if path_arms and len(path_arms) <= self.max_combination_size:
                feasible_combinations.append(path_arms)
        
        return feasible_combinations

def analyze_3hop_feasible():
    """Analyze feasible combinations with up to 3 hops"""
    
    print("Analyzing Feasible Combinations with Up to 3 Hops")
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
    
    # Use first 20 files for testing (to limit computation time)
    test_files = neighbortable_files[:20]
    print(f"Using first {len(test_files)} files for testing")
    print()
    
    # Create environment with 3-hop capability
    env = UCSBMeshnetMemmapEnvironment3Hop(
        neighbortable_files=[str(f) for f in test_files],
        source=source,
        destination=destination,
        routes_per_minute=1
    )
    
    print(f"Environment created with {env.num_rounds} rounds")
    print()
    
    # Collect data for all rounds
    feasible_counts = []
    size_distributions = []
    
    print("Analyzing each round...")
    for round_idx in range(env.num_rounds):
        # Get available arms for this round
        available_arms = set(env.get_available_arms_for_round(round_idx))
        
        if available_arms:
            # Get feasible combinations for this round
            feasible_combinations = env.get_feasible_combinations(available_arms)
            feasible_counts.append(len(feasible_combinations))
            
            # Count by size
            size_counts = {}
            for combo in feasible_combinations:
                size = len(combo)
                size_counts[size] = size_counts.get(size, 0) + 1
            size_distributions.append(size_counts)
            
            if round_idx < 5:  # Show details for first 5 rounds
                print(f"  Round {round_idx}: {len(feasible_combinations)} feasible combinations")
                if feasible_combinations:
                    print(f"    Size breakdown: {dict(sorted(size_counts.items()))}")
                    print(f"    Sample combinations: {list(feasible_combinations[:3])}")
        else:
            feasible_counts.append(0)
            size_distributions.append({})
            if round_idx < 5:
                print(f"  Round {round_idx}: No available arms")
    
    # Calculate statistics
    feasible_counts = np.array(feasible_counts)
    
    print(f"\nFEASIBLE COMBINATIONS STATISTICS (Up to 3 hops):")
    print("-" * 50)
    print(f"Total rounds: {len(feasible_counts)}")
    print(f"Rounds with feasible combinations: {np.sum(feasible_counts > 0)}")
    print(f"Rounds without feasible combinations: {np.sum(feasible_counts == 0)}")
    print()
    
    print("Feasible Combinations Count:")
    print(f"  Total feasible combinations: {np.sum(feasible_counts)}")
    print(f"  Average per round: {np.mean(feasible_counts):.2f}")
    print(f"  Median per round: {np.median(feasible_counts):.2f}")
    print(f"  Standard deviation: {np.std(feasible_counts):.2f}")
    print(f"  Minimum per round: {np.min(feasible_counts)}")
    print(f"  Maximum per round: {np.max(feasible_counts)}")
    print()
    
    # Analyze size distribution
    print("SIZE DISTRIBUTION ANALYSIS:")
    print("-" * 50)
    
    total_by_size = {}
    for size_dist in size_distributions:
        for size, count in size_dist.items():
            total_by_size[size] = total_by_size.get(size, 0) + count
    
    for size in sorted(total_by_size.keys()):
        print(f"  {size}-hop paths: {total_by_size[size]} total combinations")
    
    print()
    
    # Compare with original 2-hop results
    print("COMPARISON WITH ORIGINAL 2-HOP RESULTS:")
    print("-" * 50)
    
    # Create original environment for comparison
    env_original = UCSBMeshnetMemmapEnvironment(
        neighbortable_files=[str(f) for f in test_files],
        source=source,
        destination=destination,
        routes_per_minute=1
    )
    
    original_counts = []
    for round_idx in range(env_original.num_rounds):
        available_arms = set(env_original.get_available_arms_for_round(round_idx))
        if available_arms:
            feasible_combinations = env_original.get_feasible_combinations(available_arms)
            original_counts.append(len(feasible_combinations))
        else:
            original_counts.append(0)
    
    original_counts = np.array(original_counts)
    
    print(f"Original 2-hop results:")
    print(f"  Total feasible combinations: {np.sum(original_counts)}")
    print(f"  Average per round: {np.mean(original_counts):.2f}")
    print()
    
    print(f"3-hop results:")
    print(f"  Total feasible combinations: {np.sum(feasible_counts)}")
    print(f"  Average per round: {np.mean(feasible_counts):.2f}")
    print()
    
    improvement = (np.sum(feasible_counts) - np.sum(original_counts)) / np.sum(original_counts) * 100
    print(f"Improvement: {improvement:.1f}% more feasible combinations with 3-hop paths")

if __name__ == "__main__":
    analyze_3hop_feasible() 