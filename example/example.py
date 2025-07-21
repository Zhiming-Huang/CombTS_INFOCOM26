#!/usr/bin/env python3
"""
Simple example demonstrating the combinatorial bandit routing simulation.
"""

import sys
import os
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt

from src.simulation import RoutingSimulation
from src.utils.network_utils import (
    create_3x3_mesh_network,
    create_test_case_link_means,
    create_availability_rates
)


def main():
    """Run a simple demonstration of the simulation system."""
    print("Combinatorial Bandit Routing Simulation Demo")
    print("=" * 50)
    
    # Create network and configuration
    print("Creating 3x3 mesh network...")
    G = create_3x3_mesh_network()
    link_means = create_test_case_link_means()
    availability_rates = create_availability_rates(rate=0.9)
    
    print(f"Network has {G.number_of_nodes()} nodes and {G.number_of_edges()} links")
    print(f"Optimal path links: {set(link_id for link_id, mean in link_means.items() if mean == 0.9)}")
    print(f"Expected optimal reward: {sum(mean for mean in link_means.values() if mean == 0.9):.1f}")
    
    # Create simulation
    print("\nInitializing simulation...")
    sim = RoutingSimulation(
        network_topology=G,
        link_means=link_means,
        availability_rates=availability_rates,
        source=0,
        destination=8,
        alpha=1.0,
        beta=1.0
    )
    
    # Show network visualization
    print("\nNetwork topology:")
    sim.get_network_visualization()
    
    # Run simulation
    print("\nRunning simulation for 500 rounds...")
    results = sim.run_simulation(num_rounds=500)
    
    # Show results
    print(f"\nFinal Results:")
    print(f"Total cumulative reward: {results['cumulative_reward']:.3f}")
    print(f"Total cumulative regret: {results['cumulative_regret']:.3f}")
    print(f"Average reward per round: {results['average_reward']:.3f}")
    print(f"Average regret per round: {results['average_regret']:.3f}")
    
    # Show final arm statistics
    arm_stats = results['final_arm_statistics']
    print(f"\nFinal Arm Statistics:")
    print(f"Number of arms played: {arm_stats['played_arms']}/{arm_stats['num_arms']}")
    
    print("\nExpected rewards for each arm:")
    for arm_id in range(arm_stats['num_arms']):
        true_mean = link_means[arm_id]
        expected_reward = arm_stats['expected_rewards'].get(arm_id, "Not played")
        print(f"  Arm {arm_id}: Expected={expected_reward}, True={true_mean:.1f}")
    
    # Plot results
    print("\nGenerating plots...")
    sim.plot_results(results)
    
    # Save results
    print("\nSaving results...")
    sim.save_results(results)
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main() 