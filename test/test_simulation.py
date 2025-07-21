#!/usr/bin/env python3
"""
Test script for the combinatorial bandit routing simulation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from src.simulation import RoutingSimulation
from src.utils.network_utils import (
    create_3x3_mesh_network,
    create_test_case_link_means,
    create_availability_rates,
    print_network_info
)


def test_basic_functionality():
    """Test basic functionality of the simulation system."""
    print("Testing basic functionality...")
    
    # Create network and configuration
    G = create_3x3_mesh_network()
    link_means = create_test_case_link_means()
    availability_rates = create_availability_rates(rate=0.9)
    
    # Print network information
    print_network_info(G, link_means)
    
    # Create simulation
    sim = RoutingSimulation(
        network_topology=G,
        link_means=link_means,
        availability_rates=availability_rates,
        source=0,
        destination=8,
        alpha=1.0,
        beta=1.0
    )
    
    # Test single round
    print("\nTesting single round...")
    round_result = sim.run_round()
    print(f"Round {round_result['round']}:")
    print(f"  Available links: {round_result['available_links']}")
    print(f"  Selected path: {round_result['selected_path']}")
    print(f"  Total reward: {round_result['total_reward']}")
    print(f"  Regret: {round_result['regret']}")
    
    return sim


def test_short_simulation():
    """Test a short simulation to verify everything works."""
    print("\n" + "="*50)
    print("Testing short simulation...")
    
    # Create network and configuration
    G = create_3x3_mesh_network()
    link_means = create_test_case_link_means()
    availability_rates = create_availability_rates(rate=0.9)
    
    # Create simulation
    sim = RoutingSimulation(
        network_topology=G,
        link_means=link_means,
        availability_rates=availability_rates,
        source=0,
        destination=8,
        alpha=1.0,
        beta=1.0
    )
    
    # Run short simulation
    results = sim.run_simulation(num_rounds=100)
    
    # Print results
    print(f"\nSimulation Results:")
    print(f"Total rounds: {results['total_rounds']}")
    print(f"Cumulative reward: {results['cumulative_reward']:.3f}")
    print(f"Cumulative regret: {results['cumulative_regret']:.3f}")
    print(f"Average reward per round: {results['average_reward']:.3f}")
    print(f"Average regret per round: {results['average_regret']:.3f}")
    
    # Plot results
    sim.plot_results(results)
    
    return sim, results


def test_network_visualization():
    """Test network visualization."""
    print("\n" + "="*50)
    print("Testing network visualization...")
    
    # Create network and configuration
    G = create_3x3_mesh_network()
    link_means = create_test_case_link_means()
    
    # Create simulation
    sim = RoutingSimulation(
        network_topology=G,
        link_means=link_means,
        source=0,
        destination=8
    )
    
    # Show network visualization
    sim.get_network_visualization()


if __name__ == "__main__":
    print("Starting tests for Combinatorial Bandit Routing Simulation")
    print("="*60)
    
    try:
        # Test basic functionality
        sim = test_basic_functionality()
        
        # Test short simulation
        sim, results = test_short_simulation()
        
        # Test network visualization
        test_network_visualization()
        
        print("\n" + "="*60)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc() 