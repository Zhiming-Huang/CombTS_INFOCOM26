# Sleeping Combinatorial Bandits for Network Routing - Implementation Summary

## Overview

This implementation provides a complete solution for the Sleeping Combinatorial Bandits problem applied to network routing. The solution consists of two main classes that work together to learn optimal routing policies in stochastic networks.

## Classes Implemented

### 1. RoutingEnvironment Class

**Purpose**: Simulates the network environment with stochastic link availability and reward generation.

**Key Methods**:

- **`sample_available_graph()`**: 
  - Generates available subgraph by sampling edges based on availability probabilities
  - Uses Bernoulli sampling for each edge
  - Returns NetworkX graph representing available network at current time step

- **`get_feasible_paths(available_graph, max_paths=None)`**:
  - Finds all feasible paths from source to target in the available graph
  - These paths serve as "super arms" in the combinatorial bandit setting
  - Can limit number of paths returned for computational efficiency

- **`get_reward(path)`**:
  - Generates path reward using Bernoulli sampling for each edge
  - Each edge contributes binary (0/1) reward based on its success probability
  - Returns total path reward as sum of edge rewards

**Features**:
- Handles arbitrary network topologies using NetworkX
- Supports edge-specific availability and reward probabilities
- Validates that all edges have proper probability assignments
- Includes utility function `create_sample_network()` for testing

### 2. CombTSAgent Class

**Purpose**: Implements combinatorial Thompson Sampling strategy for path selection and learning.

**Key Methods**:

- **`sample_edge_means()`**:
  - Samples reward estimates for each edge from their Beta distributions
  - Uses current posterior parameters (α, β) for each edge
  - Returns dictionary mapping edges to sampled reward estimates

- **`select_path(feasible_paths)`**:
  - Selects best path from feasible set using Thompson Sampling
  - Computes path scores by summing sampled edge rewards along each path
  - Returns path with highest total score

- **`update(path, reward)`**:
  - Updates Beta distribution parameters based on observed path reward
  - Distributes total reward equally among all edges in the selected path
  - Increments α parameter by edge_reward, β parameter by (1 - edge_reward)
  - Maintains observation counts for each edge

**Learning Mechanism**:
- Maintains Beta distribution for each edge: Beta(α, β)
- Initial parameters: α = β = 1.0 (uniform prior)
- Updates follow Bayesian inference for Bernoulli rewards
- Estimated edge reward probability: α/(α + β)

## Key Features

### Sleeping Bandits Support
- **Dynamic Action Sets**: Each round has different feasible paths based on network availability
- **Sleeping Arms**: Edges not in available graph cannot be selected
- **Adaptive Learning**: Only observed edges get parameter updates

### Thompson Sampling Strategy
- **Exploration vs Exploitation**: Beta distributions naturally balance exploration/exploitation
- **Optimistic Selection**: High-uncertainty edges get higher sampling variance
- **Combinatorial Optimization**: Path selection considers correlations between edge rewards

### Scalability Considerations
- **Path Enumeration**: Uses NetworkX for efficient path finding
- **Computational Limits**: `max_paths` parameter controls computational complexity
- **Memory Efficiency**: Only stores parameters for edges that exist in network

## Usage Example

```python
import networkx as nx
from routing_environment import RoutingEnvironment, create_sample_network
from comb_ts_agent import CombTSAgent

# Create sample network
G, availability_probs, reward_means, source, target = create_sample_network()

# Initialize environment and agent
env = RoutingEnvironment(G, availability_probs, reward_means, source, target)
agent = CombTSAgent(list(G.edges()))

# Run simulation rounds
for round_num in range(num_rounds):
    # Sample available network
    available_graph = env.sample_available_graph()
    
    # Get feasible paths (super arms)
    feasible_paths = env.get_feasible_paths(available_graph)
    
    if feasible_paths:
        # Agent selects path using Thompson Sampling
        selected_path = agent.select_path(feasible_paths)
        
        # Observe reward
        reward = env.get_reward(selected_path)
        
        # Update agent's beliefs
        agent.update(selected_path, reward)
```

## Files Structure

- **`routing_environment.py`**: Contains RoutingEnvironment class and utilities
- **`comb_ts_agent.py`**: Contains CombTSAgent class implementation  
- **`simple_example.py`**: Basic usage demonstration
- **`test_implementation.py`**: Comprehensive tests for both classes
- **`demo.py`**: Advanced simulation with performance analysis
- **`requirements.txt`**: Required Python packages

## Dependencies

- `numpy`: Numerical computations and random sampling
- `networkx`: Graph operations and path finding
- `scipy`: Beta distribution sampling and statistics
- `matplotlib`: Visualization (for demo script)

## Theoretical Foundation

This implementation follows the theoretical framework for sleeping combinatorial bandits:

1. **Action Space**: Paths in the network (super arms)
2. **Base Arms**: Individual edges with unknown reward distributions  
3. **Sleeping Constraint**: Only edges in available graph can be used
4. **Learning Objective**: Minimize cumulative regret over time
5. **Algorithm**: Thompson Sampling with Beta-Bernoulli conjugate priors

The approach provides theoretical guarantees on regret bounds while being computationally practical for moderate-sized networks.