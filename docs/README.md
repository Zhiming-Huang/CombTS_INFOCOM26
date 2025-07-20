# Sleeping Combinatorial Bandits for Network Routing

This repository implements Sleeping Combinatorial Bandits applied to network routing problems. The implementation includes two main classes: `RoutingEnvironment` for simulating the network environment and `CombTSAgent` for implementing the combinatorial Thompson Sampling strategy.

## Overview

The Sleeping Combinatorial Bandits framework addresses the challenge of learning optimal routing paths in dynamic networks where:
- Links have varying availability (sleeping arms)
- Path rewards are combinatorial (sum of edge rewards)
- The agent must learn both link qualities and adapt to changing network topology

## Key Components

### RoutingEnvironment Class
Responsible for simulating the network environment, including link availability and reward generation:

**Methods:**
- `sample_available_graph()`: Generate available subgraph (randomly sample edges based on availability)
- `get_feasible_paths()`: Find all feasible paths from source to target in the available graph
- `get_reward(path)`: Calculate path reward (sum of Bernoulli-sampled edge rewards)

### CombTSAgent Class
Implements combinatorial Thompson Sampling strategy for path selection and posterior updates:

**Methods:**
- `sample_edge_means()`: Sample reward estimates for each edge from Beta distributions
- `select_path(feasible_paths)`: Select best path based on sampled reward estimates
- `update(path, reward)`: Update Beta distribution parameters based on observed reward

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd CombTS_INFOCOM26
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from routing_environment import RoutingEnvironment
from comb_ts_agent import CombTSAgent

# Define network
edge_availability_probs = {
    (0, 1): 0.8,  # Edge 0->1 has 80% availability
    (1, 2): 0.7,  # Edge 1->2 has 70% availability
    (0, 2): 0.6,  # Edge 0->2 has 60% availability
}

edge_reward_probs = {
    (0, 1): 0.9,  # Edge 0->1 has 90% success probability
    (1, 2): 0.8,  # Edge 1->2 has 80% success probability
    (0, 2): 0.7,  # Edge 0->2 has 70% success probability
}

# Initialize environment and agent
environment = RoutingEnvironment(
    num_nodes=3,
    edge_availability_probs=edge_availability_probs,
    edge_reward_probs=edge_reward_probs,
    source=0,
    target=2
)

agent = CombTSAgent(environment)

# Run one round
available_graph = environment.sample_available_graph()
feasible_paths = environment.get_feasible_paths(available_graph)
selected_path = agent.select_path(feasible_paths)
reward = environment.get_reward(selected_path)
agent.update(selected_path, reward)
```

### Running Simulations

```bash
# Run main simulation
python main_simulation.py

# Run example usage
python example_usage.py
```

## Features

- **Dynamic Network Topology**: Handles varying link availability
- **Combinatorial Rewards**: Path rewards are sums of edge rewards
- **Thompson Sampling**: Bayesian approach for exploration-exploitation
- **Multiple Network Topologies**: Support for grid, star, and custom networks
- **Performance Analysis**: Regret calculation and learning curves
- **Visualization**: Plotting of cumulative rewards and regret

## Network Topologies

The implementation supports various network topologies:

1. **Simple Networks**: Basic 3-node networks for testing
2. **Grid Networks**: N×N grid topologies
3. **Star Networks**: Centralized topologies with hub-and-spoke structure
4. **Custom Networks**: User-defined topologies

## Algorithm Details

### Thompson Sampling for Combinatorial Bandits

1. **Prior Initialization**: Each edge starts with Beta(α=1, β=1) prior
2. **Sampling**: Sample edge means from current Beta distributions
3. **Path Selection**: Select path with highest sum of sampled edge means
4. **Update**: Distribute observed reward among path edges and update Beta parameters

### Sleeping Arms Handling

- Edges not available in current round are not considered for path selection
- Only available edges contribute to path evaluation
- Agent learns edge qualities only when edges are available

## Performance Metrics

- **Cumulative Reward**: Total reward accumulated over time
- **Regret**: Difference between optimal and achieved performance
- **Path Selection Frequency**: How often each path is selected
- **Edge Quality Estimates**: Learned vs. true edge qualities

## Examples

See `example_usage.py` for comprehensive examples including:
- Single round demonstration
- Network topology comparison
- Edge learning demonstration
- Performance analysis

## Dependencies

- `numpy`: Numerical computations
- `networkx`: Graph operations
- `matplotlib`: Visualization
- `typing-extensions`: Type hints

## License

[Add your license information here]

## Citation

If you use this implementation in your research, please cite:

```
[Add citation information here]
```