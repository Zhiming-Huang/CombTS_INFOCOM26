# Combinatorial Bandit Routing Simulation

This project implements a simulation system for network routing with combinatorial bandits with sleeping arms. The system consists of two main components:

1. **CombTS (Combinatorial Thompson Sampling)**: A combinatorial bandit algorithm that can handle sleeping arms (arms that may not be available at every round)
2. **RoutingEnvironment**: A network routing environment that simulates link availability and generates rewards

## Project Structure

```
CombTS_INFOCOM26/
├── src/
│   ├── bandits/
│   │   ├── __init__.py
│   │   └── comb_ts.py              # Combinatorial Thompson Sampling algorithm
│   ├── environments/
│   │   ├── __init__.py
│   │   └── routing_environment.py  # Network routing environment
│   ├── utils/
│   │   ├── __init__.py
│   │   └── network_utils.py        # Network creation utilities
│   ├── __init__.py
│   └── simulation.py               # Main simulation orchestrator
├── test/
│   └── test_simulation.py          # Test script
├── notebooks/
│   └── simulation_demo.ipynb       # Jupyter notebook demo
├── output/                         # Simulation results and plots
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Key Features

### CombTS Algorithm
- **Environment-Based Design**: Takes an environment instance that provides available arms and feasible combinations
- **Posterior Sampling**: Draws samples from Beta posterior distributions for each arm
- **Combinatorial Selection**: Selects feasible combinations with highest sum of posterior samples
- **Sleeping Arms Support**: Handles arms that may not be available at every round
- **Posterior Updates**: Updates Beta posterior parameters based on observed rewards

### Routing Environment
- **Network Topology**: Supports arbitrary NetworkX graphs
- **Link Availability**: Simulates link failures with configurable availability rates
- **Path Finding**: Finds all feasible paths between source and destination
- **Reward Generation**: Generates Bernoulli rewards based on link-specific means

### Simple Environment
- **Test Environment**: Simple environment for testing CombTS algorithm
- **Configurable Arms**: 10 arms with 3 optimal (Bernoulli(0.9)) and 7 suboptimal (Bernoulli(0.1))
- **Sleeping Arms**: Each arm has 0.5 availability rate
- **Combinatorial Constraints**: Maximum combination size of 3 arms

### Simulation System
- **Modular Design**: CombTS and environment are completely separate
- **Comprehensive Tracking**: Records all simulation data for analysis
- **Visualization**: Built-in plotting and network visualization
- **Results Export**: Saves results as JSON and plots as PNG

## Test Cases

### 1. 3x3 Mesh Network

The system includes a test case with a 3x3 mesh network:

```
0 -- 1 -- 2
|    |    |
3 -- 4 -- 5
|    |    |
6 -- 7 -- 8
```

**Optimal Path**: 0 → 1 → 2 → 5 → 8 (links 0, 2, 4, 9)
- Links in optimal path: Bernoulli(0.9)
- Other links: Bernoulli(0.8)
- Expected optimal reward: 3.6

### 2. Simple Environment

A simple test environment for verifying CombTS algorithm performance:

- **10 arms**: 3 optimal arms with Bernoulli(0.9), 7 suboptimal arms with Bernoulli(0.1)
- **Sleeping arms**: Each arm has 0.5 availability rate
- **Combinatorial constraints**: Maximum combination size of 3
- **Expected optimal reward**: 2.7 (when all 3 optimal arms are available)

**Regret Analysis**: The algorithm demonstrates sublinear regret growth, confirming theoretical guarantees.

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

### Quick Start with Jupyter Notebook

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open `notebooks/simulation_demo.ipynb`

3. Run all cells to see the complete demonstration

### Python Script Usage

```python
import sys
import os
sys.path.append('.')

from src.simulation import RoutingSimulation
from src.utils.network_utils import (
    create_3x3_mesh_network,
    create_test_case_link_means,
    create_availability_rates
)

# Create network and configuration
G = create_3x3_mesh_network()
link_means = create_test_case_link_means()
availability_rates = create_availability_rates(rate=0.8)

# Create and run simulation
sim = RoutingSimulation(
    network_topology=G,
    link_means=link_means,
    availability_rates=availability_rates,
    source=0,
    destination=8
)

# Run simulation
results = sim.run_simulation(num_rounds=1000)

# Plot results
sim.plot_results(results)

# Save results
sim.save_results(results)
```

### Running Tests

```bash
# Test routing simulation
python test/test_simulation.py

# Test simple environment and regret analysis
python test/test_simple_environment.py
python test/test_regret_analysis.py

# Debug simple environment
python test/test_simple_debug.py
```

## API Reference

### CombTS Class

```python
class CombTS:
    def __init__(self, environment, alpha: float = 1.0, beta: float = 1.0)
    def select_combination(self) -> Set[int]
    def update_posterior(self, played_arms: Set[int], rewards: Dict[int, float])
    def get_arm_statistics(self) -> Dict[str, Any]
```

### RoutingEnvironment Class

```python
class RoutingEnvironment:
    def __init__(self, network_topology: nx.Graph, link_means: Dict[int, float],
                 availability_rates: Optional[Dict[int, float]] = None)
    def sample_available_links(self) -> Set[int]
    def get_feasible_paths(self, source: int, destination: int, 
                          available_links: Set[int]) -> List[Set[int]]
    def generate_reward(self, link_id: int) -> float
    def generate_path_reward(self, path: Set[int]) -> Dict[int, float]
```

### SimpleEnvironment Class

```python
class SimpleEnvironment:
    def __init__(self, num_arms: int = 10, num_optimal: int = 3, 
                 optimal_mean: float = 0.9, suboptimal_mean: float = 0.1,
                 availability_rate: float = 0.5, max_combination_size: int = 3)
    def get_available_arms(self) -> Set[int]
    def get_feasible_combinations(self, available_arms: Set[int]) -> List[Set[int]]
    def generate_reward(self, arm: int) -> float
    def generate_combination_reward(self, combination: Set[int]) -> Dict[int, float]
    def get_optimal_combination(self, available_arms: Set[int]) -> Set[int]
```

### RoutingSimulation Class

```python
class RoutingSimulation:
    def __init__(self, network_topology: nx.Graph, link_means: Dict[int, float],
                 availability_rates: Dict[int, float] = None, 
                 source: int = 0, destination: int = 8,
                 alpha: float = 1.0, beta: float = 1.0)
    def run_round(self) -> Dict[str, Any]
    def run_simulation(self, num_rounds: int) -> Dict[str, Any]
    def plot_results(self, results: Dict[str, Any], save_path: str = None)
    def save_results(self, results: Dict[str, Any], output_dir: str = "output")
    def get_network_visualization(self, save_path: str = None)
```

## Extending the System

### Adding New Bandit Algorithms

1. Create a new class in `src/bandits/`
2. Implement the same interface as `CombTS`
3. Update `src/bandits/__init__.py`

### Adding New Environments

1. Create a new class in `src/environments/`
2. Implement the same interface as `RoutingEnvironment`
3. Update `src/environments/__init__.py`

### Custom Network Topologies

Use the utility functions in `src/utils/network_utils.py` as templates for creating new network topologies.

## Output Files

The simulation generates several output files in the `output/` directory:

- `simulation_results_YYYYMMDD_HHMMSS.json`: Complete simulation results
- `simulation_plot_YYYYMMDD_HHMMSS.png`: Visualization of results
- `network_visualization_YYYYMMDD_HHMMSS.png`: Network topology visualization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here] 