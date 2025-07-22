# Combinatorial Bandit Routing Simulation

This project implements a comprehensive simulation system for network routing with multiple combinatorial bandit algorithms. The system supports various bandit algorithms and provides extensive analysis capabilities with statistical confidence intervals.

## Supported Algorithms

1. **CTS-B (Combinatorial Thompson Sampling - Bandit)**: Base combinatorial Thompson sampling algorithm
2. **CombUCB**: Combinatorial Upper Confidence Bound algorithm
3. **CTS-G (Combinatorial Thompson Sampling - Gamma)**: CTS with gamma parameter for exploration control
4. **CL-SG (Combinatorial Learning - Stochastic Gradient)**: Stochastic gradient-based combinatorial learning
5. **BG-CTS (Bayesian Gaussian - Combinatorial Thompson Sampling)**: Bayesian Gaussian variant of CTS

## Key Features

- **Multiple Bandit Algorithms**: Support for 5 different combinatorial bandit algorithms
- **Real Data Driven**: Uses actual wireless mesh network topology and link quality data from Qurinet deployment
- **Improved Availability Calculation**: Multi-factor weighted calculation using signal strength, quality, channel congestion, transmission power, distance, and frequency interference
- **Gamma Parameter Analysis**: Comprehensive comparison of different gamma values (0.01, 0.1, 0.5, 1.0) for CTS-G and CL-SG
- **Statistical Analysis**: 95% confidence intervals using t-student distribution
- **Memory-Mapped Data Storage**: Efficient handling of large-scale simulations
- **Professional Plotting**: Seaborn whitegrid style with LaTeX rendering
- **Network Routing**: 4x4 wireless mesh network environment and real Qurinet wireless mesh network

## Project Structure

```
CombTS_INFOCOM26/
├── src/
│   ├── bandits/
│   │   ├── __init__.py
│   │   ├── cts_b.py                # CTS-B algorithm
│   │   ├── comb_ucb.py             # CombUCB algorithm
│   │   ├── cts_g.py                # CTS-G algorithm
│   │   ├── cl_sg.py                # CL-SG algorithm
│   │   └── bg_cts.py               # BG-CTS algorithm
│   ├── environments/
│   │   ├── __init__.py
│   │   ├── routing_environment.py  # Network routing environment
│   │   ├── simple_environment.py   # Simple test environment
│   │   ├── real_network_environment.py  # Real network environment
│   │   └── qurinet_environment.py  # Qurinet real wireless mesh network
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── network_utils.py        # Network creation utilities
│   │   └── plotting.py             # Professional plotting utilities
│   ├── __init__.py
│   └── simulation.py               # Main simulation orchestrator
├── example/                        # Example scripts and demonstrations
│   ├── README.md                   # Examples documentation
│   ├── example_routing_4x4_memmap.py  # Comprehensive routing simulation with all algorithms
│   ├── example_real_network_memmap.py  # Real network environment simulation
│   └── example_qurinet_memmap.py   # Qurinet real wireless mesh network simulation
├── test/                           # Test scripts
│   ├── test_simple_environment_memmap_cumulative_only.py  # Memory-mapped simple environment tests
│   └── test_routing_environment_4x4.py  # 4x4 routing environment tests
├── docs/                           # Documentation
│   ├── rewards_optimization_guide.md
│   ├── new_rng_guide.md
│   ├── synchronization_fix_guide.md
│   └── progress_tracking_guide.md
├── notebooks/                      # Jupyter notebooks
│   └── simulation_demo.ipynb       # Interactive demo
├── output/                         # Simulation results and plots
├── requirements.txt                # Python dependencies
├── CHANGELOG.md                   # Version history
└── README.md                      # This file
```

## Algorithm Details

### CTS-B Algorithm
- **Posterior Sampling**: Draws samples from Beta posterior distributions for each arm
- **Combinatorial Selection**: Selects feasible combinations with highest sum of posterior samples
- **Sleeping Arms Support**: Handles arms that may not be available at every round
- **Posterior Updates**: Updates Beta posterior parameters based on observed rewards

### CombUCB Algorithm
- **Upper Confidence Bound**: Uses UCB principle for exploration-exploitation balance
- **Combinatorial Optimization**: Selects combinations based on UCB values
- **Adaptive Exploration**: Automatically adjusts exploration based on uncertainty

### CTS-G Algorithm
- **Gamma Parameter**: Controls exploration-exploitation trade-off
- **Adaptive Sampling**: Adjusts sampling strategy based on gamma value
- **Multiple Configurations**: Supports different gamma values (0.01, 0.1, 0.5, 1.0)

### CL-SG Algorithm
- **Stochastic Gradient**: Uses gradient-based optimization for combinatorial selection
- **Gamma Control**: Similar gamma parameter for exploration control
- **Learning Rate Adaptation**: Automatically adjusts learning rates

### BG-CTS Algorithm
- **Bayesian Gaussian**: Uses Gaussian posterior distributions
- **Continuous Rewards**: Designed for continuous reward spaces
- **Robust Estimation**: More robust to reward distribution assumptions

## Environment Details

### 4x4 Wireless Mesh Network
- **Topology**: 4×4 grid network with 16 nodes and 24 links
- **Source-Destination**: Node 0 (top-left) to Node 15 (bottom-right)
- **Link Availability**: 0.75 (realistic for wireless mesh networks)
- **Optimal Path**: 0 → 1 → 2 → 3 → 7 → 11 → 15 (6 links)
- **Reward Structure**: Optimal path links (0.9), other links (0.8)
- **Expected Optimal Reward**: 5.4

### Qurinet Real Wireless Mesh Network
- **Data Source**: Real deployment at Quail Ridge Natural Reserve (https://github.com/cjpatton/qr)
- **Topology**: 19 nodes with 24 links from actual wireless mesh network
- **Frequency**: 2.4GHz (channels 1, 6, 11)
- **Link Quality**: Real signal strength and quality measurements
- **Improved Availability Calculation**: Multi-factor weighted calculation using:
  - Signal Strength (40%): -100dBm to -30dBm mapping
  - Quality (20%): 0-100 quality metric
  - Channel Congestion (15%): Based on channel usage analysis
  - Transmission Power (10%): 14-19 dBm power levels
  - Distance (10%): Estimated from signal strength using free space path loss
  - Frequency Interference (5%): Adjacent channel interference modeling
- **Reward Means**: Based on improved availability calculation with realistic variation
- **Network Characteristics**: Realistic wireless mesh network behavior with environmental factors

### Memory-Mapped Data Storage
- **Efficient Storage**: Uses numpy.memmap for large-scale data
- **Persistent Data**: Results preserved for post-hoc analysis
- **Memory Efficient**: Handles 10,000+ rounds × multiple runs
- **Statistical Analysis**: Supports confidence interval calculations

## Test Cases

### 1. 4x4 Wireless Mesh Network

The main test case uses a 4x4 wireless mesh network:

```
 0 -- 1 -- 2 -- 3
 |    |    |    |
 4 -- 5 -- 6 -- 7
 |    |    |    |
 8 -- 9 -- 10 -- 11
 |    |    |    |
 12 -- 13 -- 14 -- 15
```

**Network Parameters**:
- **Source**: Node 0 (top-left)
- **Destination**: Node 15 (bottom-right)
- **Optimal Path**: 0 → 1 → 2 → 3 → 7 → 11 → 15
- **Link Availability**: 0.75 (moderate wireless conditions)
- **Reward Structure**: Optimal path links (0.9), other links (0.8)
- **Expected Optimal Reward**: 5.4

### 2. Algorithm Performance Analysis

**Statistical Results** (10,000 rounds × 5 runs):
- **CL-SG (γ=0.1)**: Best performance with 41.18 ± 13.15 final regret
- **CTSB**: Good performance with 49.08 ± 8.55 final regret
- **CombUCB**: Moderate performance with 78.60 ± 4.80 final regret
- **CTS-G (γ=0.1)**: 109.74 ± 11.76 final regret
- **BG-CTS**: 402.82 ± 10.52 final regret

**Confidence Intervals**: All results include 95% confidence intervals using t-student distribution.

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

### Quick Start

Run the comprehensive routing simulation with all algorithms:

```bash
# Quick test (500 rounds × 3 runs)
python example/example_routing_4x4_memmap.py --rounds 500 --runs 3

# Full experiment (10,000 rounds × 5 runs)
python example/example_routing_4x4_memmap.py --rounds 10000 --runs 5

# Keep memory-mapped files for post-hoc analysis
python example/example_routing_4x4_memmap.py --rounds 2000 --runs 5 --keep-memmap
```

### Output

The simulation generates three professional PDF plots:

1. **`routing_algorithm_comparison.pdf`**: Comparison of all algorithms (CTS-B, CombUCB, CTS-G(γ=0.1), CL-SG(γ=0.1), BG-CTS)
2. **`routing_ctsg_gamma_comparison.pdf`**: CTS-G performance across different gamma values
3. **`routing_clsg_gamma_comparison.pdf`**: CL-SG performance across different gamma values

All plots include:
- 95% confidence intervals (t-student distribution)
- Seaborn whitegrid style
- LaTeX rendering for mathematical symbols
- Professional formatting suitable for publication

### Running Tests

```bash
# Test simple environment with memory mapping
python test/test_simple_environment_memmap_cumulative_only.py

# Test 4x4 routing environment
python test/test_routing_environment_4x4.py
```

### Running Tests

```bash
# Test routing simulation
python test/test_simulation.py

# Test simple environment and regret analysis
python test/test_simple_environment.py

# Test progress tracking
python test/test_progress_tracking.py
python test/test_progress_performance.py
```

## API Reference

### Bandit Algorithms

All algorithms follow the same interface:

```python
class BanditAlgorithm:
    def __init__(self, environment, **kwargs)
    def select_combination(self) -> Set[int]
    def update_posterior(self, played_arms: Set[int], rewards: Dict[int, float])
    def get_arm_statistics(self) -> Dict[str, Any]
```

**Available Algorithms**:
- `CTSB(environment, alpha=1.0, beta=1.0)`: Base Thompson sampling
- `CombUCB(environment, alpha=1.0)`: Upper confidence bound
- `CTSG(environment, gamma=0.1, alpha=1.0, beta=1.0)`: Gamma-controlled Thompson sampling
- `CLSG(environment, gamma=0.1, learning_rate=0.01)`: Stochastic gradient learning
- `BGCTS(environment, alpha=1.0, beta=1.0)`: Bayesian Gaussian Thompson sampling

### RoutingEnvironment Class

```python
class RoutingEnvironment:
    def __init__(self, graph: nx.Graph, source: int, destination: int,
                 link_availability_rates: Dict[Tuple[int, int], float],
                 link_reward_means: Dict[Tuple[int, int], float],
                 num_rounds: int = 10000, pre_generate_availability: bool = True,
                 pre_generate_rewards: bool = True, rng: np.random.Generator = None)
    def get_available_arms_for_round(self, round_num: int) -> Set[int]
    def get_feasible_combinations(self, available_arms: Set[int]) -> List[Set[int]]
    def get_optimal_combination(self, available_arms: Set[int]) -> Set[int]
    def get_reward_for_round(self, combination: Set[int], round_num: int) -> Dict[int, float]
    def visualize_network(self, save_path: str = None)
```

### Plotting Utilities

```python
from src.utils.plotting import (
    plot_routing_algorithm_comparison,
    plot_gamma_comparison,
    plot_all_routing_results,
    setup_plot_style
)

# Generate all three plots
plot_all_routing_results(results, output_dir="output/images")

# Custom algorithm comparison
plot_routing_algorithm_comparison(results, output_path, default_gamma=0.1)

# Gamma parameter analysis
plot_gamma_comparison(results, "CTS-G", gamma_values, output_path)
```

## Extending the System

### Adding New Bandit Algorithms

1. Create a new class in `src/bandits/`
2. Implement the standard interface:
   ```python
   def select_combination(self) -> Set[int]
   def update_posterior(self, played_arms: Set[int], rewards: Dict[int, float])
   def get_arm_statistics(self) -> Dict[str, Any]
   ```
3. Update `src/bandits/__init__.py`

### Adding New Environments

1. Create a new class in `src/environments/`
2. Implement the standard interface:
   ```python
   def get_available_arms_for_round(self, round_num: int) -> Set[int]
   def get_feasible_combinations(self, available_arms: Set[int]) -> List[Set[int]]
   def get_optimal_combination(self, available_arms: Set[int]) -> Set[int]
   def get_reward_for_round(self, combination: Set[int], round_num: int) -> Dict[int, float]
   ```
3. Update `src/environments/__init__.py`

### Custom Network Topologies

Use NetworkX to create custom network topologies and pass them to `RoutingEnvironment`.

### Adding New Plotting Functions

1. Add new functions to `src/utils/plotting.py`
2. Follow the established style (seaborn whitegrid, LaTeX rendering, confidence intervals)
3. Update `src/utils/__init__.py`

## Output Files

The simulation generates several output files:

### Plots (`output/images/`)
- `routing_algorithm_comparison.pdf`: Main algorithm comparison with confidence intervals
- `routing_ctsg_gamma_comparison.pdf`: CTS-G gamma parameter analysis
- `routing_clsg_gamma_comparison.pdf`: CL-SG gamma parameter analysis

### Data (`output/data/`)
- Memory-mapped files (`.dat`) for each algorithm and gamma configuration
- Preserved for post-hoc analysis when using `--keep-memmap` flag

### Network Visualization
- Network topology plots showing the 4x4 mesh structure
- Source-destination paths and link configurations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here] 