# Examples

This directory contains example scripts demonstrating how to use the combinatorial bandit algorithms.

## Available Examples

### Algorithm Comparison Example

**File:** `example_algorithm_comparison.py`

This example demonstrates how to compare multiple combinatorial bandit algorithms using memory-mapped arrays for efficient large-scale simulations.

**Features:**
- Memory-efficient simulation using `numpy.memmap`
- Support for multiple algorithms: CTS-B, CombUCB, CTS-G, CL-SG, BG-CTS
- Confidence interval analysis
- Regret growth analysis
- PDF vector graphics output
- Progress tracking with different levels

**Usage:**
```bash
# Run with default settings
python example_algorithm_comparison.py

# Run with custom random seed
python example_algorithm_comparison.py --main_seed 42
```

**Configuration:**
- Rounds: 10,000 (configurable)
- Runs: 5 (configurable)
- Progress level: "normal" (options: "minimal", "normal", "detailed")
- Algorithms: CTS-B, CombUCB, CTS-G, CL-SG, BG-CTS

**Output:**
- Console output with regret analysis and algorithm comparison
- PDF plot saved to `output/images/algorithm_comparison.pdf`
- Memory-mapped data files (automatically cleaned up)

**Key Features:**
1. **Memory Efficiency**: Uses memory-mapped arrays to handle large simulations without loading everything into RAM
2. **Fair Comparison**: All algorithms use the same environment and random seeds for fair comparison
3. **Statistical Analysis**: Provides confidence intervals and regret growth analysis
4. **Professional Plotting**: Generates publication-quality PDF plots with proper formatting
5. **Progress Tracking**: Multiple levels of progress tracking for different use cases

### Routing Environment Example

**File:** `example_routing_environment.py`

This example demonstrates how to use the RoutingEnvironment for network routing problems. It creates a 3x3 mesh network where each link is modeled as an arm, and the goal is to find the optimal path from source to destination.

**Features:**
- Network topology modeling using NetworkX
- Link availability and reward modeling
- Path finding and optimization
- Multiple algorithm comparison
- Network visualization

**Usage:**
```bash
# Run the routing example
python example_routing_environment.py
```

**Configuration:**
- Network: 3x3 mesh topology
- Source: Node 0 (top-left corner)
- Destination: Node 8 (bottom-right corner)
- Link availability: 0.8 for all links
- Optimal path reward: 0.9 (path: 0→1→2→5→8)
- Suboptimal path reward: 0.8
- Rounds: 10,000, Runs: 5

**Output:**
- Network topology visualization
- Console output with routing analysis
- PDF plot saved to `output/images/routing_environment_example.pdf`

**Key Features:**
1. **Network Modeling**: Uses NetworkX for realistic network topology modeling
2. **Path Optimization**: Automatically finds optimal paths from source to destination
3. **Link Modeling**: Each network link is modeled as an arm with availability and reward
4. **Visualization**: Interactive network visualization with optimal path highlighting
5. **Algorithm Comparison**: Compares multiple algorithms on the routing problem

### 4x4 Mesh Network Example (High Availability)

**File:** `example_routing_environment_4x4.py`

This example demonstrates routing with a 4x4 mesh network topology using high link availability (0.9) representing good wireless network conditions.

**Features:**
- 4x4 mesh network topology (16 nodes, 24 links)
- High link availability (0.9) for good network conditions
- Link availability and reward modeling
- Path finding and optimization
- Multiple algorithm comparison
- Network visualization

**Usage:**
```bash
# Run with default high availability (0.9)
python example_routing_environment_4x4.py

# Run with custom availability rate
python example_routing_environment_4x4.py --availability 0.8
python example_routing_environment_4x4.py --availability 0.6

# Run with custom parameters
python example_routing_environment_4x4.py --availability 0.7 --rounds 5000 --runs 3
```

**Configuration:**
- Network: 4x4 mesh topology
- Source: Node 0 (top-left corner)
- Destination: Node 15 (bottom-right corner)
- Link availability: 0.9 for all links (high availability)
- Optimal path reward: 0.9 (path: 0→1→2→3→7→11→15)
- Suboptimal path reward: 0.8
- Rounds: 10,000, Runs: 5

**Output:**
- Network topology visualization
- Console output with routing analysis
- PDF plot saved to `output/images/routing_environment_4x4_availability_{rate}.pdf`
- Dynamic filename based on availability rate (e.g., `availability_0_8.pdf` for 0.8 rate)

**Key Features:**
1. **Larger Network**: 4x4 mesh provides more complex routing scenarios
2. **Configurable Availability**: Easy to test different network conditions (0.0-1.0)
3. **Longer Paths**: Optimal path has 6 hops, testing algorithm scalability
4. **Dynamic Analysis**: Compare algorithm performance across different availability rates
5. **Performance Comparison**: Shows how algorithms scale with network size and conditions
6. **Command Line Interface**: Easy parameter configuration via command line arguments

**Example Output:**
```
Algorithm Comparison Example with Memory-Mapped Simulation
======================================================================
Configuration:
  Rounds: 10000
  Runs: 5
  Progress level: normal
  Algorithms: CTSB, CombUCB, CTS-G, CL-SG, BG-CTS
  Main seed: 15

Running 5 simulations with 10000 rounds each...
Algorithms: CTSB, CombUCB, CTS-G, CL-SG, BG-CTS
Progress level: normal
Created memory-mapped arrays: 5 runs × 10000 rounds × 5 algorithms
Storage location: /path/to/output/data
Memory usage: 0.38 MB per array

Regret Growth Analysis (with 95% confidence intervals):
======================================================================
Round   100: CTSB:    45.23 ±   2.15 (rate: 4.5230) CombUCB:    52.18 ±   3.21 (rate: 5.2180) ...
Round  1000: CTSB:   142.67 ±   8.45 (rate: 4.5110) CombUCB:   165.23 ±  12.34 (rate: 5.2250) ...

Algorithm Comparison Results:
==================================================
CTSB final regret: 142.67 ± 8.45
CombUCB final regret: 165.23 ± 12.34
Winner: CTSB (improvement: 13.7%)

Algorithm comparison plot saved to: /path/to/output/images/algorithm_comparison.pdf
Example completed successfully!
```

## Environment Setup

Make sure you have all required dependencies installed:

```bash
pip install -r ../requirements.txt
```

## Running Examples

All examples can be run from the project root directory:

```bash
# From project root
python example/example_algorithm_comparison.py

# Or from example directory
cd example
python example_algorithm_comparison.py
```

## Customization

You can modify the example scripts to:
- Change the number of rounds and runs
- Add or remove algorithms
- Adjust environment parameters
- Modify plotting styles
- Add new analysis functions

## Troubleshooting

1. **Import Errors**: Make sure you're running from the project root or that the Python path is set correctly
2. **Memory Issues**: Reduce the number of rounds or runs for memory-constrained systems
3. **LaTeX Errors**: The plotting will automatically fall back to non-LaTeX rendering if LaTeX is not available
4. **File Permission Errors**: Ensure the output directories have write permissions 