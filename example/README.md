# Examples

This directory contains example scripts demonstrating how to use the combinatorial bandit algorithms and environments.

## Routing Environment with 4x4 Mesh Network (Memory-Mapped)

**File**: `example_routing_4x4_memmap.py`

This example demonstrates routing with a 4x4 mesh network topology using memory mapping for efficient data storage. It tests all algorithms including multiple gamma values for CTS-G and CL-SG algorithms.

**Features:**
- 4x4 mesh network topology modeling using NetworkX
- Memory-mapped data storage for efficient large-scale simulations
- Multiple gamma values for CTS-G and CL-SG algorithms (γ = 0.01, 0.1, 0.5, 1.0)
- All algorithm comparison (CTSB, CombUCB, CTS-G, CL-SG, BG-CTS)
- Path finding and optimization
- Network visualization

**Network Parameters:**
- Network: 4x4 mesh topology
- Source: Node 0 (top-left corner)
- Destination: Node 15 (bottom-right corner)
- Link availability: 0.75 (moderate wireless conditions)
- Optimal path reward: 0.9 (path: 0→1→2→3→7→11→15)
- Suboptimal path reward: 0.8
- Rounds: 10,000, Runs: 5

**Usage:**
```bash
# Run with default parameters
python example_routing_4x4_memmap.py

# Run with custom parameters
python example_routing_4x4_memmap.py --rounds 5000 --runs 3

# Keep memory-mapped files for later analysis
python example_routing_4x4_memmap.py --keep-memmap
```

**Output:**
- Console output with routing analysis
- Three PDF plots:
  1. `routing_algorithm_comparison.pdf` - Algorithm comparison (CTSB, CombUCB, CTS-G γ=0.1, CL-SG γ=0.1, BG-CTS)
  2. `routing_ctsg_gamma_comparison.pdf` - CTS-G algorithm with different gamma values
  3. `routing_clsg_gamma_comparison.pdf` - CL-SG algorithm with different gamma values
- Memory-mapped data files (if --keep-memmap is used)

**Key Features:**
1. **Memory Mapping**: Efficient data storage for large-scale simulations
2. **Multiple Gamma Values**: Tests CTS-G and CL-SG with γ = 0.01, 0.1, 0.5, 1.0
3. **Complete Algorithm Suite**: Tests all available algorithms
4. **Wireless Network Modeling**: Realistic link availability rates for mesh networks
5. **Comprehensive Visualization**: Three different plots for different analysis perspectives
6. **Statistical Analysis**: Confidence intervals and performance comparisons

**Example Output:**
```
Routing Environment Example with 4x4 Mesh Network (Memory-Mapped)
================================================================================
Network topology: 4x4 mesh
Source: 0, Destination: 15
Number of links (arms): 24
Number of nodes: 16
Max combination size: 6
Link availability rate: 0.75 (moderate wireless conditions)
Optimal path reward: 0.9, Suboptimal path reward: 0.8

Running routing simulation...
Running 5 simulations with 10000 rounds each...
Gamma values for CTS-G and CL-SG: [0.01, 0.1, 0.5, 1.0]
Created memory-mapped arrays: 5 runs × 10000 rounds × 11 algorithms
Memory usage: 0.38 MB per array

Routing Simulation Results (4x4 Mesh Network):
================================================================================

Base Algorithms:
----------------------------------------
CTSB:
  Final regret: 95.14 ± 29.47
  95% CI: [58.55, 131.73]

CombUCB:
  Final regret: 340.42 ± 26.46
  95% CI: [307.57, 373.27]

BG-CTS:
  Final regret: 847.44 ± 26.76
  95% CI: [814.21, 880.67]

Gamma Algorithms (γ=0.1):
----------------------------------------
CTS-G (γ=0.1):
  Final regret: 217.48 ± 27.49
  95% CI: [183.35, 251.61]

CL-SG (γ=0.1):
  Final regret: 123.12 ± 13.16
  95% CI: [106.78, 139.46]

Algorithm Comparison:
----------------------------------------
CombUCB vs CTSB: 72.1% improvement
BG-CTS vs CTSB: 88.8% improvement

Algorithm comparison plot saved to: output/images/routing_algorithm_comparison.pdf
CTS-G gamma comparison plot saved to: output/images/routing_ctsg_gamma_comparison.pdf
CL-SG gamma comparison plot saved to: output/images/routing_clsg_gamma_comparison.pdf

4x4 routing environment example completed!
```

## Real Data Driven Example: Qurinet Wireless Mesh Network

**File**: `example_qurinet_memmap.py`

This example demonstrates routing with **real wireless mesh network data** from the Qurinet deployment at Quail Ridge Natural Reserve. It uses actual network topology and link quality measurements from a real wireless mesh network deployment.

**Features:**
- **Real Network Topology**: 19 nodes with 24 links from actual deployment
- **Real Link Quality Data**: Signal strength and quality measurements from real wireless mesh network
- **All Algorithms**: Tests CTS-B, CombUCB, CTS-G, CL-SG, and BG-CTS
- **Multiple Gamma Values**: Compares γ = 0.01, 0.1, 0.5, 1.0 for CTS-G and CL-SG
- **Memory Mapping**: Efficient storage of all simulation data
- **Statistical Analysis**: 95% confidence intervals and regret analysis
- **Network Visualization**: Real network topology visualization

**Network Parameters:**
- **Data Source**: Real deployment at Quail Ridge Natural Reserve (https://github.com/cjpatton/qr)
- **Topology**: 19 nodes with 24 links from actual wireless mesh network
- **Frequency**: 2.4GHz (channels 1, 6, 11)
- **Link Quality**: Real signal strength (-100dBm to -30dBm) and quality measurements
- **Availability Rates**: Derived from actual signal strength data
- **Reward Means**: Based on real link quality and availability data

**Usage:**
```bash
# Run with default parameters (5,000 rounds, 5 runs)
python example_qurinet_memmap.py

# Custom parameters
python example_qurinet_memmap.py --rounds 1000 --runs 10

# Keep memory-mapped files for post-hoc analysis
python example_qurinet_memmap.py --keep-memmap

# Quiet progress (no progress bars)
python example_qurinet_memmap.py --progress quiet
```

**Output:**
- Console output with real network analysis
- Three PDF plots:
  1. `qurinet_algorithm_comparison.pdf` - Algorithm comparison on real network
  2. `qurinet_cts-g_gamma_comparison.pdf` - CTS-G algorithm with different gamma values
  3. `qurinet_cl-sg_gamma_comparison.pdf` - CL-SG algorithm with different gamma values
- Memory-mapped data files (if --keep-memmap is used)

**Key Features:**
1. **Real Data**: Uses actual wireless mesh network topology and measurements
2. **Realistic Link Quality**: Signal strength and quality from real deployment
3. **Environmental Factors**: Realistic wireless mesh network behavior
4. **Memory Mapping**: Efficient data storage for large-scale simulations
5. **Multiple Gamma Values**: Tests CTS-G and CL-SG with γ = 0.01, 0.1, 0.5, 1.0
6. **Complete Algorithm Suite**: Tests all available algorithms on real data

**Example Output:**
```
Qurinet Real Wireless Mesh Network Environment Example
================================================================================
Configuration:
  Number of rounds: 1000
  Number of runs: 5
  Keep memory-mapped files: False

Qurinet Network Statistics:
==================================================
Network: Real wireless mesh network from Quail Ridge Natural Reserve
Data date: 2-May
Number of nodes: 19
Number of edges: 24
Source: 1, Destination: 2
Max combination size: 1

Connectivity:
  Connected: True
  Number of components: 1
  Average degree: 2.53

Link Quality Statistics:
  Average availability: 0.288
  Min availability: 0.130
  Max availability: 0.700
  Std availability: 0.166

Reward Statistics:
  Average reward: 0.439
  Min reward: 0.213
  Max reward: 0.821
  Std reward: 0.146

Optimal Path:
  Path: [1, 2]
  Length: 1
  Expected reward: 0.584

Qurinet Real Wireless Mesh Network Simulation Results:
================================================================================

Base Algorithms:
----------------------------------------
CTSB:
  Final regret: 23.60 ± 124.58
  95% CI: [-131.08, 178.28]

CombUCB:
  Final regret: 23.60 ± 124.58
  95% CI: [-131.08, 178.28]

BG-CTS:
  Final regret: 23.60 ± 125.17
  95% CI: [-131.82, 179.02]

Gamma Algorithms (γ=0.1):
----------------------------------------
CTS-G (γ=0.1):
  Final regret: 23.60 ± 126.08
  95% CI: [-132.95, 180.15]

CL-SG (γ=0.1):
  Final regret: 23.00 ± 126.97
  95% CI: [-134.65, 180.65]

Algorithm Comparison:
----------------------------------------
Best performing algorithm: CL-SG_gamma_1.0 (regret: 22.60)

Algorithm comparison plot saved to: output/images/qurinet_algorithm_comparison.pdf
CTS-G gamma comparison plot saved to: output/images/qurinet_cts-g_gamma_comparison.pdf
CL-SG gamma comparison plot saved to: output/images/qurinet_cl-sg_gamma_comparison.pdf

Qurinet real wireless mesh network simulation completed!
```

## Environment Setup

To run the examples, make sure you have all required dependencies installed:

```bash
pip install -r requirements.txt
```

The examples use the following key components:
- **Environments**: `SimpleEnvironment`, `RoutingEnvironment`, `QurinetEnvironment`
- **Algorithms**: `CTSB`, `CombUCB`, `CTS-G`, `CL-SG`, `BG-CTS`
- **Utilities**: Memory mapping, plotting functions, network utilities 