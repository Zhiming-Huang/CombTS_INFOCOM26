# CTS-B Examples

This directory contains various examples demonstrating different features and optimizations of the CTS-B algorithm.

## 📁 Example Files

### 🚀 **Basic Examples**

#### `example.py`
- **Purpose**: Basic demonstration of CTS-B algorithm
- **Features**: Simple environment setup and algorithm execution
- **Use case**: Getting started with CTS-B

#### `example_simple.py`
- **Purpose**: Simple environment test with CTS-B
- **Features**: 10 arms (3 optimal, 7 suboptimal), regret analysis
- **Use case**: Basic algorithm validation

### ⚡ **Performance Optimization Examples**

#### `example_matrix_optimization.py`
- **Purpose**: Demonstrate matrix-based optimization
- **Features**: Pre-generated availability matrix, performance comparison
- **Use case**: Understanding matrix optimization benefits

#### `example_rewards_optimization.py`
- **Purpose**: Show pre-generated rewards optimization
- **Features**: 2-4x performance improvement, memory mapping
- **Use case**: Large-scale simulation optimization

#### `example_memmap_optimization.py`
- **Purpose**: Memory-mapped environment for ultra-large matrices
- **Features**: 7.89x memory efficiency, chunked generation
- **Use case**: Handling massive datasets (500 arms × 50K rounds)

### 📊 **Progress Tracking Examples**

#### `example_progress_tracking.py`
- **Purpose**: Demonstrate different progress tracking levels
- **Features**: Minimal, normal, detailed progress with performance comparison
- **Use case**: Choosing appropriate progress level for your use case

## 🎯 **Usage Guide**

### Quick Start
```bash
# Basic example
python example/example.py

# Simple environment test
python example/example_simple.py

# Performance optimization
python example/example_rewards_optimization.py

# Progress tracking
python example/example_progress_tracking.py
```

### Performance Comparison
```bash
# Compare different optimization approaches
python example/example_matrix_optimization.py
python example/example_rewards_optimization.py
python example/example_memmap_optimization.py
```

## 📈 **Performance Characteristics**

| Example | Optimization | Speedup | Memory Efficiency | Use Case |
|---------|-------------|---------|-------------------|----------|
| `example.py` | None | 1x | Standard | Basic demo |
| `example_simple.py` | None | 1x | Standard | Algorithm validation |
| `example_matrix_optimization.py` | Matrix pre-generation | 1.23x | Standard | Medium-scale |
| `example_rewards_optimization.py` | Pre-generated rewards | 2-4x | Standard | Large-scale |
| `example_memmap_optimization.py` | Memory mapping | 1.23x | 7.89x | Ultra-large scale |
| `example_progress_tracking.py` | Progress levels | Configurable | Standard | All scales |

## 🔧 **Configuration Options**

### Environment Parameters
```python
# Basic environment
env = SimpleEnvironment(
    num_arms=10,
    num_optimal=3,
    optimal_mean=0.9,
    suboptimal_mean=0.1,
    availability_rate=0.5,
    max_combination_size=3
)

# Optimized environment
env = SimpleEnvironment(
    num_arms=10,
    num_rounds=10000,  # Pre-generate matrices
    pre_generate_rewards=True,  # Pre-generate rewards
    seed=42  # Reproducible results
)
```

### Progress Tracking
```python
# Different progress levels
results = run_simple_simulation(
    num_rounds=10000,
    num_runs=5,
    progress_level="minimal"    # Production
    progress_level="normal"     # Development
    progress_level="detailed"   # Debugging
)
```

## 📊 **Expected Results**

### Regret Analysis
- **Sublinear regret**: Linear growth rate < 0.15
- **O(√T) performance**: Regret/√T growth rate bounded
- **Consistent evaluation**: Algorithm and benchmark use same arm sets

### Performance Metrics
- **Matrix optimization**: 1.23x speedup
- **Rewards optimization**: 2-4x speedup
- **Memory mapping**: 7.89x memory efficiency
- **Progress tracking**: < 5% overhead

## 🛠️ **Dependencies**

All examples require:
- `numpy` - Numerical computations
- `matplotlib` - Plotting (for visualization examples)
- `tqdm` - Progress bars (for progress tracking examples)
- `seaborn` - Enhanced plotting (for some examples)

## 📝 **Notes**

1. **Reproducibility**: All examples use fixed seeds for reproducible results
2. **Scalability**: Examples demonstrate different optimization levels
3. **Memory**: Memory-mapped examples handle ultra-large datasets
4. **Progress**: Configurable progress tracking for different use cases

## 🔗 **Related Documentation**

- `docs/rewards_optimization_guide.md` - Rewards optimization details
- `docs/new_rng_guide.md` - Random number generation guide
- `docs/synchronization_fix_guide.md` - Synchronization fixes
- `docs/progress_tracking_guide.md` - Progress tracking guide

## 🚀 **Next Steps**

1. Start with `example.py` for basic understanding
2. Try `example_simple.py` for algorithm validation
3. Explore optimization examples for performance improvements
4. Use progress tracking examples for large-scale simulations 