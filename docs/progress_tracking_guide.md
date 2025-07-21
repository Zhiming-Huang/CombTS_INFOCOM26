# Progress Tracking Guide

## Overview

This guide explains the different progress tracking options available in the CombTS simulation framework and their performance characteristics.

## Available Progress Levels

### 1. Minimal Progress (`"minimal"`)
- **Description**: No progress bars, maximum performance
- **Use case**: Production runs, large-scale simulations
- **Overhead**: ~0% (baseline)
- **Output**: Only final results

```python
results = run_simple_simulation(
    num_rounds=10000, 
    num_runs=5, 
    progress_level="minimal"
)
```

### 2. Normal Progress (`"normal"`)
- **Description**: Shows progress for each run
- **Use case**: Development, testing, monitoring
- **Overhead**: ~1-2%
- **Output**: Run progress + final results

```python
results = run_simple_simulation(
    num_rounds=10000, 
    num_runs=5, 
    progress_level="normal"
)
```

### 3. Detailed Progress (`"detailed"`)
- **Description**: Shows progress for each run and round
- **Use case**: Debugging, detailed monitoring
- **Overhead**: ~3-5%
- **Output**: Run progress + round progress + real-time metrics

```python
results = run_simple_simulation(
    num_rounds=10000, 
    num_runs=5, 
    progress_level="detailed"
)
```

## Performance Comparison

| Level | Overhead | Use Case | Features |
|-------|----------|----------|----------|
| `minimal` | ~0% | Production | No progress bars |
| `normal` | ~1-2% | Development | Run progress only |
| `detailed` | ~3-5% | Debugging | Full progress + metrics |

## Implementation Details

### Progress Bar Features

The implementation uses `tqdm` library which provides:

1. **Efficient Updates**: Minimal overhead for progress updates
2. **Nested Progress**: Support for nested progress bars
3. **Custom Postfix**: Real-time metrics display
4. **Memory Efficient**: No memory accumulation during progress tracking

### Code Structure

```python
def run_simple_simulation(num_rounds, num_runs, progress_level="normal"):
    # Configure progress bars based on level
    show_run_progress = progress_level in ["normal", "detailed"]
    show_round_progress = progress_level == "detailed"
    show_regret_updates = progress_level == "detailed"
    
    # Create progress bar for runs
    if show_run_progress:
        run_pbar = tqdm(range(num_runs), desc="Simulation runs", unit="run")
    else:
        run_pbar = range(num_runs)
    
    for run in run_pbar:
        # Run progress updates
        if show_run_progress:
            run_pbar.set_postfix({"Run": f"{run + 1}/{num_runs}"})
        
        # Create progress bar for rounds
        if show_round_progress:
            round_pbar = tqdm(range(num_rounds), desc=f"Run {run + 1} rounds", 
                             unit="round", leave=False)
        else:
            round_pbar = range(num_rounds)
        
        for round_num in round_pbar:
            # Simulation logic...
            
            # Update progress with metrics
            if show_regret_updates and isinstance(round_pbar, tqdm):
                round_pbar.set_postfix({
                    "Regret": f"{total_regret:.2f}",
                    "Reward": f"{total_reward:.2f}"
                })
```

## Best Practices

### 1. Choose Appropriate Level
- **Production**: Use `minimal` for maximum performance
- **Development**: Use `normal` for good balance
- **Debugging**: Use `detailed` for full visibility

### 2. Large Simulations
For very large simulations (100K+ rounds), consider:
```python
# For maximum performance
results = run_simple_simulation(
    num_rounds=100000,
    num_runs=10,
    progress_level="minimal"  # No overhead
)
```

### 3. Development Workflow
```python
# During development
results = run_simple_simulation(
    num_rounds=1000,
    num_runs=3,
    progress_level="detailed"  # Full visibility
)

# For production
results = run_simple_simulation(
    num_rounds=10000,
    num_runs=5,
    progress_level="minimal"  # Maximum performance
)
```

## Testing Progress Options

Run the progress tracking test to compare performance:

```bash
python test/test_progress_tracking.py
```

This will:
1. Demonstrate different progress bar features
2. Compare performance overhead of each level
3. Test memory efficiency for large simulations

## Custom Progress Bars

You can also create custom progress bars for specific needs:

```python
from tqdm import tqdm
import time

# Custom progress with specific metrics
pbar = tqdm(range(100), desc="Custom simulation")
for i in pbar:
    # Your simulation logic
    regret = calculate_regret()
    reward = calculate_reward()
    
    # Update with custom metrics
    pbar.set_postfix({
        "Regret": f"{regret:.3f}",
        "Reward": f"{reward:.3f}",
        "Efficiency": f"{reward/regret:.2f}" if regret > 0 else "N/A"
    })
    time.sleep(0.01)
```

## Memory Considerations

- **Minimal overhead**: Progress bars add negligible memory usage
- **No accumulation**: Progress bars don't accumulate data over time
- **Efficient updates**: `tqdm` uses efficient string formatting for updates

## Troubleshooting

### Common Issues

1. **Progress bars not showing**: Check if running in non-interactive environment
2. **Performance impact**: Use `minimal` level for production runs
3. **Memory usage**: Progress bars themselves don't cause memory issues

### Environment Considerations

- **Jupyter notebooks**: Progress bars work well
- **Terminal**: Full progress bar features available
- **CI/CD**: Use `minimal` level to avoid output clutter

## Summary

The progress tracking system provides three levels of detail with minimal performance impact:

- **Minimal**: 0% overhead, production-ready
- **Normal**: 1-2% overhead, development-friendly  
- **Detailed**: 3-5% overhead, debugging-focused

Choose the appropriate level based on your use case and performance requirements. 