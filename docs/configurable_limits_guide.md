# Configurable Feasible Combination Limits Guide

## Overview

The UCSB mesh network environment now supports configurable parameters to control the computational complexity of feasible combination generation. This allows users to balance between algorithm performance and computational efficiency.

## New Parameters

### `max_feasible_combinations` (default: 50)
- **Purpose**: Controls the maximum number of feasible combinations returned by the environment
- **Impact**: Limits the total number of path combinations that bandit algorithms can choose from
- **Trade-off**: Lower values reduce computation time but may limit algorithm performance

### `max_path_length` (default: 8)
- **Purpose**: Controls the maximum path length (number of hops) in feasible combinations
- **Impact**: Prevents extremely long paths that may be computationally expensive
- **Trade-off**: Lower values reduce path enumeration time but may exclude valid long paths

### `max_paths_per_algorithm` (default: 25)
- **Purpose**: Controls how many paths each path-finding algorithm can return
- **Impact**: Limits the number of paths found by shortest path and BFS algorithms
- **Trade-off**: Lower values reduce computation but may miss alternative paths

## Usage Examples

### Conservative Configuration (Fast, Limited)
```python
env = UCSBMeshnetMemmapEnvironment(
    neighbortable_files=files,
    source="10.1.1.109",
    destination="10.1.1.5",
    max_feasible_combinations=10,    # Very limited
    max_path_length=3,               # Short paths only
    max_paths_per_algorithm=5        # Few paths per algorithm
)
```

### Balanced Configuration (Default)
```python
env = UCSBMeshnetMemmapEnvironment(
    neighbortable_files=files,
    source="10.1.1.109", 
    destination="10.1.1.5",
    max_feasible_combinations=50,    # Default
    max_path_length=8,               # Default
    max_paths_per_algorithm=25       # Default
)
```

### Aggressive Configuration (Comprehensive, Slower)
```python
env = UCSBMeshnetMemmapEnvironment(
    neighbortable_files=files,
    source="10.1.1.109",
    destination="10.1.1.5", 
    max_feasible_combinations=100,   # Many combinations
    max_path_length=10,              # Long paths allowed
    max_paths_per_algorithm=50       # Many paths per algorithm
)
```

## Performance Impact

Based on testing with the UCSB dataset:

| Configuration | Avg Combinations | Computation Time | Performance Impact |
|---------------|------------------|------------------|-------------------|
| Conservative  | ~6 combinations  | Fastest          | May limit learning |
| Balanced      | ~11 combinations | Moderate         | Good balance      |
| Aggressive    | ~11 combinations | Moderate         | More exploration  |

## Recommendations

### For Quick Testing/Development
- Use conservative settings: `max_feasible_combinations=10, max_path_length=3`
- Provides fast iteration for algorithm development

### For Production/Research
- Use balanced settings (defaults): `max_feasible_combinations=50, max_path_length=8`
- Good balance between performance and computation time

### For Comprehensive Analysis
- Use aggressive settings: `max_feasible_combinations=100, max_path_length=10`
- When you need to explore all possible paths

### For Large Networks
- Start with conservative settings and gradually increase
- Monitor computation time and memory usage
- Consider network topology when setting `max_path_length`

## Implementation Details

The parameters are used in the `get_feasible_combinations()` method:

1. **Path Finding**: Multiple algorithms (shortest paths, BFS) are used with `max_paths_per_algorithm` limit
2. **Path Length Filtering**: Paths longer than `max_path_length` are discarded
3. **Combination Limiting**: Total combinations are capped at `max_feasible_combinations`

## Example Test Script

See `test/test_ucsb_configurable_limits.py` for a complete example demonstrating different configurations and their impact on:
- Number of feasible combinations per round
- Path length distributions
- Computation time
- Algorithm performance

## Example Bandit Test

See `example/example_ucsb_configurable_limits.py` for a complete example showing how these parameters affect bandit algorithm performance and computation time. 