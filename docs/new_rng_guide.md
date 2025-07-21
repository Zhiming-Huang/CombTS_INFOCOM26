# New NumPy Random Number Generator Guide

## Overview

This guide explains the migration from the old `np.random` approach to the new `np.random.default_rng()` method, which is the recommended way to generate random numbers in modern NumPy.

## Why the Change?

### **Old Approach (Deprecated)**
```python
import numpy as np

# Global state - problematic
np.random.seed(42)
np.random.binomial(1, 0.5)
```

### **New Approach (Recommended)**
```python
import numpy as np

# Explicit generator - better
rng = np.random.default_rng(42)
rng.binomial(1, 0.5)
```

## Key Improvements

### 1. **Better Algorithm**
- **Old**: Mersenne Twister (MT19937)
- **New**: PCG64 (Permuted Congruential Generator)
- **Benefits**: Better statistical properties, longer period, faster

### 2. **Thread Safety**
- **Old**: Global state, not thread-safe
- **New**: Each generator instance is independent
- **Benefits**: Safe for parallel processing

### 3. **Explicit State Management**
- **Old**: Implicit global state
- **New**: Explicit generator instances
- **Benefits**: Clearer code, easier debugging

### 4. **Future-Proof**
- **Old**: Deprecated, may be removed
- **New**: Current standard, actively maintained
- **Benefits**: Long-term compatibility

## Implementation in Our Code

### **Environment Classes Updated**

Both `SimpleEnvironment` and `SimpleEnvironmentMemmap` now use the new RNG:

```python
# Old approach (removed)
np.random.seed(seed)

# New approach (implemented)
self.rng = np.random.default_rng(seed)
```

### **Usage Examples**

```python
# Creating environment with new RNG
env = SimpleEnvironment(
    num_arms=10,
    num_rounds=1000,
    pre_generate_rewards=True,
    seed=42  # This creates a new RNG instance
)

# The RNG is automatically used for all random operations
reward = env.get_reward_for_round(arm_id, round_idx)
```

## Performance Comparison

### **Test Results**
```
Configuration: 10,000 random binomial samples

Old approach (np.random):
  Time: 0.0032s
  Mean reward: 0.492
  Total rewards: 4924

New approach (default_rng):
  Time: 0.0032s
  Mean reward: 0.498
  Total rewards: 4985

Speedup: 1.02x
Same results: False (different algorithms)
```

### **Key Observations**
1. **Similar Performance**: Both approaches are equally fast
2. **Different Results**: PCG64 produces different (but equally valid) sequences
3. **Better Quality**: New algorithm has superior statistical properties

## Statistical Quality

### **Algorithm Comparison**

| Feature | Old (MT19937) | New (PCG64) |
|---------|---------------|-------------|
| Period | 2^19937-1 | 2^128 |
| Speed | Good | Better |
| Statistical Quality | Good | Excellent |
| Thread Safety | No | Yes |
| Memory Usage | High | Low |

### **Quality Metrics**
- **Period**: How long before sequence repeats
- **Statistical Tests**: Passes more rigorous tests
- **Predictability**: Better resistance to prediction attacks

## Migration Guide

### **For Existing Code**

If you have code using the old approach:

```python
# OLD CODE (deprecated)
import numpy as np
np.random.seed(42)
rewards = np.random.binomial(1, 0.5, 1000)

# NEW CODE (recommended)
import numpy as np
rng = np.random.default_rng(42)
rewards = rng.binomial(1, 0.5, 1000)
```

### **For Our Environment Classes**

The migration is transparent - no changes needed in your code:

```python
# This works the same way with new RNG
env = SimpleEnvironment(
    num_arms=10,
    num_rounds=1000,
    pre_generate_rewards=True,
    seed=42
)

# All methods work identically
reward = env.generate_reward(arm_id)
available_arms = env.get_available_arms()
```

## Best Practices

### 1. **Always Use Explicit Generators**
```python
# Good
rng = np.random.default_rng(seed)
result = rng.binomial(1, 0.5)

# Avoid (deprecated)
np.random.seed(seed)
result = np.random.binomial(1, 0.5)
```

### 2. **Pass Generators to Functions**
```python
def generate_rewards(rng, num_samples):
    return rng.binomial(1, 0.5, num_samples)

# Usage
rng = np.random.default_rng(42)
rewards = generate_rewards(rng, 1000)
```

### 3. **Use Different Seeds for Independence**
```python
# Each gets different sequence
rng1 = np.random.default_rng(42)
rng2 = np.random.default_rng(123)
rng3 = np.random.default_rng(456)
```

### 4. **For Parallel Processing**
```python
import multiprocessing as mp

def worker(seed):
    rng = np.random.default_rng(seed)
    return rng.binomial(1, 0.5, 100)

# Each process gets independent RNG
with mp.Pool(4) as pool:
    results = pool.map(worker, [42, 123, 456, 789])
```

## Testing and Validation

### **Test Script**
```bash
# Run comprehensive RNG tests
python test/test_new_rng.py
```

### **Key Test Areas**
1. **Functionality**: Correct random number generation
2. **Consistency**: Same seed produces same results
3. **Independence**: Different seeds produce different results
4. **Thread Safety**: Multiple instances work independently
5. **Quality**: Statistical properties of generated numbers

## Common Questions

### **Q: Do I need to change my existing code?**
**A**: No, our environment classes handle the migration internally. Your code continues to work unchanged.

### **Q: Are the results different?**
**A**: Yes, but both are equally valid. The new algorithm produces different (but statistically equivalent) sequences.

### **Q: Is performance affected?**
**A**: No, performance is similar or better with the new approach.

### **Q: Can I still use the old approach?**
**A**: Technically yes, but it's deprecated and may be removed in future NumPy versions.

### **Q: What about reproducibility?**
**A**: The new approach is fully reproducible - same seed produces same sequence.

## Future Considerations

### **NumPy Version Compatibility**
- **NumPy 1.17+**: `default_rng()` available
- **NumPy 1.21+**: Recommended approach
- **Future versions**: Old approach may be removed

### **Alternative Generators**
```python
# Different generator types
rng_pcg = np.random.default_rng(42)  # PCG64 (default)
rng_mt = np.random.Generator(np.random.MT19937(42))  # Mersenne Twister
rng_philox = np.random.Generator(np.random.Philox(42))  # Philox
```

### **Integration with Other Libraries**
- **SciPy**: Compatible with new RNG
- **Pandas**: Uses NumPy RNG internally
- **Matplotlib**: Supports new RNG for plotting

## Summary

The migration to `np.random.default_rng()` provides:

1. **Better Quality**: Superior statistical properties
2. **Thread Safety**: Safe for parallel processing
3. **Future-Proof**: Current standard, actively maintained
4. **Transparent**: No changes needed in existing code
5. **Performance**: Similar or better performance

Our environment classes have been updated to use the new RNG while maintaining full backward compatibility with existing code. 