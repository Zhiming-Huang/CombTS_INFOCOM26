# Synchronization Fix Guide

## Problem Description

The original implementation had a critical synchronization issue between the environment and test program:

### **The Problem**
1. **Environment maintained internal counter**: `_current_round` was incremented each time `sample_available_arms_once()` was called
2. **Test program also called the same method**: This caused the environment's counter to advance unexpectedly
3. **Result**: Environment and test program were out of sync, leading to incorrect regret calculations

### **Example of the Issue**
```python
# Round 0
algorithm.select_combination()  # Calls env.sample_available_arms_once() → _current_round = 1
test.calculate_regret()        # Calls env.sample_available_arms_once() → _current_round = 2

# Round 1  
algorithm.select_combination()  # Calls env.sample_available_arms_once() → _current_round = 3
test.calculate_regret()        # Calls env.sample_available_arms_once() → _current_round = 4
```

**Problem**: Test program was accessing round 2 when it should be accessing round 1.

## Solution

### **New Approach**
1. **Remove internal counter**: Environment no longer maintains `_current_round`
2. **Direct matrix access**: Test program directly specifies round index
3. **Explicit round indexing**: All access is done via `get_available_arms_for_round(round_idx)`

### **Code Changes**

#### **Before (Problematic)**
```python
# Environment maintains counter
def sample_available_arms_once(self):
    if not hasattr(self, '_current_round'):
        self._current_round = 0
    return self.get_available_arms_for_round(self._current_round)

# Test program
for round_num in range(num_rounds):
    env.reset_available_arms()  # Reset counter
    selected_combination = algorithm.select_combination()  # Increments counter
    available_arms = env.sample_available_arms_once()     # Increments counter again!
    # Now environment is ahead of test program
```

#### **After (Fixed)**
```python
# Test program directly specifies round
for round_num in range(num_rounds):
    available_arms = env.get_available_arms_for_round(round_num)  # Direct access
    selected_combination = algorithm.select_combination()
    # Both use same round index - no synchronization issues
```

## Implementation Details

### **1. Environment Changes**

#### **Deprecated Methods**
```python
def sample_available_arms_once(self) -> Set[int]:
    """
    DEPRECATED - use get_available_arms_for_round(round_idx) instead.
    """
    import warnings
    warnings.warn("sample_available_arms_once is deprecated...", DeprecationWarning)
    # Still works but shows warning
```

#### **New Direct Access**
```python
def get_available_arms_for_round(self, round_idx: int) -> Set[int]:
    """
    Get available arms for specific round from matrix.
    """
    available_mask = self.availability_matrix[:, round_idx] == 1
    return set(np.where(available_mask)[0])
```

### **2. Test Program Changes**

#### **Before**
```python
for round_num in range(num_rounds):
    env.reset_available_arms()  # Reset counter
    selected_combination = algorithm.select_combination()
    rewards = env.generate_combination_reward(selected_combination)
    algorithm.update_posterior(selected_combination, rewards)
    
    # Calculate regret (potentially wrong round)
    available_arms = env.sample_available_arms_once()
    optimal_combination = env.get_optimal_combination(available_arms)
    regret = optimal_reward - total_round_reward
```

#### **After**
```python
for round_num in range(num_rounds):
    # Direct access to correct round
    available_arms = env.get_available_arms_for_round(round_num)
    
    selected_combination = algorithm.select_combination()
    
    # Get rewards from matrix
    rewards = {}
    if selected_combination:
        for arm in selected_combination:
            rewards[arm] = env.get_reward_for_round(arm, round_num)
    
    algorithm.update_posterior(selected_combination, rewards)
    
    # Calculate regret using same round
    optimal_combination = env.get_optimal_combination(available_arms)
    optimal_expected = sum(env.arm_means[arm] for arm in optimal_combination)
    selected_expected = sum(env.arm_means[arm] for arm in selected_combination)
    regret = optimal_expected - selected_expected
```

## Benefits

### **1. Correctness**
- **No more synchronization issues**: Environment and test program always access same round
- **Deterministic results**: Same round always produces same available arms
- **Accurate regret calculation**: Regret is calculated for the correct round

### **2. Clarity**
- **Explicit round indexing**: No hidden state or counters
- **Clear data flow**: Round index is explicitly passed
- **Easier debugging**: Can access any round directly

### **3. Performance**
- **No counter management**: Eliminates overhead of maintaining state
- **Direct matrix access**: O(1) access to availability and rewards
- **Consistent performance**: No unexpected state changes

### **4. Maintainability**
- **Simpler code**: No complex state management
- **Better testability**: Can test specific rounds independently
- **Future-proof**: Easy to extend for new features

## Testing Results

### **Synchronization Test**
```
=== Testing Synchronization Fix ===
--- Round 0 ---
Available arms (from matrix): {0, 1, 2, 3, 5, 8}
Selected combination: {0, 2, 3}
Optimal combination: {0, 1, 2}
Regret: 0.8
Available arms consistency: True ✓

--- Round 1 ---
Available arms (from matrix): {1, 2, 4, 5, 6}
Selected combination: {2, 4, 5}
Optimal combination: {1, 2, 4}
Regret: 0.8
Available arms consistency: True ✓
```

### **Counter Independence Test**
```
Algorithm round 0: {0, 1, 3, 4, 5, 6, 9}
Test round 0: {0, 1, 3, 4, 5, 6, 9}
Algorithm and test rounds are consistent: True ✓
```

### **Regret Calculation Test**
```
Round 0: Regret = 0.80, Total = 0.80
Round 1: Regret = 0.80, Total = 1.60
Round 2: Regret = 0.00, Total = 1.60
Round 3: Regret = 0.00, Total = 1.60
Round 4: Regret = 0.00, Total = 1.60
Final total regret: 2.40
```

## Migration Guide

### **For Existing Code**

#### **Step 1: Update Environment Creation**
```python
# Before
env = SimpleEnvironment(num_arms=10, ...)

# After
env = SimpleEnvironment(
    num_arms=10,
    num_rounds=10000,  # Specify number of rounds
    pre_generate_rewards=True,  # Enable pre-generated rewards
    seed=42
)
```

#### **Step 2: Update Simulation Loop**
```python
# Before
for round_num in range(num_rounds):
    env.reset_available_arms()
    selected_combination = algorithm.select_combination()
    # ... rest of code

# After
for round_num in range(num_rounds):
    available_arms = env.get_available_arms_for_round(round_num)
    selected_combination = algorithm.select_combination()
    # ... rest of code
```

#### **Step 3: Update Reward Generation**
```python
# Before
rewards = env.generate_combination_reward(selected_combination)

# After
rewards = {}
if selected_combination:
    for arm in selected_combination:
        rewards[arm] = env.get_reward_for_round(arm, round_num)
```

### **Backward Compatibility**

The old methods still work but show deprecation warnings:

```python
# Still works but shows warning
available_arms = env.sample_available_arms_once()  # DeprecationWarning
env.reset_available_arms()  # DeprecationWarning
```

## Best Practices

### **1. Always Use Direct Matrix Access**
```python
# Good
available_arms = env.get_available_arms_for_round(round_idx)
reward = env.get_reward_for_round(arm, round_idx)

# Avoid (deprecated)
available_arms = env.sample_available_arms_once()
```

### **2. Explicit Round Indexing**
```python
# Good
for round_idx in range(num_rounds):
    # Use round_idx explicitly

# Avoid
for round_idx in range(num_rounds):
    # Rely on internal counters
```

### **3. Consistent Round Access**
```python
# Good - same round for all operations
round_idx = 5
available_arms = env.get_available_arms_for_round(round_idx)
rewards = {arm: env.get_reward_for_round(arm, round_idx) for arm in selected_combination}
```

## Summary

The synchronization fix ensures:

1. **Correct regret calculation**: Environment and test program access same round
2. **Deterministic behavior**: Same round always produces same results
3. **Better performance**: Direct matrix access without counter overhead
4. **Improved maintainability**: Explicit round indexing, no hidden state
5. **Future compatibility**: Easy to extend and modify

This fix is crucial for accurate algorithm evaluation and reliable experimental results. 