# Matrix Optimization Guide

## Overview

This guide covers matrix optimization techniques for handling large availability matrices in combinatorial bandit simulations.

## Current Implementation (Memory-based)

### Features
- **Direct matrix generation**: `(num_arms, num_rounds)` numpy array
- **Fast random access**: O(1) access to any round's available arms
- **Memory efficient**: ~0.15 MB for 20 arms × 1000 rounds
- **Reproducible**: Seed-based generation for consistent results

### Usage
```python
# Create environment with matrix optimization
env = SimpleEnvironment(num_arms=20, num_rounds=1000, seed=42)

# Direct access to any round
available_arms = env.get_available_arms_for_round(round_idx)

# Get matrix statistics
matrix_info = env.get_availability_matrix_info()
```

## Large Matrix Optimization Techniques

### 1. Memory Mapping (Recommended for 10GB+ matrices)

**Use case**: Very large matrices that don't fit in memory
**Implementation**:
```python
import numpy as np

# Create memory-mapped file
availability_matrix = np.memmap('availability.npy', dtype='uint8', mode='w+', 
                               shape=(num_arms, num_rounds))

# Generate matrix in chunks
chunk_size = 1000
for chunk_start in range(0, num_rounds, chunk_size):
    chunk_end = min(chunk_start + chunk_size, num_rounds)
    chunk = np.random.binomial(1, availability_rate, 
                              size=(num_arms, chunk_end - chunk_start))
    availability_matrix[:, chunk_start:chunk_end] = chunk

# Flush to disk
availability_matrix.flush()
```

**Benefits**:
- Only loads needed portions into memory
- Supports matrices larger than RAM
- Fast random access

### 2. Chunked Storage (Recommended for long simulations)

**Use case**: Very long simulations with limited memory
**Implementation**:
```python
import os
import numpy as np

# Create chunks directory
os.makedirs('chunks', exist_ok=True)

# Generate and save chunks
chunk_size = 1000
for chunk_idx in range(0, num_rounds, chunk_size):
    chunk_end = min(chunk_idx + chunk_size, num_rounds)
    chunk = np.random.binomial(1, availability_rate, 
                              size=(num_arms, chunk_end - chunk_idx))
    np.save(f'chunks/chunk_{chunk_idx}.npy', chunk)

# Load chunks on demand
def get_available_arms_for_round(round_idx):
    chunk_idx = (round_idx // chunk_size) * chunk_size
    chunk = np.load(f'chunks/chunk_{chunk_idx}.npy')
    local_idx = round_idx % chunk_size
    available_mask = chunk[:, local_idx] == 1
    return set(np.where(available_mask)[0])
```

**Benefits**:
- Minimal memory usage
- Scalable to millions of rounds
- Easy parallel processing

### 3. Compressed Storage (Recommended for sparse matrices)

**Use case**: Sparse availability patterns or limited storage
**Implementation**:
```python
import h5py
import numpy as np

# Create compressed HDF5 file
with h5py.File('availability.h5', 'w') as f:
    # Generate matrix
    matrix = np.random.binomial(1, availability_rate, size=(num_arms, num_rounds))
    
    # Save with compression
    f.create_dataset('matrix', data=matrix, compression='gzip', 
                    compression_opts=9)

# Load compressed data
with h5py.File('availability.h5', 'r') as f:
    matrix = f['matrix'][:]
```

**Benefits**:
- Significant storage reduction (up to 90%)
- Fast read/write with compression
- HDF5 format for complex metadata

### 4. Database Storage (Recommended for distributed access)

**Use case**: Multiple processes or complex queries
**Implementation**:
```python
import sqlite3
import numpy as np

# Create database
conn = sqlite3.connect('availability.db')
cursor = conn.cursor()

# Create table
cursor.execute('''
    CREATE TABLE availability (
        arm_id INTEGER,
        round_id INTEGER,
        available INTEGER,
        PRIMARY KEY (arm_id, round_id)
    )
''')

# Insert data
for arm in range(num_arms):
    for round_idx in range(num_rounds):
        available = np.random.binomial(1, availability_rate)
        cursor.execute('''
            INSERT INTO availability (arm_id, round_id, available)
            VALUES (?, ?, ?)
        ''', (arm, round_idx, available))

conn.commit()

# Query available arms for a round
def get_available_arms_for_round(round_idx):
    cursor.execute('''
        SELECT arm_id FROM availability 
        WHERE round_id = ? AND available = 1
    ''', (round_idx,))
    return set(row[0] for row in cursor.fetchall())
```

**Benefits**:
- ACID transactions
- Complex queries (e.g., "arms available in rounds 100-200")
- Distributed access
- Built-in indexing

## Performance Comparison

| Method | Memory Usage | Access Speed | Scalability | Complexity |
|--------|-------------|--------------|-------------|------------|
| **Memory** | O(arms × rounds) | O(1) | ~1GB | Low |
| **Memory Mapping** | O(chunk size) | O(1) | Unlimited | Medium |
| **Chunked** | O(chunk size) | O(1) | Unlimited | Medium |
| **Compressed** | O(arms × rounds) | O(1) | ~10GB | Low |
| **Database** | O(rounds) | O(log n) | Unlimited | High |

## Implementation Recommendations

### For Different Scales

**Small Scale** (< 1GB matrix):
```python
# Use current memory-based implementation
env = SimpleEnvironment(num_arms=100, num_rounds=10000, seed=42)
```

**Medium Scale** (1-10GB matrix):
```python
# Use memory mapping
# (Implementation provided above)
```

**Large Scale** (10GB+ matrix):
```python
# Use chunked storage
# (Implementation provided above)
```

**Distributed/Complex**:
```python
# Use database storage
# (Implementation provided above)
```

### Memory Usage Estimation

```python
def estimate_memory_usage(num_arms, num_rounds):
    """Estimate memory usage in MB."""
    # uint8 = 1 byte per element
    bytes_per_element = 1
    total_elements = num_arms * num_rounds
    memory_bytes = total_elements * bytes_per_element
    memory_mb = memory_bytes / (1024 * 1024)
    return memory_mb

# Example usage
memory_mb = estimate_memory_usage(1000, 100000)
print(f"Estimated memory usage: {memory_mb:.2f} MB")
```

## Future Enhancements

### 1. Hybrid Approach
Combine memory mapping with compression for optimal performance.

### 2. Parallel Generation
Use multiprocessing for faster matrix generation.

### 3. Streaming Interface
Implement streaming for real-time availability updates.

### 4. Caching Layer
Add LRU cache for frequently accessed rounds.

## Conclusion

The current memory-based implementation is optimal for most use cases. For larger matrices, consider the techniques outlined above based on your specific requirements:

- **Memory constraints**: Use memory mapping or chunked storage
- **Storage constraints**: Use compression
- **Distributed access**: Use database storage
- **Complex queries**: Use database storage

The choice depends on your specific performance, memory, and scalability requirements. 