# Comprehensive Test: All Algorithms with 3-Hop Paths

## Overview

This document summarizes the comprehensive testing of all available bandit algorithms in the UCSB mesh network environment using up to 3-hop paths. The test included 9 different algorithms with various parameter configurations.

## Test Configuration

- **Environment**: UCSB Mesh Network (948 neighbortable files)
- **Node Pair**: 10.1.1.109 → 10.1.1.5
- **Path Limit**: Up to 3 hops
- **Rounds**: 10,428 (11 routes per minute)
- **Feasible Combinations**: 200 maximum
- **Paths per Algorithm**: 1000 maximum
- **Random Generator**: NumPy PCG64 with seed 42

## Algorithms Tested

### 1. CTSB (Combinatorial Thompson Sampling with Beta)
- **Parameters**: alpha=1.0, beta=1.0
- **Final Regret**: 41.70
- **Performance Rank**: 1st
- **Path Distribution**: 100% 3-hop paths

### 2. CL-SG Family (Combinatorial Linear Stochastic Greedy)

#### cl-sg_gamma_0.1
- **Final Regret**: 187.78
- **Performance Rank**: 2nd
- **Path Distribution**: 1% 2-hop, 99% 3-hop

#### cl-sg_gamma_0.5
- **Final Regret**: 429.95
- **Performance Rank**: 4th
- **Path Distribution**: 2% 2-hop, 98% 3-hop

#### cl-sg_gamma_1.0
- **Final Regret**: 460.74
- **Performance Rank**: 5th
- **Path Distribution**: 2% 2-hop, 98% 3-hop

### 3. CTS-G Family (Thompson Sampling with Gaussian)

#### cts-g_gamma_0.1
- **Final Regret**: 278.17
- **Performance Rank**: 3rd
- **Path Distribution**: 1% 2-hop, 99% 3-hop

#### cts-g_gamma_0.5
- **Final Regret**: 618.28
- **Performance Rank**: 7th
- **Path Distribution**: 4% 2-hop, 96% 3-hop

#### cts-g_gamma_1.0
- **Final Regret**: 868.96
- **Performance Rank**: 8th
- **Path Distribution**: 6% 2-hop, 94% 3-hop

### 4. Other Algorithms

#### CombUCB (Combinatorial Upper Confidence Bound)
- **Final Regret**: 479.38
- **Performance Rank**: 6th
- **Path Distribution**: 2% 2-hop, 98% 3-hop

#### BG-CTS (Bayesian Gaussian Combinatorial Thompson Sampling)
- **Final Regret**: 1263.79
- **Performance Rank**: 9th
- **Path Distribution**: 11% 2-hop, 89% 3-hop

## Performance Rankings

| Rank | Algorithm | Final Regret | Avg Path Length | Path Distribution |
|------|-----------|--------------|-----------------|-------------------|
| 1 | CTSB | 41.70 | 3.00 | 100% 3-hop |
| 2 | cl-sg_gamma_0.1 | 187.78 | 2.99 | 1% 2-hop, 99% 3-hop |
| 3 | cts-g_gamma_0.1 | 278.17 | 2.99 | 1% 2-hop, 99% 3-hop |
| 4 | cl-sg_gamma_0.5 | 429.95 | 2.98 | 2% 2-hop, 98% 3-hop |
| 5 | cl-sg_gamma_1.0 | 460.74 | 2.98 | 2% 2-hop, 98% 3-hop |
| 6 | CombUCB | 479.38 | 2.98 | 2% 2-hop, 98% 3-hop |
| 7 | cts-g_gamma_0.5 | 618.28 | 2.96 | 4% 2-hop, 96% 3-hop |
| 8 | cts-g_gamma_1.0 | 868.96 | 2.93 | 6% 2-hop, 94% 3-hop |
| 9 | BG-CTS | 1263.79 | 2.89 | 11% 2-hop, 89% 3-hop |

## Key Findings

### 1. Algorithm Performance
- **CTSB** significantly outperformed all other algorithms with a regret of only 41.70
- **CL-SG family** generally performed better than **CTS-G family**
- **BG-CTS** had the worst performance with regret of 1263.79
- Performance ratio between best and worst: **30.31x**

### 2. Path Length Distribution
- **3-hop paths dominated**: 96.9% average usage across all algorithms
- **2-hop paths**: 3.5% average usage
- **1-hop paths**: 0.1% average usage (minimal)
- All algorithms successfully utilized the mixed path lengths

### 3. Gamma Parameter Impact
- **CTS-G family**: Strong positive correlation (0.989) between gamma and regret
- **CL-SG family**: Moderate positive correlation (0.885) between gamma and regret
- Higher gamma values generally led to higher regret in both families

### 4. Algorithm Family Analysis

#### CL-SG Family (Best Overall)
- **cl-sg_gamma_0.1**: Best in family (187.78)
- **cl-sg_gamma_0.5**: Middle performance (429.95)
- **cl-sg_gamma_1.0**: Worst in family (460.74)
- **Trend**: Lower gamma values perform better

#### CTS-G Family
- **cts-g_gamma_0.1**: Best in family (278.17)
- **cts-g_gamma_0.5**: Middle performance (618.28)
- **cts-g_gamma_1.0**: Worst in family (868.96)
- **Trend**: Lower gamma values perform better

#### Other Algorithms
- **CTSB**: Exceptional performance (41.70)
- **CombUCB**: Moderate performance (479.38)
- **BG-CTS**: Poor performance (1263.79)

## Path Length Insights

### Average Path Length by Algorithm
- **CTSB**: 3.00 hops (100% 3-hop paths)
- **CL-SG family**: 2.98-2.99 hops
- **CTS-G family**: 2.93-2.99 hops
- **CombUCB**: 2.98 hops
- **BG-CTS**: 2.89 hops (most diverse path selection)

### Path Selection Behavior
- **CTSB**: Most conservative, always choosing 3-hop paths
- **BG-CTS**: Most exploratory, using 11% 2-hop paths
- **Other algorithms**: Balanced approach with 1-6% 2-hop paths

## Computational Efficiency

### Simulation Times
- **Fastest**: BG-CTS (28.70 seconds)
- **Slowest**: cts-g_gamma_0.1 (32.88 seconds)
- **Average**: ~30.5 seconds per algorithm
- **Total test time**: ~4.5 minutes for all 9 algorithms

### Memory Usage
- All algorithms used approximately 3.0 combinations per round on average
- Consistent memory usage across all algorithms
- No significant memory bottlenecks observed

## Recommendations

### 1. Best Algorithm Choice
- **CTSB** is the clear winner for this specific network configuration
- **cl-sg_gamma_0.1** is the best alternative if CTSB is not available

### 2. Parameter Tuning
- Use **low gamma values** (0.1) for both CTS-G and CL-SG families
- Higher gamma values significantly degrade performance

### 3. Path Length Strategy
- **3-hop paths are preferred** by all algorithms
- **2-hop paths provide diversity** but are used sparingly
- **1-hop paths are rarely used** in this network topology

### 4. Algorithm Selection Guidelines
1. **Primary choice**: CTSB
2. **Secondary choice**: cl-sg_gamma_0.1
3. **Avoid**: BG-CTS and high gamma values

## Generated Files

### Test Scripts
- `test/test_all_algorithms_3hop.py`: Main comprehensive test
- `test/analyze_all_algorithms_3hop.py`: Detailed analysis

### Results
- `output/data/ucsb_all_algorithms_3hop_results.json`: Raw results
- `output/images/ucsb_all_algorithms_3hop_comparison.pdf`: Main comparison plot
- `output/images/ucsb_all_algorithms_3hop_detailed_analysis.pdf`: Detailed analysis plots

### Documentation
- `docs/all_algorithms_3hop_comprehensive_summary.md`: This summary document

## Conclusion

The comprehensive test successfully evaluated all available bandit algorithms with 3-hop paths in the UCSB environment. Key conclusions:

1. **CTSB is the superior algorithm** for this network configuration
2. **Gamma parameter significantly affects performance** - lower values are better
3. **3-hop paths dominate** the optimal routing decisions
4. **All algorithms successfully adapt** to mixed path lengths
5. **Performance varies dramatically** between algorithms (30x difference)

The test provides a solid foundation for algorithm selection in real-world mesh network routing scenarios with path length constraints. 