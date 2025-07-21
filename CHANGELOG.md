# Changelog

## [1.4.0] - 2024-01-20

### Added
- **Pre-generated rewards optimization**: New `pre_generate_rewards` parameter for performance improvement
- **Memory-mapped rewards**: Support for large-scale reward matrix storage with memory mapping
- **New NumPy RNG**: Migrated from deprecated `np.random` to modern `np.random.default_rng()`
- **Synchronization fix**: Eliminated counter synchronization issues between environment and test programs
- **Direct matrix access**: Explicit round indexing for consistent and deterministic results
- **Efficient progress tracking**: Configurable progress levels with minimal performance overhead

### Changed
- **Random number generation**: Updated to use `np.random.default_rng()` with PCG64 algorithm
- **Reward generation**: Added pre-generated rewards matrix for 2-4x performance improvement
- **Environment interface**: Deprecated counter-based methods in favor of direct matrix access
- **Test synchronization**: Fixed environment-test program round synchronization issues
- **Regret calculation**: Fixed to compare expected rewards with expected rewards (not instantaneous)

### Performance
- **Rewards optimization**: 2-4x speedup with pre-generated rewards
- **Memory mapping**: Efficient storage for large reward matrices
- **RNG quality**: Better statistical properties with PCG64 algorithm
- **Synchronization**: Eliminated overhead of counter management
- **Progress tracking**: < 5% overhead with configurable detail levels

### Fixed
- **Critical synchronization bug**: Environment and test program now access same rounds consistently
- **Regret calculation error**: Now correctly compares expected rewards instead of mixing expected and instantaneous
- **Counter drift**: Eliminated round counter synchronization issues
- **Deterministic behavior**: Same round always produces same results

### Technical Details

#### Pre-generated Rewards
```python
# New interface with pre-generated rewards
env = SimpleEnvironment(
    num_arms=10,
    num_rounds=10000,
    pre_generate_rewards=True,  # Enable performance optimization
    seed=42
)

# Direct access to rewards
reward = env.get_reward_for_round(arm_id, round_idx)
```

#### New RNG Implementation
```python
# Old (deprecated)
np.random.seed(42)
np.random.binomial(1, 0.5)

# New (recommended)
rng = np.random.default_rng(42)
rng.binomial(1, 0.5)
```

#### Synchronization Fix
```python
# Before (problematic)
for round_num in range(num_rounds):
    env.reset_available_arms()  # Reset counter
    selected = algorithm.select_combination()  # Increments counter
    available = env.sample_available_arms_once()  # Increments counter again!

# After (fixed)
for round_num in range(num_rounds):
    available = env.get_available_arms_for_round(round_num)  # Direct access
    selected = algorithm.select_combination()
    # Both use same round - no synchronization issues
```

#### Correct Regret Calculation
```python
# Before (incorrect)
optimal_reward = sum(env.arm_means[arm] for arm in optimal_combination)  # Expected
regret = optimal_reward - total_round_reward  # Expected - Instantaneous

# After (correct)
optimal_expected = sum(env.arm_means[arm] for arm in optimal_combination)
selected_expected = sum(env.arm_means[arm] for arm in selected_combination)
regret = optimal_expected - selected_expected  # Expected - Expected
```

#### Efficient Progress Tracking
```python
# Configurable progress levels with minimal overhead
results = run_simple_simulation(
    num_rounds=10000,
    num_runs=5,
    progress_level="normal"  # Options: "minimal", "normal", "detailed"
)

# Progress overhead: minimal < 1%, normal < 2%, detailed < 5%
```

### Files Added
- `src/environments/simple_environment_memmap.py` (updated with rewards support)
- `test/test_rewards_optimization.py`
- `test/test_new_rng.py`
- `test/test_regret_calculation.py`
- `test/test_synchronization_fix.py`
- `test/test_progress_tracking.py`
- `test/test_progress_performance.py`
- `example/` directory with organized examples
- `example/README.md` - Examples documentation
- `example/example_progress_tracking.py` - Progress tracking demo
- `docs/rewards_optimization_guide.md`
- `docs/new_rng_guide.md`
- `docs/synchronization_fix_guide.md`
- `docs/progress_tracking_guide.md`

### Files Modified
- `src/environments/simple_environment.py` (added pre-generated rewards, new RNG)
- `src/environments/simple_environment_memmap.py` (added rewards support)
- `test/test_simple_environment.py` (fixed synchronization, regret calculation)
- `README.md` (updated project structure, added examples section)

### Breaking Changes
- **Deprecated methods**: `sample_available_arms_once()` and `reset_available_arms()` show deprecation warnings
- **RNG changes**: Different random sequences due to PCG64 vs MT19937 algorithms
- **Interface changes**: Recommended to use direct matrix access methods

### Migration Guide
1. **Update environment creation**: Add `num_rounds` and `pre_generate_rewards=True`
2. **Replace deprecated methods**: Use `get_available_arms_for_round(round_idx)` instead of `sample_available_arms_once()`
3. **Update reward access**: Use `get_reward_for_round(arm, round_idx)` for pre-generated rewards
4. **Fix regret calculation**: Compare expected rewards with expected rewards
5. **Configure progress tracking**: Use `progress_level` parameter for optimal performance

## [1.3.0] - 2024-01-20

### Added
- **Memory-mapped optimization**: New `SimpleEnvironmentMemmap` class for ultra-large matrices
- **Chunked generation**: Memory-friendly matrix generation in configurable chunks
- **File persistence**: Support for custom file paths and temporary file management
- **Large-scale support**: Handles matrices up to 500 arms × 50,000 rounds (25M elements)
- **Memory efficiency**: 7.89x less memory usage compared to in-memory approach

### Changed
- **Matrix optimization**: Added availability matrix generation for efficient arm access
- **Direct round access**: New `get_available_arms_for_round(round_idx)` method for direct matrix access
- **Memory efficiency**: Pre-generated availability matrix with configurable size
- **Reproducibility**: Added seed parameter for reproducible results
- **Matrix statistics**: New `get_availability_matrix_info()` method for matrix analysis

### Performance
- **Memory-mapped**: 7.89x memory efficiency improvement
- **Matrix approach**: 1.23x speedup over on-demand generation
- **Large-scale**: Supports 10M+ element matrices with minimal memory usage
- **Interface**: Backward compatible with existing `sample_available_arms_once()` and `reset_available_arms()`

### Fixed
- **Optimal combination calculation**: Fixed `get_optimal_combination()` to select arms with highest mean rewards instead of just first optimal arms
- **Regret evaluation**: Now correctly compares algorithm performance against truly optimal combinations

## [1.1.0] - 2024-01-20

### Added
- **Environment-based CombTS**: Modified `CombTS` class to accept an environment instance instead of just the number of arms
- **SimpleEnvironment**: New test environment with 10 arms (3 optimal, 7 suboptimal) for algorithm validation
- **Consistent arm sampling**: Added methods to ensure algorithm and benchmark use the same available arms
- **Comprehensive regret analysis**: New test scripts for detailed regret growth analysis

### Changed
- **CombTS interface**: 
  - Constructor now takes `environment` instead of `num_arms`
  - `select_combination()` no longer requires external parameters
- **Test organization**: Moved all test files to `test/` directory
- **Arm consistency**: Algorithm and benchmark now use identical available arm sets

### Fixed
- **Regret calculation**: Fixed inconsistency where algorithm and benchmark used different available arm sets
- **File organization**: Properly organized test files in dedicated directory

### Technical Details

#### CombTS Changes
```python
# Old interface
algorithm = CombTS(num_arms=10, alpha=1.0, beta=1.0)
selected = algorithm.select_combination(available_arms, feasible_combinations)

# New interface
algorithm = CombTS(environment=env, alpha=1.0, beta=1.0)
selected = algorithm.select_combination()  # Gets arms from environment
```

#### SimpleEnvironment Features
- 10 arms total: 3 optimal (Bernoulli(0.9)), 7 suboptimal (Bernoulli(0.1))
- 0.5 availability rate for each arm
- Maximum combination size of 3
- Expected optimal reward: 2.7

#### Regret Analysis Results
- **Sublinear regret**: Linear growth rate < 0.15
- **O(√T) performance**: Regret/√T growth rate bounded
- **Consistent evaluation**: Algorithm and benchmark use same arm sets

### Files Added
- `src/environments/simple_environment.py`
- `test/test_simple_environment.py`
- `test/test_regret_analysis.py`
- `test/test_simple_debug.py`
- `example_simple.py`
- `CHANGELOG.md`

### Files Modified
- `src/bandits/comb_ts.py`
- `src/environments/__init__.py`
- `README.md`

### Files Moved
- `test_simple_environment.py` → `test/test_simple_environment.py`
- `test_simple_debug.py` → `test/test_simple_debug.py`
- `test_regret_analysis.py` → `test/test_regret_analysis.py` 