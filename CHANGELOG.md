# Changelog

## [1.1.1] - 2024-01-20

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