# Changelog

## [1.13.0] - 2024-06-09

### Changed
- **Unified Bandit API**: All bandit algorithms (CTSB, CombUCB, CTS-G, CL-SG, BG-CTS) now have a unified `update_posterior(self, played_arms, rewards, round_idx)` interface. The `round_idx` argument is now required for all algorithms, even if unused internally. This ensures a consistent and extensible API for all bandit classes.
- **Test Program Simplification**: The main test file now always calls `update_posterior` with the round index for all algorithms, eliminating the need for dynamic argument checks or try/except logic.
- **Fairness and Reproducibility**: This change further enforces strict fairness and reproducibility in all algorithm comparisons, as all algorithms now receive the same round context in their update logic.

### Technical Details
- **API Consistency**: All bandit classes were updated to accept and document the `round_idx` argument in `update_posterior`.
- **No Behavioral Change**: For algorithms that do not use `round_idx`, the argument is accepted but ignored.
- **Cleaner Test Code**: The test harness is now simpler and more robust, with a single call signature for all algorithms.

### Benefits
- **Unified Interface**: Easier to add, test, and compare new algorithms in the future.
- **Cleaner Code**: No more special-casing or error handling for different method signatures.
- **Research-Grade**: Ensures all algorithms are evaluated under exactly the same conditions.

## [1.12.0] - 2024-01-20

### Changed
- **Randomness Fairness**: All bandit algorithms now use independent child random number generators (substreams) created from a main SeedSequence for each algorithm, ensuring strict fairness and reproducibility across all runs and algorithms.
- **API Update**: All bandit classes (CTSB, CombUCB, CTSG, CLSG, BG-CTS) now accept a `rng` argument and use it for all random sampling, instead of directly using `np.random`.
- **Test Framework**: The main test file now spawns a single set of child rngs for all algorithms at the start, and passes the correct rng to each algorithm instance in every run.
- **BG-CTS Robustness**: Improved BG-CTS to handle edge cases (e.g., available arms < m) and prevent infinite loops or crashes.

### Technical Details
- **Child RNGs**: Created using `np.random.SeedSequence.spawn`, guaranteeing non-overlapping, independent, and reproducible random streams for each algorithm.
- **Fairness**: All algorithms now draw random numbers from their own substream, eliminating cross-algorithm interference and ensuring a fair comparison.
- **Reproducibility**: Results are fully reproducible as long as the main seed is fixed.

### Benefits
- **Strict Fairness**: No random number sequence interference between algorithms.
- **Robustness**: BG-CTS and all algorithms now handle edge cases gracefully.
- **Research-Grade**: Suitable for rigorous algorithm comparison and publication.

## [1.11.0] - 2024-01-20

### Added
- **CTS-G Algorithm**: Implemented Combinatorial Thompson Sampling with Gaussian Priors
- **Gaussian Priors**: Uses \mathcal{N}(\hat{r}_{a, n_{a, t}}, \frac{\gamma \ln t}{n_{a, t}+1}) distribution
- **Tunable Parameter**: Gamma parameter for variance scaling control

### New Files
- `src/bandits/cts_g.py` - CTS-G algorithm implementation

### Files Modified
- `src/bandits/__init__.py` - Added CTSG import and export
- `test/test_simple_environment_memmap_cumulative_only.py` - Added CTS-G to algorithm comparison

### CTS-G Algorithm Details

#### Core Formula
\mathcal{N}(\hat{r}_{a, n_{a, t}}, \frac{\gamma \ln t}{n_{a, t}+1})

Where:
- \hat{r}_{a, n_{a, t}} is the empirical mean reward for arm a
- n_{a, t} is the number of times arm a has been pulled
- t is the current round number
- \gamma is the tunable parameter for variance scaling

#### Key Features
- **Gaussian Priors**: Uses normal distribution instead of Beta distribution
- **Tunable Variance**: Gamma parameter controls exploration-exploitation balance
- **Sleeping Arms Support**: Handles arms that may not be available at every round
- **Combinatorial Selection**: Selects feasible combinations with highest prior sum

#### Comparison Results
- **CTSB final regret**: 23.68 ± 2.51
- **CombUCB final regret**: 127.44 ± 4.92
- **CTS-G final regret**: 643.36 ± 3.29
- **Winner**: CTSB (improvement: 96.3%)

### Technical Implementation

#### CTSG Class Methods
- `__init__(environment, gamma)`: Initialize with environment and gamma parameter
- `select_combination()`: Select combination based on Gaussian prior samples
- `_compute_gaussian_sample(arm)`: Compute Gaussian sample for specific arm
- `update_posterior(played_arms, rewards)`: Update empirical means
- `get_arm_statistics()`: Get detailed arm statistics
- `get_algorithm_info()`: Get algorithm information

#### Gaussian Sampling
```python
# Sample from Gaussian prior
empirical_mean = self.empirical_means[arm]
variance = self.gamma * math.log(self.current_round + 1) / (self.pull_counts[arm] + 1)
sample = np.random.normal(empirical_mean, math.sqrt(variance))
```

#### Parameter Tuning
- **Gamma = 1.0**: Default setting for balanced exploration-exploitation
- **Gamma > 1.0**: More exploration (higher variance)
- **Gamma < 1.0**: Less exploration (lower variance)

### Benefits
- **Algorithm Diversity**: Three different approaches (Beta TS, UCB, Gaussian TS)
- **Parameter Control**: Tunable gamma parameter for different scenarios
- **Research Ready**: Suitable for academic research and publication
- **Extensible**: Easy to add more Thompson Sampling variants

### Current Algorithm Suite
1. **CTSB**: Thompson Sampling with Beta Prior
2. **CombUCB**: Upper Confidence Bound approach
3. **CTS-G**: Thompson Sampling with Gaussian Priors

## [1.10.0] - 2024-01-20

### Changed
- **Test File Consolidation**: Merged all algorithm testing into a single comprehensive test file
- **Algorithm Comparison Integration**: Added built-in algorithm comparison functionality to main test file
- **Simplified Test Structure**: Reduced from 3 test files to 1 unified test file

### Files Deleted
- `test/test_simple_environment_memmap_cumulative_with_ci.py` - Merged into main test file
- `test/test_algorithm_comparison.py` - Merged into main test file

### Files Modified
- `test/test_simple_environment_memmap_cumulative_only.py` - Enhanced with multi-algorithm support

### New Features

#### Multi-Algorithm Support
- **Configurable Algorithms**: Can test single algorithm or multiple algorithms
- **Flexible Configuration**: `algorithms = ["CTSB"]` or `["CombUCB"]` or `["CTSB", "CombUCB"]`
- **Unified Interface**: Single test file handles all algorithm combinations

#### Enhanced Analysis
- **Multi-Algorithm Growth Analysis**: Shows regret growth for all algorithms simultaneously
- **Algorithm Comparison**: Automatic winner determination and improvement calculation
- **Comprehensive Statistics**: Detailed statistics for each algorithm

#### Improved Output
- **Multi-Algorithm Plotting**: Different markers for each algorithm (CTSB: 'o', CombUCB: 's')
- **Unified PDF Output**: Single plot showing all algorithms
- **Professional Formatting**: Consistent style across all algorithms

### Technical Implementation

#### Algorithm Configuration
```python
algorithms = ["CTSB", "CombUCB"]  # Test both algorithms
# or
algorithms = ["CTSB"]  # Test only CTSB
# or  
algorithms = ["CombUCB"]  # Test only CombUCB
```

#### Memory Management
- **Per-Algorithm Arrays**: Separate memory-mapped arrays for each algorithm
- **Efficient Storage**: `{alg.lower()}_regrets.dat` and `{alg.lower()}_rewards.dat`
- **Automatic Cleanup**: All temporary files properly cleaned up

#### Analysis Features
- **Growth Rate Analysis**: Shows regret growth rates for each algorithm
- **Sublinearity Check**: Determines if each algorithm achieves sublinear regret
- **Performance Comparison**: Calculates improvement percentages between algorithms

### Benefits
- **Simplified Maintenance**: Only one test file to maintain
- **Flexible Testing**: Easy to test single or multiple algorithms
- **Comprehensive Analysis**: All analysis features in one place
- **Professional Output**: Publication-ready PDF plots with multiple algorithms

### Final Test Structure
```
test/
└── test_simple_environment_memmap_cumulative_only.py  # Unified test file
```

### Usage Examples
```bash
# Test both algorithms (default)
python test/test_simple_environment_memmap_cumulative_only.py

# Test only CTSB (modify algorithms list in file)
algorithms = ["CTSB"]

# Test only CombUCB (modify algorithms list in file)  
algorithms = ["CombUCB"]
```

## [1.9.0] - 2024-01-20

### Added
- **CombUCB Algorithm**: Implemented Combinatorial UCB algorithm with sleeping arms support
- **Algorithm Comparison**: Added comprehensive comparison between CTS-B and CombUCB algorithms
- **UCB Formula**: θ_{a,t} = \hat{r}_{a,n_{a,t}} + \sqrt{1.5 \ln t / n_{a,t}}

### New Files
- `src/bandits/comb_ucb.py` - CombUCB algorithm implementation
- `test/test_algorithm_comparison.py` - Algorithm comparison test

### Files Modified
- `src/bandits/__init__.py` - Added CombUCB import and export

### CombUCB Algorithm Details

#### Core Formula
θ_{a,t} = \hat{r}_{a,n_{a,t}} + \sqrt{1.5 \ln t / n_{a,t}}

Where:
- \hat{r}_{a,n_{a,t}} is the empirical mean reward for arm a
- n_{a,t} is the number of times arm a has been pulled
- t is the current round number
- 1.5 is the exploration parameter

#### Key Features
- **Empirical Mean Tracking**: Maintains running average of rewards for each arm
- **UCB Exploration**: Uses upper confidence bound for exploration-exploitation balance
- **Sleeping Arms Support**: Handles arms that may not be available at every round
- **Combinatorial Selection**: Selects feasible combinations with highest UCB sum

#### Comparison Results
- **CTSB final regret**: 25.12 ± 1.80
- **CombUCB final regret**: 127.44 ± 4.92
- **Winner**: CTSB (improvement: 80.3%)

### Technical Implementation

#### CombUCB Class Methods
- `__init__(environment)`: Initialize with environment
- `select_combination()`: Select combination based on UCB values
- `_compute_ucb_value(arm)`: Compute UCB value for specific arm
- `update_posterior(played_arms, rewards)`: Update empirical means
- `get_arm_statistics()`: Get detailed arm statistics
- `get_algorithm_info()`: Get algorithm information

#### Memory Management
- Uses memory-mapped arrays for efficient large-scale simulations
- Automatic cleanup of temporary files
- Organized output structure with PDF vector graphics

### Benefits
- **Algorithm Diversity**: Two different approaches (Thompson Sampling vs UCB)
- **Performance Comparison**: Quantitative comparison of algorithm performance
- **Research Ready**: Suitable for academic research and publication
- **Extensible**: Easy to add more algorithms in the future

## [1.8.0] - 2024-01-20

### Changed
- **Complete File Cleanup**: Removed all .dat and .png files from the entire project
- **Vector Graphics Output**: Switched from PNG to PDF vector graphics for all plots
- **High-Quality Publication Ready**: All plots now output as scalable vector graphics

### Files Deleted
- `availability_matrix_10_100.dat` - Availability matrix data file
- All PNG files in `output/images/` - Converted to PDF format

### Files Modified
- `test/test_simple_environment_memmap_cumulative_only.py` - Updated to output PDF instead of PNG
- `test/test_simple_environment_memmap_cumulative_with_ci.py` - Updated to output PDF instead of PNG

### New Output Format
- **File Extension**: Changed from `.png` to `.pdf`
- **Format**: Vector graphics (scalable, publication-ready)
- **Quality**: High-resolution, suitable for academic papers
- **Size**: Smaller file sizes while maintaining quality

### Technical Details

#### PDF Vector Graphics Benefits
- **Scalable**: No quality loss when resizing
- **Publication Ready**: Suitable for academic papers and journals
- **Smaller Size**: Efficient storage and transmission
- **Professional**: Industry standard for technical documents

#### File Naming Convention
- `simple_environment_cumulative_regret_only.pdf`
- `simple_environment_cumulative_regret_with_ci.pdf`

### Current Project State
- **Clean Structure**: No legacy .dat or .png files
- **Modern Output**: All plots in PDF vector format
- **Professional**: Ready for academic publication

## [1.7.0] - 2024-01-20

### Changed
- **Output Directory Reorganization**: Restructured output directory with separate folders for images and data
- **Memory-Mapped File Management**: Memory-mapped files now stored in `output/data/` instead of temporary directories
- **Image Organization**: All plot images moved to `output/images/` directory
- **Data Organization**: All data files (JSON, DAT) moved to `output/data/` directory

### New Directory Structure
```
output/
├── images/                    # All plot images (.png files)
│   ├── simple_environment_cumulative_regret_only.png
│   ├── simple_environment_cumulative_regret_with_ci.png
│   └── ... (other plot images)
└── data/                      # All data files (.json, .dat files)
    ├── regrets.dat            # Memory-mapped regret data
    ├── rewards.dat            # Memory-mapped reward data
    └── ... (other data files)
```

### Files Modified
- `test/test_simple_environment_memmap_cumulative_only.py` - Updated to save memory-mapped files in `output/data/` and images in `output/images/`
- `test/test_simple_environment_memmap_cumulative_with_ci.py` - Updated to save memory-mapped files in `output/data/` and images in `output/images/`

### Technical Details

#### Memory-Mapped File Storage
- Memory-mapped files (`regrets.dat`, `rewards.dat`) are now created in `output/data/`
- Files are automatically cleaned up after simulation completion
- Directory structure is preserved for future use

#### Image Storage
- All plot images are automatically saved to `output/images/`
- Consistent naming convention for easy identification
- High-quality PNG format with proper DPI settings

#### Benefits
- **Organized Structure**: Clear separation between images and data
- **Professional Layout**: Standard project organization
- **Easy Navigation**: Logical file organization
- **Scalable**: Easy to add more output types in the future

## [1.6.0] - 2024-01-20

### Changed
- **Root Directory Cleanup**: Removed all PNG image files from project root directory
- **Clean Project Structure**: Root directory now contains only essential project files

### Files Deleted
- `rewards_distribution.png` - Rewards distribution plot
- `new_rng_quality.png` - RNG quality analysis plot  