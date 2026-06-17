# CombTS INFOCOM26 - Experiment Reproduction Guide

This guide provides instructions for reproducing all experiments from the CombTS INFOCOM26 paper.

## 🚀 Quick Start

### One-Click Reproduction

We provide **multiple options** for complete experiment reproduction:

#### Option 1: Docker (Recommended - No Setup Required) 🐳

**Zero Dependencies - Just Docker:**
```bash
# Quick test run (5-10 minutes)
bash run_docker_experiments.sh quick

# Full paper experiments (30-60 minutes)
bash run_docker_experiments.sh full

# Python version
python run_docker_experiments.py quick
python run_docker_experiments.py full
```

**Docker Advantages:**
- ✅ **No local Python/package installation needed**
- ✅ **Identical environment across all systems**
- ✅ **Automatic dependency management**
- ✅ **Cross-platform compatibility** (Linux/macOS/Windows)
- ✅ **Isolated execution** (no conflicts with local setup)

#### Option 2: Native Execution (If Python Environment Available)

**Bash Script:**
```bash
# Quick test run (reduced parameters)
bash run_experiments.sh quick

# Full paper experiments (longer runtime)
bash run_experiments.sh full
```

**Python Script:**
```bash
# Quick test run
python run_experiments.py quick

# Full paper experiments  
python run_experiments.py full
```

### Parameters

| Mode | Routing 4x4 | UCSB Network | Runtime | Use Case |
|------|-------------|--------------|---------|----------|
| `quick` | 1000 rounds × 3 runs | 3000 rounds × 3 runs | ~5-10 min | Testing & validation |
| `full` | 10000 rounds × 5 runs | 10000 rounds × 5 runs | ~30-60 min | Paper reproduction |

## 🐳 Docker Usage (Detailed)

### Prerequisites

1. **Install Docker**: Download from [docker.com](https://docs.docker.com/get-docker/)
2. **Start Docker**: Ensure Docker daemon is running
3. **Disk Space**: ~2GB for image + output data

### Docker Commands

**Basic Usage:**
```bash
# Quick test with auto-build
bash run_docker_experiments.sh quick

# Full experiments with cleanup after
bash run_docker_experiments.sh full --cleanup

# Force rebuild image
bash run_docker_experiments.sh quick --rebuild
```

**Advanced Options:**
```bash
# Get help
bash run_docker_experiments.sh --help

# Manual container run
docker run -it --rm \
  -v "$(pwd)/output:/app/output" \
  combts-infocom26:latest bash

# Check image size
docker images combts-infocom26:latest
```

### Docker Features

- **Volume Mounting**: `src/`, `data/`, `output/`, `example/`, `test/` are mounted
- **Automatic Building**: Image builds automatically on first run
- **Data Persistence**: Results saved to local `output/` directory
- **Security**: Runs as non-root user inside container
- **Health Checks**: Built-in dependency verification

### Troubleshooting Docker

**Docker not found:**
```bash
# Install Docker first
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

**Permission denied:**
```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
# Log out and back in
```

**Build failures:**
```bash
# Clean rebuild
docker system prune
bash run_docker_experiments.sh quick --rebuild
```

**Memory issues:**
```bash
# Increase Docker memory limit in Docker Desktop
# or use quick mode
bash run_docker_experiments.sh quick
```

## 📊 Generated Results

### Output Structure
```
output/
├── data/           # Saved simulation results (.pkl files)
│   ├── routing_4x4_results_*.pkl
│   └── ucsb_results_*.pkl
└── images/         # Generated plots (.pdf files)
    ├── routing_4x4_algorithm_comparison.pdf
    ├── routing_4x4_tclsg_gamma_comparison.pdf
    ├── routing_4x4_combined_gamma_comparison.pdf
    ├── ucsb_comprehensive_parallel_algorithm_comparison.pdf
    ├── ucsb_comprehensive_parallel_tclsg_gamma_comparison.pdf
    └── ucsb_comprehensive_parallel_combined_gamma_comparison.pdf
```

### Key Plots

1. **Algorithm Comparison**: Performance of all algorithms (CTSB, CombUCB, BG-CTS, CTS-G, CL-SG, T-CL-SG)
2. **Gamma Comparison**: Effect of different γ values for CTS-G, CL-SG, and T-CL-SG algorithms
3. **Combined Gamma**: Unified view of both algorithms across all γ values

## 🔧 Manual Execution

### Individual Experiments

**Routing 4x4 Mesh Network:**
```bash
# Run and save data
python example/example_routing_4x4_memmap.py --rounds 10000 --runs 5 --keep-memmap

# Load and replot existing data
python example/example_routing_4x4_memmap.py --load-data "output/data/routing_4x4_results_*.pkl"
```

**UCSB Real Network:**
```bash
# Run new simulation
python example/example_ucsb_comprehensive_parallel.py --rounds 10000 --runs 5

# Load existing data (example)
python example/example_ucsb_comprehensive_parallel.py --load-data "output/data/ucsb_results_*.pkl"
```

### Quick Plot Regeneration

After running experiments, you can quickly regenerate plots from saved data:

```bash
# Find latest data files and regenerate plots
python example/example_routing_4x4_memmap.py --load-data "$(ls output/data/routing_4x4_results_*.pkl | tail -1)"
python example/example_ucsb_comprehensive_parallel.py --load-data "$(ls output/data/ucsb_results_*.pkl | tail -1)"
```

## 🎯 Experiment Details

### Routing 4x4 Mesh Network
- **Topology**: 4×4 grid with 16 nodes, 24 directed links
- **Route**: Source node 0 → Destination node 15
- **Environment**: 75% link availability, optimal path length 6
- **Algorithms**: All 5 algorithms with 4 γ values each for CTS-G/CL-SG

### UCSB Real Wireless Network
- **Dataset**: UC Santa Barbara mesh network traces
- **Trace Period**: 1144393236-1144450070 (real timestamps)
- **Route**: Node 10.1.1.102 → Node 10.1.1.25  
- **Constraint**: Maximum 3-hop paths
- **Environment**: Real wireless link variations

## 📈 Visualization Features

All plots include:
- ✅ **Log-scale y-axis** for better regret visualization
- ✅ **Legend inside plots** (lower right, 8pt font)
- ✅ **Multi-marker system** for γ value distinction
- ✅ **LaTeX rendering** for mathematical symbols
- ✅ **PDF output** with publication quality
- ✅ **Confidence intervals** with transparent shading

## ⚡ Performance Tips

1. **Use Docker for consistency** across different systems
2. **Use `quick` mode** for initial testing and validation
3. **Save data** with `--keep-memmap` for later plot regeneration
4. **Parallel execution** is enabled by default for UCSB experiments
5. **Memory mapping** reduces memory usage for large-scale simulations

## 🔍 Troubleshooting

### Common Issues

**Memory errors:**
```bash
# Docker: Use quick mode
bash run_docker_experiments.sh quick

# Native: Reduce parameters
python run_experiments.py quick
```

**Missing plots:**
```bash
# Check if data was saved and regenerate
ls output/data/*.pkl
python example/example_*_memmap.py --load-data "output/data/[latest_file].pkl"
```

**Permission errors:**
```bash
# Ensure scripts are executable
chmod +x run_docker_experiments.sh run_docker_experiments.py
chmod +x run_experiments.sh run_experiments.py
```

**Docker build errors:**
```bash
# Clean and rebuild
docker system prune
bash run_docker_experiments.sh quick --rebuild
```

## 📝 Customization

### Modify Experiment Parameters

**Docker version:** Edit `run_docker_experiments.sh` or `run_docker_experiments.py`
**Native version:** Edit `run_experiments.sh` or `run_experiments.py`

Example modifications:
```python
# In run_experiments.py, modify these lines:
routing_rounds, routing_runs = 5000, 10  # Custom parameters
ucsb_rounds, ucsb_runs = 8000, 7        # Custom parameters
```

### Docker Image Customization

Modify `Dockerfile` to:
- Add additional Python packages
- Change base image version
- Include custom dependencies
- Modify user permissions

## 📚 File References

### Docker Files
- `Dockerfile` - Container image definition
- `run_docker_experiments.sh` - Bash Docker runner
- `run_docker_experiments.py` - Python Docker runner
- `.dockerignore` - Build context optimization

### Native Execution Files
- `run_experiments.sh` - Bash reproduction script
- `run_experiments.py` - Python reproduction script  
- `example/example_routing_4x4_memmap.py` - Routing experiment
- `example/example_ucsb_comprehensive_parallel.py` - UCSB experiment
- `src/utils/plotting.py` - Visualization functions
- `output/` - Results directory

## 🎯 Expected Results

### Performance Ranking (Typical)

**Routing 4x4:**
1. CL-SG (γ=0.01) - Best performance
2. CTS-G variants - Good performance  
3. CTSB - Baseline
4. CombUCB - Moderate
5. BG-CTS - Highest regret

**UCSB Network:**
1. CL-SG (γ=0.1) - Best for real networks
2. CTSB - Strong baseline
3. CTS-G variants - Good adaptation
4. CombUCB - Moderate  
5. BG-CTS - Highest regret

The exact values will vary between runs due to randomness, but the relative ranking should be consistent.

## 🚀 Recommended Workflow

1. **First Time Setup**: Use Docker for guaranteed compatibility
   ```bash
   bash run_docker_experiments.sh quick
   ```

2. **Verify Results**: Check generated plots and data files

3. **Full Reproduction**: Run complete experiments
   ```bash
   bash run_docker_experiments.sh full
   ```

4. **Analysis**: Use saved data for quick plot modifications
   ```bash
   # Docker
   docker run -it --rm -v "$(pwd)/output:/app/output" combts-infocom26:latest bash
   
   # Native  
   python example/example_routing_4x4_memmap.py --load-data "output/data/latest_file.pkl"
   ```

5. **Publication**: Use generated PDFs in `output/images/` for papers 
