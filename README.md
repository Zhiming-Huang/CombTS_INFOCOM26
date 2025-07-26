# CombTS INFOCOM26 - Combinatorial Thompson Sampling for Wireless Networks

[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](Dockerfile)
[![Python](https://img.shields.io/badge/Python-3.11+-green?logo=python)](requirements.txt)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the implementation and experiments for the INFOCOM26 paper on Combinatorial Thompson Sampling algorithms for wireless network optimization.

## 🚀 Quick Start

### Zero-Setup Docker Reproduction (Recommended)

**Requirements**: Only Docker installed

```bash
# Quick test (5-10 minutes)
bash run_docker_experiments.sh quick

# Full experiments (30-60 minutes) 
bash run_docker_experiments.sh full
```

### Native Python Reproduction

**Requirements**: Python 3.11+, dependencies installed

```bash
# Install dependencies
pip install -r requirements.txt

# Quick test
bash run_experiments.sh quick

# Full experiments
bash run_experiments.sh full
```

## 📋 Available Methods

| Method | Requirements | Setup Time | Advantages |
|--------|--------------|------------|------------|
| 🐳 **Docker** | Docker only | Auto | ✅ Zero config, identical environment |
| 🐍 **Native** | Python 3.11+ | Manual | ✅ Direct execution, customizable |

## 🐳 Docker Usage

### One-Command Execution
```bash
# Test Docker setup
bash test_docker.sh

# Run experiments with automatic cleanup
bash run_docker_experiments.sh full --cleanup

# Python version
python run_docker_experiments.py quick
```

### Docker Features
- ✅ **Automatic building**: Image builds on first run
- ✅ **Volume mounting**: Code and data automatically mounted
- ✅ **Cross-platform**: Works on Linux, macOS, Windows
- ✅ **Isolation**: No conflicts with local Python environment
- ✅ **Reproducibility**: Identical environment across systems

### Manual Docker Commands
```bash
# Build image manually
docker build -t combts-infocom26:latest .

# Run interactive container
docker run -it --rm \
  -v "$(pwd)/output:/app/output" \
  combts-infocom26:latest bash

# Run specific experiment
docker run --rm \
  -v "$(pwd)/src:/app/src:ro" \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/output:/app/output" \
  combts-infocom26:latest \
  python run_experiments.py quick
```

## 🔬 Experiments

### Network Environments

1. **Routing 4x4 Mesh Network**
   - 4×4 grid topology, 16 nodes, 24 links
   - Source: node 0 → Destination: node 15
   - Synthetic environment with 75% link availability

2. **UCSB Real Wireless Network**
   - Real deployment traces from UC Santa Barbara
   - Node pair: 10.1.1.102 → 10.1.1.25
   - Maximum 3-hop constraint

### Algorithms Tested

- **CTS-B**: Combinatorial Thompson Sampling with Beta Prior
- **CombUCB**: Combinatorial Upper Confidence Bound
- **BG-CTS**: Boosted Gaussian-Combinatorial Thompson Sampling  
- **CTS-G**: Combinatorial Thompson Sampling with Gaussian Priors (γ ∈ {0.01, 0.1, 0.5, 1.0})
- **CL-SG**: Combinatorial Learning with Single Gaussian (γ ∈ {0.01, 0.1, 0.5, 1.0})

### Generated Results

```
output/
├── data/           # Simulation data (.pkl files)
└── images/         # Publication-quality plots (.pdf)
    ├── routing_4x4_algorithm_comparison.pdf
    ├── routing_4x4_combined_gamma_comparison.pdf
    ├── ucsb_comprehensive_parallel_algorithm_comparison.pdf
    └── ucsb_comprehensive_parallel_combined_gamma_comparison.pdf
```

## 📊 Visualization Features

All plots include:
- **Log-scale y-axis** for better regret visualization
- **Multi-marker system** for algorithm distinction
- **LaTeX rendering** for mathematical notation
- **Confidence intervals** with transparency
- **Publication-ready PDF** output

## 📖 Documentation

- **[EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)** - Comprehensive reproduction guide
- **[Dockerfile](Dockerfile)** - Container environment specification
- **[requirements.txt](requirements.txt)** - Python dependencies

## 🛠️ Development

### Project Structure
```
CombTS_INFOCOM26/
├── src/                    # Core algorithms
│   ├── bandits/           # Bandit algorithms
│   ├── environments/      # Network environments  
│   └── utils/             # Plotting and utilities
├── example/               # Experiment scripts
├── data/                  # Network datasets
├── test/                  # Unit tests
├── output/                # Results directory
├── Dockerfile             # Docker configuration
├── run_docker_experiments.sh    # Docker runner (Bash)
├── run_docker_experiments.py    # Docker runner (Python)
├── run_experiments.sh           # Native runner (Bash)
└── run_experiments.py           # Native runner (Python)
```

### Adding New Algorithms

1. Implement algorithm in `src/bandits/`
2. Add to experiment scripts in `example/`
3. Update plotting utilities in `src/utils/`
4. Test with Docker: `bash run_docker_experiments.sh quick`

### Environment Variables

Docker containers use:
- `PYTHONPATH=/app` - Module resolution
- `PYTHONUNBUFFERED=1` - Real-time output
- Non-root user for security

## 🔧 Troubleshooting

### Docker Issues
```bash
# Test Docker setup
bash test_docker.sh

# Clean rebuild
docker system prune
bash run_docker_experiments.sh quick --rebuild

# Check image
docker images combts-infocom26:latest
```

### Native Issues
```bash
# Install dependencies
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.11+

# Test imports
python -c "import numpy, matplotlib, networkx, scipy"
```

### Memory Issues
```bash
# Use quick mode
bash run_docker_experiments.sh quick

# Monitor Docker resources
docker stats
```

## 📈 Performance

| Mode | Routing | UCSB | Total Time | Memory |
|------|---------|------|------------|--------|
| Quick | 1K×3 | 3K×3 | ~5-10 min | ~2GB |
| Full | 10K×5 | 10K×5 | ~30-60 min | ~4GB |

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Test with Docker: `bash run_docker_experiments.sh quick`
4. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📝 Citation

```bibtex
@inproceedings{combts-infocom26,
  title={Combinatorial Thompson Sampling for Wireless Network Optimization},
  author={Authors},
  booktitle={IEEE INFOCOM 2026},
  year={2026}
}
```

## 🔗 Links

- **Paper**: [arXiv:xxxx.xxxxx](https://arxiv.org/abs/xxxx.xxxxx)
- **Docker Hub**: [combts-infocom26](https://hub.docker.com/r/combts-infocom26) (if published)
- **Documentation**: [GitHub Pages](https://username.github.io/CombTS_INFOCOM26)

---

**Quick Start Summary:**
1. **Docker**: `bash run_docker_experiments.sh quick` (No setup required)
2. **Native**: `pip install -r requirements.txt && bash run_experiments.sh quick`
3. **Results**: Check `output/images/*.pdf` for publication-quality plots 