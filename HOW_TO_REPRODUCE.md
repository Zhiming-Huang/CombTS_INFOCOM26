# 🎯 One-Click to Reproduce All Results in the Paper

## TL;DR - Fastest Reproduction

```bash
# If you have Docker (RECOMMENDED - zero setup):
bash run_docker_experiments.sh full

# If you have Python 3.11+:
bash run_experiments.sh full
```

**Results**: All experiments, data, and publication-ready plots will be automatically generated in `output/`

---

## 📋 What Gets Reproduced

✅ **Routing 4x4 Mesh Network Experiments**
- Algorithm comparison (CTSB, CombUCB, CTS-G, CL-SG, BG-CTS)  
- Gamma parameter analysis for CTS-G and CL-SG
- Performance plots with confidence intervals

✅ **UCSB Real Wireless Network Experiments**
- Real network topology analysis
- 3-hop routing optimization
- Algorithm performance comparison

✅ **Publication-Ready Figures**
- Algorithm comparison plots
- Gamma parameter sensitivity analysis
- Performance metrics and statistics

---

## 🚀 Step-by-Step Instructions

### Method 1: Docker (Zero Setup) 🐳

**Requirements**: Only Docker

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd CombTS_INFOCOM26
   ```

2. **Run experiments**:
   ```bash
   # Quick test (5-10 minutes)
   bash run_docker_experiments.sh quick
   
   # Full experiments (30-60 minutes)  
   bash run_docker_experiments.sh full
   ```

3. **Check results**:
   ```bash
   ls output/images/  # Generated plots
   ls output/data/    # Simulation data
   ```

### Method 2: Native Python 🐍

**Requirements**: Python 3.11+

1. **Setup environment**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run experiments**:
   ```bash
   # Quick test
   bash run_experiments.sh quick
   
   # Full experiments
   bash run_experiments.sh full
   ```

3. **Check results**:
   ```bash
   ls output/images/  # Generated plots
   ls output/data/    # Simulation data
   ```

---

## 📊 Expected Results

After running the experiments, you should find these files in `output/images/`:

### Routing 4x4 Experiments
- `routing_4x4_algorithm_comparison.pdf` - Main algorithm comparison
- `routing_4x4_ctsg_gamma_comparison.pdf` - CTS-G gamma analysis
- `routing_4x4_clsg_gamma_comparison.pdf` - CL-SG gamma analysis
- `routing_4x4_combined_gamma_comparison.pdf` - Combined gamma analysis

### UCSB Network Experiments  
- `ucsb_comprehensive_parallel_algorithm_comparison.pdf` - Algorithm comparison
- `ucsb_comprehensive_parallel_ctsg_gamma_comparison.pdf` - CTS-G gamma analysis
- `ucsb_comprehensive_parallel_clsg_gamma_comparison.pdf` - CL-SG gamma analysis
- `ucsb_comprehensive_parallel_combined_gamma_comparison.pdf` - Combined analysis

---

## ⚡ Quick Options

### Only Generate Plots (No Re-simulation)
```bash
# Use existing data to regenerate plots
python run_experiments.py --plot-only
bash run_experiments.sh --plot-only
```

### Test Quick Setup
```bash
# Quick verification (reduced settings)
python run_experiments.py quick
bash run_experiments.sh quick
```

---

## 🔧 Troubleshooting

### Docker Issues
```bash
# Check Docker installation
docker --version

# Test Docker permissions
docker run hello-world

# Clean Docker cache if needed
docker system prune
```

### Python Issues
```bash
# Check Python version
python --version  # Should be 3.11+

# Install dependencies
pip install -r requirements.txt

# Check imports
python -c "from src.bandits import CTSB, CTSG, CLSG, BGCTS; print('All imports successful!')"
```

### Permission Issues
```bash
# Make scripts executable
chmod +x run_experiments.sh run_docker_experiments.sh
```

---

## 📞 Support

If you encounter any issues:

1. Check that your system meets the requirements
2. Verify Docker/Python installation
3. Ensure all dependencies are installed
4. Check the `EXPERIMENT_GUIDE.md` for detailed instructions

---

**🎉 Congratulations!** You've successfully reproduced all results from the paper! 