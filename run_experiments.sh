#!/bin/bash
# =============================================================================
# CombTS INFOCOM26 - One-Click Experiment Reproduction Script
# =============================================================================
# This script reproduces all experiments with the exact settings used in the paper
# 
# Features:
# - Routing 4x4 mesh network simulation
# - UCSB real wireless mesh network simulation  
# - Automatic data saving and plot generation
# - Log scale visualization with optimized legends
# - Memory-mapped efficient computation
#
# Usage:
#   bash run_experiments.sh [quick|full]
#   
# Arguments:
#   quick - Run with reduced rounds/runs for fast testing (default)
#   full  - Run with full paper settings (longer computation time)
#
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
MODE=${1:-quick}  # Default to quick mode
PROJECT_ROOT=$(pwd)
OUTPUT_DIR="$PROJECT_ROOT/output"
DATA_DIR="$OUTPUT_DIR/data" 
IMAGES_DIR="$OUTPUT_DIR/images"

# Create output directories
mkdir -p "$DATA_DIR" "$IMAGES_DIR"

echo -e "${BLUE}==============================================================================${NC}"
echo -e "${BLUE}CombTS INFOCOM26 - Experiment Reproduction Script${NC}"
echo -e "${BLUE}==============================================================================${NC}"
echo -e "Mode: ${YELLOW}$MODE${NC}"
echo -e "Project root: ${YELLOW}$PROJECT_ROOT${NC}"
echo -e "Output directory: ${YELLOW}$OUTPUT_DIR${NC}"
echo ""

# Set experiment parameters based on mode
if [ "$MODE" = "full" ]; then
    echo -e "${YELLOW}Running FULL experiments (paper settings)${NC}"
    ROUTING_ROUNDS=10000
    ROUTING_RUNS=5
    UCSB_ROUNDS=10000
    UCSB_RUNS=100  # Changed to 100 runs for better statistics
else
    echo -e "${YELLOW}Running QUICK experiments (reduced settings for testing)${NC}"
    ROUTING_ROUNDS=1000
    ROUTING_RUNS=3
    UCSB_ROUNDS=3000
    UCSB_RUNS=3
fi

echo "Routing 4x4: $ROUTING_ROUNDS rounds × $ROUTING_RUNS runs"
echo "UCSB: $UCSB_ROUNDS rounds × $UCSB_RUNS runs"
echo ""

# =============================================================================
# Experiment 1: Routing 4x4 Mesh Network
# =============================================================================
echo -e "${GREEN}==============================================================================${NC}"
echo -e "${GREEN}Experiment 1: Routing 4x4 Mesh Network${NC}"
echo -e "${GREEN}==============================================================================${NC}"
echo "Configuration:"
echo "  - Network: 4×4 mesh topology"
echo "  - Source: node 0, Destination: node 15" 
echo "  - Links: 24 arms with 75% availability"
echo "  - Algorithms: CTSB, CombUCB, BG-CTS, CTS-G, CL-SG"
echo "  - Gamma values: [0.01, 0.1, 0.5, 1.0]"
echo ""

echo -e "${BLUE}Running routing 4x4 simulation...${NC}"
python example/example_routing_4x4_memmap.py \
    --rounds $ROUTING_ROUNDS \
    --runs $ROUTING_RUNS \
    --keep-memmap \
    --default-gamma 0.01

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Routing 4x4 experiment completed successfully${NC}"
else
    echo -e "${RED}✗ Routing 4x4 experiment failed${NC}"
    exit 1
fi

# =============================================================================
# Experiment 2: UCSB Real Wireless Mesh Network
# =============================================================================
echo ""
echo -e "${GREEN}==============================================================================${NC}"
echo -e "${GREEN}Experiment 2: UCSB Real Wireless Mesh Network${NC}"
echo -e "${GREEN}==============================================================================${NC}"
echo "Configuration:"
echo "  - Network: Real deployment at UC Santa Barbara"
echo "  - Trace period: 1144393236-1144450070"
echo "  - Node pair: 10.1.1.102 → 10.1.1.25"
echo "  - Max path length: 3 hops"
echo "  - Algorithms: CTSB, CombUCB, BG-CTS, CTS-G, CL-SG"
echo "  - Gamma values: [0.01, 0.1, 0.5, 1.0]"
echo ""

echo -e "${BLUE}Running UCSB mesh network simulation...${NC}"
python example/example_ucsb_comprehensive_parallel.py \
    --fixed-source 10.1.1.100 \
    --fixed-destination 10.1.1.102 \
    --trace-period 1144393236-1144450070 \
    --max-path-length 3 \
    --rounds $UCSB_ROUNDS \
    --runs $UCSB_RUNS \
    --main-seed 123 \
    --sequential \
    --save-data output/data

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ UCSB mesh network experiment completed successfully${NC}"
else
    echo -e "${RED}✗ UCSB mesh network experiment failed${NC}"
    exit 1
fi

# =============================================================================
# Results Summary
# =============================================================================
echo ""
echo -e "${GREEN}==============================================================================${NC}"
echo -e "${GREEN}Experiment Completion Summary${NC}"
echo -e "${GREEN}==============================================================================${NC}"

echo -e "${BLUE}Generated Data Files:${NC}"
echo "📁 Data directory: $DATA_DIR"
find "$DATA_DIR" -name "*.pkl" -newer "$0" 2>/dev/null | head -10 | while read file; do
    size=$(du -h "$file" | cut -f1)
    echo "  └── $(basename "$file") ($size)"
done

echo ""
echo -e "${BLUE}Generated Plot Files:${NC}"
echo "📊 Images directory: $IMAGES_DIR"

# List routing plots
echo "  Routing 4x4 plots:"
for plot in algorithm_comparison ctsg_gamma_comparison clsg_gamma_comparison combined_gamma_comparison; do
    file="$IMAGES_DIR/routing_4x4_${plot}.pdf"
    if [ -f "$file" ]; then
        echo -e "    ✓ routing_4x4_${plot}.pdf"
    else
        echo -e "    ✗ routing_4x4_${plot}.pdf ${RED}(missing)${NC}"
    fi
done

# List UCSB plots  
echo "  UCSB mesh network plots:"
for plot in algorithm_comparison ctsg_gamma_comparison clsg_gamma_comparison combined_gamma_comparison; do
    file="$IMAGES_DIR/ucsb_comprehensive_parallel_${plot}.pdf"
    if [ -f "$file" ]; then
        echo -e "    ✓ ucsb_comprehensive_parallel_${plot}.pdf"
    else
        echo -e "    ✗ ucsb_comprehensive_parallel_${plot}.pdf ${RED}(missing)${NC}"
    fi
done

# =============================================================================
# Visualization Features Summary
# =============================================================================
echo ""
echo -e "${BLUE}Visualization Features:${NC}"
echo "  ✓ Log-scale y-axis for better regret visualization"
echo "  ✓ Legend positioned inside plots (lower right)"
echo "  ✓ Multi-marker system for gamma value distinction"
echo "  ✓ LaTeX rendering for mathematical symbols"
echo "  ✓ PDF output with publication-quality formatting"
echo "  ✓ Confidence intervals with transparent shading"

# =============================================================================
# Performance Statistics
# =============================================================================
echo ""
echo -e "${BLUE}Quick Analysis Commands:${NC}"
echo "To reload and regenerate plots from saved data:"
echo "  # Routing 4x4:"
echo "  python example/example_routing_4x4_memmap.py --load-data \"\$(ls $DATA_DIR/routing_4x4_results_*.pkl | tail -1)\""
echo ""
echo "  # UCSB network:"  
echo "  python example/example_ucsb_comprehensive_parallel.py --load-data \"\$(ls $DATA_DIR/ucsb_results_*.pkl | tail -1)\""

echo ""
echo -e "${GREEN}==============================================================================${NC}"
echo -e "${GREEN}🎉 All experiments completed successfully!${NC}"
echo -e "${GREEN}==============================================================================${NC}"
echo "Results are saved in: $OUTPUT_DIR"
echo "📊 Plots: $IMAGES_DIR/*.pdf"
echo "📁 Data: $DATA_DIR/*.pkl"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Review generated plots in $IMAGES_DIR"
echo "2. Use saved data files for quick plot regeneration"
echo "3. Modify experiment parameters in this script for custom runs" 