#!/bin/bash
# =============================================================================
# CombTS INFOCOM26 - One-Click Docker Experiment Runner
# =============================================================================
# This script builds a Docker image and runs experiments in an isolated 
# container environment with no local dependencies required.
#
# Features:
# - Automatic Docker image building
# - Volume mounting for data persistence  
# - Cross-platform compatibility (Linux/macOS/Windows)
# - No local Python/package installation needed
# - Automatic cleanup options
#
# Usage:
#   bash run_docker_experiments.sh [quick|full] [--rebuild] [--cleanup]
#   
# Arguments:
#   quick     - Run with reduced rounds/runs for fast testing (default)
#   full      - Run with full paper settings (longer computation time)
#   --rebuild - Force rebuild of Docker image
#   --cleanup - Remove Docker image after completion
#   --help    - Show this help message
#
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="combts-infocom26"
IMAGE_TAG="latest"
CONTAINER_NAME="combts-experiment"
PROJECT_ROOT=$(pwd)

# Parse command line arguments
MODE="quick"
REBUILD_IMAGE=false
CLEANUP_IMAGE=false
SHOW_HELP=false

for arg in "$@"; do
    case $arg in
        quick|full)
            MODE="$arg"
            ;;
        --rebuild)
            REBUILD_IMAGE=true
            ;;
        --cleanup)
            CLEANUP_IMAGE=true
            ;;
        --help|-h)
            SHOW_HELP=true
            ;;
        *)
            echo -e "${RED}Unknown argument: $arg${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Show help if requested
if [ "$SHOW_HELP" = true ]; then
    echo -e "${BLUE}CombTS INFOCOM26 - Docker Experiment Runner${NC}"
    echo ""
    echo -e "${YELLOW}Usage:${NC}"
    echo "  bash run_docker_experiments.sh [quick|full] [--rebuild] [--cleanup]"
    echo ""
    echo -e "${YELLOW}Arguments:${NC}"
    echo "  quick     - Run with reduced rounds/runs for fast testing (default)"
    echo "  full      - Run with full paper settings (longer computation time)"
    echo "  --rebuild - Force rebuild of Docker image"
    echo "  --cleanup - Remove Docker image after completion"
    echo "  --help    - Show this help message"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  bash run_docker_experiments.sh                    # Quick test run"
    echo "  bash run_docker_experiments.sh full               # Full experiments"
    echo "  bash run_docker_experiments.sh quick --rebuild    # Rebuild and test"
    echo "  bash run_docker_experiments.sh full --cleanup     # Full run + cleanup"
    echo ""
    echo -e "${YELLOW}Requirements:${NC}"
    echo "  - Docker installed and running"
    echo "  - Sufficient disk space (~2GB for image + data)"
    echo "  - Internet connection for initial build"
    exit 0
fi

echo -e "${BLUE}==============================================================================${NC}"
echo -e "${BLUE}CombTS INFOCOM26 - Docker Experiment Runner${NC}"
echo -e "${BLUE}==============================================================================${NC}"
echo -e "Mode: ${YELLOW}$MODE${NC}"
echo -e "Project root: ${YELLOW}$PROJECT_ROOT${NC}"
echo -e "Docker image: ${YELLOW}$IMAGE_NAME:$IMAGE_TAG${NC}"
echo -e "Rebuild image: ${YELLOW}$REBUILD_IMAGE${NC}"
echo -e "Cleanup after: ${YELLOW}$CLEANUP_IMAGE${NC}"
echo ""

# Check if Docker is installed and running
echo -e "${CYAN}Checking Docker installation...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed or not in PATH${NC}"
    echo -e "${YELLOW}Please install Docker from: https://docs.docker.com/get-docker/${NC}"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker is not running${NC}"
    echo -e "${YELLOW}Please start Docker and try again${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker is installed and running${NC}"

# Check if image exists or needs rebuilding
IMAGE_EXISTS=$(docker images -q $IMAGE_NAME:$IMAGE_TAG 2> /dev/null)
if [ -z "$IMAGE_EXISTS" ] || [ "$REBUILD_IMAGE" = true ]; then
    echo ""
    echo -e "${CYAN}Building Docker image...${NC}"
    echo "This may take 5-10 minutes on first run (downloads packages)"
    
    docker build -t $IMAGE_NAME:$IMAGE_TAG . \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        || {
            echo -e "${RED}❌ Docker build failed${NC}"
            exit 1
        }
    
    echo -e "${GREEN}✓ Docker image built successfully${NC}"
else
    echo -e "${GREEN}✓ Docker image already exists${NC}"
fi

# Create local output directories
echo ""
echo -e "${CYAN}Preparing output directories...${NC}"
mkdir -p "$PROJECT_ROOT/output/data" "$PROJECT_ROOT/output/images"

# Stop and remove any existing container
if docker ps -a --format 'table {{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
    echo -e "${YELLOW}Removing existing container...${NC}"
    docker rm -f $CONTAINER_NAME &> /dev/null || true
fi

# Run the experiment in Docker container
echo ""
echo -e "${CYAN}Starting experiment container...${NC}"
echo -e "${YELLOW}Running experiments in isolated Docker environment${NC}"
echo -e "${YELLOW}Output will be saved to: $PROJECT_ROOT/output${NC}"
echo ""

# Calculate expected runtime
if [ "$MODE" = "full" ]; then
    echo -e "${YELLOW}⏱️  Expected runtime: 30-60 minutes (full experiments)${NC}"
else
    echo -e "${YELLOW}⏱️  Expected runtime: 5-10 minutes (quick test)${NC}"
fi
echo ""

# Run Docker container with volume mounts
docker run --rm \
    --name $CONTAINER_NAME \
    -v "$PROJECT_ROOT/src:/app/src:ro" \
    -v "$PROJECT_ROOT/data:/app/data:ro" \
    -v "$PROJECT_ROOT/output:/app/output" \
    -v "$PROJECT_ROOT/example:/app/example:ro" \
    -v "$PROJECT_ROOT/test:/app/test:ro" \
    -e PYTHONPATH=/app \
    $IMAGE_NAME:$IMAGE_TAG \
    python run_experiments.py $MODE

# Check if experiments completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}==============================================================================${NC}"
    echo -e "${GREEN}🎉 Docker experiments completed successfully!${NC}"
    echo -e "${GREEN}==============================================================================${NC}"
    
    # Show generated files
    echo -e "${BLUE}Generated files in output directory:${NC}"
    echo ""
    
    # List data files
    if [ -d "$PROJECT_ROOT/output/data" ] && [ "$(ls -A $PROJECT_ROOT/output/data 2>/dev/null)" ]; then
        echo -e "${CYAN}📁 Data files:${NC}"
        ls -la "$PROJECT_ROOT/output/data"/*.pkl 2>/dev/null | while read -r line; do
            file=$(echo $line | awk '{print $9}')
            size=$(echo $line | awk '{print $5}')
            echo "  └── $(basename $file) ($(echo $size | numfmt --to=iec --suffix=B))"
        done
    fi
    
    echo ""
    
    # List image files  
    if [ -d "$PROJECT_ROOT/output/images" ] && [ "$(ls -A $PROJECT_ROOT/output/images 2>/dev/null)" ]; then
        echo -e "${CYAN}📊 Plot files:${NC}"
        find "$PROJECT_ROOT/output/images" -name "*.pdf" -type f | sort | while read -r file; do
            echo "  ✓ $(basename $file)"
        done
    fi
    
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Review plots in: $PROJECT_ROOT/output/images/"
    echo "2. Examine data files in: $PROJECT_ROOT/output/data/"
    echo "3. Use saved data for quick plot regeneration"
    
else
    echo -e "${RED}❌ Docker experiments failed${NC}"
    exit 1
fi

# Cleanup if requested
if [ "$CLEANUP_IMAGE" = true ]; then
    echo ""
    echo -e "${CYAN}Cleaning up Docker image...${NC}"
    docker rmi $IMAGE_NAME:$IMAGE_TAG &> /dev/null || true
    echo -e "${GREEN}✓ Docker image removed${NC}"
fi

# Show Docker usage information
echo ""
echo -e "${BLUE}Docker Information:${NC}"
echo -e "Image size: $(docker images $IMAGE_NAME:$IMAGE_TAG --format 'table {{.Size}}' | tail -n +2)"
echo -e "To manually run container: ${CYAN}docker run -it --rm -v \"$PROJECT_ROOT/output:/app/output\" $IMAGE_NAME:$IMAGE_TAG bash${NC}"
echo -e "To remove image: ${CYAN}docker rmi $IMAGE_NAME:$IMAGE_TAG${NC}"

echo ""
echo -e "${GREEN}Docker experiment runner completed!${NC}" 