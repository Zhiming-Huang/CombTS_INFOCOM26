#!/usr/bin/env python3
"""
CombTS INFOCOM26 - One-Click Docker Experiment Runner (Python Version)
=======================================================================

This script builds a Docker image and runs experiments in an isolated 
container environment with no local dependencies required.

Features:
- Automatic Docker image building
- Volume mounting for data persistence  
- Cross-platform compatibility (Linux/macOS/Windows)
- No local Python/package installation needed
- Automatic cleanup options

Usage:
    python run_docker_experiments.py [quick|full] [--rebuild] [--cleanup]
    
Arguments:
    quick     - Run with reduced rounds/runs for fast testing (default)
    full      - Run with full paper settings (longer computation time)
    --rebuild - Force rebuild of Docker image
    --cleanup - Remove Docker image after completion
    --help    - Show this help message

=======================================================================
"""

import sys
import os
import subprocess
import argparse
import shutil
from pathlib import Path

# ANSI color codes for pretty output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    BLUE = '\033[0;34m'
    YELLOW = '\033[1;33m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color

def print_colored(text, color):
    """Print text with color."""
    print(f"{color}{text}{Colors.NC}")

def print_header(text):
    """Print a header with formatting."""
    print_colored("=" * 80, Colors.BLUE)
    print_colored(text, Colors.BLUE)
    print_colored("=" * 80, Colors.BLUE)

def run_command(cmd, description="", capture_output=True):
    """Run a command and handle errors."""
    if description:
        print_colored(f"Running {description}...", Colors.CYAN)
    
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, check=True, 
                                  capture_output=True, text=True)
            return result.stdout.strip()
        else:
            subprocess.run(cmd, shell=True, check=True)
            return True
    except subprocess.CalledProcessError as e:
        print_colored(f"❌ Command failed: {e}", Colors.RED)
        if capture_output and e.stdout:
            print(f"stdout: {e.stdout}")
        if capture_output and e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def check_docker():
    """Check if Docker is installed and running."""
    print_colored("Checking Docker installation...", Colors.CYAN)
    
    # Check if docker command exists
    if not shutil.which("docker"):
        print_colored("❌ Docker is not installed or not in PATH", Colors.RED)
        print_colored("Please install Docker from: https://docs.docker.com/get-docker/", Colors.YELLOW)
        return False
    
    # Check if Docker daemon is running
    if not run_command("docker info", capture_output=True):
        print_colored("❌ Docker is not running", Colors.RED)
        print_colored("Please start Docker and try again", Colors.YELLOW)
        return False
    
    print_colored("✓ Docker is installed and running", Colors.GREEN)
    return True

def image_exists(image_name, tag):
    """Check if Docker image exists."""
    result = run_command(f"docker images -q {image_name}:{tag}")
    return bool(result)

def build_docker_image(image_name, tag, project_root):
    """Build Docker image."""
    print()
    print_colored("Building Docker image...", Colors.CYAN)
    print("This may take 5-10 minutes on first run (downloads packages)")
    
    build_cmd = f"docker build -t {image_name}:{tag} {project_root} --build-arg BUILDKIT_INLINE_CACHE=1"
    
    if run_command(build_cmd, capture_output=False):
        print_colored("✓ Docker image built successfully", Colors.GREEN)
        return True
    else:
        print_colored("❌ Docker build failed", Colors.RED)
        return False

def remove_existing_container(container_name):
    """Remove existing container if it exists."""
    # Check if container exists
    result = run_command(f"docker ps -a --format 'table {{{{.Names}}}}' | grep -q '^{container_name}$'")
    if result is not False:  # Container exists
        print_colored("Removing existing container...", Colors.YELLOW)
        run_command(f"docker rm -f {container_name}")

def run_docker_experiment(image_name, tag, container_name, project_root, mode):
    """Run the experiment in Docker container."""
    print()
    print_colored("Starting experiment container...", Colors.CYAN)
    print_colored("Running experiments in isolated Docker environment", Colors.YELLOW)
    print_colored(f"Output will be saved to: {project_root}/output", Colors.YELLOW)
    print()
    
    # Calculate expected runtime
    if mode == "full":
        print_colored("⏱️  Expected runtime: 30-60 minutes (full experiments)", Colors.YELLOW)
    else:
        print_colored("⏱️  Expected runtime: 5-10 minutes (quick test)", Colors.YELLOW)
    print()
    
    # Build Docker run command
    docker_cmd = (
        f"docker run --rm "
        f"--name {container_name} "
        f"-v \"{project_root}/src:/app/src:ro\" "
        f"-v \"{project_root}/data:/app/data:ro\" "
        f"-v \"{project_root}/output:/app/output\" "
        f"-v \"{project_root}/example:/app/example:ro\" "
        f"-v \"{project_root}/test:/app/test:ro\" "
        f"-e PYTHONPATH=/app "
        f"{image_name}:{tag} "
        f"python run_experiments.py {mode}"
    )
    
    return run_command(docker_cmd, capture_output=False)

def show_results(project_root):
    """Show generated results."""
    print()
    print_colored("=" * 80, Colors.GREEN)
    print_colored("🎉 Docker experiments completed successfully!", Colors.GREEN)
    print_colored("=" * 80, Colors.GREEN)
    
    print_colored("Generated files in output directory:", Colors.BLUE)
    print()
    
    # List data files
    data_dir = Path(project_root) / "output" / "data"
    if data_dir.exists() and any(data_dir.glob("*.pkl")):
        print_colored("📁 Data files:", Colors.CYAN)
        for pkl_file in sorted(data_dir.glob("*.pkl")):
            size = pkl_file.stat().st_size
            size_str = f"{size / (1024*1024):.1f}MB" if size > 1024*1024 else f"{size / 1024:.1f}KB"
            print(f"  └── {pkl_file.name} ({size_str})")
    
    print()
    
    # List image files
    images_dir = Path(project_root) / "output" / "images"
    if images_dir.exists() and any(images_dir.glob("*.pdf")):
        print_colored("📊 Plot files:", Colors.CYAN)
        for pdf_file in sorted(images_dir.glob("*.pdf")):
            print(f"  ✓ {pdf_file.name}")
    
    print()
    print_colored("Next steps:", Colors.YELLOW)
    print(f"1. Review plots in: {project_root}/output/images/")
    print(f"2. Examine data files in: {project_root}/output/data/")
    print("3. Use saved data for quick plot regeneration")

def cleanup_image(image_name, tag):
    """Remove Docker image."""
    print()
    print_colored("Cleaning up Docker image...", Colors.CYAN)
    if run_command(f"docker rmi {image_name}:{tag}"):
        print_colored("✓ Docker image removed", Colors.GREEN)
    else:
        print_colored("⚠️  Failed to remove Docker image", Colors.YELLOW)

def show_docker_info(image_name, tag, project_root):
    """Show Docker usage information."""
    print()
    print_colored("Docker Information:", Colors.BLUE)
    
    # Get image size
    size_result = run_command(f"docker images {image_name}:{tag} --format 'table {{{{.Size}}}}' | tail -n +2")
    if size_result:
        print(f"Image size: {size_result}")
    
    print_colored(f"To manually run container: docker run -it --rm -v \"{project_root}/output:/app/output\" {image_name}:{tag} bash", Colors.CYAN)
    print_colored(f"To remove image: docker rmi {image_name}:{tag}", Colors.CYAN)

def main():
    """Main Docker experiment runner."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run CombTS INFOCOM26 experiments in Docker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_docker_experiments.py                    # Quick test run
  python run_docker_experiments.py full               # Full experiments
  python run_docker_experiments.py quick --rebuild    # Rebuild and test
  python run_docker_experiments.py full --cleanup     # Full run + cleanup

Requirements:
  - Docker installed and running
  - Sufficient disk space (~2GB for image + data)
  - Internet connection for initial build
        """
    )
    
    parser.add_argument('mode', nargs='?', default='quick', 
                       choices=['quick', 'full'],
                       help='Experiment mode: quick (fast testing) or full (paper settings)')
    parser.add_argument('--rebuild', action='store_true',
                       help='Force rebuild of Docker image')
    parser.add_argument('--cleanup', action='store_true',
                       help='Remove Docker image after completion')
    
    args = parser.parse_args()
    
    # Configuration
    IMAGE_NAME = "combts-infocom26"
    IMAGE_TAG = "latest" 
    CONTAINER_NAME = "combts-experiment"
    project_root = Path.cwd()
    
    # Print header
    print_header("CombTS INFOCOM26 - Docker Experiment Runner")
    print_colored(f"Mode: {args.mode}", Colors.YELLOW)
    print_colored(f"Project root: {project_root}", Colors.YELLOW)
    print_colored(f"Docker image: {IMAGE_NAME}:{IMAGE_TAG}", Colors.YELLOW)
    print_colored(f"Rebuild image: {args.rebuild}", Colors.YELLOW)
    print_colored(f"Cleanup after: {args.cleanup}", Colors.YELLOW)
    print()
    
    # Check Docker installation
    if not check_docker():
        sys.exit(1)
    
    # Check if image exists or needs rebuilding
    if not image_exists(IMAGE_NAME, IMAGE_TAG) or args.rebuild:
        if not build_docker_image(IMAGE_NAME, IMAGE_TAG, project_root):
            sys.exit(1)
    else:
        print_colored("✓ Docker image already exists", Colors.GREEN)
    
    # Create local output directories
    print()
    print_colored("Preparing output directories...", Colors.CYAN)
    (project_root / "output" / "data").mkdir(parents=True, exist_ok=True)
    (project_root / "output" / "images").mkdir(parents=True, exist_ok=True)
    
    # Remove any existing container
    remove_existing_container(CONTAINER_NAME)
    
    # Run the experiment
    if run_docker_experiment(IMAGE_NAME, IMAGE_TAG, CONTAINER_NAME, project_root, args.mode):
        show_results(project_root)
    else:
        print_colored("❌ Docker experiments failed", Colors.RED)
        sys.exit(1)
    
    # Cleanup if requested
    if args.cleanup:
        cleanup_image(IMAGE_NAME, IMAGE_TAG)
    
    # Show Docker information
    show_docker_info(IMAGE_NAME, IMAGE_TAG, project_root)
    
    print()
    print_colored("Docker experiment runner completed!", Colors.GREEN)

if __name__ == "__main__":
    main() 