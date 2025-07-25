FROM python:3.11-slim

LABEL maintainer="CombTS INFOCOM26 Team"
LABEL description="Docker image for CombTS INFOCOM26 combinatorial bandit experiments"
LABEL version="1.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    git \
    wget \
    curl \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    dvipng \
    cm-super \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Install additional packages for scientific computing and visualization
RUN pip install --no-cache-dir \
    matplotlib==3.8.* \
    seaborn==0.13.* \
    networkx==3.2.* \
    scipy==1.11.* \
    scikit-learn==1.3.* \
    pandas==2.1.* \
    tqdm==4.66.* \
    joblib==1.3.*

# Copy the entire project
COPY . .

# Create output directories
RUN mkdir -p output/data output/images

# Set proper permissions
RUN chmod +x run_experiments.sh run_experiments.py

# Create a non-root user for security
RUN useradd -m -u 1000 combts && \
    chown -R combts:combts /app
USER combts

# Set default command
CMD ["python", "run_experiments.py", "quick"]

# Expose volume mount points for data persistence
VOLUME ["/app/output", "/app/data"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import numpy, matplotlib, networkx, scipy; print('Dependencies OK')" || exit 1 