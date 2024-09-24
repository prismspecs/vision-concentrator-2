# Use NVIDIA's CUDA base image with Ubuntu 22.04 and cuDNN 8
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MODEL_FILE="/app/models/sd_xl_turbo_1.0_fp16.safetensors"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    git \
    wget \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create a symbolic link for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip using the correct python interpreter
RUN python -m pip install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA support using the correct python interpreter
RUN python -m pip install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# Install other Python dependencies using the correct python interpreter
RUN python -m pip install --no-cache-dir \
    diffusers>=0.20.0 \
    transformers>=4.30.0 \
    moviepy \
    tqdm \
    numpy \
    accelerate \
    safetensors \
    flask

# Install xformers
RUN python -m pip install --no-cache-dir xformers==0.0.20

# Verify that torch, diffusers, and xformers are installed
RUN python -c "import torch; import diffusers; import xformers; print('torch:', torch.__version__, 'diffusers:', diffusers.__version__, 'xformers:', xformers.__version__)"

# Set the working directory
WORKDIR /app

# Copy the contents of the app directory into /app in the container
#COPY app/ /app/

# Expose port 5000 for the Flask app
EXPOSE 5000

# Command to run the Flask application
CMD ["python", "app.py"]
