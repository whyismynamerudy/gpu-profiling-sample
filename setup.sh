#!/bin/bash

# Exit immediately if any command fails
# This prevents partial installations that might leave the system in an inconsistent state
set -e

# Update package lists and upgrade existing packages
# apt update: Refreshes the list of available packages
# apt upgrade: Actually upgrades the packages to their latest versions
# -y: Automatically answers "yes" to prompts
echo "--- Updating System ---"
sudo apt update && sudo apt upgrade -y

# Install NVIDIA drivers if they're not already installed
# nvidia-smi is a utility that comes with NVIDIA drivers
# command -v checks if nvidia-smi exists in system PATH
echo "--- Installing NVIDIA Drivers ---"
if ! command -v nvidia-smi &> /dev/null; then
    # Install NVIDIA driver version 525
    # This version is compatible with the T4 GPU in g4dn.xlarge
    sudo apt install -y nvidia-driver-525
fi

# Download and install CUDA toolkit
# wget: Downloads the CUDA installer
# --silent: Runs installer without interactive prompts
# --toolkit: Installs CUDA toolkit (compiler, libraries, etc.)
# --driver: Includes driver components
# --override: Overwrites existing installations
echo "--- Installing CUDA ---"
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run --silent --toolkit --driver --override

# Add CUDA binaries and libraries to system PATH
# These lines ensure the system can find CUDA executables and libraries
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Install Docker
# docker.io is the Ubuntu package for Docker
echo "--- Installing Docker ---"
sudo apt install -y docker.io
# Start Docker daemon now and enable it to start on boot
sudo systemctl start docker
sudo systemctl enable docker
# Add current user to docker group to allow running docker without sudo
sudo usermod -aG docker $USER

# Set up NVIDIA Container Toolkit
# This allows Docker containers to access the GPU
echo "--- Installing NVIDIA Container Toolkit ---"
# Get the OS distribution info for the repository configuration
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
# Add NVIDIA's package repositories
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
# Update package lists to include NVIDIA packages
sudo apt update
# Install the NVIDIA Container Toolkit
sudo apt install -y nvidia-container-toolkit
# Configure Docker to use NVIDIA Container Runtime
sudo nvidia-ctk runtime configure --runtime=docker
# Restart Docker to apply changes
sudo systemctl restart docker

# Configure Docker daemon to use NVIDIA runtime by default
# This makes the NVIDIA runtime available to all containers
echo "--- Setting up Docker with NVIDIA Runtime ---"
sudo tee /etc/docker/daemon.json << EOF
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF
# Restart Docker again to apply runtime changes
sudo systemctl restart docker

# Clone the repository containing CUDA code
echo "--- Cloning CUDA Matrix Multiplication Repository ---"
git clone https://github.com/whyismynamerudy/gpu-profiling-sample.git

echo "--- Setup Complete ---"
# Notify user about needed logout for group changes
echo "NOTE: You may need to log out and back in for group changes to take effect"

# Verify installations
# Test NVIDIA driver installation
nvidia-smi
# Test Docker GPU access by running a test container
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi