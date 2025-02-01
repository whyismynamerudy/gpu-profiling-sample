#!/bin/bash
set -e  # Exit on error

echo "--- Updating System ---"
sudo apt update && sudo apt upgrade -y

echo "--- Installing NVIDIA Drivers ---"
if ! command -v nvidia-smi &> /dev/null; then
    sudo apt install -y nvidia-driver-525
fi

echo "--- Installing CUDA ---"
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run --silent --toolkit --driver --override
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

echo "--- Installing Docker ---"
sudo apt install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

echo "--- Installing NVIDIA Container Toolkit ---"
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

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
sudo systemctl restart docker

echo "--- Cloning CUDA Matrix Multiplication Repository ---"
git clone https://github.com/whyismynamerudy/gpu-profiling-sample.git  # Replace with your actual repository URL

echo "--- Setup Complete ---"
echo "NOTE: You may need to log out and back in for group changes to take effect"
nvidia-smi  # Verify NVIDIA setup
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi  # Verify Docker GPU access