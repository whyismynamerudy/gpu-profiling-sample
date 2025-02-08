#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <volume_name>"
    echo "Example: $0 nvme1n1"
    exit 1
fi

VOLUME_NAME=$1

# Check if volume exists
if [ ! -b "/dev/$VOLUME_NAME" ]; then
    echo "Error: Volume /dev/$VOLUME_NAME does not exist"
    echo "Available volumes:"
    lsblk
    exit 1
fi

echo "Setting up Docker to use /dev/$VOLUME_NAME..."

# Stop Docker
echo "Stopping Docker..."
sudo systemctl stop docker.socket
sudo systemctl stop docker.service

# Set up instance store
echo "Setting up instance store..."
sudo mkfs -t xfs "/dev/$VOLUME_NAME"
sudo mkdir -p /mnt/docker
sudo mount "/dev/$VOLUME_NAME" /mnt/docker

# Configure Docker
echo "Configuring Docker..."
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json << EOF
{
    "data-root": "/mnt/docker",
    "storage-driver": "overlay2"
}
EOF

# Start Docker
echo "Starting Docker..."
sudo systemctl daemon-reload
sudo systemctl start docker

# Verify setup
echo -e "\nVerification:"
echo "Mount point status:"
df -h /mnt/docker
echo -e "\nDocker root directory:"
docker info | grep "Docker Root Dir"

echo -e "\nSetup complete!"