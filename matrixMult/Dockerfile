FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Copy the source file
COPY matrixMul.cu .

# Install necessary tools
RUN apt-get update && apt-get install -y \
    nvidia-cuda-toolkit \
    build-essential \
    time \
    nvidia-utils-535 \
    && rm -rf /var/lib/apt/lists/*

# Compile the CUDA code with timing and maximum optimization
RUN /usr/bin/time -v nvcc -O3 \
    --generate-line-info \
    --compiler-options -Wall \
    matrixMul.cu -o matrix_mult

# Set executable permissions
RUN chmod +x matrix_mult

ENTRYPOINT ["./matrix_mult"]