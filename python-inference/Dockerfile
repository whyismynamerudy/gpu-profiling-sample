# Use NVIDIA's PyTorch container as base for optimal GPU support
FROM nvcr.io/nvidia/pytorch:23.12-py3

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script
COPY ml_workload.py .

# Create directory for profiling results
RUN mkdir -p /app/profiling_results

# Set environment variables for better GPU utilization
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Cache directory for Hugging Face models
RUN mkdir -p /root/.cache/huggingface

# Expose port for Prometheus metrics
EXPOSE 8000

# Default command to run the workload
ENTRYPOINT ["python", "ml_workload.py"]

# Default arguments (no run duration specified = run indefinitely)
CMD ["--output-dir", "/app/profiling_results"]