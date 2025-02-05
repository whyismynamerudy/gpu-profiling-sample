# Use NVIDIA's PyTorch container as base for optimal GPU support
FROM nvcr.io/nvidia/pytorch:23.12-py3

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script
COPY ml_workload.py .

# Set environment variables for better GPU utilization
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Cache directory for Hugging Face models
RUN mkdir -p /root/.cache/huggingface

# Default command to run the workload
ENTRYPOINT ["python", "ml_workload.py"]

# Allow for parameter overrides at runtime
CMD []