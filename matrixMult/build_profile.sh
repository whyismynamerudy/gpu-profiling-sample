#!/bin/bash

# Create timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="cuda_profile_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$OUTPUT_DIR/full_process.log"
}

# Function to cleanup background processes
cleanup() {
    log "Cleaning up background processes..."
    kill $NVIDIA_SMI_PID 2>/dev/null
    exit
}

trap cleanup EXIT INT TERM

# Verify Docker is installed and running
if ! command -v docker &> /dev/null; then
    log "Error: Docker is not installed or not in PATH"
    exit 1
fi

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    log "Error: nvidia-smi is not installed or not in PATH"
    exit 1
fi

# Get GPU information
log "Getting GPU information..."
nvidia-smi --query-gpu=gpu_name,compute_cap,driver_version --format=csv,noheader > "$OUTPUT_DIR/gpu_info.csv"

# Start monitoring GPU metrics before build
log "Starting GPU monitoring..."
nvidia-smi --query-gpu=timestamp,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw,clocks.current.graphics \
    --format=csv -l 1 > "$OUTPUT_DIR/gpu_metrics.csv" &
NVIDIA_SMI_PID=$!

# Time the Docker build process
log "Starting Docker build process..."
BUILD_START=$(date +%s.%N)
docker build --no-cache -t cuda-matrix . 2>&1 | tee "$OUTPUT_DIR/docker_build.log"
BUILD_STATUS=${PIPESTATUS[0]}
BUILD_END=$(date +%s.%N)
BUILD_TIME=$(echo "$BUILD_END - $BUILD_START" | bc)

if [ $BUILD_STATUS -ne 0 ]; then
    log "Error: Docker build failed"
    cleanup
    exit 1
fi

log "Docker build completed in $BUILD_TIME seconds"

# Extract compilation time from Docker build log
log "Extracting compilation metrics..."
grep "usr/bin/time" "$OUTPUT_DIR/docker_build.log" > "$OUTPUT_DIR/compilation_time.log"

# Verify the image exists
if ! docker image inspect cuda-matrix >/dev/null 2>&1; then
    log "Error: cuda-matrix image not found after build"
    cleanup
    exit 1
fi

# Run the container and measure execution time
log "Running containerized workload..."
EXEC_START=$(date +%s.%N)
docker run --gpus all cuda-matrix 2>&1 | tee "$OUTPUT_DIR/container_output.log"
EXEC_STATUS=${PIPESTATUS[0]}
EXEC_END=$(date +%s.%N)
EXEC_TIME=$(echo "$EXEC_END - $EXEC_START" | bc)

if [ $EXEC_STATUS -ne 0 ]; then
    log "Error: Container execution failed"
    cleanup
    exit 1
fi

log "Container execution completed in $EXEC_TIME seconds"

# Wait for final GPU metrics
sleep 2
kill $NVIDIA_SMI_PID

# Process GPU metrics
log "Processing GPU metrics..."
{
    echo "=== Performance Summary ==="
    echo "Date: $(date)"
    
    echo -e "\nGPU Information:"
    cat "$OUTPUT_DIR/gpu_info.csv"
    
    echo -e "\nBuild and Compilation Times:"
    echo "Total Docker build time: $BUILD_TIME seconds"
    echo "CUDA compilation details:"
    grep "Maximum resident set size" "$OUTPUT_DIR/compilation_time.log" || echo "Compilation details not available"
    
    echo -e "\nExecution Statistics:"
    echo "Container execution time: $EXEC_TIME seconds"
    
    echo -e "\nGPU Performance Metrics:"
    if [ -s "$OUTPUT_DIR/gpu_metrics.csv" ]; then
        echo "Peak GPU Utilization:"
        tail -n +2 "$OUTPUT_DIR/gpu_metrics.csv" | \
            awk -F, '{print $3}' | sort -rn | head -n1 | \
            awk '{print $1 "%"}'
        
        echo "Peak Memory Usage:"
        tail -n +2 "$OUTPUT_DIR/gpu_metrics.csv" | \
            awk -F, '{print $5}' | sort -rn | head -n1 | \
            awk '{print $1 " MiB"}'
        
        echo "Average GPU Clock Speed:"
        tail -n +2 "$OUTPUT_DIR/gpu_metrics.csv" | \
            awk -F, '{sum += $8; count++} END {print sum/count " MHz"}'
        
        echo -e "\nResource Usage Analysis:"
        echo "Memory Usage Pattern:"
        tail -n +2 "$OUTPUT_DIR/gpu_metrics.csv" | \
            awk -F, 'BEGIN {max=$5; min=$5} 
                {sum+=$5; count++; 
                 if($5>max) max=$5; 
                 if($5<min) min=$5} 
            END {
                print "Min: " min " MiB"
                print "Max: " max " MiB"
                print "Avg: " sum/count " MiB"
            }'
    else
        echo "No GPU metrics collected"
    fi
} > "$OUTPUT_DIR/performance_summary.txt"

# Create a simple CSV with key metrics
{
    echo "Build Time,Execution Time,Peak GPU Util,Peak Memory,Avg Clock Speed"
    if [ -s "$OUTPUT_DIR/gpu_metrics.csv" ]; then
        echo "$BUILD_TIME,$EXEC_TIME,$(tail -n +2 "$OUTPUT_DIR/gpu_metrics.csv" | awk -F, '{print $3}' | sort -rn | head -n1),$(tail -n +2 "$OUTPUT_DIR/gpu_metrics.csv" | awk -F, '{print $5}' | sort -rn | head -n1),$(tail -n +2 "$OUTPUT_DIR/gpu_metrics.csv" | awk -F, '{sum += $8; count++} END {print sum/count}')"
    else
        echo "$BUILD_TIME,$EXEC_TIME,N/A,N/A,N/A"
    fi
} > "$OUTPUT_DIR/key_metrics.csv"

log "Results saved in $OUTPUT_DIR/"
echo "Key files:"
echo "- Full process log: full_process.log"
echo "- Performance summary: performance_summary.txt"
echo "- GPU metrics: gpu_metrics.csv"
echo "- Key metrics: key_metrics.csv"
echo "- Docker build log: docker_build.log"
echo "- Container output: container_output.log"