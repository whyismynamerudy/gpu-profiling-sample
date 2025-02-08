#!/bin/bash

# Create timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="ml_profile_${TIMESTAMP}"
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

# Check Docker GPU support
if ! docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    log "Error: Docker GPU support not properly configured"
    exit 1
fi

# Get GPU information
log "Getting GPU information..."
nvidia-smi --query-gpu=gpu_name,compute_cap,memory.total,driver_version --format=csv,noheader > "$OUTPUT_DIR/gpu_info.csv"

# Start monitoring GPU metrics
log "Starting GPU monitoring..."
nvidia-smi --query-gpu=timestamp,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw,clocks.current.graphics,clocks.current.memory \
    --format=csv -l 1 > "$OUTPUT_DIR/gpu_metrics.csv" &
NVIDIA_SMI_PID=$!

# Time the Docker build process
log "Starting Docker build process..."
BUILD_START=$(date +%s.%N)
docker build --no-cache -t ml-workload . 2>&1 | tee "$OUTPUT_DIR/docker_build.log"
BUILD_STATUS=${PIPESTATUS[0]}
BUILD_END=$(date +%s.%N)
BUILD_TIME=$(echo "$BUILD_END - $BUILD_START" | bc)

if [ $BUILD_STATUS -ne 0 ]; then
    log "Error: Docker build failed"
    cleanup
    exit 1
fi

log "Docker build completed in $BUILD_TIME seconds"

# Run the container with GPU support and mount the output directory
log "Running ML workload..."
EXEC_START=$(date +%s.%N)
docker run --gpus all \
    -v "$PWD/$OUTPUT_DIR:/app/profiling_results" \
    ml-workload \
    --output-dir /app/profiling_results 2>&1 | tee "$OUTPUT_DIR/container_output.log"
EXEC_STATUS=${PIPESTATUS[0]}
EXEC_END=$(date +%s.%N)
EXEC_TIME=$(echo "$EXEC_END - $EXEC_START" | bc)

if [ $EXEC_STATUS -ne 0 ]; then
    log "Error: Container execution failed"
    cleanup
    exit 1
fi

log "ML workload completed in $EXEC_TIME seconds"

# Wait for final GPU metrics
sleep 2
kill $NVIDIA_SMI_PID

# Process GPU metrics and create summary
log "Processing GPU metrics..."
{
    echo "=== ML Workload Performance Summary ==="
    echo "Date: $(date)"
    
    echo -e "\nGPU Information:"
    cat "$OUTPUT_DIR/gpu_info.csv"
    
    echo -e "\nBuild and Execution Times:"
    echo "Docker build time: $BUILD_TIME seconds"
    echo "Workload execution time: $EXEC_TIME seconds"
    
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
        
        echo "Memory Clock Speed:"
        tail -n +2 "$OUTPUT_DIR/gpu_metrics.csv" | \
            awk -F, '{sum += $9; count++} END {print sum/count " MHz"}'
        
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
        
        echo -e "\nGPU Utilization Pattern:"
        tail -n +2 "$OUTPUT_DIR/gpu_metrics.csv" | \
            awk -F, 'BEGIN {max=$3; min=$3} 
                {sum+=$3; count++; 
                 if($3>max) max=$3; 
                 if($3<min) min=$3} 
            END {
                print "Min: " min "%"
                print "Max: " max "%"
                print "Avg: " sum/count "%"
            }'
    else
        echo "No GPU metrics collected"
    fi
    
    # Add ML-specific metrics if available
    if ls "$OUTPUT_DIR"/ml_summary_*.json 1> /dev/null 2>&1; then
        echo -e "\nML Workload Metrics:"
        cat "$OUTPUT_DIR"/ml_summary_*.json
    fi
} > "$OUTPUT_DIR/performance_summary.txt"

# Create a simple CSV with key metrics
{
    echo "Build Time,Execution Time,Peak GPU Util,Peak Memory,Avg GPU Clock,Avg Memory Clock"
    if [ -s "$OUTPUT_DIR/gpu_metrics.csv" ]; then
        echo "$BUILD_TIME,$EXEC_TIME,$(tail -n +2 "$OUTPUT_DIR/gpu_metrics.csv" | awk -F, '{print $3}' | sort -rn | head -n1),$(tail -n +2 "$OUTPUT_DIR/gpu_metrics.csv" | awk -F, '{print $5}' | sort -rn | head -n1),$(tail -n +2 "$OUTPUT_DIR/gpu_metrics.csv" | awk -F, '{sum += $8; count++} END {print sum/count}'),$(tail -n +2 "$OUTPUT_DIR/gpu_metrics.csv" | awk -F, '{sum += $9; count++} END {print sum/count}')"
    else
        echo "$BUILD_TIME,$EXEC_TIME,N/A,N/A,N/A,N/A"
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
echo "- ML metrics: ml_metrics_*.json"
echo "- ML summary: ml_summary_*.json"