#!/bin/bash

# Exit on error
set -e

# Create directory structure
mkdir -p prometheus/rules grafana/{dashboards,provisioning/{datasources,dashboards}} profiling_results

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check prerequisites
log "Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    log "Error: Docker not found. Please install Docker first."
    exit 1
fi

# Check Docker Compose
if ! command -v docker compose &> /dev/null; then
    log "Error: Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

# Check NVIDIA Docker
if ! docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    log "Error: NVIDIA Docker not properly configured. Please install NVIDIA Container Toolkit."
    exit 1
fi

# Create Prometheus recording rules
cat > prometheus/rules/recording_rules.yml << 'EOF'
groups:
  - name: ml_workload_rules
    rules:
      - record: ml:inference_latency:p95
        expr: histogram_quantile(0.95, rate(ml_inference_time_seconds_bucket[5m]))
      
      - record: ml:tokens_per_second
        expr: rate(ml_tokens_processed_total[5m])
      
      - record: gpu:memory_efficiency
        expr: DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_TOTAL * 100
      
      - record: ml:cost_per_1k_tokens
        expr: (DCGM_FI_DEV_POWER_USAGE * 0.12) / (rate(ml_tokens_processed_total[5m]) * 1000) # Assuming $0.12 per kWh
EOF

# Create Prometheus alert rules
cat > prometheus/rules/alert_rules.yml << 'EOF'
groups:
  - name: ml_alerts
    rules:
      - alert: HighInferenceLatency
        expr: ml:inference_latency:p95 > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High inference latency detected"
          description: "95th percentile inference latency is {{ $value }}s"

      - alert: LowTokenThroughput
        expr: ml:tokens_per_second < 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Low token throughput"
          description: "Token processing rate is {{ $value }} tokens/sec"

      - alert: HighGPUMemoryUsage
        expr: gpu:memory_efficiency > 90
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High GPU memory usage"
          description: "GPU memory usage is {{ $value }}%"
EOF

# Create Grafana datasource configuration
cat > grafana/provisioning/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
    jsonData:
      timeInterval: "15s"
EOF

# Create Grafana dashboard provisioning configuration
cat > grafana/provisioning/dashboards/ml-dashboards.yml << 'EOF'
apiVersion: 1

providers:
  - name: 'ML Workload Dashboards'
    orgId: 1
    folder: 'ML Monitoring'
    folderUid: ''
    type: file
    disableDeletion: false
    editable: true
    updateIntervalSeconds: 30
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF

# Configure Docker to use NVIDIA Container Runtime
cat > /etc/docker/daemon.json << 'EOF'
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

# Create a script to verify metrics collection
cat > verify_metrics.sh << 'EOF'
#!/bin/bash

echo "Verifying DCGM-Exporter metrics..."
curl -s http://localhost:9400/metrics | grep DCGM_FI_DEV

echo -e "\nVerifying ML workload metrics..."
curl -s http://localhost:8000/metrics | grep ml_

echo -e "\nVerifying Prometheus targets..."
curl -s http://localhost:9090/api/v1/targets | jq .

echo -e "\nChecking Grafana health..."
curl -s http://localhost:3000/api/health
EOF
chmod +x verify_metrics.sh

# Start the monitoring stack
log "Starting monitoring stack..."
docker compose up -d

# Wait for services to be ready
log "Waiting for services to be ready..."
sleep 30

# Verify setup
log "Verifying setup..."
./verify_metrics.sh

log "Setup complete! Access your dashboards at:"
log "Grafana: http://localhost:3000 (admin/admin)"
log "Prometheus: http://localhost:9090"
log "DCGM-Exporter metrics: http://localhost:9400/metrics"
log "ML Workload metrics: http://localhost:8000/metrics"