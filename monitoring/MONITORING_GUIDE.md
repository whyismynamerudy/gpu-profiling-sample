# ML Workload Monitoring Guide

## Useful PromQL Queries

### GPU Metrics

1. GPU Utilization (average over 5m):
```promql
avg_over_time(DCGM_FI_DEV_GPU_UTIL[5m])
```

2. GPU Memory Usage:
```promql
DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_TOTAL * 100
```

3. Power Efficiency (tokens per watt-hour):
```promql
rate(ml_tokens_processed_total[5m]) / DCGM_FI_DEV_POWER_USAGE
```

### ML Workload Metrics

1. Average Inference Time (95th percentile):
```promql
histogram_quantile(0.95, rate(ml_inference_time_seconds_bucket[5m]))
```

2. Token Processing Rate:
```promql
rate(ml_tokens_processed_total[5m])
```

3. Cost per 1000 tokens (assuming $0.12/kWh):
```promql
(DCGM_FI_DEV_POWER_USAGE * 0.12) / (rate(ml_tokens_processed_total[5m]) * 1000)
```

## Common Alert Conditions

1. High GPU Memory Usage:
```promql
DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_TOTAL * 100 > 90
```

2. High GPU Temperature:
```promql
DCGM_FI_DEV_GPU_TEMP > 80
```

3. Low Inference Throughput:
```promql
rate(ml_inference_time_seconds_count[5m]) < 1
```

## Backup and Restore

### Prometheus Data

Backup:
```bash
docker run --rm -v prometheus_data:/data -v $(pwd):/backup ubuntu tar czf /backup/prometheus_backup.tar.gz /data
```

Restore:
```bash
docker run --rm -v prometheus_data:/data -v $(pwd):/backup ubuntu bash -c "cd /data && tar xzf /backup/prometheus_backup.tar.gz --strip 1"
```

### Grafana Data

Backup:
```bash
docker run --rm -v grafana_data:/data -v $(pwd):/backup ubuntu tar czf /backup/grafana_backup.tar.gz /data
```

Restore:
```bash
docker run --rm -v grafana_data:/data -v $(pwd):/backup ubuntu bash -c "cd /data && tar xzf /backup/grafana_backup.tar.gz --strip 1"
```

## Scaling Considerations

1. Storage:
   - Monitor Prometheus storage usage
   - Adjust retention period based on needs
   - Consider remote storage for long-term metrics

2. Network:
   - Monitor network bandwidth between components
   - Adjust scrape intervals if needed
   - Consider using recording rules for frequently used queries

3. Resources:
   - Monitor container resource usage
   - Adjust resource limits based on usage patterns
   - Consider using service discovery for dynamic scaling

## Troubleshooting

1. Check component status:
```bash
docker-compose ps
```

2. View component logs:
```bash
docker-compose logs [service_name]
```

3. Verify metrics collection:
```bash
./verify_metrics.sh
```

4. Common issues:
   - DCGM-Exporter not starting: Check NVIDIA driver installation
   - Missing metrics: Check scrape configurations
   - High latency: Check resource usage and network connectivity
   - Dashboard not loading: Verify Prometheus data source configuration