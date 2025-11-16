# Server Log Anomaly Detection

## Overview
This example demonstrates anomaly detection in server logs to identify performance issues, crashes, and potential security threats using clustering-based (DBSCAN), tree-based (Isolation Forest), and statistical threshold methods.

## Problem Description
Server anomalies can indicate:
- **Performance degradation**: Slow responses, high resource usage
- **Security threats**: DDoS attacks, unauthorized access
- **System failures**: Crashes, memory leaks
- **Configuration issues**: Misconfigurations causing errors

Early detection enables rapid response and minimizes downtime.

## Dataset
Synthetic server logs with 25,000 entries:
- **Normal operations (96%)**: Standard server behavior
- **Anomalies (4%)**: Various failure modes

### Log Features
- `response_time_ms`: HTTP response time
- `cpu_usage_pct`: CPU utilization percentage
- `memory_usage_pct`: Memory utilization percentage
- `requests_per_minute`: Request rate
- `error_rate`: Proportion of errors
- `active_connections`: Concurrent connections
- `disk_io_mbps`: Disk I/O throughput
- `network_io_mbps`: Network I/O throughput
- `http_200/404/500`: HTTP status code flags

### Derived Features
- `cpu_memory_product`: Combined resource usage
- `requests_per_connection`: Request efficiency
- `response_time_per_request`: Average latency

## Anomaly Types

### 1. DDoS Attack
- Very high request volume (500+ req/min)
- Maxed CPU/memory (80-100%)
- Slow responses (500-5000 ms)
- High error rate (20-50%)
- Many concurrent connections (1000+)

### 2. Server Crash
- Timeout responses (1000-10000 ms)
- Extreme memory usage (90-100%)
- Very high error rate (50-100%)
- Few successful requests
- Mostly HTTP 500 errors

### 3. Memory Leak
- Gradually increasing memory (85-99%)
- Moderate CPU usage (50-80%)
- Elevated error rate (10-30%)
- Slower responses over time

### 4. Slow Query/Database Issue
- Very slow responses (1000-3000 ms)
- High CPU usage (60-90%)
- High disk I/O
- Normal request volume

## Methods Used

### 1. DBSCAN (Density-Based Spatial Clustering)
- **Approach**: Identifies points in low-density regions as outliers
- **Parameters**: eps=3, min_samples=50
- **Strengths**: No assumption about cluster shape, finds arbitrary anomalies
- **Weaknesses**: Sensitive to parameter tuning

### 2. Isolation Forest
- **Approach**: Isolates anomalies using random partitioning
- **Configuration**: 100 estimators, 4% contamination
- **Strengths**: Fast, scalable, handles high dimensions
- **Weaknesses**: Assumes anomalies are rare and different

### 3. Statistical Threshold
- **Approach**: Flags values exceeding percentile thresholds
- **Thresholds**:
  - Response time > 95th percentile
  - CPU usage > 90th percentile
  - Error rate > 95th percentile
- **Strengths**: Interpretable, easy to implement
- **Weaknesses**: Treats features independently

## Evaluation Metrics
- **Precision**: Accuracy of anomaly alerts
- **Recall**: Percentage of anomalies detected
- **F1-Score**: Harmonic mean of precision and recall

## Results Visualizations
1. **server_metrics_timeline.png**: Metrics over time with anomalies marked
2. **detection_results.png**: Confusion matrices and performance comparison

## Key Insights
- Different anomaly types have distinct signatures
- DBSCAN effective for density-based outliers
- Isolation Forest balances speed and accuracy
- Statistical methods good for real-time monitoring
- Combining methods improves reliability

## Usage
```bash
python solution.py
```

## Requirements
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- scipy

## Production Deployment

### Real-Time Monitoring Pipeline
1. Collect server metrics every 10 seconds
2. Maintain sliding window (last 1000 samples)
3. Update scaler statistics periodically
4. Run all three detectors
5. Alert if 2+ methods agree

### Alert Severity Levels
- **Critical**: All 3 methods detect anomaly
- **High**: 2 methods detect anomaly
- **Medium**: 1 method detects anomaly
- **Info**: Close to threshold

### Threshold Tuning
- Adjust based on alert fatigue vs. miss rate
- Consider time of day patterns
- Account for expected traffic spikes
- Seasonal adjustments for load changes

### Integration Points
- Log aggregation (ELK Stack, Splunk)
- Monitoring systems (Prometheus, Grafana)
- Incident management (PagerDuty, OpsGenie)
- Auto-scaling triggers

## Extensions
1. Add temporal features (hour of day, day of week)
2. Implement LSTM for sequence anomaly detection
3. Use Prophet for seasonal decomposition
4. Add user behavior analysis
5. Implement automatic threshold adjustment
6. Add root cause analysis module
7. Create anomaly explanation dashboard
8. Implement distributed detection for microservices

## Real-World Applications
- Web server monitoring
- Application performance management (APM)
- Cloud infrastructure monitoring
- Microservices health checking
- Database performance monitoring
- CDN anomaly detection
