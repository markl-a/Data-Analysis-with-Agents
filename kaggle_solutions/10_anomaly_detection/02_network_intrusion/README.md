# Network Intrusion Detection System

## Overview
This example implements a comprehensive network intrusion detection system (NIDS) using multiple anomaly detection techniques to identify malicious network traffic including DoS attacks, port scans, and intrusion attempts.

## Problem Description
Network security requires detecting anomalous traffic patterns that may indicate:
- **DoS/DDoS attacks**: High packet rates, small packet sizes
- **Port scanning**: Many connections to different ports
- **Intrusion attempts**: Unusual authentication patterns, privilege escalation

## Dataset
Synthetic network traffic with 15,000 connections:
- **Normal traffic (95%)**: Regular web, email, SSH connections
- **Attack traffic (5%)**: DoS, probe, and intrusion attempts

### Features
- `duration`: Connection duration (seconds)
- `src_bytes`: Bytes sent from source
- `dst_bytes`: Bytes sent to destination
- `packet_size`: Average packet size
- `packets_per_sec`: Packet transmission rate
- `failed_login_attempts`: Number of failed authentications
- `num_compromised`: Compromised conditions detected
- `root_shell`: Root shell obtained (0/1)
- `dst_port`: Destination port number
- `byte_ratio`: Ratio of sent/received bytes
- `connection_rate`: Packets per second per duration

## Attack Types Simulated

### 1. DoS Attacks
- Very high packet rates (100+ packets/sec)
- Small packet sizes (~100 bytes)
- Short duration connections
- Target common ports (80, 443)

### 2. Port Scans
- Minimal data transfer
- Very short duration
- Random high port numbers
- Medium packet rates

### 3. Intrusion Attempts
- Multiple failed login attempts
- Unusual ports (Telnet, RDP, SMB)
- Evidence of compromise
- Longer duration connections

## Methods Used

### 1. Isolation Forest
- **Application**: General anomaly detection
- **Strengths**: Fast, handles mixed attack types well
- **Configuration**: 200 estimators, 5% contamination

### 2. One-Class SVM
- **Application**: Boundary-based detection
- **Strengths**: Good for well-separated anomalies
- **Configuration**: Auto gamma, 5% nu parameter

### 3. Statistical Z-Score
- **Application**: Statistical outlier detection
- **Strengths**: Interpretable, fast
- **Method**: Flags points with z-score > 95th percentile

## Evaluation Metrics
- **Precision**: Accuracy of attack predictions
- **Recall (Detection Rate)**: Percentage of attacks detected
- **F1-Score**: Balance between precision and recall
- **False Positive Rate**: Normal traffic flagged as attacks
- **ROC AUC**: Overall discriminative ability

## Results Visualizations
1. **network_feature_distributions.png**: Traffic pattern analysis
2. **network_intrusion_results.png**: Model performance and ROC curves

## Key Insights
- DoS attacks have distinctive high packet rates
- Port scans show unique connection patterns
- Intrusion attempts identifiable by failed logins
- Combined approach improves detection reliability

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

## Operational Considerations
1. **False Positive Rate**: Must be low for production use
2. **Detection Speed**: Real-time systems need fast inference
3. **Alert Prioritization**: Combine severity scores
4. **Model Updates**: Retrain on new attack patterns

## Extensions
1. Add protocol-specific features (TCP flags, HTTP methods)
2. Implement temporal pattern analysis
3. Use deep learning (LSTM) for sequence modeling
4. Add geographic IP analysis
5. Implement ensemble voting system
6. Create real-time streaming detection pipeline

## Real-World Applications
- Enterprise network security
- Cloud infrastructure monitoring
- IoT device protection
- Financial transaction networks
- Critical infrastructure defense
