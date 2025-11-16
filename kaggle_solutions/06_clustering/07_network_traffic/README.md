# Network Traffic Clustering Analysis

## Overview
This solution demonstrates clustering of network traffic flows for pattern recognition, traffic classification, and anomaly detection in cybersecurity applications.

## Problem Statement
Network administrators need to classify traffic types, detect anomalies, and identify potential security threats. Clustering helps automatically categorize traffic patterns without labeled data.

## Dataset Features
- **bytes_sent**: Number of bytes transmitted
- **bytes_received**: Number of bytes received
- **packets_sent**: Number of packets transmitted
- **packets_received**: Number of packets received
- **duration**: Flow duration in seconds
- **port**: Destination port number
- **protocol**: Network protocol (TCP/UDP/ICMP)
- **flow_rate**: Bytes per second
- **packet_rate**: Packets per second

## Derived Features
- **bytes_ratio**: Ratio of sent to received bytes
- **packets_ratio**: Ratio of sent to received packets
- **avg_packet_size**: Average size of packets in flow

## Traffic Types Generated
1. **Normal Web Traffic** (40%): HTTP/HTTPS browsing
2. **Video Streaming** (25%): High bandwidth, long duration
3. **File Transfer** (15%): FTP/SFTP with large data volumes
4. **P2P Traffic** (10%): Peer-to-peer file sharing
5. **DDoS Attack** (10%): Malicious flood traffic

## Clustering Algorithms
1. **K-Means**: Fast partitioning for traffic classification
2. **DBSCAN**: Density-based clustering excellent for anomaly detection
   - Identifies outliers as noise (-1 label)
   - Finds arbitrarily shaped clusters

## Evaluation Metrics
- **Silhouette Score**: Cluster quality measure
- **Davies-Bouldin Index**: Cluster separation metric
- **Calinski-Harabasz Score**: Variance ratio criterion
- **Anomaly Count**: Number of outliers detected (DBSCAN)

## Analysis Steps
1. Generate 2000 realistic network flow records
2. Engineer features including ratios and rates
3. Apply RobustScaler (handles outliers better than StandardScaler)
4. Determine optimal cluster count using elbow and silhouette
5. Compare K-Means and DBSCAN clustering
6. Visualize clusters using PCA
7. Profile each cluster's traffic characteristics

## Key Insights
- Different traffic types have distinct patterns
- DBSCAN identifies anomalies (potential attacks)
- Flow rate and packet size are strong discriminators
- Duration helps separate streaming from browsing

## Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Usage
```bash
python solution.py
```

## Output
- Optimal cluster count visualization
- PCA-based cluster visualization
- Traffic profile for each cluster
- Anomaly detection results (DBSCAN)
- Performance metrics comparison

## Cybersecurity Applications
- **Traffic Classification**: Automatically categorize network flows
- **Anomaly Detection**: Identify suspicious or malicious traffic
- **Bandwidth Management**: Prioritize traffic by type
- **Intrusion Detection**: Detect attack patterns (DDoS, port scans)
- **Network Forensics**: Analyze historical traffic patterns
- **QoS Optimization**: Quality of service by traffic class

## Technical Advantages
- **RobustScaler**: Better handles outliers than StandardScaler
- **DBSCAN**: No need to specify cluster count, finds anomalies
- **Feature Engineering**: Ratios and rates improve discrimination
- **PCA Visualization**: Reduces dimensions for interpretation

## Real-World Extensions
- Real-time traffic analysis
- Integration with IDS/IPS systems
- Deep packet inspection features
- Time-series pattern analysis
- Protocol-specific feature extraction

## Author
Kaggle Competition Solution - Network Security Clustering
