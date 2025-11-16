# IoT Sensor Network Anomaly Detection

## Overview
Detects sensor failures, environmental anomalies, and connectivity issues in IoT sensor networks using machine learning and rule-based methods.

## Problem Description
IoT sensors can fail due to:
- Hardware malfunction (stuck values, erratic readings)
- Battery depletion
- Connectivity issues
- Environmental events (fire, flooding)

## Dataset
- 50 sensors with 400 readings each (20,000 total)
- 4% anomaly rate
- Environmental monitoring (temperature, humidity, light, sound)

### Features
- `temperature`: Ambient temperature (°C)
- `humidity`: Relative humidity (%)
- `light`: Light intensity (lux)
- `sound`: Sound level (dB)
- `battery_voltage`: Battery voltage (V)
- `rssi`: Signal strength (dBm)

## Anomaly Types
1. **Sensor Failure**: Unrealistic/stuck values
2. **Battery Failure**: Low voltage, weak signal
3. **Environmental Spike**: Fire, flood events
4. **Connectivity Issue**: Poor signal quality

## Methods
1. **Isolation Forest**: General anomaly detection
2. **Elliptic Envelope**: Gaussian assumption-based
3. **Hardware Rules**: Physical limit thresholds

## Usage
```bash
python solution.py
```

## Requirements
- numpy, pandas, matplotlib, seaborn, scikit-learn

## Applications
- Smart buildings, agriculture, industrial monitoring, environmental sensing
