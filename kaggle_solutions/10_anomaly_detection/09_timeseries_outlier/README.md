# Time Series Outlier Detection

## Overview
Detects anomalies in time series data including spikes, dips, level shifts, and trend changes using statistical decomposition, Isolation Forest, and moving average methods.

## Problem Description
Time series anomalies include:
- **Spikes**: Sudden large increases
- **Dips**: Sudden decreases
- **Level Shifts**: Sustained changes in baseline
- **Trend Changes**: Temporary trend deviations

## Dataset
- 2000 time points with trend + seasonality
- 5% anomaly rate
- Multiple anomaly types

### Signal Components
- Trend: Linear increase over time
- Seasonality: Multiple periodic components
- Noise: Random variation
- Anomalies: Injected outliers

## Methods

### 1. Statistical (Z-Score on Residuals)
- Decompose into trend, seasonal, residual
- Apply z-score threshold (3σ) on residuals
- Detects deviations from expected pattern

### 2. Isolation Forest
- Features: Value, rolling stats, differences
- Captures multivariate patterns
- Good for complex anomalies

### 3. Moving Average
- Compare to local moving average
- Threshold: 3 standard deviations
- Simple and interpretable

## Usage
```bash
python solution.py
```

## Requirements
- numpy, pandas, matplotlib, seaborn, scikit-learn, scipy

## Applications
- Stock market monitoring, sensor data, web traffic, system metrics, energy consumption
