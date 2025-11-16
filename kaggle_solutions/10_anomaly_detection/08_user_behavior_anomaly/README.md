# User Behavior Anomaly Detection

## Overview
Detects compromised accounts and insider threats by analyzing user behavior patterns including login times, session activity, and access patterns.

## Problem Description
Identifies:
- Compromised accounts (unauthorized access)
- Data exfiltration attempts
- Unusual resource access
- Suspicious login patterns

## Dataset
- 200 users, 150 sessions each (30,000 total)
- 5% anomaly rate
- User-specific behavior profiles

### Features
- Login hour, session duration, pages visited, clicks
- Downloads, failed logins, location/device changes
- Derived: Clicks per page, session intensity

## Anomaly Types
1. **Compromised Account**: Unusual time/location, failed logins
2. **Data Exfiltration**: Excessive downloads, long sessions
3. **Unusual Access**: Exploration of unfamiliar resources
4. **Suspicious Login**: Multiple failures, new devices

## Methods
1. **Isolation Forest**: General behavioral anomalies
2. **User-Specific Baseline**: Individual behavior profiles
3. **Rule-Based**: Security policy violations

## Usage
```bash
python solution.py
```

## Requirements
- numpy, pandas, matplotlib, seaborn, scikit-learn

## Applications
- Corporate security, insider threat detection, account security, SIEM systems
