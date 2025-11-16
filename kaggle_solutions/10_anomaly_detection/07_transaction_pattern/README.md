# Transaction Pattern Anomaly Detection

## Overview
Detects unusual transaction patterns including fraud, account compromise, and money laundering using One-Class SVM, Local Outlier Factor, and rule-based methods.

## Problem Description
Identifies anomalous patterns:
- Large unusual amounts
- Transactions at unusual times
- Geographic anomalies
- Rapid transaction sequences
- New merchant patterns

## Dataset
- 500 customers, 100 transactions each (50,000 total)
- 3% anomaly rate
- Customer-specific spending patterns

### Features
- Amount, time of day, merchant category
- Distance from home, time since last transaction
- Merchant transaction history

## Anomaly Types
1. **Large Amount**: 10-50x normal spending
2. **Unusual Time**: Middle of night (2-5 AM)
3. **Unusual Location**: Far from home (200-2000 km)
4. **Rapid Sequence**: Multiple transactions minutes apart
5. **New Merchant**: First time, large amount

## Methods
1. **One-Class SVM**: Boundary-based detection
2. **Local Outlier Factor**: Local density comparison
3. **Rule-Based**: Business logic thresholds

## Usage
```bash
python solution.py
```

## Requirements
- numpy, pandas, matplotlib, seaborn, scikit-learn

## Applications
- Credit card fraud, banking security, payment processing, anti-money laundering
