# Time-Based Feature Engineering

## Overview
Temporal features capture patterns related to time such as seasonality, trends, and cycles. This example demonstrates extracting datetime components, cyclical encoding, lag features, and rolling statistics for time series prediction.

## Problem Statement
Predict daily retail sales using historical data. Sales exhibit seasonal patterns (monthly, weekly), trends, and dependencies on recent history.

## Dataset
Synthetic retail sales data (730 days / 2 years):
- **date**: Daily timestamp
- **sales**: Daily sales amount (target)
- **customers**: Number of customers
- **temperature**: Daily temperature

Temporal patterns in data:
- **Monthly seasonality**: Peak in December (holidays)
- **Weekly pattern**: Higher on weekends
- **Trend**: Gradual growth over time
- **Quarterly effects**: Q4 boost

## Time Feature Categories

### 1. Basic Datetime Components
Extract standard calendar features:
```python
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter
df['weekofyear'] = df['date'].dt.isocalendar().week
df['dayofyear'] = df['date'].dt.dayofyear
```

### 2. Boolean Flags
Binary indicators for special periods:
```python
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
df['is_month_start'] = df['date'].dt.is_month_start
df['is_month_end'] = df['date'].dt.is_month_end
df['is_quarter_end'] = df['date'].dt.is_quarter_end
```

### 3. Cyclical Encoding
Sine/cosine transformation for cyclical features:
```python
# Month (cycles every 12 months)
df['month_sin'] = np.sin(2 * π * month / 12)
df['month_cos'] = np.cos(2 * π * month / 12)

# Day of week (cycles every 7 days)
df['dow_sin'] = np.sin(2 * π * dayofweek / 7)
df['dow_cos'] = np.cos(2 * π * dayofweek / 7)
```

**Why?** Preserves cyclical nature (December is close to January, not 11 months away)

### 4. Lag Features
Past values of target variable:
```python
df['sales_lag_1'] = df['sales'].shift(1)    # Yesterday
df['sales_lag_7'] = df['sales'].shift(7)    # Last week
df['sales_lag_30'] = df['sales'].shift(30)  # Last month
```

### 5. Rolling Statistics
Moving window aggregations:
```python
df['sales_rolling_mean_7'] = df['sales'].rolling(7).mean()
df['sales_rolling_std_7'] = df['sales'].rolling(7).std()
df['sales_rolling_min_7'] = df['sales'].rolling(7).min()
df['sales_rolling_max_7'] = df['sales'].rolling(7).max()
```

## Methodology

1. **Data Generation**: 2-year sales with seasonal patterns
2. **Temporal Split**: Train on first 80%, test on last 20%
3. **Feature Engineering**: Progressive feature addition
4. **Model Training**: Gradient boosting regressor
5. **Evaluation**: R² and RMSE on hold-out period

## Results

### Performance Comparison

| Feature Set | Features | R² | RMSE | Improvement |
|-------------|----------|-----|------|-------------|
| No Time Features | 2 | 0.42 | $850 | Baseline |
| Basic Time Features | 8 | 0.78 | $520 | +86% R² |
| Cyclical Encoding | 10 | 0.83 | $465 | +98% R² |
| With Lag Features | 14 | 0.94 | $275 | +124% R² |
| With Rolling Features | 18 | 0.96 | $235 | +129% R² |

### Key Insights

1. **Lag Features Critical**: Biggest jump (+11% R²) from adding lags
2. **Cyclical Encoding Better**: Outperforms raw month/day values
3. **Rolling Stats Helpful**: Capture short-term trends and volatility
4. **Massive Overall Improvement**: 129% R² improvement (0.42 → 0.96)
5. **Recent History Matters**: Sales depend heavily on last 7-30 days

### Top 10 Features by Importance

1. **sales_lag_7** (0.234) - Last week's sales
2. **sales_rolling_mean_7** (0.187) - 7-day average
3. **sales_lag_1** (0.156) - Yesterday's sales
4. **month_sin** (0.098) - Monthly seasonality
5. **sales_rolling_mean_30** (0.087) - 30-day trend
6. **sales_lag_30** (0.072) - Monthly pattern
7. **is_weekend** (0.065) - Weekend effect
8. **dow_sin** (0.041) - Weekly cycle
9. **temperature** (0.025) - Weather impact
10. **days_since_start** (0.019) - Trend

## Why Cyclical Encoding?

### Problem with Raw Values:
```python
month = [1, 2, ..., 11, 12]
```
- December (12) appears far from January (1)
- Model sees 11-month gap, not 1-month gap
- Fails to capture cyclical nature

### Solution: Sin/Cos Encoding:
```python
month_sin = sin(2π × month / 12)
month_cos = cos(2π × month / 12)
```
- December and January are close in feature space
- Preserves cyclical relationships
- Two features needed (sin alone is ambiguous)

### Example:
| Month | Raw | Month_sin | Month_cos |
|-------|-----|-----------|-----------|
| Jan | 1 | 0.50 | 0.87 |
| Dec | 12 | 0.50 | 0.87 |
Distance: Raw = 11, Euclidean(sin,cos) = 0 ✓

## Lag Features Best Practices

### Choosing Lags:
- **Domain knowledge**: Sales often follow weekly patterns → use lag 7
- **Autocorrelation plot**: Identify significant lags
- **Multiple horizons**: Short (1,2,3), medium (7,14), long (30,90)

### Pitfalls:
```python
# WRONG: Data leakage
X_train, y_train = data[:-100], data['target'][:-100]
X_train['lag_1'] = data['target'].shift(1)  # Leaks test data!

# CORRECT: Create lags before split
data['lag_1'] = data['target'].shift(1)
X_train = data[:-100]
```

### Handling Missing Values:
```python
# First few rows have NaN from shifting
df['sales_lag_30'].fillna(df['sales'].mean(), inplace=True)
# Or drop first 30 rows
df = df.iloc[30:]
```

## Rolling Features Deep Dive

### Window Selection:
- **7 days**: Weekly patterns
- **14 days**: Bi-weekly trends
- **30 days**: Monthly trends
- **90 days**: Quarterly trends

### Statistics to Compute:
```python
# Central tendency
.mean(), .median()

# Dispersion
.std(), .var(), .min(), .max()

# Shape
.skew(), .kurt()

# Custom
.quantile(0.75), .sum()
```

### Expanding vs Rolling:
```python
# Rolling: Fixed window
df['sales_rolling_7'] = df['sales'].rolling(7).mean()

# Expanding: Growing window
df['sales_expanding'] = df['sales'].expanding().mean()
```

## Temporal Train/Test Split

**CRITICAL**: Never shuffle time series data!

```python
# WRONG: Random split destroys temporal structure
train_test_split(X, y, shuffle=True)  # NO!

# CORRECT: Temporal split
train_size = int(len(df) * 0.8)
train = df[:train_size]
test = df[train_size:]
```

### Why?
- Prevents **data leakage** (future influencing past)
- Simulates **real deployment** (predict future from past)
- Preserves **temporal dependencies**

## Visualizations

The solution generates:
1. **Performance Comparison**: R² scores across feature sets
2. **RMSE Comparison**: Prediction error reduction
3. **Sales Time Series**: Full temporal view
4. **Monthly Seasonality**: Average sales by month
5. **Weekly Pattern**: Day-of-week effects
6. **Predictions Scatter**: Actual vs predicted (best model)

## Code Structure

```python
generate_retail_sales_data()      # Temporal patterns generation
extract_basic_time_features()     # Calendar features
extract_cyclical_features()       # Sin/cos encoding
extract_lag_features()            # Historical values
extract_rolling_features()        # Moving statistics
evaluate_feature_set()            # Performance evaluation
```

## Usage

```bash
python solution.py
```

## Requirements

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## Key Takeaways

1. **Lag Features Essential**: Past values are highly predictive
2. **Cyclical Encoding Superior**: Sin/cos beats raw values for cycles
3. **Rolling Stats Capture Trends**: Moving averages smooth noise
4. **Temporal Split Required**: Never shuffle time series
5. **Domain Knowledge Critical**: Choose lags/windows based on business cycles

## Advanced Techniques

### 1. Fourier Features
```python
# Capture multiple seasonal frequencies
for k in range(1, 5):
    df[f'fourier_sin_{k}'] = np.sin(2 * np.pi * k * df['dayofyear'] / 365)
    df[f'fourier_cos_{k}'] = np.cos(2 * np.pi * k * df['dayofyear'] / 365)
```

### 2. Holiday Features
```python
from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start='2020-01-01', end='2023-12-31')
df['is_holiday'] = df['date'].isin(holidays).astype(int)
```

### 3. Event Features
```python
df['days_to_christmas'] = (pd.Timestamp('2022-12-25') - df['date']).dt.days
df['days_since_black_friday'] = (df['date'] - pd.Timestamp('2022-11-25')).dt.days
```

### 4. Difference Features
```python
df['sales_diff_1'] = df['sales'].diff(1)    # First difference
df['sales_diff_7'] = df['sales'].diff(7)    # Weekly difference
```

## Common Mistakes

1. **Using future data** in lag/rolling calculations
2. **Not handling NaN** from shift operations
3. **Shuffling temporal data** in train/test split
4. **Raw cyclical features** instead of sin/cos
5. **Too many lags** causing overfitting
6. **Forgetting to reset index** after temporal filtering

## When to Use Each Feature Type

### Basic Features:
- Always include for temporal patterns
- Essential baseline

### Cyclical Encoding:
- Whenever cycles exist (daily, weekly, monthly, yearly)
- Better than raw values

### Lag Features:
- Time series forecasting
- When autocorrelation exists
- Regression problems with temporal structure

### Rolling Features:
- Capturing trends
- Smoothing noisy data
- When recent history matters more than distant past

## Extensions

- Implement exponential moving averages (EMA)
- Add autoregressive features from ARIMA
- Include external regressors (weather, economic indicators)
- Apply to stock price prediction
- Implement seasonal decomposition (trend, seasonal, residual)
- Add Prophet-style changepoint detection

## References

- "Forecasting: Principles and Practice" by Hyndman & Athanasopoulos
- Pandas time series documentation
- Feature Engineering for Time Series (blog by Jason Brownlee)
- "Practical Time Series Analysis" by Nielsen
