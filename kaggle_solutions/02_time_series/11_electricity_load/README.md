# Electricity Load Forecasting with Hourly Patterns

## Overview
This project implements comprehensive electricity load forecasting using time series analysis techniques. It demonstrates hourly load prediction with multiple seasonal patterns and temperature correlation.

## Problem Statement
Accurate electricity load forecasting is critical for:
- Grid stability and reliability
- Efficient power generation scheduling
- Cost optimization
- Renewable energy integration
- Demand response planning

## Dataset
Synthetic electricity load data with realistic patterns:
- **180 days** of hourly data (4,320 hours)
- **Features**: Load (MW), Temperature, Hour, Day of Week, Month
- **Patterns**: Daily cycles, weekly patterns, temperature effects, trends

### Data Characteristics
- Daily seasonality with peak hours (morning and evening)
- Weekly seasonality (weekday vs. weekend differences)
- Temperature correlation
- Gradual upward trend
- Random noise component

## Methodology

### 1. Data Generation
- Hourly timestamps with multiple seasonal components
- Base load + trend + daily pattern + weekly pattern + temperature effect
- Realistic noise levels

### 2. Feature Engineering
- **Lag features**: 1h, 2h, 3h, 24h (1 day), 168h (1 week)
- **Rolling statistics**: 24-hour moving average and standard deviation
- **Cyclical encoding**: Hour and day of week (sine/cosine transformation)
- **Categorical features**: Weekend indicator, month

### 3. Models

#### Random Forest Regressor
- Ensemble method capturing non-linear patterns
- Handles multiple features effectively
- Provides feature importance

#### Holt-Winters Exponential Smoothing
- Triple exponential smoothing
- Additive trend and seasonality
- Specialized for time series forecasting

### 4. Seasonal Decomposition
- Trend component extraction
- Seasonal pattern identification
- Residual analysis

## Results

### Model Performance
- **Random Forest**: High accuracy with lag features
- **Holt-Winters**: Strong performance on seasonal patterns
- **Metrics**: MAE, RMSE, R² score

### Key Findings
1. **Daily Pattern**: Clear peaks during morning (7-9 AM) and evening (6-9 PM)
2. **Weekly Pattern**: Lower load on weekends (15-20% reduction)
3. **Temperature Effect**: Significant correlation with extreme temperatures
4. **Important Features**: Recent lags (24h, 168h) most predictive

## Visualizations
1. Time series overview (first 30 days)
2. Trend component from decomposition
3. Average load by hour of day
4. Average load by day of week
5. Forecast comparison (Random Forest vs. Holt-Winters)
6. Feature importance ranking

## Requirements
```bash
numpy
pandas
matplotlib
seaborn
scikit-learn
statsmodels
```

## Usage
```bash
python solution.py
```

## Output
- Performance metrics for both models
- Feature importance analysis
- Comprehensive visualizations saved as `electricity_load_forecast.png`

## Real-World Applications
- **Utility Companies**: Generation and distribution planning
- **Energy Trading**: Price forecasting and optimization
- **Grid Operators**: Load balancing and stability
- **Renewable Integration**: Managing variable generation
- **Demand Response**: Identifying peak reduction opportunities

## Extensions
1. Incorporate weather forecasts (temperature, humidity)
2. Add calendar effects (holidays, special events)
3. Multi-step ahead forecasting
4. Probabilistic forecasting with prediction intervals
5. Deep learning models (LSTM, Transformer)
6. Regional load aggregation
7. Real-time forecast updates

## Key Insights
- Short-term lags (1-3 hours) capture immediate patterns
- Daily and weekly seasonality are dominant components
- Ensemble methods outperform simple statistical models
- Cyclical encoding preserves temporal relationships
- Temperature is a significant external factor

## Author
Created as part of the Kaggle Solutions Collection for Time Series Analysis
