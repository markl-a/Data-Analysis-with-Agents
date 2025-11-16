# Seasonal Sales Decomposition Analysis

## Overview
This project provides comprehensive seasonal decomposition and forecasting for retail sales data with strong seasonal patterns. It demonstrates advanced time series analysis techniques including STL decomposition and Holt-Winters forecasting.

## Problem Statement
Retail businesses experience complex seasonal patterns:
- **Quarterly seasonality**: Holiday shopping peaks
- **Monthly effects**: Specific holidays (Black Friday, Back-to-School)
- **Growth trends**: Year-over-year expansion
- **Cyclical patterns**: Economic cycles

Accurate decomposition helps understand these components and improve forecasts.

## Dataset
Synthetic retail sales data with realistic patterns:
- **3 years** of weekly sales data (156 weeks)
- **Base sales**: ~$100,000 per week
- **Seasonal variation**: ±40% from baseline
- **Growth rate**: 5% annual

### Seasonal Components
1. **Quarterly Pattern**:
   - Q1: Post-holiday slump (-15%)
   - Q2: Spring moderate (-5%)
   - Q3: Summer above average (+5%)
   - Q4: Holiday peak (+25%)

2. **Holiday Spikes**:
   - Valentine's Day: +8%
   - Back-to-School: +12%
   - Black Friday/Cyber Monday: +30%

3. **Cyclical**: 2-year economic cycle

## Methodology

### 1. Classical Decomposition
- Additive model: Y = Trend + Seasonal + Residual
- 52-week seasonal period
- Moving average trend extraction

### 2. STL Decomposition
- **S**easonal-**T**rend decomposition using **L**OESS
- More flexible than classical methods
- Robust to outliers
- Better handles changing seasonality

### 3. Seasonality Strength Metrics
- **Trend Strength**: Measures trend vs. noise
- **Seasonal Strength**: Measures seasonality vs. noise
- Values range 0-1 (higher = stronger)

### 4. Forecasting Models

#### Holt-Winters Triple Exponential Smoothing
- Level, trend, and seasonal components
- Multiplicative seasonality (seasonal amplitude grows with level)
- Damped trend (prevents unrealistic long-term growth)

#### Naive Seasonal Baseline
- Uses same week from previous year
- Simple but effective benchmark
- Captures pure seasonality

### 5. Evaluation Metrics
- **MAE**: Mean Absolute Error (dollar amount)
- **RMSE**: Root Mean Squared Error (penalizes large errors)
- **MAPE**: Mean Absolute Percentage Error (relative accuracy)
- **R²**: Coefficient of determination (variance explained)

## Results

### Decomposition Insights
- **Strong Seasonality**: Seasonal strength typically > 0.8
- **Clear Trend**: Growth trend visible over years
- **Manageable Residuals**: Low noise indicates good model fit

### Forecast Performance
- **Holt-Winters**: Typically 5-10% MAPE
- **Naive Seasonal**: Baseline performance ~10-15% MAPE
- **Best Periods**: Mid-quarter weeks (stable patterns)
- **Challenging Periods**: Holiday weeks (high variance)

### Seasonal Patterns
1. **Q4 Dominance**: 30-40% higher than Q1
2. **Black Friday Effect**: Largest single-week spike
3. **February Drop**: Post-holiday decline
4. **Steady Summer**: Consistent mid-year performance

## Visualizations
1. **Complete Time Series**: Full 3-year view with train/test split
2. **Trend Component**: Long-term growth pattern
3. **Seasonal Component**: Repeating 52-week pattern
4. **Residuals**: Random fluctuations after decomposition
5. **Residual Distribution**: Normality check
6. **Quarterly Box Plots**: Distribution by quarter
7. **Monthly Averages**: Average performance by month
8. **Forecast Comparison**: Holt-Winters vs. Naive vs. Actual

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
- Detailed seasonal pattern statistics
- Component strength metrics
- Forecast accuracy for multiple models
- Comprehensive visualizations saved as `seasonal_sales_analysis.png`

## Real-World Applications
- **Inventory Planning**: Stock levels based on seasonal demand
- **Staff Scheduling**: Workforce allocation for peak periods
- **Budget Forecasting**: Revenue projections by quarter
- **Marketing Campaigns**: Timing promotions with seasonality
- **Supply Chain**: Order quantities and timing
- **Capacity Planning**: Warehouse and logistics resources

## Key Techniques

### STL vs. Classical Decomposition
- **STL Advantages**:
  - Handles non-constant seasonality
  - Robust to outliers
  - More accurate trend estimation
  - Can specify seasonal window

### Multiplicative vs. Additive Seasonality
- **Additive**: Seasonal effect is constant (±$X)
- **Multiplicative**: Seasonal effect scales with level (±X%)
- Retail often uses multiplicative (higher sales = larger swings)

### Damped Trend
- Prevents unrealistic exponential growth
- More conservative long-term forecasts
- Better for stable businesses

## Extensions
1. **Multiple Seasonalities**: Daily + Weekly + Annual
2. **External Regressors**: Promotions, weather, competitors
3. **Hierarchical Forecasting**: Product categories aggregation
4. **Prophet**: Facebook's forecasting tool
5. **SARIMAX**: Seasonal ARIMA with exogenous variables
6. **Machine Learning**: XGBoost, LSTM for complex patterns
7. **Probabilistic Forecasting**: Prediction intervals
8. **Intermittent Demand**: For low-volume products

## Statistical Insights

### Variance Decomposition
- Total Variance = Trend Var + Seasonal Var + Residual Var
- High seasonal variance indicates strong patterns
- Low residual variance indicates good model fit

### Autocorrelation
- Strong autocorrelation at lag 52 (yearly)
- Moderate at lag 13 (quarterly)
- Indicates predictable patterns

### Holiday Effects
- Non-periodic spikes require special handling
- Calendar adjustments improve accuracy
- External regressor variables for known events

## Limitations
- Assumes patterns repeat consistently
- May not capture structural changes
- External shocks (pandemics, regulations) not modeled
- Single-series (no cross-product effects)

## Best Practices
1. **Visual Inspection**: Always plot before modeling
2. **Multiple Methods**: Compare classical and STL
3. **Baseline Models**: Include naive forecasts for comparison
4. **Residual Analysis**: Check for remaining patterns
5. **Backtesting**: Test on multiple historical periods
6. **Domain Knowledge**: Incorporate business insights

## Key Insights
- Strong seasonality dominates sales patterns
- Q4 consistently highest performing quarter
- Holiday effects require special modeling
- STL decomposition provides superior component separation
- Holt-Winters effectively captures seasonal patterns
- Year-over-year comparison useful for stable businesses

## Author
Created as part of the Kaggle Solutions Collection for Time Series Analysis
