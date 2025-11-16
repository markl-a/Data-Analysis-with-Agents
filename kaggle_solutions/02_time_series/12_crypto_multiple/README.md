# Multi-Cryptocurrency Price Prediction

## Overview
This project implements multi-variate time series forecasting for cryptocurrency prices using Vector Autoregression (VAR). It analyzes the interdependencies between Bitcoin, Ethereum, Litecoin, and Ripple to predict future price movements.

## Problem Statement
Cryptocurrency markets exhibit complex relationships where:
- Major coins (like Bitcoin) influence others
- Correlations change over time
- Returns are more predictable than raw prices
- Multiple series provide better predictions than univariate models

## Dataset
Synthetic data for four major cryptocurrencies:
- **Bitcoin (BTC)**: Market leader, ~$30,000-$45,000
- **Ethereum (ETH)**: Highly correlated with BTC, ~$1,800-$2,700
- **Litecoin (LTC)**: Moderately correlated, ~$90-$135
- **Ripple (XRP)**: Less correlated, ~$0.40-$0.55

### Data Characteristics
- **365 days** of daily price data
- Realistic trends, cycles, and volatility
- Correlation structure: BTC-ETH (0.8), BTC-LTC (0.6), BTC-XRP (0.4)
- Occasional price jumps and regime changes

## Methodology

### 1. Data Preprocessing
- Log return calculation for stationarity
- Stationarity testing (Augmented Dickey-Fuller)
- Normalization for visualization

### 2. Correlation Analysis
- **Price correlation**: Shows long-term relationships
- **Return correlation**: Captures short-term co-movements
- Heatmap visualization

### 3. Vector Autoregression (VAR)
- Multivariate time series model
- Captures interdependencies between cryptocurrencies
- Automatic lag order selection (AIC criterion)
- Simultaneous forecasting of all series

### 4. Granger Causality Tests
- Tests if Bitcoin "Granger-causes" other cryptocurrencies
- Identifies leading/lagging relationships
- Statistical significance testing

### 5. Evaluation
- Return-level metrics (MAE, RMSE)
- Price-level metrics (MAPE)
- Individual cryptocurrency accuracy

## Key Features

### Statistical Tests
1. **Stationarity**: ADF test for returns
2. **Causality**: Granger causality from BTC to others
3. **Correlation**: Pearson correlation matrices

### Model Components
- **VAR Model**: Captures cross-correlation dynamics
- **Optimal Lag Selection**: Data-driven parameter tuning
- **Multi-step Forecasting**: Predicts test period

### Return to Price Conversion
- Cumulative return application
- Exponential transformation
- Last known price anchoring

## Results

### Model Performance
- **Returns**: Low MAE/RMSE (0.001-0.01 range)
- **Prices**: MAPE typically 5-15%
- **Best Performance**: BTC and ETH (high correlation)
- **Challenging**: XRP (lower correlation)

### Key Findings
1. **Strong BTC Influence**: Bitcoin drives the market
2. **ETH-BTC Correlation**: Highest at ~0.8
3. **Granger Causality**: BTC leads other cryptocurrencies
4. **Return Stationarity**: All return series are stationary
5. **Volatility Clustering**: Periods of high/low volatility

## Visualizations
1. Normalized price trends (all cryptos)
2. Price correlation heatmap
3. Return correlation heatmap
4. Returns distribution (histogram)
5. Individual cryptocurrency forecasts (4 plots)
6. Mean absolute error comparison

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
- Correlation matrices (prices and returns)
- Stationarity test results
- Granger causality test results
- Forecast accuracy metrics
- Comprehensive visualizations saved as `crypto_multiple_forecast.png`

## Real-World Applications
- **Trading Strategies**: Pair trading and arbitrage
- **Risk Management**: Portfolio diversification
- **Market Analysis**: Understanding crypto relationships
- **Price Discovery**: Leading/lagging indicators
- **Hedging**: Cross-crypto hedging strategies

## Technical Concepts

### VAR Model
- Extension of AR (autoregression) to multiple series
- Each variable depends on its own lags and lags of other variables
- Captures dynamic relationships

### Log Returns
- Returns = log(P_t / P_{t-1})
- Ensures stationarity
- Normally distributed
- Additive over time

### Granger Causality
- Tests if past values of X help predict Y
- Does not imply true causality
- Useful for identifying predictive relationships

## Extensions
1. Add more cryptocurrencies (Cardano, Polkadot, Solana)
2. Include external factors (stock market, gold prices)
3. Implement GARCH models for volatility forecasting
4. Use deep learning (LSTM, GRU) for comparison
5. Add trading volume and market cap
6. Real-time data integration
7. Sentiment analysis from social media
8. Regime switching models for different market conditions

## Limitations
- Synthetic data may not capture all real market dynamics
- VAR assumes linear relationships
- Past performance doesn't guarantee future results
- Model doesn't account for extreme events (black swans)
- Correlation structures can change rapidly in crypto markets

## Key Insights
- Bitcoin dominance affects all other cryptocurrencies
- Return-based models more stable than price-based
- Cross-correlation improves forecast accuracy
- Short-term predictions more reliable than long-term
- Multivariate approach captures market structure better

## Author
Created as part of the Kaggle Solutions Collection for Time Series Analysis
