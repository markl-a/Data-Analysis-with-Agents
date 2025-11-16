# Bitcoin Price Prediction with LSTM

## 📊 Project Overview

This project demonstrates Bitcoin price prediction using Long Short-Term Memory (LSTM) neural networks, a powerful deep learning architecture specifically designed for time series forecasting.

**Difficulty Level:** ⭐⭐⭐ Advanced
**Category:** Time Series Forecasting
**Techniques:** LSTM, Deep Learning, Technical Analysis

## 🎯 Objective

Predict future Bitcoin prices based on historical price data and technical indicators using LSTM neural networks that can capture complex temporal dependencies and non-linear patterns.

## 📁 Dataset Description

The solution generates realistic Bitcoin price data with the following characteristics:

- **Time Period:** 3 years (1095 days) of daily prices
- **Features:**
  - `date`: Trading date
  - `price`: Daily closing price (target variable)
  - `returns`: Daily percentage returns
  - `volatility`: 7-day rolling volatility
  - `ma_7`: 7-day moving average
  - `ma_30`: 30-day moving average
  - `ma_90`: 90-day moving average
  - `price_to_ma7`: Price relative to 7-day MA
  - `price_to_ma30`: Price relative to 30-day MA

### Data Generation Features:
- **Exponential trend** mimicking Bitcoin's long-term growth
- **Cyclical patterns** representing quarterly market cycles
- **Random volatility** simulating market uncertainty
- **Flash events** representing sudden price movements
- **Technical indicators** for enhanced prediction

## 🧠 Methodology

### 1. Data Preprocessing
- Generate synthetic Bitcoin price data with realistic patterns
- Calculate technical indicators (moving averages, volatility)
- Normalize data using MinMaxScaler
- Create sliding window sequences (60-day lookback)

### 2. Model Architecture (LSTM)
```
Input Layer (60 timesteps × 5 features)
    ↓
LSTM Layer (50 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM Layer (50 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM Layer (50 units)
    ↓
Dropout (0.2)
    ↓
Dense Layer (25 units)
    ↓
Output Layer (1 unit)
```

### 3. Training Configuration
- **Optimizer:** Adam (learning rate: 0.001)
- **Loss Function:** Mean Squared Error (MSE)
- **Epochs:** 20
- **Batch Size:** 32
- **Validation Split:** 10%

### 4. Evaluation Metrics
- **RMSE** (Root Mean Squared Error) - measures average prediction error
- **MAE** (Mean Absolute Error) - average absolute deviation
- **MAPE** (Mean Absolute Percentage Error) - percentage-based error metric

## 📊 Visualizations

The solution generates comprehensive visualizations:

1. **Price History and Predictions** - Full timeline with train/test split
2. **Test Set Zoom** - Detailed view of predictions vs actual values
3. **Prediction Error Distribution** - Histogram of prediction errors
4. **Moving Averages** - Technical indicators overlaid on price
5. **Training History** - Loss curves during model training
6. **Scatter Plot** - Actual vs predicted prices correlation

## 🚀 How to Run

```bash
# Navigate to the project directory
cd /home/user/Data-Analysis-with-Chatbots/kaggle_solutions/02_time_series/06_bitcoin_price

# Run the solution
python solution.py
```

### Dependencies
```python
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow  # Optional - falls back to simple method if not available
```

## 📈 Expected Results

Typical performance metrics:
- **RMSE:** $500 - $2,000 (depending on price range and volatility)
- **MAE:** $300 - $1,500
- **MAPE:** 2-8%

The model performs better during stable periods and faces challenges during high volatility events.

## 🔍 Key Insights

1. **LSTM Advantages:**
   - Captures long-term dependencies in price movements
   - Handles sequential data with memory cells
   - Can learn complex non-linear patterns

2. **Technical Indicators:**
   - Moving averages help identify trends
   - Volatility measures market uncertainty
   - Price ratios indicate overbought/oversold conditions

3. **Challenges:**
   - Cryptocurrency markets are highly volatile
   - External events can cause unpredictable movements
   - Model performance degrades for longer horizons

4. **Best Practices:**
   - Use multiple features beyond just price
   - Implement proper train/test temporal splits
   - Monitor validation loss to prevent overfitting
   - Consider ensemble methods for production

## 💡 Extensions and Improvements

1. **Advanced Features:**
   - Trading volume data
   - Sentiment analysis from news/social media
   - Market indicators (RSI, MACD, Bollinger Bands)
   - Correlation with other cryptocurrencies

2. **Model Enhancements:**
   - Bidirectional LSTM layers
   - Attention mechanisms
   - Transformer architectures
   - Ensemble with other models (ARIMA, Prophet)

3. **Production Considerations:**
   - Real-time data streaming
   - Model retraining schedule
   - Confidence intervals for predictions
   - Risk management strategies

## 📚 Learning Resources

- **LSTM Networks:** [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- **Time Series with Deep Learning:** [TensorFlow Time Series Guide](https://www.tensorflow.org/tutorials/structured_data/time_series)
- **Cryptocurrency Analysis:** Technical analysis fundamentals
- **Feature Engineering:** Creating effective time series features

## ⚠️ Disclaimer

This project is for educational purposes only. Cryptocurrency markets are highly volatile and unpredictable. Never use these predictions for actual trading or financial decisions without proper risk assessment and professional advice.

## 📄 License

This project is part of the Data Analysis with Chatbots educational repository.
