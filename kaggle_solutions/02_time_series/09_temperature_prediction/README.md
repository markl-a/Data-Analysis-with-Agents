# Temperature Forecasting

## 📊 Project Overview

This project demonstrates temperature forecasting using Gradient Boosting with extensive temporal feature engineering, showcasing how machine learning can effectively capture seasonal patterns and weather dynamics.

**Difficulty Level:** ⭐⭐ Intermediate
**Category:** Time Series Forecasting
**Techniques:** Gradient Boosting, Feature Engineering, Cyclical Encoding

## 🎯 Objective

Forecast daily temperatures using machine learning techniques that leverage temporal patterns, lagged values, and engineered features to predict weather conditions for energy planning, agriculture, and climate analysis.

## 📁 Dataset Description

The solution generates realistic temperature data with the following characteristics:

- **Time Period:** 1095 days (3 years) of daily temperatures
- **Temporal Features:**
  - `date`: Observation date
  - `temperature`: Daily temperature in Celsius (target variable)
  - `year`, `month`, `day`: Calendar components
  - `day_of_year`: Day number (1-365)
  - `day_of_week`: Weekday (0=Monday, 6=Sunday)
  - `quarter`: Quarter (1-4)
  - `week_of_year`: ISO week number

- **Engineered Features:**
  - `month_sin`, `month_cos`: Cyclical month encoding
  - `day_of_year_sin`, `day_of_year_cos`: Cyclical day encoding
  - `temp_lag1`, `temp_lag7`, `temp_lag30`, `temp_lag365`: Lagged temperatures
  - `temp_ma7`, `temp_ma30`: Moving averages
  - `temp_std7`, `temp_std30`: Rolling standard deviations
  - `temp_range7`: 7-day temperature range

### Data Generation Features:
- **Seasonal cycle** - sinusoidal yearly pattern
- **Climate trend** - gradual warming over time
- **Weather autocorrelation** - each day related to previous day
- **Extreme events** - heatwaves and cold snaps
- **Realistic noise** - day-to-day variability

## 🧠 Methodology

### 1. Feature Engineering

**Cyclical Encoding:**
Temperature patterns are cyclical (seasons repeat yearly). We encode this using sine/cosine transformations:

```python
month_sin = sin(2π × month / 12)
month_cos = cos(2π × month / 12)
```

This ensures December (12) is close to January (1) in feature space.

**Lagged Features:**
- `temp_lag1`: Yesterday's temperature
- `temp_lag7`: Temperature 7 days ago
- `temp_lag30`: Temperature 30 days ago
- `temp_lag365`: Temperature same day last year

**Rolling Statistics:**
- Moving averages capture short/medium-term trends
- Standard deviations measure recent volatility
- Range indicates temperature variability

### 2. Gradient Boosting Model

**Configuration:**
```python
GradientBoostingRegressor(
    n_estimators=200,        # Number of boosting stages
    learning_rate=0.1,       # Shrinkage parameter
    max_depth=5,             # Maximum tree depth
    min_samples_split=10,    # Minimum samples to split
    min_samples_leaf=5,      # Minimum samples in leaf
    subsample=0.8,           # Fraction of samples per tree
    random_state=42
)
```

**How It Works:**
1. Builds trees sequentially
2. Each tree corrects errors of previous trees
3. Combines predictions through weighted sum
4. Regularization prevents overfitting

### 3. Evaluation Metrics

- **RMSE** (Root Mean Squared Error) - penalizes large errors
- **MAE** (Mean Absolute Error) - average absolute deviation
- **MAPE** (Mean Absolute Percentage Error) - relative error
- **R²** (Coefficient of Determination) - proportion of variance explained

## 📊 Visualizations

The solution generates comprehensive visualizations:

1. **Temperature Timeline** - Full history with train/test split
2. **Test Period Detail** - Zoomed predictions vs actual
3. **Error Distribution** - Histogram of prediction errors
4. **Seasonal Pattern** - Average temperature by month with std dev
5. **Feature Importance** - Top 10 most influential features
6. **Scatter Plot** - Actual vs predicted correlation

## 🚀 How to Run

```bash
# Navigate to the project directory
cd /home/user/Data-Analysis-with-Chatbots/kaggle_solutions/02_time_series/09_temperature_prediction

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
```

## 📈 Expected Results

Typical performance metrics:
- **Training RMSE:** 0.5-1.5°C
- **Test RMSE:** 1.5-3.0°C
- **Test MAE:** 1.0-2.5°C
- **Test R²:** 0.85-0.95
- **Test MAPE:** 1-3%

Performance is better for:
- Stable seasonal patterns
- Locations with predictable weather
- Short-term forecasts (1-7 days)

## 🔍 Key Insights

### 1. Feature Importance Hierarchy
Typical ranking:
1. **temp_lag1** (yesterday's temperature) - strongest predictor
2. **temp_ma7** (7-day moving average) - recent trend
3. **day_of_year_sin/cos** - seasonal position
4. **temp_lag7** (week-ago temperature) - weekly pattern
5. **month_sin/cos** - monthly seasonality

### 2. Seasonal Patterns
- Summer (Jun-Aug): Peak temperatures
- Winter (Dec-Feb): Minimum temperatures
- Spring/Fall: Transition periods
- Annual range: ~20-25°C in temperate regions

### 3. Forecasting Challenges
- **Extreme events** are difficult to predict
- **Long-term forecasts** (>14 days) degrade rapidly
- **Climate change** requires model updates
- **Local effects** (urban heat islands) not captured

### 4. Model Advantages
- **Non-linear relationships** captured naturally
- **Feature interactions** learned automatically
- **Robust to outliers** compared to linear models
- **Interpretable** through feature importance

## 💡 Extensions and Improvements

### 1. Additional Features
- **Weather variables:**
  - Humidity, pressure, wind speed
  - Cloud cover, precipitation
  - Solar radiation

- **Spatial features:**
  - Latitude, longitude, elevation
  - Distance to coast/mountains
  - Urban/rural classification

- **External data:**
  - Ocean temperatures (El Niño/La Niña)
  - Atmospheric indices (NAO, AO)
  - Historical climate data

### 2. Advanced Models
- **Ensemble methods:**
  - Combine GBM + Random Forest + LSTM
  - Weighted averaging
  - Stacking

- **Deep learning:**
  - LSTM for sequential patterns
  - CNN for spatial-temporal data
  - Transformer architectures

- **Probabilistic forecasting:**
  - Quantile regression
  - Gaussian processes
  - Bayesian neural networks

### 3. Multi-step Forecasting
Current approach predicts one day ahead. Extensions:
- **Direct strategy:** Separate model for each horizon
- **Recursive strategy:** Feed predictions back as features
- **Multi-output:** Single model predicting multiple days

### 4. Real-world Integration
- **Data pipeline:** Automated weather data ingestion
- **Model retraining:** Weekly/monthly updates
- **API deployment:** Real-time predictions
- **Monitoring:** Track forecast accuracy over time

## 📚 Learning Resources

- **Gradient Boosting:** [Scikit-learn User Guide](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)
- **Feature Engineering:** "Feature Engineering for Machine Learning" by Alice Zheng
- **Time Series ML:** "Machine Learning for Time Series Forecasting" by Jason Brownlee
- **Cyclical Features:** Understanding periodic variable encoding

## 🎯 Real-World Applications

### 1. Energy Management
- **Heating/cooling demand:** Plan HVAC system operation
- **Grid load forecasting:** Anticipate peak demand
- **Renewable energy:** Solar/wind generation estimates

### 2. Agriculture
- **Crop planning:** Planting and harvesting schedules
- **Frost protection:** Early warning systems
- **Irrigation scheduling:** Water resource management

### 3. Transportation
- **Aviation:** Flight planning and fuel optimization
- **Railways:** Track maintenance scheduling
- **Shipping:** Route optimization

### 4. Health & Safety
- **Heat warnings:** Public health alerts
- **Cold weather preparedness:** Emergency services
- **Air quality:** Temperature-pollution relationships

### 5. Tourism & Events
- **Event planning:** Outdoor activity scheduling
- **Tourism forecasting:** Visitor demand prediction
- **Insurance:** Weather-related risk assessment

## 🔧 Hyperparameter Tuning

### Grid Search Suggestions
```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [5, 10, 20],
    'subsample': [0.7, 0.8, 0.9]
}
```

### Cross-Validation Strategy
Use **TimeSeriesSplit** to respect temporal ordering:
```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

## ⚠️ Limitations

1. **Short-term focus:** Best for 1-7 day forecasts
2. **Historical dependence:** Assumes past patterns continue
3. **No physics:** Doesn't use atmospheric models
4. **Local model:** Trained for specific location
5. **Extreme events:** Black swan events not predictable
6. **Feature drift:** Changing climate may invalidate features

## 📊 Comparison with Other Methods

| Method | RMSE | Pros | Cons |
|--------|------|------|------|
| **Persistence** | 3-5°C | Simple | No trend capture |
| **Linear Regression** | 2-4°C | Interpretable | Assumes linearity |
| **ARIMA** | 2-3°C | Statistical rigor | Limited features |
| **Gradient Boosting** | 1.5-3°C | Flexible, accurate | Less interpretable |
| **LSTM** | 1-2.5°C | Sequential learning | Needs more data |
| **Numerical Weather** | 1-2°C | Physics-based | Computationally expensive |

## 📄 License

This project is part of the Data Analysis with Chatbots educational repository.
