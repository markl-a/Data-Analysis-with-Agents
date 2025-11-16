# 10. Air Quality Spatial Analysis

## 📋 Project Overview

Analyze and predict air quality patterns across urban areas using spatial interpolation and machine learning. Map pollution hotspots and understand environmental factors affecting air quality.

**Difficulty**: ⭐⭐⭐ Advanced

## 🎯 Objective

Analyze air quality by:
- Mapping pollution distributions
- Identifying hotspots and patterns
- Building predictive models
- Understanding environmental factors

## 📊 Dataset Description

### Generated Data Features

| Feature | Description | Type |
|---------|-------------|------|
| station_id | Monitoring station ID | String |
| latitude/longitude | Station coordinates | Float |
| pm25 | PM2.5 concentration | Float |
| pm10 | PM10 concentration | Float |
| no2 | NO2 concentration | Float |
| o3 | O3 concentration | Float |
| dist_to_center_km | Distance from city center | Float |
| dist_to_industrial_km | Distance from industry | Float |
| land_use | Land use category | Categorical |
| traffic_density | Traffic volume | Float |
| green_space_pct | Green space coverage | Float |
| temperature | Air temperature | Float |
| wind_speed | Wind speed | Float |
| humidity | Relative humidity | Float |
| season | Season | Categorical |
| aqi_category | AQI category | Categorical |

### Dataset Size
- Monitoring Stations: 150
- Pollutants: PM2.5, PM10, NO2, O3
- AQI Categories: Good, Moderate, Unhealthy, etc.

## 🔍 Key Features

1. **Spatial Interpolation**: Create continuous pollution surfaces
2. **Multi-pollutant Analysis**: PM2.5, PM10, NO2, O3
3. **Random Forest Prediction**: Predict PM2.5 levels
4. **Factor Analysis**: Traffic, land use, meteorology
5. **Hotspot Identification**: High pollution areas

## 🛠️ Technical Approach

### 1. PM2.5 Modeling
```python
PM2.5 = Urban_Effect + Industrial_Effect + Traffic_Effect -
        Green_Space_Effect + Weather_Effect × Season_Multiplier

Components:
- Urban: 50 × exp(-distance/8)
- Industrial: 40 × exp(-distance/3)
- Traffic: density / 500
- Green space: -0.3 × coverage%
- Wind: Dispersion effect
```

### 2. Prediction Model
- Random Forest Regressor (100 trees)
- Features: 7 numeric + land use + season
- Target: PM2.5 concentration
- Metrics: MAE, R²

### 3. Spatial Mapping
- Cubic interpolation
- Gaussian smoothing
- Contour visualization

## 📈 Results & Insights

### Model Performance
- **Test MAE**: 3-6 μg/m³
- **Test R²**: 0.80-0.92
- **Top Predictors**: Distance to center, traffic, wind speed

### Key Insights
1. **Urban Gradient**: PM2.5 decreases with distance from center
2. **Traffic Impact**: Major contributor to NO2 and PM levels
3. **Green Space**: 1% increase → 0.3 μg/m³ reduction
4. **Wind Effect**: >2 m/s wind reduces pollution by 40%
5. **Seasonal Pattern**: Winter 50% higher than summer

## 🎨 Visualizations

1. **PM2.5 Spatial Map**: Contour map with monitoring stations
2. **PM2.5 vs Distance**: Radial decay pattern
3. **AQI Distribution**: Air quality categories
4. **Pollutants Comparison**: PM2.5, PM10, NO2, O3 levels
5. **Land Use Impact**: Pollution by category
6. **Prediction Accuracy**: Actual vs predicted scatter

## 💡 Applications

- **Public Health**: Exposure assessment and health advisories
- **Urban Planning**: Green space allocation
- **Traffic Management**: Congestion reduction strategies
- **Policy Making**: Emission standards and regulations
- **Real-time Monitoring**: Air quality forecasting

## 🚀 Usage

```bash
python solution.py
```

## 📚 Libraries Used

- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **matplotlib/seaborn**: Visualization
- **scikit-learn**: Machine learning
- **scipy**: Interpolation and filtering

## 🔗 Extensions

1. Time series forecasting
2. Source apportionment analysis
3. Health impact assessment
4. Mobile sensor networks
5. Deep learning (LSTM, CNN)
6. Satellite data integration

## 📖 Learning Outcomes

- Air quality modeling
- Spatial interpolation
- Environmental factor analysis
- Pollution hotspot detection
- Public health analytics
