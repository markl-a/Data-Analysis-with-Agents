# 08. Wildfire Risk Prediction

## 📋 Project Overview

Predict wildfire risk using environmental, topographic, and climate data. Build machine learning models to identify high-risk areas for prevention and resource allocation.

**Difficulty**: ⭐⭐⭐ Advanced

## 🎯 Objective

Predict wildfire risk by:
- Analyzing environmental factors
- Building classification models
- Identifying high-risk zones
- Understanding risk drivers

## 📊 Dataset Description

### Generated Data Features

| Feature | Description | Type |
|---------|-------------|------|
| location_id | Location identifier | String |
| latitude/longitude | Geographic coordinates | Float |
| elevation_m | Elevation above sea level | Float |
| slope_degrees | Terrain slope | Float |
| aspect_degrees | Slope direction | Float |
| vegetation_density | Vegetation coverage (%) | Float |
| vegetation_type | Vegetation category | Categorical |
| temperature_avg | Average temperature | Float |
| precipitation_mm | Annual precipitation | Float |
| humidity_avg | Average humidity (%) | Float |
| wind_speed_kmh | Wind speed | Float |
| dist_to_road_km | Distance to roads | Float |
| dist_to_urban_km | Distance to urban areas | Float |
| fires_in_10km | Historical fires nearby | Integer |
| years_since_fire | Years since last fire | Float |
| risk_class | Risk category | Categorical |

### Dataset Size
- Locations: 1,000 assessment points
- Risk Classes: Low, Moderate, High, Extreme
- Vegetation Types: 5 categories

## 🔍 Key Features

1. **Multi-factor Risk Assessment**: Environmental + topographic + human factors
2. **Random Forest Classification**: Predict high-risk areas
3. **Feature Importance**: Identify key risk drivers
4. **Spatial Risk Mapping**: Geographic risk visualization
5. **Vegetation Analysis**: Fire behavior by vegetation type

## 🛠️ Technical Approach

### 1. Risk Score Calculation
```python
Risk = Vegetation_Risk + Climate_Risk + Topography_Risk + Human_Risk + Historical_Risk

Components:
- Vegetation: Type-specific flammability
- Climate: Temperature, precipitation, humidity, wind
- Topography: Slope, aspect (south-facing)
- Human: Road proximity, wildland-urban interface
- Historical: Recent fires, fuel load
```

### 2. Classification Model
- Random Forest (100 trees)
- Binary: High/Extreme vs Low/Moderate
- Features: 12 numeric + 5 vegetation types
- Metrics: Accuracy, AUC-ROC

### 3. Risk Factors
- **High Risk**: Hot, dry, windy, steep, shrubland
- **Low Risk**: Cool, wet, flat, sparse vegetation

## 📈 Results & Insights

### Model Performance
- **Test Accuracy**: 85-92%
- **AUC-ROC**: 0.88-0.95
- **Top Predictors**: Precipitation, temperature, vegetation type

### Key Insights
1. **Climate Dominant**: Temperature and precipitation are strongest predictors
2. **Vegetation Critical**: Shrubland and forest most risky
3. **Topography Effect**: Slopes >20° significantly increase risk
4. **Human Factor**: Roads increase ignition risk
5. **Historical Pattern**: Recent burns reduce short-term risk

## 🎨 Visualizations

1. **Risk Map**: Color-coded risk levels across geography
2. **Risk by Vegetation**: Average risk scores by type
3. **Climate Factors**: Temperature vs precipitation risk
4. **Risk Distribution**: Bar chart of risk classes
5. **Topography**: Elevation and slope patterns
6. **Prediction Probability**: Model confidence distribution

## 💡 Applications

- **Fire Prevention**: Resource pre-positioning
- **Evacuation Planning**: High-risk area identification
- **Land Management**: Controlled burns, fuel reduction
- **Insurance**: Premium calculation
- **Policy Making**: Building codes in fire-prone areas

## 🚀 Usage

```bash
python solution.py
```

## 📚 Libraries Used

- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **matplotlib/seaborn**: Visualization
- **scikit-learn**: Machine learning

## 🔗 Extensions

1. Time series forecasting
2. Real-time fire weather indices
3. Deep learning models
4. Satellite imagery integration
5. Fire spread simulation
6. Economic impact modeling

## 📖 Learning Outcomes

- Environmental risk modeling
- Random Forest classification
- Feature importance analysis
- Geospatial risk mapping
- Disaster risk assessment
