# 09. Flood Risk Assessment

## 📋 Project Overview

Assess flood risk using topographic, hydrological, and land use data. Build predictive models to identify vulnerable areas for emergency planning and infrastructure development.

**Difficulty**: ⭐⭐⭐ Advanced

## 🎯 Objective

Assess flood risk by:
- Analyzing topographic factors
- Evaluating drainage and soil properties
- Building classification models
- Mapping high-risk zones

## 📊 Dataset Description

### Generated Data Features

| Feature | Description | Type |
|---------|-------------|------|
| location_id | Location identifier | String |
| latitude/longitude | Geographic coordinates | Float |
| elevation_m | Elevation above sea level | Float |
| slope_degrees | Terrain slope | Float |
| dist_to_river_km | Distance to nearest river | Float |
| soil_type | Soil category | Categorical |
| permeability | Soil permeability rating | Integer |
| land_use | Land use category | Categorical |
| impervious_surface_pct | Impervious surface (%) | Float |
| drainage_density | Drainage network density | Float |
| annual_rainfall_mm | Annual precipitation | Float |
| storm_frequency | Severe storms per year | Integer |
| has_levee | Levee protection | Binary |
| has_drainage | Drainage system | Binary |
| historical_floods | Past flood events | Integer |
| risk_level | Risk category | Categorical |

### Dataset Size
- Locations: 800 assessment points
- Risk Levels: Low, Moderate, High, Extreme
- Soil Types: 5 categories
- Land Use Types: 5 categories

## 🔍 Key Features

1. **Multi-factor Risk Model**: Topography + hydrology + infrastructure
2. **Random Forest Classification**: Predict high-risk areas
3. **Proximity Analysis**: Distance to water bodies
4. **Infrastructure Impact**: Levees and drainage effects
5. **Historical Analysis**: Past flood patterns

## 🛠️ Technical Approach

### 1. Risk Score Calculation
```python
Risk = Distance_Risk + Elevation_Risk + Slope_Risk + Soil_Risk +
       Surface_Risk + Rainfall_Risk - Drainage_Benefit - Protection_Benefit +
       Historical_Risk

Key Factors:
- Distance to river: <0.5km = very high risk
- Elevation: <10m = high risk
- Slope: <2° = poor drainage
- Impervious surfaces: Increase runoff
- Levees: Reduce risk by 15 points
```

### 2. Classification Model
- Random Forest (100 trees, depth=12)
- Binary: High/Extreme vs Low/Moderate
- Features: 11 numeric + categorical encodings
- Metrics: Accuracy, AUC-ROC

### 3. Critical Factors
- **High Risk**: Low elevation, near river, clay soil, urban
- **Low Risk**: High elevation, far from water, good drainage

## 📈 Results & Insights

### Model Performance
- **Test Accuracy**: 87-94%
- **AUC-ROC**: 0.90-0.96
- **Top Predictors**: Elevation, distance to river, impervious surfaces

### Key Insights
1. **Elevation Critical**: Most important predictor
2. **Distance Decay**: Risk drops exponentially with distance from river
3. **Urban Vulnerability**: High impervious surfaces increase risk
4. **Soil Matters**: Clay soils (low permeability) increase flooding
5. **Infrastructure Helps**: Levees and drainage reduce risk significantly

## 🎨 Visualizations

1. **Flood Risk Map**: Color-coded risk levels across geography
2. **Elevation Map**: Topographic context
3. **Distance vs Risk**: Proximity to water relationship
4. **Risk Distribution**: Bar chart of risk levels
5. **Elevation vs Risk**: Scatter plot analysis
6. **Risk by Land Use**: Category comparisons

## 💡 Applications

- **Emergency Planning**: Evacuation route planning
- **Insurance**: Premium calculation and coverage
- **Urban Planning**: Development restrictions
- **Infrastructure**: Levee and drainage placement
- **Climate Adaptation**: Sea level rise preparation

## 🚀 Usage

```bash
python solution.py
```

## 📚 Libraries Used

- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **matplotlib/seaborn**: Visualization
- **scikit-learn**: Machine learning
- **scipy**: Spatial analysis

## 🔗 Extensions

1. Storm surge modeling
2. Climate change scenarios
3. Real-time precipitation integration
4. Economic impact assessment
5. Evacuation route optimization
6. Flood insurance mapping

## 📖 Learning Outcomes

- Flood risk modeling
- Topographic analysis
- Hydrological factors
- Infrastructure assessment
- Disaster risk management
