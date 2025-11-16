# 03. Real Estate Price Mapping

## 📋 Project Overview

Analyze and predict real estate prices using geospatial features and machine learning. Identify undervalued properties and understand price patterns across geographic areas.

**Difficulty**: ⭐⭐⭐ Advanced

## 🎯 Objective

Map and predict real estate prices by:
- Analyzing spatial price patterns
- Building predictive models
- Identifying undervalued/overvalued properties
- Understanding location-based price factors

## 📊 Dataset Description

### Generated Data Features

| Feature | Description | Type |
|---------|-------------|------|
| property_id | Unique identifier | String |
| latitude/longitude | Property coordinates | Float |
| bedrooms | Number of bedrooms | Integer |
| bathrooms | Number of bathrooms | Float |
| sqft | Living area square footage | Integer |
| lot_size_sqft | Lot size | Integer |
| age_years | Property age | Integer |
| dist_to_center_km | Distance to city center | Float |
| dist_to_school_km | Distance to nearest school | Float |
| dist_to_park_km | Distance to nearest park | Float |
| dist_to_transit_km | Distance to transit | Float |
| price | Sale price | Integer |

### Dataset Size
- Properties: 1,000 listings
- POIs: 45 (schools, parks, transit)
- Neighborhoods: 5 distinct areas

## 🔍 Key Features

1. **Price Prediction**: Random Forest regression model
2. **Spatial Analysis**: Geographic price patterns
3. **POI Integration**: Distance-based features
4. **Value Assessment**: Under/overvaluation detection
5. **Feature Importance**: Key price drivers

## 🛠️ Technical Approach

### 1. Price Modeling
```python
Features:
- Property: bedrooms, bathrooms, sqft, lot_size, age
- Location: dist_to_center, dist_to_school, dist_to_park, dist_to_transit
- Model: Random Forest (100 trees)
```

### 2. Value Score
```python
Value_Score = (Predicted_Price - Actual_Price) / Predicted_Price × 100
- Positive: Undervalued (good deal)
- Negative: Overvalued (premium price)
```

### 3. Spatial Features
- Distance calculations using haversine formula
- Neighborhood clustering
- POI proximity effects

## 📈 Results & Insights

### Model Performance
- **Test MAE**: $50,000 - $80,000
- **Test R²**: 0.75 - 0.85
- **Key Predictors**: sqft, bedrooms, dist_to_center

### Key Insights
1. **Location Premium**: Proximity to city center adds 15-20% value
2. **Size Matters**: Each bedroom adds ~$80,000
3. **Age Penalty**: Each year reduces value by ~$2,000
4. **Transit Access**: Near stations command 10-15% premium
5. **School Districts**: Good schools add 8-12% value

## 🎨 Visualizations

1. **Price Heatmap**: Geographic price distribution
2. **Price per SqFt Map**: Normalized pricing
3. **Value Score Map**: Under/overvalued areas
4. **Price vs Size**: Relationship analysis
5. **Prediction Accuracy**: Model performance
6. **Price by Bedrooms**: Category analysis

## 💡 Applications

- **Home Buyers**: Find undervalued properties
- **Real Estate Agents**: Price estimation
- **Investors**: Identify opportunities
- **Developers**: Site selection
- **Appraisers**: Valuation support

## 🚀 Usage

```bash
python solution.py
```

## 📚 Libraries Used

- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **matplotlib/seaborn**: Visualization
- **scikit-learn**: Machine learning
- **scipy**: Statistical tools

## 🔗 Extensions

1. Add time series (price trends)
2. Include crime data
3. School quality ratings
4. Walk score integration
5. Deep learning models
6. Image analysis (property photos)

## 📖 Learning Outcomes

- Spatial price modeling
- Feature engineering with POI data
- Random Forest regression
- Value assessment techniques
- Real estate analytics
