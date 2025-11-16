#!/usr/bin/env python3
"""
Electricity Load Forecasting with Hourly Patterns
==================================================
Predicts hourly electricity load using time series analysis with seasonal patterns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


def generate_electricity_data(n_days=180):
    """
    Generate synthetic electricity load data with realistic patterns.

    Includes:
    - Daily seasonality (peak hours)
    - Weekly seasonality (weekday vs weekend)
    - Temperature correlation
    - Trend component
    """
    # Generate hourly timestamps
    dates = pd.date_range(start='2023-01-01', periods=n_days*24, freq='H')
    n_hours = len(dates)

    # Base load (MW)
    base_load = 500

    # Trend component (slight increase over time)
    trend = np.linspace(0, 50, n_hours)

    # Daily seasonality (hourly pattern)
    hour = dates.hour
    daily_pattern = (
        -100 * np.cos(2 * np.pi * hour / 24) +  # Main daily cycle
        -50 * np.cos(4 * np.pi * hour / 24) +   # Secondary peak
        80  # Offset to ensure positivity
    )

    # Weekly seasonality (lower on weekends)
    day_of_week = dates.dayofweek
    weekly_pattern = np.where(day_of_week >= 5, -80, 0)  # Weekend reduction

    # Temperature effect (simplified)
    # Assume temperature varies daily and affects cooling/heating demand
    temp_cycle = 15 * np.sin(2 * np.pi * np.arange(n_hours) / (24*7))  # Weekly temperature variation
    temp_effect = 2 * temp_cycle  # Load increases with extreme temperatures

    # Random noise
    noise = np.random.normal(0, 15, n_hours)

    # Combine all components
    load = base_load + trend + daily_pattern + weekly_pattern + temp_effect + noise

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'load_mw': load,
        'temperature': 20 + temp_cycle + np.random.normal(0, 2, n_hours),
        'hour': hour,
        'day_of_week': day_of_week,
        'is_weekend': (day_of_week >= 5).astype(int),
        'month': dates.month
    })

    return df


def create_lag_features(df, lag_hours=[1, 2, 3, 24, 168]):
    """Create lag features for the model."""
    df_features = df.copy()

    for lag in lag_hours:
        df_features[f'load_lag_{lag}h'] = df_features['load_mw'].shift(lag)

    # Rolling statistics
    df_features['load_rolling_mean_24h'] = df_features['load_mw'].shift(1).rolling(window=24).mean()
    df_features['load_rolling_std_24h'] = df_features['load_mw'].shift(1).rolling(window=24).std()

    # Cyclical encoding for hour
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)

    # Cyclical encoding for day of week
    df_features['dow_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['dow_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)

    return df_features


def main():
    """Main execution function."""
    print("=" * 80)
    print("ELECTRICITY LOAD FORECASTING WITH HOURLY PATTERNS")
    print("=" * 80)

    # Generate data
    print("\n1. Generating synthetic electricity load data...")
    df = generate_electricity_data(n_days=180)
    print(f"   Generated {len(df)} hours of data ({len(df)//24} days)")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Load range: {df['load_mw'].min():.2f} - {df['load_mw'].max():.2f} MW")

    # Set timestamp as index
    df.set_index('timestamp', inplace=True)

    # Seasonal decomposition
    print("\n2. Performing seasonal decomposition...")
    decomposition = seasonal_decompose(df['load_mw'], model='additive', period=24)

    # Create features
    print("\n3. Creating temporal features...")
    df_features = create_lag_features(df)
    df_features.dropna(inplace=True)

    # Split data (last 14 days for testing)
    split_idx = len(df_features) - 14*24
    train_df = df_features[:split_idx]
    test_df = df_features[split_idx:]

    print(f"   Training set: {len(train_df)} hours")
    print(f"   Test set: {len(test_df)} hours")

    # Define features
    feature_cols = [col for col in df_features.columns if col not in ['load_mw']]
    X_train = train_df[feature_cols]
    y_train = train_df['load_mw']
    X_test = test_df[feature_cols]
    y_test = test_df['load_mw']

    # Train Random Forest model
    print("\n4. Training Random Forest model...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred_rf = rf_model.predict(X_test)

    # Holt-Winters model for comparison
    print("\n5. Training Holt-Winters Exponential Smoothing model...")
    hw_model = ExponentialSmoothing(
        train_df['load_mw'],
        seasonal_periods=24,
        trend='add',
        seasonal='add'
    ).fit()

    y_pred_hw = hw_model.forecast(steps=len(test_df))

    # Evaluate models
    print("\n6. Model Evaluation:")
    print("   Random Forest:")
    print(f"      MAE: {mean_absolute_error(y_test, y_pred_rf):.2f} MW")
    print(f"      RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f} MW")
    print(f"      R²: {r2_score(y_test, y_pred_rf):.4f}")

    print("\n   Holt-Winters:")
    print(f"      MAE: {mean_absolute_error(y_test, y_pred_hw):.2f} MW")
    print(f"      RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_hw)):.2f} MW")
    print(f"      R²: {r2_score(y_test, y_pred_hw):.4f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n7. Top 10 Most Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")

    # Visualization
    print("\n8. Creating visualizations...")
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    # Plot 1: Time series overview
    axes[0, 0].plot(df.index[:24*30], df['load_mw'][:24*30], label='Actual Load')
    axes[0, 0].set_title('Electricity Load - First 30 Days', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Load (MW)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Seasonal decomposition
    axes[0, 1].plot(decomposition.trend.index[:24*30], decomposition.trend[:24*30])
    axes[0, 1].set_title('Trend Component', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Trend (MW)')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Daily pattern
    hourly_avg = df.groupby('hour')['load_mw'].mean()
    axes[1, 0].plot(hourly_avg.index, hourly_avg.values, marker='o')
    axes[1, 0].set_title('Average Load by Hour of Day', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Hour of Day')
    axes[1, 0].set_ylabel('Average Load (MW)')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Weekly pattern
    dow_avg = df.groupby('day_of_week')['load_mw'].mean()
    dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[1, 1].bar(range(7), dow_avg.values)
    axes[1, 1].set_title('Average Load by Day of Week', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Day of Week')
    axes[1, 1].set_ylabel('Average Load (MW)')
    axes[1, 1].set_xticks(range(7))
    axes[1, 1].set_xticklabels(dow_labels)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # Plot 5: Predictions comparison
    test_hours = 24 * 7  # Show 1 week
    axes[2, 0].plot(test_df.index[:test_hours], y_test.values[:test_hours],
                    label='Actual', linewidth=2)
    axes[2, 0].plot(test_df.index[:test_hours], y_pred_rf[:test_hours],
                    label='RF Prediction', linewidth=2, alpha=0.7)
    axes[2, 0].plot(test_df.index[:test_hours], y_pred_hw.values[:test_hours],
                    label='HW Prediction', linewidth=2, alpha=0.7)
    axes[2, 0].set_title('Forecast Comparison (First Week)', fontsize=12, fontweight='bold')
    axes[2, 0].set_xlabel('Date')
    axes[2, 0].set_ylabel('Load (MW)')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # Plot 6: Feature importance
    top_features = feature_importance.head(10)
    axes[2, 1].barh(range(len(top_features)), top_features['importance'].values)
    axes[2, 1].set_yticks(range(len(top_features)))
    axes[2, 1].set_yticklabels(top_features['feature'].values)
    axes[2, 1].set_title('Top 10 Feature Importance', fontsize=12, fontweight='bold')
    axes[2, 1].set_xlabel('Importance')
    axes[2, 1].invert_yaxis()
    axes[2, 1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('electricity_load_forecast.png', dpi=300, bbox_inches='tight')
    print("   Saved: electricity_load_forecast.png")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
