#!/usr/bin/env python3
"""
Call Center Volume Forecasting
===============================
Predicts call center volume using Poisson regression and time series methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


def generate_call_center_data(n_weeks=52):
    """
    Generate synthetic call center volume data.

    Includes:
    - Hourly call volumes with daily patterns
    - Day of week effects
    - Seasonal trends
    - Special events (promotions, outages)
    """
    # Generate hourly data
    dates = pd.date_range(start='2023-01-01', periods=n_weeks*7*24, freq='H')

    # Base call rate (calls per hour)
    base_rate = 50

    # Hour of day pattern (9 AM - 5 PM peak)
    hour = dates.hour
    hour_pattern = np.where(
        (hour >= 9) & (hour <= 17),
        40 + 30 * np.sin(np.pi * (hour - 9) / 8),  # Peak during business hours
        5  # Low overnight
    )

    # Day of week pattern (higher Mon-Fri)
    day_of_week = dates.dayofweek
    dow_pattern = np.where(day_of_week < 5, 20, -15)  # Weekday vs weekend

    # Weekly seasonality (beginning of week higher)
    weekly_cycle = 10 * np.cos(2 * np.pi * day_of_week / 7)

    # Monthly pattern (end of month higher - billing cycle)
    day_of_month = dates.day
    monthly_pattern = np.where(day_of_month >= 25, 15, 0)

    # Seasonal trend (higher in Q1 and Q4)
    week_of_year = dates.isocalendar().week.values
    seasonal_pattern = 20 * np.sin(2 * np.pi * week_of_year / 52)

    # Special events (product launches, system outages)
    special_events = np.zeros(len(dates))
    # Simulate 5 major events throughout the year
    event_hours = np.random.choice(len(dates), size=5, replace=False)
    for event_hour in event_hours:
        # Event lasts 12 hours with elevated volume
        end_hour = min(event_hour + 12, len(dates))
        special_events[event_hour:end_hour] = 100

    # Poisson-distributed call arrivals
    lambda_rate = base_rate + hour_pattern + dow_pattern + weekly_cycle + monthly_pattern + seasonal_pattern + special_events
    lambda_rate = np.maximum(lambda_rate, 1)  # Ensure positive rate

    # Generate call counts from Poisson distribution
    call_volume = np.random.poisson(lambda_rate)

    # Average handle time (minutes) - varies by time of day
    # Complex calls in morning, simpler later
    avg_handle_time = 8 + 3 * np.cos(2 * np.pi * hour / 24) + np.random.normal(0, 1, len(dates))
    avg_handle_time = np.maximum(avg_handle_time, 3)  # Minimum 3 minutes

    # Calculate required staff (Erlang C approximation)
    total_call_minutes = call_volume * avg_handle_time
    required_staff = np.ceil(total_call_minutes / 60 * 1.2)  # 20% buffer for efficiency

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'call_volume': call_volume,
        'avg_handle_time': avg_handle_time,
        'required_staff': required_staff,
        'hour': hour,
        'day_of_week': day_of_week,
        'day_of_month': day_of_month,
        'week_of_year': week_of_year,
        'is_weekend': (day_of_week >= 5).astype(int),
        'is_business_hours': ((hour >= 9) & (hour <= 17)).astype(int)
    })

    return df


def create_time_features(df):
    """Create temporal features for modeling."""
    df_features = df.copy()

    # Lag features (previous hours)
    for lag in [1, 2, 3, 24, 168]:  # 1h, 2h, 3h, 1day, 1week
        df_features[f'volume_lag_{lag}h'] = df_features['call_volume'].shift(lag)

    # Rolling statistics
    df_features['volume_rolling_mean_24h'] = df_features['call_volume'].shift(1).rolling(window=24).mean()
    df_features['volume_rolling_std_24h'] = df_features['call_volume'].shift(1).rolling(window=24).std()
    df_features['volume_rolling_mean_168h'] = df_features['call_volume'].shift(1).rolling(window=168).mean()

    # Cyclical encodings
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
    df_features['dow_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['dow_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)

    return df_features


def calculate_staffing_metrics(df):
    """Calculate call center staffing metrics."""
    metrics = {}

    # Total calls
    metrics['total_calls'] = df['call_volume'].sum()

    # Peak hour
    peak_hour_data = df.groupby('hour')['call_volume'].mean().idxmax()
    metrics['peak_hour'] = peak_hour_data

    # Average calls by shift
    morning_shift = df[df['hour'].between(6, 14)]['call_volume'].mean()
    afternoon_shift = df[df['hour'].between(14, 22)]['call_volume'].mean()
    night_shift = df[(df['hour'] < 6) | (df['hour'] >= 22)]['call_volume'].mean()

    metrics['morning_shift_avg'] = morning_shift
    metrics['afternoon_shift_avg'] = afternoon_shift
    metrics['night_shift_avg'] = night_shift

    # Staffing requirements
    metrics['avg_staff_required'] = df['required_staff'].mean()
    metrics['peak_staff_required'] = df['required_staff'].max()

    return metrics


def main():
    """Main execution function."""
    print("=" * 80)
    print("CALL CENTER VOLUME FORECASTING")
    print("=" * 80)

    # Generate data
    print("\n1. Generating synthetic call center data...")
    df = generate_call_center_data(n_weeks=52)
    print(f"   Generated {len(df)} hours of data ({len(df)//24} days)")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Call volume range: {df['call_volume'].min()} - {df['call_volume'].max()} calls/hour")
    print(f"   Average volume: {df['call_volume'].mean():.2f} calls/hour")

    # Set timestamp as index
    df.set_index('timestamp', inplace=True)

    # Calculate metrics
    print("\n2. Call Center Metrics:")
    metrics = calculate_staffing_metrics(df)
    print(f"   Total calls (annual): {metrics['total_calls']:,}")
    print(f"   Peak hour: {metrics['peak_hour']}:00")
    print(f"   Morning shift avg (6-14): {metrics['morning_shift_avg']:.2f} calls/hour")
    print(f"   Afternoon shift avg (14-22): {metrics['afternoon_shift_avg']:.2f} calls/hour")
    print(f"   Night shift avg (22-6): {metrics['night_shift_avg']:.2f} calls/hour")
    print(f"   Average staff required: {metrics['avg_staff_required']:.2f} agents")
    print(f"   Peak staff required: {metrics['peak_staff_required']:.0f} agents")

    # Volume patterns
    print("\n3. Volume Patterns:")
    print("   By Day of Week:")
    dow_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_avg = df.groupby('day_of_week')['call_volume'].mean()
    for dow, label in enumerate(dow_labels):
        print(f"      {label}: {dow_avg[dow]:.2f} calls/hour")

    print("\n   Peak Hours (Top 5):")
    hourly_avg = df.groupby('hour')['call_volume'].mean().sort_values(ascending=False)
    for hour, volume in hourly_avg.head().items():
        print(f"      {hour}:00 - {volume:.2f} calls/hour")

    # Create features
    print("\n4. Creating temporal features...")
    df_features = create_time_features(df)
    df_features.dropna(inplace=True)

    # Split data
    split_idx = int(len(df_features) * 0.85)
    train_df = df_features[:split_idx]
    test_df = df_features[split_idx:]

    print(f"   Training set: {len(train_df)} hours ({len(train_df)//24} days)")
    print(f"   Test set: {len(test_df)} hours ({len(test_df)//24} days)")

    # Define features
    feature_cols = [col for col in df_features.columns
                    if col not in ['call_volume', 'avg_handle_time', 'required_staff']]
    X_train = train_df[feature_cols]
    y_train = train_df['call_volume']
    X_test = test_df[feature_cols]
    y_test = test_df['call_volume']

    # Train Poisson Regression
    print("\n5. Training Poisson Regression model...")
    poisson_model = PoissonRegressor(max_iter=500, alpha=0.1)
    poisson_model.fit(X_train, y_train)
    y_pred_poisson = poisson_model.predict(X_test)

    # Train Random Forest
    print("\n6. Training Random Forest model...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    # Aggregate to daily for Holt-Winters
    daily_df = df.resample('D')['call_volume'].sum()
    train_size_daily = int(len(daily_df) * 0.85)
    train_daily = daily_df[:train_size_daily]
    test_daily = daily_df[train_size_daily:]

    # Holt-Winters
    print("\n7. Training Holt-Winters model (daily aggregation)...")
    hw_model = ExponentialSmoothing(
        train_daily,
        seasonal_periods=7,
        trend='add',
        seasonal='add'
    ).fit()
    hw_forecast = hw_model.forecast(steps=len(test_daily))

    # Evaluate models
    print("\n8. Model Evaluation (Hourly Predictions):")

    # Poisson Regression
    mae_poisson = mean_absolute_error(y_test, y_pred_poisson)
    rmse_poisson = np.sqrt(mean_squared_error(y_test, y_pred_poisson))
    r2_poisson = r2_score(y_test, y_pred_poisson)

    print("\n   Poisson Regression:")
    print(f"      MAE: {mae_poisson:.2f} calls/hour")
    print(f"      RMSE: {rmse_poisson:.2f} calls/hour")
    print(f"      R²: {r2_poisson:.4f}")

    # Random Forest
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)

    print("\n   Random Forest:")
    print(f"      MAE: {mae_rf:.2f} calls/hour")
    print(f"      RMSE: {rmse_rf:.2f} calls/hour")
    print(f"      R²: {r2_rf:.4f}")

    # Holt-Winters (daily)
    mae_hw = mean_absolute_error(test_daily, hw_forecast)
    rmse_hw = np.sqrt(mean_squared_error(test_daily, hw_forecast))

    print("\n   Holt-Winters (Daily Total):")
    print(f"      MAE: {mae_hw:.2f} calls/day")
    print(f"      RMSE: {rmse_hw:.2f} calls/day")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n9. Top 10 Most Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")

    # Visualization
    print("\n10. Creating visualizations...")
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

    # Plot 1: Hourly volume (first 2 weeks)
    ax1 = fig.add_subplot(gs[0, :])
    display_hours = 24 * 14
    ax1.plot(df.index[:display_hours], df['call_volume'][:display_hours], linewidth=1)
    ax1.set_title('Hourly Call Volume (First 2 Weeks)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Calls/Hour')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Daily pattern
    ax2 = fig.add_subplot(gs[1, 0])
    hourly_pattern = df.groupby('hour')['call_volume'].mean()
    ax2.plot(hourly_pattern.index, hourly_pattern.values, marker='o', linewidth=2, markersize=6)
    ax2.set_title('Average Calls by Hour of Day', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Average Calls')
    ax2.grid(True, alpha=0.3)
    ax2.axvspan(9, 17, alpha=0.2, color='green', label='Business Hours')
    ax2.legend()

    # Plot 3: Weekly pattern
    ax3 = fig.add_subplot(gs[1, 1])
    dow_pattern = df.groupby('day_of_week')['call_volume'].mean()
    ax3.bar(range(7), dow_pattern.values, color='steelblue')
    ax3.set_title('Average Calls by Day of Week', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Day of Week')
    ax3.set_ylabel('Average Calls')
    ax3.set_xticks(range(7))
    ax3.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Volume distribution
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(df['call_volume'], bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax4.set_title('Call Volume Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Calls/Hour')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3, axis='y')

    # Plot 5: Business hours heatmap
    ax5 = fig.add_subplot(gs[2, 0])
    pivot_data = df.groupby(['day_of_week', 'hour'])['call_volume'].mean().unstack()
    sns.heatmap(pivot_data, cmap='YlOrRd', ax=ax5, cbar_kws={'label': 'Avg Calls'})
    ax5.set_title('Call Volume Heatmap', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Day of Week')
    ax5.set_xlabel('Hour')
    ax5.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

    # Plot 6: Staffing requirements
    ax6 = fig.add_subplot(gs[2, 1])
    staff_by_hour = df.groupby('hour')['required_staff'].mean()
    ax6.plot(staff_by_hour.index, staff_by_hour.values, marker='s', linewidth=2, markersize=5, color='green')
    ax6.set_title('Average Staff Requirements by Hour', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Hour')
    ax6.set_ylabel('Required Staff')
    ax6.grid(True, alpha=0.3)

    # Plot 7: Actual vs Predicted (RF)
    ax7 = fig.add_subplot(gs[2, 2])
    display_test_hours = min(24 * 7, len(y_test))  # 1 week
    ax7.plot(test_df.index[:display_test_hours], y_test.values[:display_test_hours],
             label='Actual', linewidth=2, alpha=0.8)
    ax7.plot(test_df.index[:display_test_hours], y_pred_rf[:display_test_hours],
             label='RF Prediction', linewidth=2, alpha=0.8)
    ax7.set_title('Forecast: Actual vs Random Forest', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Date')
    ax7.set_ylabel('Calls/Hour')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # Plot 8: Feature importance
    ax8 = fig.add_subplot(gs[3, 0])
    top_features = feature_importance.head(10)
    ax8.barh(range(len(top_features)), top_features['importance'].values)
    ax8.set_yticks(range(len(top_features)))
    ax8.set_yticklabels(top_features['feature'].values, fontsize=9)
    ax8.set_title('Top 10 Feature Importance', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Importance')
    ax8.invert_yaxis()
    ax8.grid(True, alpha=0.3, axis='x')

    # Plot 9: Model comparison
    ax9 = fig.add_subplot(gs[3, 1])
    models = ['Poisson', 'Random\nForest']
    maes = [mae_poisson, mae_rf]
    colors_bar = ['#ff7f0e', '#2ca02c']
    bars = ax9.bar(models, maes, color=colors_bar, alpha=0.7)
    ax9.set_title('Model Comparison (MAE)', fontsize=12, fontweight='bold')
    ax9.set_ylabel('MAE (calls/hour)')
    ax9.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bar, mae in zip(bars, maes):
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height,
                f'{mae:.2f}', ha='center', va='bottom', fontweight='bold')

    # Plot 10: Prediction scatter
    ax10 = fig.add_subplot(gs[3, 2])
    ax10.scatter(y_test, y_pred_rf, alpha=0.3, s=10)
    ax10.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', linewidth=2, label='Perfect Prediction')
    ax10.set_title('Actual vs Predicted (RF)', fontsize=12, fontweight='bold')
    ax10.set_xlabel('Actual Calls')
    ax10.set_ylabel('Predicted Calls')
    ax10.legend()
    ax10.grid(True, alpha=0.3)

    plt.savefig('call_center_forecast.png', dpi=300, bbox_inches='tight')
    print("   Saved: call_center_forecast.png")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
