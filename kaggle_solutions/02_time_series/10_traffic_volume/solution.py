"""
Traffic Volume Prediction
Predict hourly traffic volume using XGBoost with temporal and contextual features

Dataset: Simulated traffic sensor data
Difficulty: ⭐⭐⭐ Advanced
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost, fallback to Random Forest
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Using Random Forest fallback.")


class TrafficVolumePredictor:
    """Traffic volume forecasting using XGBoost with rich feature engineering"""

    def __init__(self):
        self.model = None
        self.label_encoders = {}

    def generate_traffic_data(self, n_hours=8760):
        """Generate realistic hourly traffic data (1 year)"""
        np.random.seed(42)

        # Create hourly date range (1 year = 8760 hours)
        start_date = pd.Timestamp.now() - pd.Timedelta(days=365)
        dates = pd.date_range(start=start_date, periods=n_hours, freq='H')

        # Base traffic volume
        base_volume = 1000

        # Hour of day pattern (rush hours in morning and evening)
        hour_pattern = np.array([
            0.3, 0.2, 0.15, 0.15, 0.2, 0.4,   # 0-5 AM: very low
            0.7, 1.2, 1.5, 1.3, 1.0, 0.9,     # 6-11 AM: morning rush
            0.95, 1.0, 1.05, 1.1, 1.3, 1.6,   # 12-5 PM: afternoon increase
            1.7, 1.4, 1.0, 0.8, 0.6, 0.4      # 6-11 PM: evening rush then decrease
        ])

        # Day of week pattern (weekdays higher than weekends)
        day_of_week_multiplier = {
            0: 1.1,  # Monday
            1: 1.15, # Tuesday
            2: 1.2,  # Wednesday
            3: 1.15, # Thursday
            4: 1.1,  # Friday
            5: 0.7,  # Saturday
            6: 0.6   # Sunday
        }

        # Monthly pattern (summer months slightly different)
        month_multiplier = {
            1: 0.95, 2: 0.95, 3: 1.0, 4: 1.05, 5: 1.1, 6: 1.15,
            7: 1.15, 8: 1.1, 9: 1.05, 10: 1.0, 11: 0.95, 12: 0.9
        }

        # Generate traffic volume
        volume = []
        weather_conditions = []
        holiday_flags = []

        for date in dates:
            hour = date.hour
            dow = date.dayofweek
            month = date.month

            # Base calculation
            vol = base_volume * hour_pattern[hour] * day_of_week_multiplier[dow] * month_multiplier[month]

            # Weather effects (random weather)
            weather = np.random.choice(['Clear', 'Rain', 'Snow', 'Fog'],
                                      p=[0.6, 0.25, 0.1, 0.05])
            weather_conditions.append(weather)

            if weather == 'Rain':
                vol *= np.random.uniform(0.85, 0.95)
            elif weather == 'Snow':
                vol *= np.random.uniform(0.6, 0.8)
            elif weather == 'Fog':
                vol *= np.random.uniform(0.75, 0.85)

            # Holiday effect (reduced traffic)
            is_holiday = (month == 12 and date.day in [24, 25, 31]) or \
                        (month == 1 and date.day == 1) or \
                        (month == 7 and date.day == 4) or \
                        (month == 11 and date.day in [24, 25])  # Thanksgiving
            holiday_flags.append(1 if is_holiday else 0)

            if is_holiday:
                vol *= np.random.uniform(0.4, 0.6)

            # Add noise
            vol *= np.random.uniform(0.95, 1.05)

            # Ensure positive
            volume.append(max(int(vol), 10))

        # Create dataframe
        df = pd.DataFrame({
            'datetime': dates,
            'traffic_volume': volume,
            'weather': weather_conditions,
            'is_holiday': holiday_flags
        })

        # Extract temporal features
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['day_name'] = df['datetime'].dt.day_name()
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['week_of_year'] = df['datetime'].dt.isocalendar().week
        df['quarter'] = df['datetime'].dt.quarter

        # Rush hour indicators
        df['is_morning_rush'] = df['hour'].isin([7, 8, 9]).astype(int)
        df['is_evening_rush'] = df['hour'].isin([17, 18, 19]).astype(int)
        df['is_business_hours'] = df['hour'].isin(range(9, 18)).astype(int)

        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Lagged features
        df['volume_lag1'] = df['traffic_volume'].shift(1)
        df['volume_lag24'] = df['traffic_volume'].shift(24)  # Same hour yesterday
        df['volume_lag168'] = df['traffic_volume'].shift(168)  # Same hour last week

        # Rolling statistics
        df['volume_ma24'] = df['traffic_volume'].rolling(window=24, center=False).mean()
        df['volume_ma168'] = df['traffic_volume'].rolling(window=168, center=False).mean()
        df['volume_std24'] = df['traffic_volume'].rolling(window=24, center=False).std()

        # Fill NaN
        df = df.fillna(method='bfill')

        return df

    def prepare_features(self, df):
        """Prepare feature matrix for training"""
        # Encode categorical variables
        df_encoded = df.copy()

        if 'weather' in df.columns:
            if 'weather' not in self.label_encoders:
                self.label_encoders['weather'] = LabelEncoder()
                df_encoded['weather_encoded'] = self.label_encoders['weather'].fit_transform(df['weather'])
            else:
                df_encoded['weather_encoded'] = self.label_encoders['weather'].transform(df['weather'])

        # Select features
        feature_cols = [
            'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday',
            'is_morning_rush', 'is_evening_rush', 'is_business_hours',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            'day_of_week_sin', 'day_of_week_cos',
            'volume_lag1', 'volume_lag24', 'volume_lag168',
            'volume_ma24', 'volume_ma168', 'volume_std24',
            'weather_encoded', 'week_of_year'
        ]

        X = df_encoded[feature_cols].copy()
        y = df['traffic_volume'].copy()

        return X, y, feature_cols

    def train_and_evaluate(self):
        """Train model and evaluate performance"""
        print("=" * 70)
        print("Traffic Volume Prediction with XGBoost")
        print("=" * 70)

        # Generate data
        print("\n1. Generating hourly traffic data...")
        df = self.generate_traffic_data()
        print(f"   Generated {len(df)} hours of traffic data (1 year)")
        print(f"   Volume range: {df['traffic_volume'].min():,} - {df['traffic_volume'].max():,} vehicles/hour")
        print(f"   Average volume: {df['traffic_volume'].mean():,.0f} vehicles/hour")

        # Data exploration
        print("\n2. Analyzing traffic patterns...")
        print(f"   Peak traffic hour: {df.groupby('hour')['traffic_volume'].mean().idxmax()}:00")
        print(f"   Busiest day: {df.groupby('day_name')['traffic_volume'].mean().idxmax()}")
        weather_impact = df.groupby('weather')['traffic_volume'].mean().sort_values(ascending=False)
        print(f"   Weather impact:")
        for weather, vol in weather_impact.items():
            print(f"     {weather}: {vol:,.0f} avg vehicles/hour")

        # Prepare features
        print("\n3. Engineering features...")
        X, y, feature_cols = self.prepare_features(df)
        print(f"   Created {len(feature_cols)} features")
        print(f"   - Temporal: hour, day_of_week, month, week_of_year")
        print(f"   - Cyclical: sine/cosine encodings")
        print(f"   - Contextual: weather, holidays, rush hours")
        print(f"   - Lagged: 1hr, 24hr, 168hr lags")
        print(f"   - Rolling: moving averages and std")

        # Split data (80% train, 20% test) - respecting temporal order
        train_size = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        test_dates = df['datetime'].iloc[train_size:]

        print(f"\n4. Splitting data...")
        print(f"   Training hours: {len(X_train):,}")
        print(f"   Test hours: {len(X_test):,}")

        # Train model
        if XGBOOST_AVAILABLE:
            print(f"\n5. Training XGBoost model...")
            self.model = xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        else:
            print(f"\n5. Training Random Forest model...")
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                verbose=0
            )

        self.model.fit(X_train, y_train)
        print(f"   Model trained successfully!")

        # Make predictions
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        # Calculate metrics
        print("\n" + "=" * 70)
        print("EVALUATION METRICS")
        print("=" * 70)

        # Training metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)

        print(f"\nTraining Set Performance:")
        print(f"  RMSE: {train_rmse:.2f} vehicles/hour")
        print(f"  MAE:  {train_mae:.2f} vehicles/hour")
        print(f"  R²:   {train_r2:.4f}")

        # Test metrics
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_mae = mean_absolute_error(y_test, test_pred)
        test_mape = mean_absolute_percentage_error(y_test, test_pred) * 100
        test_r2 = r2_score(y_test, test_pred)

        print(f"\nTest Set Performance:")
        print(f"  RMSE: {test_rmse:.2f} vehicles/hour")
        print(f"  MAE:  {test_mae:.2f} vehicles/hour")
        print(f"  MAPE: {test_mape:.2f}%")
        print(f"  R²:   {test_r2:.4f}")

        # Feature importance
        if XGBOOST_AVAILABLE:
            importance_type = 'gain'
            importances = self.model.feature_importances_
        else:
            importance_type = 'impurity'
            importances = self.model.feature_importances_

        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print(f"\nTop 5 Most Important Features ({importance_type}):")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        # Visualizations
        self.create_visualizations(df, train_size, train_pred, test_pred,
                                   y_train, y_test, test_dates, feature_importance)

        return {
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'test_mape': test_mape
        }

    def create_visualizations(self, df, train_size, train_pred, test_pred,
                             y_train, y_test, test_dates, feature_importance):
        """Create comprehensive visualizations"""
        fig = plt.figure(figsize=(16, 14))

        # 1. Traffic volume timeline
        ax1 = plt.subplot(4, 2, 1)
        # Plot every 24th point for clarity
        plot_indices = range(0, len(df), 24)
        plt.plot(df['datetime'].iloc[plot_indices], df['traffic_volume'].iloc[plot_indices],
                label='Actual', linewidth=1, alpha=0.7)
        test_plot_indices = range(0, len(test_dates), 24)
        plt.plot(test_dates.iloc[test_plot_indices], test_pred[test_plot_indices],
                label='Predictions', linewidth=2, alpha=0.8, color='red')
        plt.axvline(x=df['datetime'].iloc[train_size], color='green',
                   linestyle='--', label='Train/Test Split', alpha=0.5)
        plt.xlabel('Date')
        plt.ylabel('Vehicles/Hour')
        plt.title('Traffic Volume: Actual vs Predicted (Daily Sampling)',
                 fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # 2. Test period detail (first week)
        ax2 = plt.subplot(4, 2, 2)
        week_hours = min(168, len(test_dates))
        plt.plot(range(week_hours), y_test.iloc[:week_hours].values,
                label='Actual', marker='o', markersize=3, linewidth=1.5)
        plt.plot(range(week_hours), test_pred[:week_hours],
                label='Predicted', marker='s', markersize=3, linewidth=1.5)
        plt.xlabel('Hours from Test Start')
        plt.ylabel('Vehicles/Hour')
        plt.title('First Week of Test Set', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 3. Hourly pattern
        ax3 = plt.subplot(4, 2, 3)
        hourly_avg = df.groupby('hour')['traffic_volume'].agg(['mean', 'std'])
        plt.errorbar(range(24), hourly_avg['mean'], yerr=hourly_avg['std'],
                    marker='o', linewidth=2, markersize=6, capsize=4)
        plt.axvspan(7, 9, alpha=0.2, color='orange', label='Morning Rush')
        plt.axvspan(17, 19, alpha=0.2, color='red', label='Evening Rush')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Vehicles/Hour')
        plt.title('Traffic Pattern by Hour', fontsize=12, fontweight='bold')
        plt.xticks(range(0, 24, 2))
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 4. Day of week pattern
        ax4 = plt.subplot(4, 2, 4)
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_avg = df.groupby('day_name')['traffic_volume'].mean().reindex(day_order)
        colors = ['steelblue' if day not in ['Saturday', 'Sunday'] else 'coral'
                 for day in day_order]
        plt.bar(range(7), dow_avg.values, color=colors,
               edgecolor='black', alpha=0.7)
        plt.xlabel('Day of Week')
        plt.ylabel('Average Vehicles/Hour')
        plt.title('Traffic Pattern by Day of Week', fontsize=12, fontweight='bold')
        plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        plt.grid(True, alpha=0.3, axis='y')

        # 5. Weather impact
        ax5 = plt.subplot(4, 2, 5)
        weather_avg = df.groupby('weather')['traffic_volume'].mean().sort_values(ascending=True)
        plt.barh(range(len(weather_avg)), weather_avg.values,
                color='skyblue', edgecolor='black', alpha=0.7)
        plt.yticks(range(len(weather_avg)), weather_avg.index)
        plt.xlabel('Average Vehicles/Hour')
        plt.ylabel('Weather Condition')
        plt.title('Traffic Volume by Weather', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')

        # 6. Feature importance
        ax6 = plt.subplot(4, 2, 6)
        top_features = feature_importance.head(10)
        plt.barh(range(len(top_features)), top_features['importance'],
                color='mediumseagreen', edgecolor='black', alpha=0.7)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Top 10 Feature Importance', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')

        # 7. Prediction errors
        ax7 = plt.subplot(4, 2, 7)
        errors = test_pred - y_test.values
        plt.hist(errors, bins=40, edgecolor='black', alpha=0.7, color='orange')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Prediction Error (vehicles/hour)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Errors', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # 8. Actual vs Predicted scatter
        ax8 = plt.subplot(4, 2, 8)
        plt.scatter(y_test, test_pred, alpha=0.3, s=10)
        plt.plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()],
                'r--', linewidth=2, label='Perfect Prediction')
        plt.xlabel('Actual Volume (vehicles/hour)')
        plt.ylabel('Predicted Volume (vehicles/hour)')
        plt.title('Actual vs Predicted (Test Set)', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/02_time_series/10_traffic_volume/traffic_forecast.png',
                   dpi=300, bbox_inches='tight')
        print("\n📊 Visualizations saved to 'traffic_forecast.png'")
        plt.close()


def main():
    """Main execution function"""
    # Create and run predictor
    predictor = TrafficVolumePredictor()
    results = predictor.train_and_evaluate()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nKey Insights:")
    print("1. Traffic shows strong hourly patterns (rush hours)")
    print("2. Weekday traffic significantly higher than weekends")
    print("3. Weather conditions notably impact traffic volume")
    print("4. Recent lags (especially 24hr) are highly predictive")

    if XGBOOST_AVAILABLE:
        print("\n✅ XGBoost model successfully trained and evaluated")
    else:
        print("\n⚠️  Random Forest used (install xgboost for optimal performance)")

    print("\nApplications:")
    print("• Traffic management and signal optimization")
    print("• Infrastructure planning and maintenance scheduling")
    print("• Emergency response resource allocation")
    print("• Environmental impact assessment (emissions)")


if __name__ == "__main__":
    main()
