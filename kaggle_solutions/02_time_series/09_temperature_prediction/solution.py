"""
Temperature Forecasting
Predict daily temperatures using Gradient Boosting and temporal features

Dataset: Simulated temperature data with seasonal patterns
Difficulty: ⭐⭐ Intermediate
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class TemperaturePredictor:
    """Temperature forecasting using Gradient Boosting with temporal features"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def generate_temperature_data(self, n_days=1095, location='temperate'):
        """Generate realistic temperature data with seasonal patterns"""
        np.random.seed(42)

        # Create daily date range (3 years)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq='D')

        # Base temperature (varies by location)
        if location == 'temperate':
            base_temp = 15  # Celsius
            amplitude = 12   # Seasonal variation
        elif location == 'tropical':
            base_temp = 27
            amplitude = 5
        else:  # arctic
            base_temp = -5
            amplitude = 20

        # Seasonal component (yearly cycle)
        day_of_year = np.array([d.dayofyear for d in dates])
        seasonal = amplitude * np.sin(2 * np.pi * (day_of_year - 80) / 365)

        # Long-term trend (climate change)
        trend = np.linspace(0, 0.8, n_days)  # 0.8°C warming over 3 years

        # Weather variability (day-to-day fluctuations)
        # Use autocorrelated noise to simulate weather patterns
        noise = np.zeros(n_days)
        noise[0] = np.random.normal(0, 2)
        for i in range(1, n_days):
            # Each day's temp related to previous day
            noise[i] = 0.7 * noise[i-1] + np.random.normal(0, 2)

        # Occasional extreme events (heatwaves, cold snaps)
        for _ in range(8):  # 8 extreme events over 3 years
            event_start = np.random.randint(0, n_days - 7)
            event_magnitude = np.random.choice([-1, 1]) * np.random.uniform(5, 10)
            for j in range(7):
                if event_start + j < n_days:
                    noise[event_start + j] += event_magnitude * np.exp(-j / 3)

        # Combine all components
        temperature = base_temp + seasonal + trend + noise

        # Create dataframe
        df = pd.DataFrame({
            'date': dates,
            'temperature': temperature
        })

        # Add temporal features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_year'] = df['date'].dt.dayofyear
        df['day_of_week'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['week_of_year'] = df['date'].dt.isocalendar().week

        # Add cyclical encoding for temporal features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

        # Add lagged features
        df['temp_lag1'] = df['temperature'].shift(1)
        df['temp_lag7'] = df['temperature'].shift(7)
        df['temp_lag30'] = df['temperature'].shift(30)
        df['temp_lag365'] = df['temperature'].shift(365)

        # Rolling statistics
        df['temp_ma7'] = df['temperature'].rolling(window=7, center=False).mean()
        df['temp_ma30'] = df['temperature'].rolling(window=30, center=False).mean()
        df['temp_std7'] = df['temperature'].rolling(window=7, center=False).std()
        df['temp_std30'] = df['temperature'].rolling(window=30, center=False).std()

        # Temperature ranges
        df['temp_range7'] = df['temperature'].rolling(window=7).max() - df['temperature'].rolling(window=7).min()

        # Fill NaN values
        df = df.fillna(method='bfill')

        return df

    def prepare_features(self, df):
        """Prepare feature matrix for training"""
        feature_cols = [
            'month', 'day_of_year', 'day_of_week', 'quarter',
            'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos',
            'temp_lag1', 'temp_lag7', 'temp_lag30',
            'temp_ma7', 'temp_ma30', 'temp_std7', 'temp_range7'
        ]

        # Include lag365 only if we have enough data
        if 'temp_lag365' in df.columns and not df['temp_lag365'].isna().all():
            feature_cols.append('temp_lag365')

        X = df[feature_cols].copy()
        y = df['temperature'].copy()

        return X, y, feature_cols

    def train_and_evaluate(self):
        """Train model and evaluate performance"""
        print("=" * 70)
        print("Temperature Forecasting with Gradient Boosting")
        print("=" * 70)

        # Generate data
        print("\n1. Generating temperature data...")
        df = self.generate_temperature_data(location='temperate')
        print(f"   Generated {len(df)} days of temperature data")
        print(f"   Temperature range: {df['temperature'].min():.1f}°C - {df['temperature'].max():.1f}°C")
        print(f"   Average temperature: {df['temperature'].mean():.1f}°C")

        # Prepare features
        print("\n2. Engineering temporal features...")
        X, y, feature_cols = self.prepare_features(df)
        print(f"   Created {len(feature_cols)} features:")
        print(f"   - Temporal: month, day_of_year, day_of_week")
        print(f"   - Cyclical: sine/cosine encodings")
        print(f"   - Lagged: 1, 7, 30-day lags")
        print(f"   - Rolling: moving averages and standard deviations")

        # Split data (80% train, 20% test) - respecting temporal order
        train_size = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        test_dates = df['date'].iloc[train_size:]

        print(f"\n3. Splitting data...")
        print(f"   Training days: {len(X_train)}")
        print(f"   Test days: {len(X_test)}")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Gradient Boosting model
        print(f"\n4. Training Gradient Boosting model...")
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42,
            verbose=0
        )

        self.model.fit(X_train_scaled, y_train)
        print(f"   Model trained successfully!")

        # Make predictions
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)

        # Calculate metrics
        print("\n" + "=" * 70)
        print("EVALUATION METRICS")
        print("=" * 70)

        # Training metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)

        print(f"\nTraining Set Performance:")
        print(f"  RMSE: {train_rmse:.2f}°C")
        print(f"  MAE:  {train_mae:.2f}°C")
        print(f"  R²:   {train_r2:.4f}")

        # Test metrics
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_mae = mean_absolute_error(y_test, test_pred)
        test_mape = mean_absolute_percentage_error(y_test + 273.15, test_pred + 273.15) * 100  # Convert to Kelvin for MAPE
        test_r2 = r2_score(y_test, test_pred)

        print(f"\nTest Set Performance:")
        print(f"  RMSE: {test_rmse:.2f}°C")
        print(f"  MAE:  {test_mae:.2f}°C")
        print(f"  MAPE: {test_mape:.2f}%")
        print(f"  R²:   {test_r2:.4f}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\nTop 5 Most Important Features:")
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
        fig = plt.figure(figsize=(16, 12))

        # 1. Temperature history and predictions
        ax1 = plt.subplot(3, 2, 1)
        plt.plot(df['date'], df['temperature'], label='Actual Temperature',
                color='blue', linewidth=1, alpha=0.7)
        plt.plot(test_dates, test_pred, label='Predictions',
                color='red', linewidth=2, alpha=0.8)
        plt.axvline(x=df['date'].iloc[train_size], color='green',
                   linestyle='--', label='Train/Test Split', alpha=0.5)
        plt.xlabel('Date')
        plt.ylabel('Temperature (°C)')
        plt.title('Temperature: Actual vs Predicted', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # 2. Test set detail
        ax2 = plt.subplot(3, 2, 2)
        plt.plot(test_dates, y_test.values, label='Actual',
                marker='o', markersize=2, linewidth=1.5, alpha=0.7)
        plt.plot(test_dates, test_pred, label='Predicted',
                marker='s', markersize=2, linewidth=1.5, alpha=0.7)
        plt.xlabel('Date')
        plt.ylabel('Temperature (°C)')
        plt.title('Test Set: Predictions vs Actual', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # 3. Prediction errors
        ax3 = plt.subplot(3, 2, 3)
        errors = test_pred - y_test.values
        plt.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='coral')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Prediction Error (°C)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Errors', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # 4. Seasonal pattern
        ax4 = plt.subplot(3, 2, 4)
        monthly_avg = df.groupby('month')['temperature'].agg(['mean', 'std'])
        months = range(1, 13)
        plt.errorbar(months, monthly_avg['mean'], yerr=monthly_avg['std'],
                    marker='o', linewidth=2, markersize=8, capsize=5)
        plt.xlabel('Month')
        plt.ylabel('Temperature (°C)')
        plt.title('Seasonal Pattern: Average Temperature by Month',
                 fontsize=12, fontweight='bold')
        plt.xticks(months)
        plt.grid(True, alpha=0.3)

        # 5. Feature importance
        ax5 = plt.subplot(3, 2, 5)
        top_features = feature_importance.head(10)
        plt.barh(range(len(top_features)), top_features['importance'],
                color='steelblue', edgecolor='black', alpha=0.7)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Top 10 Feature Importance', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')

        # 6. Actual vs Predicted scatter
        ax6 = plt.subplot(3, 2, 6)
        plt.scatter(y_test, test_pred, alpha=0.5, s=20)
        plt.plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()],
                'r--', linewidth=2, label='Perfect Prediction')
        plt.xlabel('Actual Temperature (°C)')
        plt.ylabel('Predicted Temperature (°C)')
        plt.title('Actual vs Predicted (Test Set)', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/02_time_series/09_temperature_prediction/temperature_forecast.png',
                   dpi=300, bbox_inches='tight')
        print("\n📊 Visualizations saved to 'temperature_forecast.png'")
        plt.close()


def main():
    """Main execution function"""
    # Create and run predictor
    predictor = TemperaturePredictor()
    results = predictor.train_and_evaluate()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nKey Insights:")
    print("1. Temperature shows strong seasonal patterns (summer/winter)")
    print("2. Recent temperatures (lag features) are highly predictive")
    print("3. Cyclical encoding captures periodic nature effectively")
    print("4. Gradient Boosting handles non-linear relationships well")
    print("\nApplications:")
    print("• Energy demand forecasting (heating/cooling)")
    print("• Agricultural planning (crop cycles)")
    print("• Climate analysis and monitoring")
    print("• Event planning and tourism")

    print("\n✅ Temperature forecasting model successfully trained and evaluated")


if __name__ == "__main__":
    main()
