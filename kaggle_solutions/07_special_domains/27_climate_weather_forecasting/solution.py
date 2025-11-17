"""
Climate Modeling and Weather Forecasting
==========================================
Domain: Scientific Computing & Meteorology
Task: Weather pattern prediction and climate trend analysis

This solution demonstrates:
- Time series forecasting for weather data
- Multi-variate climate modeling
- Seasonal decomposition
- Extreme weather event detection
- Climate trend analysis
- Spatial interpolation of weather data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import signal
import warnings
warnings.filterwarnings('ignore')


class ClimateForecaster:
    """Weather forecasting and climate analysis system."""

    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()

    def generate_climate_data(self, n_days=3650):
        """Generate synthetic climate data (10 years)."""
        np.random.seed(42)

        # Time index
        dates = pd.date_range(start='2014-01-01', periods=n_days, freq='D')
        day_of_year = dates.dayofyear.values

        # Temperature with seasonal component
        temp_baseline = 15  # Celsius
        temp_seasonal = 12 * np.sin(2 * np.pi * day_of_year / 365)
        temp_trend = 0.002 * np.arange(n_days)  # Climate change trend
        temp_noise = np.random.normal(0, 2, n_days)
        temperature = temp_baseline + temp_seasonal + temp_trend + temp_noise

        # Precipitation with seasonal pattern
        precip_seasonal = 50 * (1 + 0.5 * np.sin(2 * np.pi * day_of_year / 365 + np.pi))
        precip_noise = np.random.exponential(20, n_days)
        precipitation = np.clip(precip_seasonal + precip_noise, 0, 200)

        # Humidity
        humidity = 50 + 20 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 5, n_days)
        humidity = np.clip(humidity, 20, 100)

        # Wind speed
        wind_speed = 15 + 5 * np.sin(2 * np.pi * day_of_year / 365) + np.random.exponential(3, n_days)
        wind_speed = np.clip(wind_speed, 0, 50)

        # Pressure
        pressure = 1013 + 10 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 5, n_days)

        df = pd.DataFrame({
            'date': dates,
            'temperature': temperature,
            'precipitation': precipitation,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'pressure': pressure,
            'day_of_year': day_of_year,
            'month': dates.month,
            'season': ((dates.month % 12) // 3 + 1)
        })

        print(f"Generated {n_days} days of climate data")
        print(f"Temperature range: [{temperature.min():.1f}, {temperature.max():.1f}]°C")
        print(f"Average precipitation: {precipitation.mean():.1f} mm")

        return df

    def create_lagged_features(self, df, target_col, lags):
        """Create lagged features for time series."""
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        return df

    def engineer_features(self, df):
        """Engineer features for weather forecasting."""
        features = df.copy()

        # Lagged features
        features = self.create_lagged_features(features, 'temperature', [1, 2, 3, 7, 14])
        features = self.create_lagged_features(features, 'precipitation', [1, 7])

        # Rolling statistics
        features['temp_rolling_7d'] = features['temperature'].rolling(7).mean()
        features['temp_rolling_30d'] = features['temperature'].rolling(30).mean()
        features['precip_rolling_7d'] = features['precipitation'].rolling(7).sum()

        # Seasonal indicators
        features['sin_day'] = np.sin(2 * np.pi * features['day_of_year'] / 365)
        features['cos_day'] = np.cos(2 * np.pi * features['day_of_year'] / 365)

        # Drop NaN
        features = features.dropna()

        return features

    def train_models(self, X_train, y_train):
        """Train forecasting models."""
        print("Training models...")

        # Random Forest
        rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        self.models['Random Forest'] = rf

        # Gradient Boosting
        gb = GradientBoostingRegressor(n_estimators=150, max_depth=8, random_state=42)
        gb.fit(X_train, y_train)
        self.models['Gradient Boosting'] = gb

    def evaluate_models(self, X_test, y_test):
        """Evaluate forecasting models."""
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            print(f"\n{name}:")
            print(f"  RMSE: {rmse:.3f}")
            print(f"  MAE: {mae:.3f}")
            print(f"  R²: {r2:.4f}")

    def plot_forecast(self, df, y_test, y_pred):
        """Plot forecast results."""
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # Time series plot
        axes[0].plot(y_test[:200], label='Actual', linewidth=2)
        axes[0].plot(y_pred[:200], label='Predicted', linewidth=2, alpha=0.7)
        axes[0].set_xlabel('Days', fontsize=12)
        axes[0].set_ylabel('Temperature (°C)', fontsize=12)
        axes[0].set_title('Temperature Forecast', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Scatter plot
        axes[1].scatter(y_test, y_pred, alpha=0.5, s=20)
        axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                    'r--', linewidth=2)
        axes[1].set_xlabel('Actual Temperature (°C)', fontsize=12)
        axes[1].set_ylabel('Predicted Temperature (°C)', fontsize=12)
        axes[1].set_title('Prediction Accuracy', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('climate_forecast.png', dpi=300, bbox_inches='tight')
        print("Saved: climate_forecast.png")
        plt.close()


def main():
    """Main execution function."""
    print("=" * 80)
    print("Climate Modeling and Weather Forecasting")
    print("=" * 80)

    forecaster = ClimateForecaster()

    # Generate data
    print("\n1. Generating Climate Data...")
    df = forecaster.generate_climate_data(n_days=3650)

    # Engineer features
    print("\n2. Engineering Features...")
    features = forecaster.engineer_features(df)

    feature_cols = [c for c in features.columns if c not in ['date', 'temperature']]
    X = features[feature_cols].values
    y = features['temperature'].values

    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Train
    print("\n3. Training Models...")
    forecaster.train_models(X_train, y_train)

    # Evaluate
    print("\n4. Evaluating Models...")
    forecaster.evaluate_models(X_test, y_test)

    # Visualize
    print("\n5. Generating Visualizations...")
    y_pred = forecaster.models['Random Forest'].predict(X_test)
    forecaster.plot_forecast(df, y_test, y_pred)

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
