"""
Air Quality Prediction - Environmental Time Series Analysis

This module predicts air quality index (AQI) using historical pollution
data and meteorological factors.

Dataset: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
Difficulty: ⭐⭐⭐ Advanced Level
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)


class AirQualityPredictor:
    """Air Quality Index Prediction Model."""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_names = []

    def create_sample_data(self) -> pd.DataFrame:
        """Create realistic air quality dataset."""
        np.random.seed(42)

        # Generate 2 years of hourly data
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='H')
        n_hours = len(dates)

        # Generate seasonal patterns
        day_of_year = np.array([d.timetuple().tm_yday for d in dates])
        hour_of_day = np.array([d.hour for d in dates])

        # Base pollution levels with seasonal variation (higher in winter)
        seasonal_factor = 1 + 0.5 * np.cos(2 * np.pi * day_of_year / 365)

        # Daily pattern (rush hours have higher pollution)
        daily_factor = 1 + 0.3 * np.sin(2 * np.pi * (hour_of_day - 8) / 24)

        # Generate pollutants
        pm25_base = 35 * seasonal_factor * daily_factor
        pm25 = (pm25_base + np.random.normal(0, 10, n_hours)).clip(5, 300)

        pm10 = (pm25 * 1.5 + np.random.normal(0, 15, n_hours)).clip(10, 500)

        no2 = (25 * daily_factor + np.random.normal(0, 8, n_hours)).clip(5, 150)

        so2 = (10 + np.random.exponential(5, n_hours)).clip(2, 100)

        co = (0.8 * daily_factor + np.random.exponential(0.3, n_hours)).clip(0.1, 5)

        o3 = (30 + 20 * np.sin(2 * np.pi * hour_of_day / 24) +
              np.random.normal(0, 10, n_hours)).clip(5, 150)

        # Weather features
        temperature = (15 + 10 * np.cos(2 * np.pi * day_of_year / 365) +
                      5 * np.sin(2 * np.pi * hour_of_day / 24) +
                      np.random.normal(0, 3, n_hours))

        humidity = (60 - 20 * np.cos(2 * np.pi * day_of_year / 365) +
                   np.random.normal(0, 10, n_hours)).clip(20, 100)

        wind_speed = (np.random.exponential(3, n_hours) + 1).clip(0.5, 20)

        # Calculate AQI (simplified formula based on PM2.5)
        aqi = np.where(pm25 <= 12, pm25 * 50 / 12,
              np.where(pm25 <= 35.4, 50 + (pm25 - 12) * 50 / 23.4,
              np.where(pm25 <= 55.4, 100 + (pm25 - 35.4) * 50 / 20,
              np.where(pm25 <= 150.4, 150 + (pm25 - 55.4) * 50 / 95,
              np.where(pm25 <= 250.4, 200 + (pm25 - 150.4) * 100 / 100,
              300 + (pm25 - 250.4) * 100 / 99.6)))))

        aqi = aqi.clip(0, 500).round(0).astype(int)

        return pd.DataFrame({
            'datetime': dates,
            'PM2.5': pm25.round(1),
            'PM10': pm10.round(1),
            'NO2': no2.round(1),
            'SO2': so2.round(1),
            'CO': co.round(2),
            'O3': o3.round(1),
            'temperature': temperature.round(1),
            'humidity': humidity.round(1),
            'wind_speed': wind_speed.round(1),
            'AQI': aqi
        })

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time and lag features."""
        df = df.copy()

        # Time features
        df['hour'] = df['datetime'].dt.hour
        df['day'] = df['datetime'].dt.day
        df['month'] = df['datetime'].dt.month
        df['dayofweek'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Rush hour indicator
        df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)

        # Lag features for AQI
        for lag in [1, 6, 12, 24]:
            df[f'AQI_lag_{lag}'] = df['AQI'].shift(lag)

        # Rolling features
        for window in [6, 12, 24]:
            df[f'AQI_rolling_mean_{window}'] = df['AQI'].rolling(window).mean()
            df[f'PM25_rolling_mean_{window}'] = df['PM2.5'].rolling(window).mean()

        # Weather interaction
        df['temp_humidity'] = df['temperature'] * df['humidity'] / 100

        return df.dropna()

    def plot_analysis(self, df: pd.DataFrame, output_dir: str = '.') -> None:
        """Generate analysis visualizations."""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Air Quality Analysis', fontsize=16)

        # AQI time series (sample)
        sample = df.head(24*30)  # First month
        axes[0, 0].plot(sample['datetime'], sample['AQI'], linewidth=0.8)
        axes[0, 0].set_title('AQI Time Series (30 days)')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # AQI distribution
        df['AQI'].hist(bins=50, ax=axes[0, 1], color='steelblue', alpha=0.7)
        axes[0, 1].set_title('AQI Distribution')
        axes[0, 1].axvline(x=50, color='green', linestyle='--', label='Good')
        axes[0, 1].axvline(x=100, color='yellow', linestyle='--', label='Moderate')
        axes[0, 1].axvline(x=150, color='orange', linestyle='--', label='Unhealthy')
        axes[0, 1].legend()

        # Hourly pattern
        hourly = df.groupby('hour')['AQI'].mean()
        hourly.plot(ax=axes[0, 2], marker='o', color='coral')
        axes[0, 2].set_title('Average AQI by Hour')
        axes[0, 2].set_xlabel('Hour')

        # Monthly pattern
        monthly = df.groupby('month')['AQI'].mean()
        monthly.plot(kind='bar', ax=axes[1, 0], color='purple')
        axes[1, 0].set_title('Average AQI by Month')

        # PM2.5 vs AQI
        axes[1, 1].scatter(df['PM2.5'], df['AQI'], alpha=0.1, s=1)
        axes[1, 1].set_title('PM2.5 vs AQI')
        axes[1, 1].set_xlabel('PM2.5')
        axes[1, 1].set_ylabel('AQI')

        # Wind speed effect
        wind_bins = pd.cut(df['wind_speed'], bins=5)
        df.groupby(wind_bins)['AQI'].mean().plot(kind='bar', ax=axes[1, 2], color='green')
        axes[1, 2].set_title('AQI by Wind Speed')
        axes[1, 2].tick_params(axis='x', rotation=45)

        # Pollutant correlations
        pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'AQI']
        sns.heatmap(df[pollutants].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=axes[2, 0])
        axes[2, 0].set_title('Pollutant Correlations')

        # Temperature effect
        axes[2, 1].scatter(df['temperature'], df['AQI'], alpha=0.1, s=1)
        axes[2, 1].set_title('Temperature vs AQI')
        axes[2, 1].set_xlabel('Temperature')
        axes[2, 1].set_ylabel('AQI')

        # Weekend vs Weekday
        df.groupby('is_weekend')['AQI'].mean().plot(kind='bar', ax=axes[2, 2], color='teal')
        axes[2, 2].set_title('AQI: Weekday vs Weekend')
        axes[2, 2].set_xticklabels(['Weekday', 'Weekend'], rotation=0)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/air_quality_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Analysis saved to {output_dir}/air_quality_analysis.png")
        plt.close()

    def prepare_data(self, df: pd.DataFrame) -> Tuple:
        """Prepare train/test data."""
        df = self.create_features(df)

        # Feature columns (excluding target and datetime)
        feature_cols = [col for col in df.columns
                       if col not in ['datetime', 'AQI'] and not col.startswith('AQI_lag')]
        feature_cols += [f'AQI_lag_{lag}' for lag in [1, 6, 12, 24]]
        feature_cols += [f'AQI_rolling_mean_{w}' for w in [6, 12, 24]]
        feature_cols += [f'PM25_rolling_mean_{w}' for w in [6, 12, 24]]

        self.feature_names = [col for col in feature_cols if col in df.columns]

        # Time-based split
        split_idx = int(len(df) * 0.8)
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]

        X_train = train[self.feature_names]
        y_train = train['AQI']
        X_test = test[self.feature_names]
        y_test = test['AQI']

        return X_train, X_test, y_train, y_test, test['datetime']

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train forecasting models."""
        X_scaled = self.scaler.fit_transform(X_train)

        print("\nTraining models...")

        self.models['Linear Regression'] = LinearRegression()
        self.models['Linear Regression'].fit(X_scaled, y_train)

        self.models['Random Forest'] = RandomForestRegressor(
            n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
        )
        self.models['Random Forest'].fit(X_scaled, y_train)

        self.models['Gradient Boosting'] = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
        )
        self.models['Gradient Boosting'].fit(X_scaled, y_train)

        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            )
            self.models['XGBoost'].fit(X_scaled, y_train)

        if LIGHTGBM_AVAILABLE:
            self.models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, verbose=-1
            )
            self.models['LightGBM'].fit(X_scaled, y_train)

        print(f"Trained {len(self.models)} models!")

    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """Evaluate all models."""
        X_scaled = self.scaler.transform(X_test)
        results = []

        for name, model in self.models.items():
            y_pred = model.predict(X_scaled)
            results.append({
                'Model': name,
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAE': mean_absolute_error(y_test, y_pred),
                'MAPE': np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
                'R2': r2_score(y_test, y_pred)
            })

        results_df = pd.DataFrame(results).sort_values('RMSE')
        self.best_model = self.models[results_df.iloc[0]['Model']]
        return results_df

    def plot_results(self, X_test: pd.DataFrame, y_test: pd.Series,
                    test_dates: pd.Series, output_dir: str = '.') -> None:
        """Visualize forecast results."""
        X_scaled = self.scaler.transform(X_test)
        y_pred = self.best_model.predict(X_scaled)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Forecast vs Actual (first week)
        n_plot = min(168, len(y_test))  # One week
        axes[0, 0].plot(range(n_plot), y_test.values[:n_plot], label='Actual', linewidth=1.5)
        axes[0, 0].plot(range(n_plot), y_pred[:n_plot], label='Predicted', linewidth=1.5, alpha=0.8)
        axes[0, 0].set_title('AQI Forecast vs Actual (1 week)')
        axes[0, 0].legend()

        # Scatter plot
        axes[0, 1].scatter(y_test, y_pred, alpha=0.3, s=5)
        axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        axes[0, 1].set_title('Predicted vs Actual')
        axes[0, 1].set_xlabel('Actual AQI')
        axes[0, 1].set_ylabel('Predicted AQI')

        # Residuals
        residuals = y_test.values - y_pred
        axes[1, 0].hist(residuals, bins=50, color='steelblue', alpha=0.7)
        axes[1, 0].set_title('Residual Distribution')

        # Feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            indices = np.argsort(importance)[-10:]
            axes[1, 1].barh(range(10), importance[indices], color='coral')
            axes[1, 1].set_yticks(range(10))
            axes[1, 1].set_yticklabels([self.feature_names[i] for i in indices])
            axes[1, 1].set_title('Top 10 Features')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/forecast_results.png', dpi=300, bbox_inches='tight')
        print(f"Results saved to {output_dir}/forecast_results.png")
        plt.close()


def main():
    """Main execution."""
    print("=" * 70)
    print("AIR QUALITY INDEX PREDICTION")
    print("=" * 70)

    predictor = AirQualityPredictor()

    # Create data
    df = predictor.create_sample_data()
    print(f"\nDataset: {df.shape}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    # Analysis
    predictor.plot_analysis(df)

    # Prepare and train
    X_train, X_test, y_train, y_test, test_dates = predictor.prepare_data(df)
    print(f"\nTraining: {X_train.shape}, Test: {X_test.shape}")

    predictor.train_models(X_train, y_train)
    results = predictor.evaluate_models(X_test, y_test)

    print(f"\n{results.to_string(index=False)}")

    predictor.plot_results(X_test, y_test, test_dates)

    print("\n" + "=" * 70)
    print(f"Best Model: {results.iloc[0]['Model']}")
    print(f"Best RMSE: {results.iloc[0]['RMSE']:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
