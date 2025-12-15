"""
Weather Prediction

Predict future weather conditions based on historical meteorological data
for agriculture planning, energy management and daily life decisions.

Dataset: https://www.kaggle.com/datasets/ananthr1/weather-prediction
Difficulty: ⭐⭐⭐ Advanced Level
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class WeatherPredictor:
    """Weather Prediction Model."""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scaler = StandardScaler()
        self.best_model = None

    def create_sample_data(self, n_days: int = 1095) -> pd.DataFrame:
        """Create synthetic weather dataset (3 years)."""
        np.random.seed(42)

        dates = pd.date_range(start='2021-01-01', periods=n_days, freq='D')

        data = []
        for i, date in enumerate(dates):
            day_of_year = date.dayofyear

            # Temperature: seasonal pattern + noise
            seasonal_temp = 15 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            temp_noise = np.random.normal(0, 3)
            temperature = seasonal_temp + temp_noise

            # Previous day influence
            if i > 0:
                temperature = 0.3 * data[i-1]['temperature'] + 0.7 * temperature

            temperature = np.clip(temperature, -10, 40)

            # Humidity: inverse relationship with temperature + noise
            base_humidity = 70 - 0.5 * (temperature - 15)
            humidity = base_humidity + np.random.normal(0, 10)
            humidity = np.clip(humidity, 20, 100)

            # Pressure: seasonal + random variations
            pressure = 1013 + 10 * np.sin(2 * np.pi * day_of_year / 365)
            pressure += np.random.normal(0, 8)
            pressure = np.clip(pressure, 980, 1040)

            # Wind speed
            wind_speed = 10 + np.random.exponential(5)
            if np.random.random() > 0.9:  # Occasional high winds
                wind_speed += np.random.uniform(10, 30)
            wind_speed = np.clip(wind_speed, 0, 80)

            # Precipitation (depends on humidity and pressure)
            precip_prob = (humidity / 100) * (1 - (pressure - 990) / 50)
            precip_prob = np.clip(precip_prob, 0, 1)

            if np.random.random() < precip_prob:
                precipitation = np.random.exponential(5)
            else:
                precipitation = 0
            precipitation = np.clip(precipitation, 0, 50)

            # Weather type
            if precipitation > 10:
                if temperature < 2:
                    weather_type = 'Snowy'
                elif wind_speed > 40:
                    weather_type = 'Stormy'
                else:
                    weather_type = 'Rainy'
            elif precipitation > 0:
                weather_type = 'Rainy'
            elif humidity > 70:
                weather_type = 'Cloudy'
            else:
                weather_type = 'Sunny'

            data.append({
                'date': date,
                'temperature': round(temperature, 1),
                'humidity': round(humidity, 1),
                'pressure': round(pressure, 1),
                'wind_speed': round(wind_speed, 1),
                'precipitation': round(precipitation, 1),
                'weather_type': weather_type
            })

        return pd.DataFrame(data)

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time series features."""
        df = df.copy()

        # Date features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_year'] = df['date'].dt.dayofyear
        df['day_of_week'] = df['date'].dt.dayofweek
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)

        # Season
        df['season'] = df['month'].apply(lambda x: (x % 12 + 3) // 3)

        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

        # Lag features (for temperature prediction)
        for lag in [1, 2, 3, 7, 14, 30]:
            df[f'temp_lag_{lag}'] = df['temperature'].shift(lag)
            df[f'humidity_lag_{lag}'] = df['humidity'].shift(lag)
            df[f'pressure_lag_{lag}'] = df['pressure'].shift(lag)

        # Rolling statistics
        for window in [3, 7, 14, 30]:
            df[f'temp_rolling_mean_{window}'] = df['temperature'].rolling(window).mean()
            df[f'temp_rolling_std_{window}'] = df['temperature'].rolling(window).std()
            df[f'humidity_rolling_mean_{window}'] = df['humidity'].rolling(window).mean()
            df[f'pressure_rolling_mean_{window}'] = df['pressure'].rolling(window).mean()

        # Pressure change (weather fronts indicator)
        df['pressure_change_1d'] = df['pressure'].diff(1)
        df['pressure_change_3d'] = df['pressure'].diff(3)

        # Drop rows with NaN from lag features
        df = df.dropna()

        return df

    def analyze_data(self, df: pd.DataFrame, output_dir: str = '.') -> None:
        """Perform exploratory data analysis."""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Weather Data Analysis', fontsize=16)

        # Temperature time series
        axes[0, 0].plot(df['date'], df['temperature'], 'b-', alpha=0.7, linewidth=0.5)
        axes[0, 0].set_title('Temperature Time Series')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Temperature (°C)')

        # Monthly temperature pattern
        monthly_temp = df.groupby('month')['temperature'].agg(['mean', 'std'])
        axes[0, 1].errorbar(monthly_temp.index, monthly_temp['mean'],
                           yerr=monthly_temp['std'], fmt='o-', capsize=5)
        axes[0, 1].set_title('Monthly Temperature Pattern')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Temperature (°C)')
        axes[0, 1].set_xticks(range(1, 13))

        # Weather type distribution
        df['weather_type'].value_counts().plot(kind='bar', ax=axes[0, 2], color='steelblue')
        axes[0, 2].set_title('Weather Type Distribution')
        axes[0, 2].tick_params(axis='x', rotation=45)

        # Temperature vs Humidity
        axes[1, 0].scatter(df['temperature'], df['humidity'], alpha=0.3, s=10)
        axes[1, 0].set_title('Temperature vs Humidity')
        axes[1, 0].set_xlabel('Temperature (°C)')
        axes[1, 0].set_ylabel('Humidity (%)')

        # Pressure distribution
        axes[1, 1].hist(df['pressure'], bins=50, edgecolor='black', alpha=0.7)
        axes[1, 1].set_title('Pressure Distribution')
        axes[1, 1].set_xlabel('Pressure (hPa)')
        axes[1, 1].set_ylabel('Count')

        # Precipitation by month
        monthly_precip = df.groupby('month')['precipitation'].sum()
        monthly_precip.plot(kind='bar', ax=axes[1, 2], color='steelblue')
        axes[1, 2].set_title('Total Precipitation by Month')
        axes[1, 2].set_xlabel('Month')
        axes[1, 2].set_ylabel('Precipitation (mm)')

        # Seasonal temperature boxplot
        season_names = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
        df['season_name'] = df['season'].map(season_names)
        df.boxplot(column='temperature', by='season_name', ax=axes[2, 0])
        axes[2, 0].set_title('Temperature by Season')
        plt.suptitle('')

        # Correlation heatmap
        numeric_cols = ['temperature', 'humidity', 'pressure', 'wind_speed', 'precipitation']
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[2, 1], center=0)
        axes[2, 1].set_title('Feature Correlations')

        # Wind speed vs Weather type
        df.boxplot(column='wind_speed', by='weather_type', ax=axes[2, 2])
        axes[2, 2].set_title('Wind Speed by Weather Type')
        axes[2, 2].tick_params(axis='x', rotation=45)
        plt.suptitle('')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/weather_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Analysis saved to {output_dir}/weather_analysis.png")
        plt.close()

    def prepare_features(self, df: pd.DataFrame, target: str = 'temperature',
                        fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for modeling."""
        feature_cols = [col for col in df.columns if col not in
                       ['date', 'weather_type', 'temperature', 'year', 'season_name']]

        X = df[feature_cols].values

        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        y = df[target].values

        return X, y

    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train regression models."""
        print("\nTraining models...")

        self.models['Linear Regression'] = LinearRegression()
        self.models['Linear Regression'].fit(X_train, y_train)

        self.models['Ridge Regression'] = Ridge(alpha=1.0)
        self.models['Ridge Regression'].fit(X_train, y_train)

        self.models['Random Forest'] = RandomForestRegressor(
            n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
        )
        self.models['Random Forest'].fit(X_train, y_train)

        self.models['Gradient Boosting'] = GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )
        self.models['Gradient Boosting'].fit(X_train, y_train)

        print(f"Trained {len(self.models)} models!")

    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """Evaluate all models."""
        results = []

        for name, model in self.models.items():
            y_pred = model.predict(X_test)

            results.append({
                'Model': name,
                'MAE': mean_absolute_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'R²': r2_score(y_test, y_pred)
            })

        results_df = pd.DataFrame(results).sort_values('MAE')
        self.best_model = self.models[results_df.iloc[0]['Model']]

        return results_df

    def plot_results(self, results: pd.DataFrame, df_test: pd.DataFrame,
                    X_test: np.ndarray, y_test: np.ndarray, output_dir: str = '.') -> None:
        """Visualize results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Weather Prediction Results', fontsize=16)

        # Model comparison
        results.set_index('Model')[['MAE', 'RMSE']].plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Model Error Comparison')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_ylabel('Error (°C)')

        # R² comparison
        results.set_index('Model')['R²'].plot(kind='bar', ax=axes[0, 1], color='steelblue')
        axes[0, 1].set_title('R² Score Comparison')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].set_ylim(0, 1)

        # Predictions vs Actual (time series)
        y_pred = self.best_model.predict(X_test)
        n_show = min(100, len(y_test))
        axes[1, 0].plot(range(n_show), y_test[:n_show], 'b-', label='Actual', linewidth=2)
        axes[1, 0].plot(range(n_show), y_pred[:n_show], 'r--', label='Predicted', linewidth=2)
        axes[1, 0].set_title('Temperature: Actual vs Predicted')
        axes[1, 0].set_xlabel('Days')
        axes[1, 0].set_ylabel('Temperature (°C)')
        axes[1, 0].legend()

        # Scatter plot
        axes[1, 1].scatter(y_test, y_pred, alpha=0.5)
        min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[1, 1].set_xlabel('Actual Temperature (°C)')
        axes[1, 1].set_ylabel('Predicted Temperature (°C)')
        axes[1, 1].set_title('Actual vs Predicted Scatter')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/weather_results.png', dpi=300, bbox_inches='tight')
        print(f"Results saved to {output_dir}/weather_results.png")
        plt.close()


def main():
    """Main execution."""
    print("=" * 70)
    print("WEATHER PREDICTION")
    print("=" * 70)

    predictor = WeatherPredictor()

    # Create data
    print("\nCreating synthetic dataset...")
    df = predictor.create_sample_data(n_days=1095)  # 3 years
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Analysis
    predictor.analyze_data(df)

    # Feature engineering
    df_fe = predictor.feature_engineering(df)
    print(f"\nFeatures created: {df_fe.shape[1]} columns")

    # Time series split (use last 20% for testing)
    train_size = int(len(df_fe) * 0.8)
    df_train = df_fe.iloc[:train_size]
    df_test = df_fe.iloc[train_size:]

    # Prepare features
    X_train, y_train = predictor.prepare_features(df_train, fit=True)
    X_test, y_test = predictor.prepare_features(df_test, fit=False)

    print(f"Train: {X_train.shape[0]} days, Test: {X_test.shape[0]} days")

    # Train and evaluate
    predictor.train_models(X_train, y_train)
    results = predictor.evaluate_models(X_test, y_test)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(results.to_string(index=False))

    # Visualize
    predictor.plot_results(results, df_test, X_test, y_test)

    print("\n" + "=" * 70)
    best = results.iloc[0]
    print(f"Best Model: {best['Model']}")
    print(f"MAE: {best['MAE']:.2f}°C")
    print(f"R²: {best['R²']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
