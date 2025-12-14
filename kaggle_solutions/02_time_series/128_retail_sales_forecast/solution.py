"""
Retail Sales Forecasting - Time Series Analysis

This module implements retail sales forecasting using various time series
and machine learning methods.

Dataset: https://www.kaggle.com/datasets/mohammadtalib786/retail-sales-dataset
Difficulty: ⭐⭐ Intermediate Level
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


class RetailSalesForecaster:
    """Retail Sales Forecasting using Time Series methods."""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_names = []

    def create_sample_data(self) -> pd.DataFrame:
        """Create realistic retail sales time series data."""
        np.random.seed(42)

        # Generate 3 years of daily data
        dates = pd.date_range(start='2021-01-01', end='2023-12-31', freq='D')
        n_days = len(dates)

        # Base sales with trend
        trend = np.linspace(100, 150, n_days)

        # Yearly seasonality
        yearly_seasonal = 20 * np.sin(2 * np.pi * np.arange(n_days) / 365)

        # Weekly seasonality (weekends have higher sales)
        weekly_seasonal = np.array([10 if d.weekday() >= 5 else 0 for d in dates])

        # Monthly effects (holiday seasons)
        monthly_effect = np.array([
            30 if d.month == 12 else  # December boost
            15 if d.month == 11 else  # November boost
            -10 if d.month in [1, 2] else  # Post-holiday dip
            0 for d in dates
        ])

        # Random noise
        noise = np.random.normal(0, 10, n_days)

        # Combine all components
        sales = trend + yearly_seasonal + weekly_seasonal + monthly_effect + noise
        sales = np.maximum(sales, 10)  # Ensure positive sales

        return pd.DataFrame({
            'date': dates,
            'sales': sales.round(0).astype(int)
        })

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features from date column."""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        # Basic time features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['dayofyear'] = df['date'].dt.dayofyear
        df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
        df['quarter'] = df['date'].dt.quarter

        # Cyclical encoding for time features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

        # Weekend flag
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

        # Month start/end flags
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

        # Holiday season flags
        df['is_holiday_season'] = df['month'].isin([11, 12]).astype(int)

        return df

    def create_lag_features(self, df: pd.DataFrame, lags: list = [1, 7, 14, 30]) -> pd.DataFrame:
        """Create lag features for time series."""
        df = df.copy()

        for lag in lags:
            df[f'sales_lag_{lag}'] = df['sales'].shift(lag)

        return df

    def create_rolling_features(self, df: pd.DataFrame,
                               windows: list = [7, 14, 30]) -> pd.DataFrame:
        """Create rolling window statistics."""
        df = df.copy()

        for window in windows:
            df[f'sales_rolling_mean_{window}'] = df['sales'].rolling(window=window).mean()
            df[f'sales_rolling_std_{window}'] = df['sales'].rolling(window=window).std()
            df[f'sales_rolling_min_{window}'] = df['sales'].rolling(window=window).min()
            df[f'sales_rolling_max_{window}'] = df['sales'].rolling(window=window).max()

        return df

    def plot_time_series_analysis(self, df: pd.DataFrame, output_dir: str = '.') -> None:
        """Generate time series analysis visualizations."""
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        fig.suptitle('Retail Sales Time Series Analysis', fontsize=16)

        # Overall time series
        axes[0, 0].plot(df['date'], df['sales'], linewidth=0.8)
        axes[0, 0].set_title('Daily Sales Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Sales')

        # Monthly aggregation
        monthly = df.groupby(df['date'].dt.to_period('M'))['sales'].mean()
        monthly.plot(ax=axes[0, 1], marker='o')
        axes[0, 1].set_title('Monthly Average Sales')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Average Sales')

        # Day of week pattern
        dow_sales = df.groupby('dayofweek')['sales'].mean()
        dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[1, 0].bar(dow_labels, dow_sales.values, color='steelblue')
        axes[1, 0].set_title('Average Sales by Day of Week')
        axes[1, 0].set_ylabel('Average Sales')

        # Monthly pattern
        month_sales = df.groupby('month')['sales'].mean()
        axes[1, 1].bar(range(1, 13), month_sales.values, color='coral')
        axes[1, 1].set_title('Average Sales by Month')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Average Sales')
        axes[1, 1].set_xticks(range(1, 13))

        # Sales distribution
        df['sales'].hist(bins=50, ax=axes[2, 0], color='green', alpha=0.7)
        axes[2, 0].set_title('Sales Distribution')
        axes[2, 0].set_xlabel('Sales')

        # Year-over-year comparison
        yearly = df.groupby(['year', 'month'])['sales'].mean().unstack(level=0)
        yearly.plot(ax=axes[2, 1], marker='o')
        axes[2, 1].set_title('Year-over-Year Monthly Comparison')
        axes[2, 1].set_xlabel('Month')
        axes[2, 1].set_ylabel('Average Sales')
        axes[2, 1].legend(title='Year')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/sales_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Analysis saved to {output_dir}/sales_analysis.png")
        plt.close()

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare train/test data."""
        df = self.create_time_features(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)

        # Drop rows with NaN from lag/rolling features
        df = df.dropna()

        # Feature columns
        feature_cols = [col for col in df.columns if col not in ['date', 'sales']]
        self.feature_names = feature_cols

        # Time-based split (last 30 days for test)
        split_date = df['date'].max() - timedelta(days=30)
        train = df[df['date'] <= split_date]
        test = df[df['date'] > split_date]

        X_train = train[feature_cols]
        y_train = train['sales']
        X_test = test[feature_cols]
        y_test = test['sales']

        return X_train, X_test, y_train, y_test

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train multiple forecasting models."""
        X_train_scaled = self.scaler.fit_transform(X_train)

        print("\nTraining models...")

        # Linear Regression
        self.models['Linear Regression'] = LinearRegression()
        self.models['Linear Regression'].fit(X_train_scaled, y_train)

        # Random Forest
        self.models['Random Forest'] = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        self.models['Random Forest'].fit(X_train_scaled, y_train)

        # Gradient Boosting
        self.models['Gradient Boosting'] = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
        )
        self.models['Gradient Boosting'].fit(X_train_scaled, y_train)

        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            )
            self.models['XGBoost'].fit(X_train_scaled, y_train)

        if LIGHTGBM_AVAILABLE:
            self.models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, verbose=-1
            )
            self.models['LightGBM'].fit(X_train_scaled, y_train)

        print(f"Trained {len(self.models)} models!")

    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """Evaluate all models."""
        X_test_scaled = self.scaler.transform(X_test)
        results = []

        print("\n=== Model Evaluation ===")

        for name, model in self.models.items():
            y_pred = model.predict(X_test_scaled)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            r2 = r2_score(y_test, y_pred)

            results.append({
                'Model': name,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'R2': r2
            })

            print(f"{name}: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%, R2={r2:.4f}")

        results_df = pd.DataFrame(results).sort_values('RMSE')
        self.best_model = self.models[results_df.iloc[0]['Model']]
        return results_df

    def plot_forecast_results(self, df: pd.DataFrame, X_test: pd.DataFrame,
                             y_test: pd.Series, output_dir: str = '.') -> None:
        """Visualize forecast results."""
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.best_model.predict(X_test_scaled)

        test_dates = df[df['date'] > df['date'].max() - timedelta(days=30)]['date'].values[-len(y_test):]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Actual vs Predicted
        axes[0, 0].plot(test_dates, y_test.values, label='Actual', linewidth=2)
        axes[0, 0].plot(test_dates, y_pred, label='Predicted', linewidth=2, linestyle='--')
        axes[0, 0].set_title('Actual vs Predicted Sales')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Residuals
        residuals = y_test.values - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_title('Residuals Plot')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Residuals')

        # Residual distribution
        axes[1, 0].hist(residuals, bins=30, color='steelblue', alpha=0.7)
        axes[1, 0].set_title('Residual Distribution')
        axes[1, 0].set_xlabel('Residual')

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
        print(f"Forecast results saved to {output_dir}/forecast_results.png")
        plt.close()


def main():
    """Main execution."""
    print("=" * 70)
    print("RETAIL SALES FORECASTING")
    print("=" * 70)

    forecaster = RetailSalesForecaster()

    # Create sample data
    df = forecaster.create_sample_data()
    print(f"\nDataset: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Plot analysis
    df_with_features = forecaster.create_time_features(df)
    forecaster.plot_time_series_analysis(df_with_features)

    # Prepare data
    X_train, X_test, y_train, y_test = forecaster.prepare_data(df)
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Train and evaluate
    forecaster.train_models(X_train, y_train)
    results = forecaster.evaluate_models(X_test, y_test)

    print(f"\n{results.to_string(index=False)}")

    # Plot results
    forecaster.plot_forecast_results(df, X_test, y_test)

    print("\n" + "=" * 70)
    print(f"Best Model: {results.iloc[0]['Model']}")
    print(f"Best RMSE: {results.iloc[0]['RMSE']:.2f}")
    print(f"Best MAPE: {results.iloc[0]['MAPE']:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
