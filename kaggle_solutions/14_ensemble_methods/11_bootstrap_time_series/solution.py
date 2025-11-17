"""
Bootstrap Aggregating for Time Series
======================================

This solution demonstrates bootstrap aggregating techniques specifically
designed for time series forecasting, including moving block bootstrap
and other time-aware resampling methods.

Key Concepts:
- Moving block bootstrap preserves temporal dependencies
- Circular block bootstrap for periodic patterns
- Stationary bootstrap with geometric block lengths
- Bagging for time series forecasting

Author: Kaggle Solutions Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class TimeSeriesBagging:
    """Time series bagging with various bootstrap methods."""

    def __init__(self, block_length=10, random_state=42):
        """Initialize time series bagging."""
        self.block_length = block_length
        self.random_state = random_state
        self.models = {}
        self.results = {}
        np.random.seed(random_state)

    def generate_time_series(self, n_points=1000):
        """Generate synthetic time series data."""
        t = np.arange(n_points)

        # Trend component
        trend = 0.05 * t

        # Seasonal component
        seasonal = 10 * np.sin(2 * np.pi * t / 50) + 5 * np.cos(2 * np.pi * t / 30)

        # Cyclical component
        cyclical = 15 * np.sin(2 * np.pi * t / 200)

        # Noise
        noise = np.random.normal(0, 2, n_points)

        # Combine components
        y = trend + seasonal + cyclical + noise

        # Create features
        df = pd.DataFrame({
            'time': t,
            'value': y
        })

        return df

    def create_features(self, df, lag_features=10):
        """Create lag features for time series."""
        feature_df = df.copy()

        # Lag features
        for i in range(1, lag_features + 1):
            feature_df[f'lag_{i}'] = feature_df['value'].shift(i)

        # Rolling statistics
        for window in [5, 10, 20]:
            feature_df[f'rolling_mean_{window}'] = feature_df['value'].rolling(window).mean()
            feature_df[f'rolling_std_{window}'] = feature_df['value'].rolling(window).std()

        # Time-based features
        feature_df['time_sin'] = np.sin(2 * np.pi * feature_df['time'] / 50)
        feature_df['time_cos'] = np.cos(2 * np.pi * feature_df['time'] / 50)

        # Drop NaN values
        feature_df = feature_df.dropna()

        return feature_df

    def moving_block_bootstrap(self, X, y, n_samples=None):
        """Perform moving block bootstrap."""
        if n_samples is None:
            n_samples = len(X)

        n_blocks = int(np.ceil(n_samples / self.block_length))
        max_start = len(X) - self.block_length

        indices = []
        for _ in range(n_blocks):
            start = np.random.randint(0, max_start + 1)
            block_indices = list(range(start, min(start + self.block_length, len(X))))
            indices.extend(block_indices)

        indices = indices[:n_samples]
        return X.iloc[indices], y.iloc[indices]

    def circular_block_bootstrap(self, X, y, n_samples=None):
        """Perform circular block bootstrap."""
        if n_samples is None:
            n_samples = len(X)

        n_blocks = int(np.ceil(n_samples / self.block_length))
        n_data = len(X)

        indices = []
        for _ in range(n_blocks):
            start = np.random.randint(0, n_data)
            for i in range(self.block_length):
                indices.append((start + i) % n_data)

        indices = indices[:n_samples]
        return X.iloc[indices], y.iloc[indices]

    def stationary_bootstrap(self, X, y, avg_block_length=10):
        """Perform stationary bootstrap with geometric block lengths."""
        n_samples = len(X)
        p = 1.0 / avg_block_length

        indices = []
        while len(indices) < n_samples:
            start = np.random.randint(0, len(X))
            indices.append(start)

            while len(indices) < n_samples and np.random.random() > p:
                start = (start + 1) % len(X)
                indices.append(start)

        indices = indices[:n_samples]
        return X.iloc[indices], y.iloc[indices]

    def train_bagged_models(self, X_train, y_train):
        """Train bagged models with different bootstrap methods."""
        base_estimator = DecisionTreeRegressor(max_depth=10, random_state=self.random_state)

        # Standard bagging (IID bootstrap)
        print("Training standard bagging model...")
        self.models['standard'] = BaggingRegressor(
            estimator=base_estimator,
            n_estimators=50,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.models['standard'].fit(X_train, y_train)

        # Custom bagging with moving block bootstrap
        print("Training moving block bootstrap model...")
        self.models['moving_block'] = self._custom_bagging(
            X_train, y_train, 'moving_block', n_estimators=50
        )

        # Custom bagging with circular block bootstrap
        print("Training circular block bootstrap model...")
        self.models['circular_block'] = self._custom_bagging(
            X_train, y_train, 'circular_block', n_estimators=50
        )

        # Custom bagging with stationary bootstrap
        print("Training stationary bootstrap model...")
        self.models['stationary'] = self._custom_bagging(
            X_train, y_train, 'stationary', n_estimators=50
        )

    def _custom_bagging(self, X_train, y_train, method, n_estimators=50):
        """Custom bagging implementation with different bootstrap methods."""
        estimators = []

        for i in range(n_estimators):
            if method == 'moving_block':
                X_boot, y_boot = self.moving_block_bootstrap(X_train, y_train)
            elif method == 'circular_block':
                X_boot, y_boot = self.circular_block_bootstrap(X_train, y_train)
            elif method == 'stationary':
                X_boot, y_boot = self.stationary_bootstrap(X_train, y_train)

            estimator = DecisionTreeRegressor(
                max_depth=10,
                random_state=self.random_state + i
            )
            estimator.fit(X_boot, y_boot)
            estimators.append(estimator)

        return estimators

    def predict_ensemble(self, estimators, X):
        """Make predictions with custom ensemble."""
        predictions = np.array([est.predict(X) for est in estimators])
        return np.mean(predictions, axis=0)

    def evaluate_models(self, X_test, y_test):
        """Evaluate all models."""
        print("\nEvaluating models...")

        for name, model in self.models.items():
            if isinstance(model, list):
                y_pred = self.predict_ensemble(model, X_test)
            else:
                y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            self.results[name] = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred
            }

            print(f"{name}:")
            print(f"  MSE: {mse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R²: {r2:.4f}")

    def plot_forecasts(self, X_test, y_test, n_points=200):
        """Plot forecasts from all models."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.ravel()

        plot_points = min(n_points, len(y_test))

        for idx, (name, results) in enumerate(self.results.items()):
            y_pred = results['predictions'][:plot_points]
            y_true = y_test.values[:plot_points]

            axes[idx].plot(y_true, label='Actual', linewidth=2)
            axes[idx].plot(y_pred, label='Predicted', linewidth=2, alpha=0.7)
            axes[idx].set_title(f'{name.replace("_", " ").title()}\nR² = {results["r2"]:.4f}')
            axes[idx].set_xlabel('Time')
            axes[idx].set_ylabel('Value')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('time_series_forecasts.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Forecasts plot saved!")

    def plot_residuals(self, y_test):
        """Plot residuals analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.ravel()

        for idx, (name, results) in enumerate(self.results.items()):
            residuals = y_test.values - results['predictions']

            axes[idx].scatter(range(len(residuals)), residuals, alpha=0.5)
            axes[idx].axhline(y=0, color='r', linestyle='--')
            axes[idx].set_title(f'{name.replace("_", " ").title()} - Residuals')
            axes[idx].set_xlabel('Time')
            axes[idx].set_ylabel('Residual')
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('residuals_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Residuals analysis plot saved!")

    def plot_performance_metrics(self):
        """Plot performance metrics comparison."""
        metrics = ['mse', 'mae', 'r2']
        names = list(self.results.keys())

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, metric in enumerate(metrics):
            values = [self.results[name][metric] for name in names]

            axes[idx].bar(names, values, alpha=0.7)
            axes[idx].set_xlabel('Method')
            axes[idx].set_ylabel(metric.upper())
            axes[idx].set_title(f'{metric.upper()} Comparison')
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Performance metrics plot saved!")

    def plot_block_bootstrap_illustration(self, df):
        """Illustrate different bootstrap methods."""
        n_samples = 100
        original_series = df['value'].values[:n_samples]

        fig, axes = plt.subplots(4, 1, figsize=(14, 12))

        # Original
        axes[0].plot(original_series, marker='o', markersize=3)
        axes[0].set_title('Original Time Series')
        axes[0].set_ylabel('Value')
        axes[0].grid(True, alpha=0.3)

        # Moving block bootstrap
        X_temp = df.iloc[:n_samples].drop('value', axis=1)
        y_temp = df.iloc[:n_samples]['value']
        _, y_boot = self.moving_block_bootstrap(X_temp, y_temp)
        axes[1].plot(y_boot.values, marker='o', markersize=3, alpha=0.7)
        axes[1].set_title(f'Moving Block Bootstrap (block_length={self.block_length})')
        axes[1].set_ylabel('Value')
        axes[1].grid(True, alpha=0.3)

        # Circular block bootstrap
        _, y_boot = self.circular_block_bootstrap(X_temp, y_temp)
        axes[2].plot(y_boot.values, marker='o', markersize=3, alpha=0.7)
        axes[2].set_title(f'Circular Block Bootstrap (block_length={self.block_length})')
        axes[2].set_ylabel('Value')
        axes[2].grid(True, alpha=0.3)

        # Stationary bootstrap
        _, y_boot = self.stationary_bootstrap(X_temp, y_temp)
        axes[3].plot(y_boot.values, marker='o', markersize=3, alpha=0.7)
        axes[3].set_title('Stationary Bootstrap')
        axes[3].set_xlabel('Time')
        axes[3].set_ylabel('Value')
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('bootstrap_methods_illustration.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Bootstrap methods illustration saved!")

    def plot_autocorrelation_comparison(self, df, y_test):
        """Compare autocorrelation preservation."""
        from statsmodels.graphics.tsaplots import plot_acf

        fig, axes = plt.subplots(3, 2, figsize=(14, 12))

        # Original series ACF
        plot_acf(df['value'][:200], lags=40, ax=axes[0, 0])
        axes[0, 0].set_title('Original Series - ACF')

        # Residuals ACF for each method
        for idx, (name, results) in enumerate(list(self.results.items())[:5], 1):
            residuals = y_test.values - results['predictions']
            row = idx // 2
            col = idx % 2

            if row < 3:
                plot_acf(residuals[:200], lags=40, ax=axes[row, col])
                axes[row, col].set_title(f'{name.replace("_", " ").title()} - Residuals ACF')

        plt.tight_layout()
        plt.savefig('autocorrelation_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Autocorrelation comparison plot saved!")

    def analyze_block_length_sensitivity(self, X_train, y_train, X_test, y_test):
        """Analyze sensitivity to block length."""
        block_lengths = [5, 10, 15, 20, 30, 50]
        moving_block_scores = []
        circular_block_scores = []

        print("\nAnalyzing block length sensitivity...")

        for bl in block_lengths:
            self.block_length = bl

            # Moving block
            estimators = self._custom_bagging(X_train, y_train, 'moving_block', n_estimators=30)
            y_pred = self.predict_ensemble(estimators, X_test)
            moving_block_scores.append(r2_score(y_test, y_pred))

            # Circular block
            estimators = self._custom_bagging(X_train, y_train, 'circular_block', n_estimators=30)
            y_pred = self.predict_ensemble(estimators, X_test)
            circular_block_scores.append(r2_score(y_test, y_pred))

        # Reset to default
        self.block_length = 10

        plt.figure(figsize=(10, 6))
        plt.plot(block_lengths, moving_block_scores, marker='o', label='Moving Block', linewidth=2)
        plt.plot(block_lengths, circular_block_scores, marker='s', label='Circular Block', linewidth=2)
        plt.xlabel('Block Length')
        plt.ylabel('R² Score')
        plt.title('Block Length Sensitivity Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('block_length_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Block length sensitivity plot saved!")


def main():
    """Main execution function."""
    print("=" * 80)
    print("Bootstrap Aggregating for Time Series")
    print("=" * 80)

    # Initialize
    ts_bagging = TimeSeriesBagging(block_length=10, random_state=42)

    # Generate data
    print("\nGenerating time series data...")
    df = ts_bagging.generate_time_series(n_points=1000)

    # Create features
    print("Creating time series features...")
    df_features = ts_bagging.create_features(df, lag_features=10)

    # Split data
    train_size = int(0.8 * len(df_features))
    train_df = df_features.iloc[:train_size]
    test_df = df_features.iloc[train_size:]

    X_train = train_df.drop('value', axis=1)
    y_train = train_df['value']
    X_test = test_df.drop('value', axis=1)
    y_test = test_df['value']

    print(f"Training size: {len(X_train)}")
    print(f"Testing size: {len(X_test)}")

    # Train models
    ts_bagging.train_bagged_models(X_train, y_train)

    # Evaluate
    ts_bagging.evaluate_models(X_test, y_test)

    # Visualizations
    ts_bagging.plot_forecasts(X_test, y_test)
    ts_bagging.plot_residuals(y_test)
    ts_bagging.plot_performance_metrics()
    ts_bagging.plot_block_bootstrap_illustration(df_features)
    ts_bagging.plot_autocorrelation_comparison(df, y_test)
    ts_bagging.analyze_block_length_sensitivity(X_train, y_train, X_test, y_test)

    print("\n" + "=" * 80)
    print("Analysis complete! All visualizations saved.")
    print("=" * 80)


if __name__ == "__main__":
    main()
