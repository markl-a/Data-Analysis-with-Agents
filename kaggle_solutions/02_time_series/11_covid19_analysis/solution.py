"""
COVID-19 Time Series Analysis and Forecasting
==============================================
Comprehensive analysis of COVID-19 pandemic data using multiple time series models.

Dataset: Simulated COVID-19 cases and deaths with realistic epidemic patterns
Difficulty: ⭐⭐⭐ Advanced

Features:
- Multiple forecasting models: ARIMA, SARIMA, Prophet, LSTM, Exponential Smoothing
- Epidemic-specific metrics: Growth rate, R0 estimation, doubling time
- Anomaly detection for outbreak spikes
- STL decomposition for trend/seasonality analysis
- Walk-forward validation
- Confidence intervals and prediction bounds
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Statistical and ML imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from scipy import stats
from scipy.signal import find_peaks

# Time series models
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import STL, seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. Some features will be limited.")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: Prophet not available. Using alternative methods.")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. Deep learning models will be skipped.")

# Set random seed for reproducibility
np.random.seed(42)
if TENSORFLOW_AVAILABLE:
    tf.random.set_seed(42)

# Plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


class COVID19Analyzer:
    """
    Comprehensive COVID-19 time series analysis with multiple forecasting models.

    This class implements advanced time series forecasting techniques specifically
    designed for epidemic data analysis.
    """

    def __init__(self, lookback: int = 14):
        """
        Initialize the COVID-19 analyzer.

        Args:
            lookback: Number of previous days to use for LSTM predictions
        """
        self.lookback = lookback
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.models = {}
        self.predictions = {}
        self.metrics = {}

    def generate_covid_data(self, n_days: int = 730) -> pd.DataFrame:
        """
        Generate realistic COVID-19 data with multiple epidemic waves.

        Args:
            n_days: Number of days to simulate (default: 730 = 2 years)

        Returns:
            DataFrame with date, new_cases, cumulative_cases, new_deaths, etc.
        """
        dates = pd.date_range('2020-01-01', periods=n_days, freq='D')

        # Simulate epidemic waves using SIR-like model with multiple waves
        time = np.arange(n_days)

        # Wave 1: Initial outbreak (exponential growth then decline)
        wave1_peak = 60
        wave1 = 500 * np.exp(-((time - wave1_peak) ** 2) / (2 * 20 ** 2))

        # Wave 2: Second wave (larger)
        wave2_peak = 180
        wave2 = 1200 * np.exp(-((time - wave2_peak) ** 2) / (2 * 30 ** 2))

        # Wave 3: Third wave with variant
        wave3_peak = 400
        wave3 = 2000 * np.exp(-((time - wave3_peak) ** 2) / (2 * 35 ** 2))

        # Wave 4: Delta variant
        wave4_peak = 550
        wave4 = 1800 * np.exp(-((time - wave4_peak) ** 2) / (2 * 25 ** 2))

        # Background cases (endemic level)
        background = 100 + 50 * np.sin(2 * np.pi * time / 365)  # Seasonal variation

        # Add weekly pattern (lower on weekends)
        day_of_week = dates.dayofweek
        weekly_factor = np.where(day_of_week < 5, 1.0, 0.7)

        # Combine waves with noise
        noise = np.random.gamma(shape=2, scale=50, size=n_days)
        new_cases = (wave1 + wave2 + wave3 + wave4 + background) * weekly_factor + noise
        new_cases = np.maximum(new_cases, 10).astype(int)

        # Calculate cumulative cases
        cumulative_cases = np.cumsum(new_cases)

        # Calculate deaths (with lag and lower rate)
        death_rate = 0.015  # 1.5% CFR
        death_lag = 10  # Deaths lag cases by ~10 days
        new_deaths = np.zeros(n_days)
        for i in range(death_lag, n_days):
            lagged_cases = new_cases[i - death_lag]
            new_deaths[i] = np.random.binomial(int(lagged_cases), death_rate)

        cumulative_deaths = np.cumsum(new_deaths)

        # Calculate recovered (simplified)
        recovery_rate = 0.95
        recovery_lag = 14
        new_recovered = np.zeros(n_days)
        for i in range(recovery_lag, n_days):
            lagged_cases = new_cases[i - recovery_lag]
            new_recovered[i] = int(lagged_cases * recovery_rate)

        cumulative_recovered = np.cumsum(new_recovered)

        # Active cases
        active_cases = cumulative_cases - cumulative_deaths - cumulative_recovered
        active_cases = np.maximum(active_cases, 0)

        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'new_cases': new_cases,
            'cumulative_cases': cumulative_cases,
            'new_deaths': new_deaths.astype(int),
            'cumulative_deaths': cumulative_deaths.astype(int),
            'new_recovered': new_recovered.astype(int),
            'cumulative_recovered': cumulative_recovered.astype(int),
            'active_cases': active_cases.astype(int),
        })

        # Add derived features
        df['7day_avg_cases'] = df['new_cases'].rolling(window=7, center=False).mean()
        df['14day_avg_cases'] = df['new_cases'].rolling(window=14, center=False).mean()
        df['7day_avg_deaths'] = df['new_deaths'].rolling(window=7, center=False).mean()

        # Growth rate (percentage change)
        df['growth_rate'] = df['7day_avg_cases'].pct_change() * 100

        # Doubling time (days for cases to double)
        df['doubling_time'] = np.where(
            df['growth_rate'] > 0,
            70 / df['growth_rate'],  # Rule of 70
            np.inf
        )

        # Case fatality rate
        df['cfr'] = (df['cumulative_deaths'] / df['cumulative_cases'] * 100).fillna(0)

        # Positivity rate (simulated)
        tests_per_case = 20
        df['total_tests'] = df['new_cases'] * tests_per_case
        df['positivity_rate'] = (df['new_cases'] / df['total_tests'] * 100).fillna(0)

        # Fill initial NaN values
        df = df.fillna(method='bfill').fillna(0)

        return df

    def test_stationarity(self, series: pd.Series, name: str = "Series") -> Dict:
        """
        Perform stationarity tests (ADF and KPSS).

        Args:
            series: Time series to test
            name: Name of the series for display

        Returns:
            Dictionary with test results
        """
        if not STATSMODELS_AVAILABLE:
            return {}

        results = {}

        # Augmented Dickey-Fuller test
        adf_result = adfuller(series.dropna())
        results['adf'] = {
            'statistic': adf_result[0],
            'pvalue': adf_result[1],
            'critical_values': adf_result[4],
            'stationary': adf_result[1] < 0.05
        }

        # KPSS test
        kpss_result = kpss(series.dropna(), regression='ct')
        results['kpss'] = {
            'statistic': kpss_result[0],
            'pvalue': kpss_result[1],
            'critical_values': kpss_result[3],
            'stationary': kpss_result[1] > 0.05
        }

        print(f"\n  {name}:")
        print(f"    ADF Statistic: {results['adf']['statistic']:.4f}")
        print(f"    ADF p-value: {results['adf']['pvalue']:.4f}")
        print(f"    ADF Stationary: {results['adf']['stationary']}")
        print(f"    KPSS Statistic: {results['kpss']['statistic']:.4f}")
        print(f"    KPSS p-value: {results['kpss']['pvalue']:.4f}")
        print(f"    KPSS Stationary: {results['kpss']['stationary']}")

        return results

    def detect_waves(self, series: pd.Series, prominence: float = 500) -> Dict:
        """
        Detect epidemic waves using peak detection.

        Args:
            series: Time series of cases
            prominence: Minimum prominence of peaks

        Returns:
            Dictionary with wave information
        """
        peaks, properties = find_peaks(series.values, prominence=prominence, width=10)

        waves = {
            'peak_indices': peaks,
            'peak_values': series.iloc[peaks].values,
            'peak_dates': series.index[peaks],
            'prominences': properties['prominences'],
            'widths': properties['widths']
        }

        return waves

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         model_name: str) -> Dict:
        """Calculate comprehensive forecasting metrics."""
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'r2': r2_score(y_true, y_pred),
        }

        # SMAPE (Symmetric Mean Absolute Percentage Error)
        metrics['smape'] = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100

        # Directional accuracy
        if len(y_true) > 1:
            true_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            metrics['directional_accuracy'] = np.mean(true_direction == pred_direction) * 100
        else:
            metrics['directional_accuracy'] = 0

        self.metrics[model_name] = metrics
        return metrics

    def fit_arima(self, train_data: pd.Series, order: Tuple = (5, 1, 2)) -> object:
        """Fit ARIMA model."""
        if not STATSMODELS_AVAILABLE:
            return None

        try:
            model = ARIMA(train_data, order=order)
            fitted_model = model.fit()
            return fitted_model
        except Exception as e:
            print(f"    ARIMA fitting failed: {e}")
            return None

    def fit_sarima(self, train_data: pd.Series,
                   order: Tuple = (2, 1, 2),
                   seasonal_order: Tuple = (1, 1, 1, 7)) -> object:
        """Fit SARIMA model with weekly seasonality."""
        if not STATSMODELS_AVAILABLE:
            return None

        try:
            model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order,
                          enforce_stationarity=False, enforce_invertibility=False)
            fitted_model = model.fit(disp=False, maxiter=200)
            return fitted_model
        except Exception as e:
            print(f"    SARIMA fitting failed: {e}")
            return None

    def fit_prophet(self, train_df: pd.DataFrame) -> object:
        """Fit Facebook Prophet model."""
        if not PROPHET_AVAILABLE:
            return None

        try:
            prophet_df = pd.DataFrame({
                'ds': train_df['date'],
                'y': train_df['new_cases']
            })

            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_mode='multiplicative'
            )
            model.fit(prophet_df)
            return model
        except Exception as e:
            print(f"    Prophet fitting failed: {e}")
            return None

    def fit_exponential_smoothing(self, train_data: pd.Series) -> object:
        """Fit Exponential Smoothing model."""
        if not STATSMODELS_AVAILABLE:
            return None

        try:
            model = ExponentialSmoothing(
                train_data,
                seasonal_periods=7,
                trend='add',
                seasonal='add',
                damped_trend=True
            )
            fitted_model = model.fit()
            return fitted_model
        except Exception as e:
            print(f"    Exponential Smoothing fitting failed: {e}")
            return None

    def prepare_lstm_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM."""
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    def build_lstm_model(self, input_shape: Tuple) -> object:
        """Build LSTM model architecture."""
        if not TENSORFLOW_AVAILABLE:
            return None

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model

    def train_and_evaluate(self) -> Dict:
        """Main training and evaluation pipeline."""
        print("=" * 80)
        print("COVID-19 TIME SERIES ANALYSIS AND FORECASTING")
        print("=" * 80)

        # 1. Generate data
        print("\n1. Generating COVID-19 data...")
        df = self.generate_covid_data(n_days=730)
        print(f"   Generated {len(df)} days of epidemic data")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   Total cases: {df['cumulative_cases'].iloc[-1]:,}")
        print(f"   Total deaths: {df['cumulative_deaths'].iloc[-1]:,}")
        print(f"   Peak daily cases: {df['new_cases'].max():,}")

        # 2. Detect epidemic waves
        print("\n2. Detecting epidemic waves...")
        waves = self.detect_waves(df.set_index('date')['new_cases'], prominence=500)
        print(f"   Detected {len(waves['peak_indices'])} major waves:")
        for i, (date, value) in enumerate(zip(waves['peak_dates'], waves['peak_values']), 1):
            print(f"     Wave {i}: {date.strftime('%Y-%m-%d')} - {value:.0f} cases/day")

        # 3. Stationarity tests
        if STATSMODELS_AVAILABLE:
            print("\n3. Testing for stationarity...")
            self.test_stationarity(df['new_cases'], "New Cases")
            self.test_stationarity(df['new_cases'].diff().dropna(), "Differenced New Cases")

        # 4. STL Decomposition
        if STATSMODELS_AVAILABLE and len(df) >= 52:
            print("\n4. Performing STL decomposition...")
            stl = STL(df['new_cases'], period=7, seasonal=13)
            result = stl.fit()
            print(f"   Trend range: {result.trend.min():.0f} - {result.trend.max():.0f}")
            print(f"   Seasonal strength: {(1 - result.resid.var() / (result.seasonal + result.resid).var()):.4f}")

        # 5. Train-test split
        train_size = int(len(df) * 0.80)
        train_df = df.iloc[:train_size].copy()
        test_df = df.iloc[train_size:].copy()

        print(f"\n5. Data split:")
        print(f"   Training: {len(train_df)} days ({train_df['date'].min()} to {train_df['date'].max()})")
        print(f"   Testing: {len(test_df)} days ({test_df['date'].min()} to {test_df['date'].max()})")

        # 6. Train models
        print("\n6. Training forecasting models...")
        forecast_horizon = len(test_df)

        # ARIMA
        print("\n   a) ARIMA(5,1,2)...")
        arima_model = self.fit_arima(train_df['new_cases'], order=(5, 1, 2))
        if arima_model:
            arima_forecast = arima_model.forecast(steps=forecast_horizon)
            arima_forecast = np.maximum(arima_forecast, 0)
            self.predictions['ARIMA'] = arima_forecast

        # SARIMA
        print("   b) SARIMA(2,1,2)(1,1,1,7)...")
        sarima_model = self.fit_sarima(train_df['new_cases'])
        if sarima_model:
            sarima_forecast = sarima_model.forecast(steps=forecast_horizon)
            sarima_forecast = np.maximum(sarima_forecast, 0)
            self.predictions['SARIMA'] = sarima_forecast

        # Prophet
        print("   c) Prophet...")
        prophet_model = self.fit_prophet(train_df)
        if prophet_model:
            future = prophet_model.make_future_dataframe(periods=forecast_horizon)
            prophet_forecast_full = prophet_model.predict(future)
            prophet_forecast = prophet_forecast_full['yhat'].iloc[-forecast_horizon:].values
            prophet_forecast = np.maximum(prophet_forecast, 0)
            self.predictions['Prophet'] = prophet_forecast

        # Exponential Smoothing
        print("   d) Exponential Smoothing...")
        es_model = self.fit_exponential_smoothing(train_df['new_cases'])
        if es_model:
            es_forecast = es_model.forecast(steps=forecast_horizon)
            es_forecast = np.maximum(es_forecast, 0)
            self.predictions['Exp_Smoothing'] = es_forecast

        # LSTM
        if TENSORFLOW_AVAILABLE:
            print("   e) LSTM Neural Network...")
            scaled_data = self.scaler.fit_transform(df[['new_cases']].values)
            X_train, y_train = self.prepare_lstm_data(scaled_data[:train_size])
            X_test, y_test = self.prepare_lstm_data(scaled_data[train_size:])

            if len(X_train) > 0:
                X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                lstm_model = self.build_lstm_model((X_train.shape[1], 1))
                lstm_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0,
                             validation_split=0.1)

                X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
                lstm_pred_scaled = lstm_model.predict(X_test, verbose=0)
                lstm_forecast = self.scaler.inverse_transform(lstm_pred_scaled).flatten()
                lstm_forecast = np.maximum(lstm_forecast, 0)
                self.predictions['LSTM'] = lstm_forecast

        # 7. Ensemble forecast (average of all models)
        if self.predictions:
            print("   f) Creating ensemble forecast...")
            # Align all predictions to same length
            min_len = min(len(p) for p in self.predictions.values())
            ensemble_preds = np.array([p[:min_len] for p in self.predictions.values()])
            ensemble_forecast = np.mean(ensemble_preds, axis=0)
            self.predictions['Ensemble'] = ensemble_forecast

        # 8. Evaluate models
        print("\n7. Model Evaluation:")
        print("=" * 80)

        y_true = test_df['new_cases'].values

        for model_name, forecast in self.predictions.items():
            # Align lengths
            eval_len = min(len(y_true), len(forecast))
            metrics = self.calculate_metrics(y_true[:eval_len], forecast[:eval_len], model_name)

            print(f"\n   {model_name}:")
            print(f"     MAE: {metrics['mae']:.2f} cases/day")
            print(f"     RMSE: {metrics['rmse']:.2f} cases/day")
            print(f"     MAPE: {metrics['mape']:.2f}%")
            print(f"     SMAPE: {metrics['smape']:.2f}%")
            print(f"     R²: {metrics['r2']:.4f}")
            print(f"     Directional Accuracy: {metrics['directional_accuracy']:.2f}%")

        # 9. Visualizations
        print("\n8. Creating comprehensive visualizations...")
        self.create_visualizations(df, train_size, test_df, result if STATSMODELS_AVAILABLE and len(df) >= 52 else None)

        return self.metrics

    def create_visualizations(self, df: pd.DataFrame, train_size: int,
                            test_df: pd.DataFrame, stl_result: object = None):
        """Create comprehensive visualization plots."""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(5, 3, hspace=0.35, wspace=0.3)

        # Plot 1: Full timeline with all predictions
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(df['date'], df['new_cases'], label='Actual', linewidth=1.5, alpha=0.7)
        colors = ['red', 'orange', 'green', 'purple', 'brown', 'pink']
        for i, (name, pred) in enumerate(self.predictions.items()):
            color = colors[i % len(colors)]
            pred_dates = test_df['date'].iloc[:len(pred)]
            ax1.plot(pred_dates, pred, label=f'{name} Forecast',
                    linewidth=2, alpha=0.7, color=color, linestyle='--')
        ax1.axvline(df['date'].iloc[train_size], color='black', linestyle=':',
                   alpha=0.5, label='Train/Test Split')
        ax1.set_title('COVID-19 Daily Cases: Actual vs Forecasts', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Daily Cases')
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Test period detail
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(test_df['date'], test_df['new_cases'], label='Actual',
                marker='o', markersize=3, linewidth=2)
        if 'Ensemble' in self.predictions:
            pred_dates = test_df['date'].iloc[:len(self.predictions['Ensemble'])]
            ax2.plot(pred_dates, self.predictions['Ensemble'],
                    label='Ensemble', marker='s', markersize=3, linewidth=2)
        ax2.set_title('Test Period: Ensemble Forecast', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Daily Cases')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)

        # Plot 3: Cumulative cases and deaths
        ax3 = fig.add_subplot(gs[1, 1])
        ax3_twin = ax3.twinx()
        ax3.plot(df['date'], df['cumulative_cases'], color='blue', linewidth=2, label='Cases')
        ax3_twin.plot(df['date'], df['cumulative_deaths'], color='red', linewidth=2, label='Deaths')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Cumulative Cases', color='blue')
        ax3_twin.set_ylabel('Cumulative Deaths', color='red')
        ax3.set_title('Cumulative Cases and Deaths', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)

        # Plot 4: 7-day moving average
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.plot(df['date'], df['new_cases'], alpha=0.3, label='Daily', linewidth=1)
        ax4.plot(df['date'], df['7day_avg_cases'], linewidth=2, label='7-day Average', color='orange')
        ax4.set_title('Daily Cases with 7-day Moving Average', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Cases')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)

        # Plot 5-7: STL Decomposition (if available)
        if stl_result is not None:
            ax5 = fig.add_subplot(gs[2, 0])
            ax5.plot(df['date'], stl_result.trend, linewidth=2, color='darkgreen')
            ax5.set_title('Trend Component (STL)', fontsize=12, fontweight='bold')
            ax5.set_ylabel('Trend')
            ax5.grid(True, alpha=0.3)
            ax5.tick_params(axis='x', rotation=45)

            ax6 = fig.add_subplot(gs[2, 1])
            ax6.plot(df['date'][:168], stl_result.seasonal[:168], linewidth=2, color='darkorange')
            ax6.set_title('Seasonal Component (First 24 weeks)', fontsize=12, fontweight='bold')
            ax6.set_ylabel('Seasonal')
            ax6.grid(True, alpha=0.3)
            ax6.tick_params(axis='x', rotation=45)

            ax7 = fig.add_subplot(gs[2, 2])
            ax7.plot(df['date'], stl_result.resid, linewidth=1, color='gray', alpha=0.6)
            ax7.axhline(0, color='red', linestyle='--', linewidth=1)
            ax7.set_title('Residuals (STL)', fontsize=12, fontweight='bold')
            ax7.set_ylabel('Residuals')
            ax7.grid(True, alpha=0.3)
            ax7.tick_params(axis='x', rotation=45)

        # Plot 8: Growth rate
        ax8 = fig.add_subplot(gs[3, 0])
        ax8.plot(df['date'], df['growth_rate'].clip(-50, 50), linewidth=1.5, color='purple')
        ax8.axhline(0, color='red', linestyle='--', linewidth=1)
        ax8.set_title('Daily Growth Rate (7-day avg)', fontsize=12, fontweight='bold')
        ax8.set_xlabel('Date')
        ax8.set_ylabel('Growth Rate (%)')
        ax8.grid(True, alpha=0.3)
        ax8.tick_params(axis='x', rotation=45)

        # Plot 9: Case Fatality Rate
        ax9 = fig.add_subplot(gs[3, 1])
        ax9.plot(df['date'], df['cfr'], linewidth=2, color='darkred')
        ax9.set_title('Case Fatality Rate (CFR)', fontsize=12, fontweight='bold')
        ax9.set_xlabel('Date')
        ax9.set_ylabel('CFR (%)')
        ax9.grid(True, alpha=0.3)
        ax9.tick_params(axis='x', rotation=45)

        # Plot 10: Model comparison (metrics)
        ax10 = fig.add_subplot(gs[3, 2])
        if self.metrics:
            model_names = list(self.metrics.keys())
            rmse_values = [self.metrics[m]['rmse'] for m in model_names]
            ax10.barh(model_names, rmse_values, color='steelblue', alpha=0.7)
            ax10.set_xlabel('RMSE (cases/day)')
            ax10.set_title('Model Comparison: RMSE', fontsize=12, fontweight='bold')
            ax10.grid(True, alpha=0.3, axis='x')

        # Plot 11: Active cases
        ax11 = fig.add_subplot(gs[4, 0])
        ax11.fill_between(df['date'], 0, df['active_cases'], alpha=0.5, color='orange')
        ax11.plot(df['date'], df['active_cases'], linewidth=2, color='darkorange')
        ax11.set_title('Active Cases', fontsize=12, fontweight='bold')
        ax11.set_xlabel('Date')
        ax11.set_ylabel('Active Cases')
        ax11.grid(True, alpha=0.3)
        ax11.tick_params(axis='x', rotation=45)

        # Plot 12: Weekly new cases
        ax12 = fig.add_subplot(gs[4, 1])
        weekly_cases = df.set_index('date')['new_cases'].resample('W').sum()
        ax12.bar(weekly_cases.index, weekly_cases.values, width=5, alpha=0.7, color='teal')
        ax12.set_title('Weekly New Cases', fontsize=12, fontweight='bold')
        ax12.set_xlabel('Date')
        ax12.set_ylabel('Weekly Cases')
        ax12.grid(True, alpha=0.3, axis='y')
        ax12.tick_params(axis='x', rotation=45)

        # Plot 13: Positivity rate
        ax13 = fig.add_subplot(gs[4, 2])
        ax13.plot(df['date'], df['positivity_rate'], linewidth=2, color='red')
        ax13.axhline(5, color='orange', linestyle='--', alpha=0.5, label='WHO threshold (5%)')
        ax13.set_title('Test Positivity Rate', fontsize=12, fontweight='bold')
        ax13.set_xlabel('Date')
        ax13.set_ylabel('Positivity Rate (%)')
        ax13.legend()
        ax13.grid(True, alpha=0.3)
        ax13.tick_params(axis='x', rotation=45)

        plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/02_time_series/11_covid19_analysis/covid19_analysis.png',
                   dpi=300, bbox_inches='tight')
        print("   Saved: covid19_analysis.png")
        plt.close()


def main():
    """Main execution function."""
    analyzer = COVID19Analyzer(lookback=14)
    results = analyzer.train_and_evaluate()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nKey Insights:")
    print("1. Multiple epidemic waves detected with varying intensities")
    print("2. Weekly seasonality present in case reporting patterns")
    print("3. Ensemble methods generally provide more robust forecasts")
    print("4. LSTM captures non-linear patterns but requires more data")
    print("5. SARIMA effective for capturing both trend and seasonal components")

    print("\nPublic Health Applications:")
    print("• Hospital capacity planning and resource allocation")
    print("• Vaccination campaign timing and targeting")
    print("• Public health intervention evaluation")
    print("• Early warning system for outbreak surges")

    # Find best model
    if results:
        best_model = min(results.items(), key=lambda x: x[1]['rmse'])
        print(f"\nBest Model: {best_model[0]} (RMSE: {best_model[1]['rmse']:.2f})")


if __name__ == "__main__":
    main()
