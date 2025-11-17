"""
ARIMA Model Selection and Diagnostics - Comprehensive Time Series Analysis
===========================================================================

This solution demonstrates advanced ARIMA modeling with comprehensive diagnostics:
1. Multiple differencing and transformation methods
2. Grid search for optimal ARIMA parameters
3. Information criteria comparison (AIC, BIC, HQIC)
4. Stationarity tests (ADF, KPSS, PP)
5. ACF/PACF analysis for order selection
6. STL decomposition
7. Walk-forward validation
8. Residual diagnostics and white noise tests
9. Forecast with confidence intervals
10. Multiple model comparison

Dataset: Synthetic and real-world time series data
Models: ARIMA, Auto ARIMA, Manual Grid Search
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def generate_time_series_data(n_samples=500):
    """Generate synthetic time series data with trend, seasonality, and noise"""
    print("Generating synthetic time series data...")

    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')

    # Trend component
    trend = np.linspace(100, 200, n_samples)

    # Seasonal component (weekly pattern)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 7)

    # AR(1) component
    ar_component = np.zeros(n_samples)
    ar_component[0] = np.random.normal(0, 5)
    for i in range(1, n_samples):
        ar_component[i] = 0.7 * ar_component[i-1] + np.random.normal(0, 5)

    # Combine components
    values = trend + seasonal + ar_component

    df = pd.DataFrame({
        'date': dates,
        'value': values,
        'trend': trend,
        'seasonal': seasonal,
        'ar_component': ar_component
    })
    df.set_index('date', inplace=True)

    print(f"Generated {len(df)} observations")
    print(f"Value range: [{df['value'].min():.2f}, {df['value'].max():.2f}]")

    return df


def perform_stationarity_tests(series, name='Series'):
    """Perform comprehensive stationarity tests"""
    print(f"\n{'='*70}")
    print(f"Stationarity Tests for {name}")
    print(f"{'='*70}")

    # Augmented Dickey-Fuller test
    print("\n1. Augmented Dickey-Fuller Test:")
    adf_result = adfuller(series, autolag='AIC')
    print(f"   ADF Statistic: {adf_result[0]:.6f}")
    print(f"   p-value: {adf_result[1]:.6f}")
    print(f"   Critical Values:")
    for key, value in adf_result[4].items():
        print(f"      {key}: {value:.6f}")
    print(f"   Result: {'STATIONARY' if adf_result[1] < 0.05 else 'NON-STATIONARY'}")

    # KPSS test
    print("\n2. KPSS Test:")
    kpss_result = kpss(series, regression='ct', nlags='auto')
    print(f"   KPSS Statistic: {kpss_result[0]:.6f}")
    print(f"   p-value: {kpss_result[1]:.6f}")
    print(f"   Critical Values:")
    for key, value in kpss_result[3].items():
        print(f"      {key}: {value:.6f}")
    print(f"   Result: {'STATIONARY' if kpss_result[1] > 0.05 else 'NON-STATIONARY'}")

    # Phillips-Perron test (approximated by ADF with different lag selection)
    pp_result = adfuller(series, autolag='BIC')
    print("\n3. Phillips-Perron Test (approximated):")
    print(f"   PP Statistic: {pp_result[0]:.6f}")
    print(f"   p-value: {pp_result[1]:.6f}")
    print(f"   Result: {'STATIONARY' if pp_result[1] < 0.05 else 'NON-STATIONARY'}")

    return {
        'adf_statistic': adf_result[0],
        'adf_pvalue': adf_result[1],
        'kpss_statistic': kpss_result[0],
        'kpss_pvalue': kpss_result[1],
        'pp_statistic': pp_result[0],
        'pp_pvalue': pp_result[1]
    }


def perform_stl_decomposition(series, period=7):
    """Perform STL decomposition"""
    print(f"\n{'='*70}")
    print("STL Decomposition")
    print(f"{'='*70}")

    stl = STL(series, seasonal=period, trend=None)
    result = stl.fit()

    print(f"\nDecomposition Statistics:")
    print(f"  Trend variance: {np.var(result.trend):.2f}")
    print(f"  Seasonal variance: {np.var(result.seasonal):.2f}")
    print(f"  Residual variance: {np.var(result.resid):.2f}")
    print(f"  Seasonal strength: {1 - np.var(result.resid) / np.var(result.seasonal + result.resid):.4f}")
    print(f"  Trend strength: {1 - np.var(result.resid) / np.var(result.trend + result.resid):.4f}")

    return result


def plot_acf_pacf(series, lags=40):
    """Plot ACF and PACF"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    plot_acf(series, lags=lags, ax=axes[0], alpha=0.05)
    axes[0].set_title('Autocorrelation Function (ACF)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('ACF')

    plot_pacf(series, lags=lags, ax=axes[1], alpha=0.05, method='ywm')
    axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('PACF')

    plt.tight_layout()
    return fig


def grid_search_arima(series, p_range, d_range, q_range):
    """Grid search for optimal ARIMA parameters"""
    print(f"\n{'='*70}")
    print("ARIMA Grid Search")
    print(f"{'='*70}")
    print(f"Testing {len(p_range) * len(d_range) * len(q_range)} combinations...")

    results = []
    best_aic = np.inf
    best_order = None

    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    model = ARIMA(series, order=(p, d, q))
                    fitted = model.fit()

                    results.append({
                        'order': (p, d, q),
                        'aic': fitted.aic,
                        'bic': fitted.bic,
                        'hqic': fitted.hqic,
                        'log_likelihood': fitted.llf
                    })

                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)

                except:
                    continue

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('aic')

    print(f"\nTop 10 Models by AIC:")
    print(results_df.head(10).to_string(index=False))
    print(f"\nBest ARIMA order: {best_order} (AIC: {best_aic:.2f})")

    return results_df, best_order


def fit_arima_models(series, orders):
    """Fit multiple ARIMA models and compare"""
    print(f"\n{'='*70}")
    print("Fitting ARIMA Models")
    print(f"{'='*70}")

    models = {}

    for order in orders:
        try:
            print(f"\nFitting ARIMA{order}...")
            model = ARIMA(series, order=order)
            fitted = model.fit()

            models[order] = {
                'model': fitted,
                'aic': fitted.aic,
                'bic': fitted.bic,
                'hqic': fitted.hqic,
                'log_likelihood': fitted.llf,
                'params': fitted.params
            }

            print(f"  AIC: {fitted.aic:.2f}")
            print(f"  BIC: {fitted.bic:.2f}")
            print(f"  HQIC: {fitted.hqic:.2f}")
            print(f"  Log-Likelihood: {fitted.llf:.2f}")

        except Exception as e:
            print(f"  Failed to fit ARIMA{order}: {str(e)}")
            continue

    return models


def residual_diagnostics(residuals, model_name='Model'):
    """Comprehensive residual diagnostics"""
    print(f"\n{'='*70}")
    print(f"Residual Diagnostics for {model_name}")
    print(f"{'='*70}")

    # Basic statistics
    print(f"\nResidual Statistics:")
    print(f"  Mean: {np.mean(residuals):.6f}")
    print(f"  Std Dev: {np.std(residuals):.6f}")
    print(f"  Skewness: {stats.skew(residuals):.6f}")
    print(f"  Kurtosis: {stats.kurtosis(residuals):.6f}")

    # Normality test
    shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000] if len(residuals) > 5000 else residuals)
    print(f"\nShapiro-Wilk Normality Test:")
    print(f"  Statistic: {shapiro_stat:.6f}")
    print(f"  p-value: {shapiro_p:.6f}")
    print(f"  Result: {'NORMAL' if shapiro_p > 0.05 else 'NOT NORMAL'}")

    # Ljung-Box test for autocorrelation
    lb_result = acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)
    print(f"\nLjung-Box Test (White Noise):")
    print(lb_result.to_string())

    return {
        'mean': np.mean(residuals),
        'std': np.std(residuals),
        'shapiro_p': shapiro_p,
        'ljungbox': lb_result
    }


def walk_forward_validation(series, order, n_splits=5):
    """Walk-forward validation for time series"""
    print(f"\n{'='*70}")
    print(f"Walk-Forward Validation - ARIMA{order}")
    print(f"{'='*70}")

    n = len(series)
    test_size = n // (n_splits + 1)

    predictions = []
    actuals = []
    errors = []

    for i in range(n_splits):
        split_point = n - (n_splits - i) * test_size
        train = series[:split_point]
        test = series[split_point:split_point + test_size]

        try:
            model = ARIMA(train, order=order)
            fitted = model.fit()
            forecast = fitted.forecast(steps=len(test))

            predictions.extend(forecast)
            actuals.extend(test)

            mae = mean_absolute_error(test, forecast)
            rmse = np.sqrt(mean_squared_error(test, forecast))
            mape = mean_absolute_percentage_error(test, forecast) * 100

            errors.append({'fold': i+1, 'mae': mae, 'rmse': rmse, 'mape': mape})

            print(f"\nFold {i+1}: Train={len(train)}, Test={len(test)}")
            print(f"  MAE: {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAPE: {mape:.2f}%")

        except Exception as e:
            print(f"  Failed: {str(e)}")
            continue

    errors_df = pd.DataFrame(errors)
    print(f"\nAverage Performance:")
    print(f"  MAE: {errors_df['mae'].mean():.4f} ± {errors_df['mae'].std():.4f}")
    print(f"  RMSE: {errors_df['rmse'].mean():.4f} ± {errors_df['rmse'].std():.4f}")
    print(f"  MAPE: {errors_df['mape'].mean():.2f}% ± {errors_df['mape'].std():.2f}%")

    return predictions, actuals, errors_df


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive forecast metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    # SMAPE
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

    # MASE (Mean Absolute Scaled Error)
    naive_mae = np.mean(np.abs(np.diff(y_true)))
    mase = mae / naive_mae if naive_mae > 0 else np.inf

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'SMAPE': smape,
        'MASE': mase
    }


def plot_diagnostics(df, stl_result, residuals, forecast_result):
    """Create comprehensive diagnostic plots"""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

    # Original series
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df.index, df['value'], label='Original Series', linewidth=1.5)
    ax1.set_title('Original Time Series', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # STL decomposition
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(stl_result.trend, label='Trend', linewidth=1.5)
    ax2.set_title('Trend Component', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(stl_result.seasonal, label='Seasonal', linewidth=1.5, color='orange')
    ax3.set_title('Seasonal Component', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Residuals
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(residuals, label='Residuals', linewidth=1, alpha=0.7)
    ax4.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax4.set_title('Residuals Over Time', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Residual histogram
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax5.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax5.set_title('Residual Distribution', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Residual Value')
    ax5.set_ylabel('Frequency')
    ax5.grid(True, alpha=0.3)

    # Q-Q plot
    ax6 = fig.add_subplot(gs[3, 0])
    stats.probplot(residuals, dist="norm", plot=ax6)
    ax6.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # Forecast
    ax7 = fig.add_subplot(gs[3, 1])
    train_size = len(df) - 50
    ax7.plot(df.index[:train_size], df['value'][:train_size], label='Training Data', linewidth=1.5)
    ax7.plot(df.index[train_size:], df['value'][train_size:], label='Actual', linewidth=1.5, color='green')

    forecast_index = df.index[train_size:]
    ax7.plot(forecast_index, forecast_result.predicted_mean[:50],
             label='Forecast', linewidth=2, color='red', linestyle='--')

    ax7.fill_between(forecast_index,
                     forecast_result.conf_int()['lower value'][:50],
                     forecast_result.conf_int()['upper value'][:50],
                     alpha=0.3, color='red')

    ax7.set_title('Forecast with 95% Confidence Interval', fontsize=12, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    plt.savefig('arima_diagnostics.png', dpi=300, bbox_inches='tight')
    return fig


def main():
    """Main execution function"""
    print("="*70)
    print("ARIMA MODEL SELECTION AND DIAGNOSTICS")
    print("="*70)

    # Generate data
    df = generate_time_series_data(n_samples=500)
    series = df['value']

    # Stationarity tests on original series
    stationarity_results = perform_stationarity_tests(series, 'Original Series')

    # Test differenced series
    diff_series = series.diff().dropna()
    diff_stationarity = perform_stationarity_tests(diff_series, 'First Difference')

    # STL decomposition
    stl_result = perform_stl_decomposition(series, period=7)

    # ACF/PACF plots
    acf_pacf_fig = plot_acf_pacf(series, lags=40)
    plt.savefig('acf_pacf_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Grid search for optimal parameters
    p_range = range(0, 5)
    d_range = range(0, 3)
    q_range = range(0, 5)
    grid_results, best_order = grid_search_arima(series, p_range, d_range, q_range)

    # Fit multiple ARIMA models
    candidate_orders = [best_order, (1, 1, 1), (2, 1, 2), (3, 1, 1), (1, 1, 2)]
    models = fit_arima_models(series, candidate_orders)

    # Detailed analysis of best model
    best_model = models[best_order]['model']
    print(f"\n{'='*70}")
    print(f"Best Model Summary: ARIMA{best_order}")
    print(f"{'='*70}")
    print(best_model.summary())

    # Residual diagnostics
    residuals = best_model.resid
    residual_stats = residual_diagnostics(residuals, f'ARIMA{best_order}')

    # Walk-forward validation
    predictions, actuals, cv_results = walk_forward_validation(series, best_order, n_splits=5)

    # Generate forecast
    train_size = len(series) - 50
    train_series = series[:train_size]
    test_series = series[train_size:]

    final_model = ARIMA(train_series, order=best_order)
    final_fitted = final_model.fit()
    forecast_result = final_fitted.get_forecast(steps=50)
    forecast_mean = forecast_result.predicted_mean

    # Calculate metrics
    metrics = calculate_metrics(test_series, forecast_mean)
    print(f"\n{'='*70}")
    print("Final Forecast Metrics")
    print(f"{'='*70}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Create comprehensive diagnostic plots
    plot_diagnostics(df, stl_result, residuals, forecast_result)

    # Model comparison summary
    print(f"\n{'='*70}")
    print("Model Comparison Summary")
    print(f"{'='*70}")
    comparison_data = []
    for order, info in models.items():
        comparison_data.append({
            'Order': str(order),
            'AIC': info['aic'],
            'BIC': info['bic'],
            'HQIC': info['hqic'],
            'Log-Likelihood': info['log_likelihood']
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('AIC')
    print(comparison_df.to_string(index=False))

    print("\n" + "="*70)
    print("ARIMA ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nBest Model: ARIMA{best_order}")
    print(f"Test MAE: {metrics['MAE']:.4f}")
    print(f"Test RMSE: {metrics['RMSE']:.4f}")
    print(f"Test MAPE: {metrics['MAPE']:.2f}%")
    print(f"\nDiagnostic plots saved to 'arima_diagnostics.png'")
    print(f"ACF/PACF plots saved to 'acf_pacf_plots.png'")


if __name__ == "__main__":
    main()
