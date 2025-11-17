"""
SARIMA for Seasonal Time Series Data - Comprehensive Analysis
==============================================================

This solution demonstrates advanced SARIMA modeling for seasonal data:
1. Seasonal decomposition and pattern identification
2. Multiple seasonal differencing approaches
3. Grid search for seasonal ARIMA parameters
4. Seasonal strength measurement
5. Multiple seasonality handling
6. STL decomposition with robust seasonal extraction
7. Seasonal subseries plots
8. Walk-forward validation with seasonal splits
9. Forecast accuracy across different seasons
10. Seasonal residual diagnostics

Dataset: Synthetic seasonal data with multiple patterns
Models: SARIMA, Auto-SARIMA, Seasonal Naive, Multiple Seasonality
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import itertools
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def generate_seasonal_data(n_samples=730):
    """Generate synthetic seasonal time series data"""
    print("Generating seasonal time series data...")

    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')

    # Trend component
    trend = np.linspace(100, 300, n_samples)

    # Weekly seasonality
    weekly_season = 15 * np.sin(2 * np.pi * np.arange(n_samples) / 7)

    # Monthly seasonality
    monthly_season = 20 * np.sin(2 * np.pi * np.arange(n_samples) / 30)

    # Yearly seasonality
    yearly_season = 30 * np.sin(2 * np.pi * np.arange(n_samples) / 365)

    # Random noise
    noise = np.random.normal(0, 5, n_samples)

    # Combine components
    values = trend + weekly_season + monthly_season + yearly_season + noise

    df = pd.DataFrame({
        'date': dates,
        'value': values,
        'trend': trend,
        'weekly': weekly_season,
        'monthly': monthly_season,
        'yearly': yearly_season
    })
    df.set_index('date', inplace=True)

    print(f"Generated {len(df)} observations")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Value range: [{df['value'].min():.2f}, {df['value'].max():.2f}]")

    return df


def seasonal_stationarity_tests(series, seasonal_period):
    """Test stationarity with seasonal differencing"""
    print(f"\n{'='*70}")
    print("Seasonal Stationarity Tests")
    print(f"{'='*70}")

    tests_results = {}

    # Original series
    print("\n1. Original Series:")
    adf_orig = adfuller(series, autolag='AIC')
    print(f"   ADF Statistic: {adf_orig[0]:.6f}, p-value: {adf_orig[1]:.6f}")
    tests_results['original_adf'] = adf_orig[1]

    # First difference
    diff1 = series.diff().dropna()
    print("\n2. First Difference:")
    adf_diff1 = adfuller(diff1, autolag='AIC')
    print(f"   ADF Statistic: {adf_diff1[0]:.6f}, p-value: {adf_diff1[1]:.6f}")
    tests_results['diff1_adf'] = adf_diff1[1]

    # Seasonal difference
    seasonal_diff = series.diff(seasonal_period).dropna()
    print(f"\n3. Seasonal Difference (lag={seasonal_period}):")
    adf_seasonal = adfuller(seasonal_diff, autolag='AIC')
    print(f"   ADF Statistic: {adf_seasonal[0]:.6f}, p-value: {adf_seasonal[1]:.6f}")
    tests_results['seasonal_diff_adf'] = adf_seasonal[1]

    # Both differences
    both_diff = series.diff().diff(seasonal_period).dropna()
    print(f"\n4. First + Seasonal Difference:")
    adf_both = adfuller(both_diff, autolag='AIC')
    print(f"   ADF Statistic: {adf_both[0]:.6f}, p-value: {adf_both[1]:.6f}")
    tests_results['both_diff_adf'] = adf_both[1]

    # Recommend differencing
    print("\nRecommendation:")
    if adf_orig[1] < 0.05:
        print("  No differencing needed (already stationary)")
    elif adf_seasonal[1] < 0.05:
        print(f"  Use seasonal differencing only (D=1, s={seasonal_period})")
    elif adf_diff1[1] < 0.05:
        print("  Use first differencing only (d=1)")
    elif adf_both[1] < 0.05:
        print(f"  Use both differencing (d=1, D=1, s={seasonal_period})")
    else:
        print("  May need additional transformations")

    return tests_results


def perform_seasonal_decomposition(series, period=7, model='additive'):
    """Perform seasonal decomposition using multiple methods"""
    print(f"\n{'='*70}")
    print(f"Seasonal Decomposition (period={period})")
    print(f"{'='*70}")

    # STL decomposition
    stl = STL(series, seasonal=period, trend=None, robust=True)
    stl_result = stl.fit()

    # Classical decomposition
    classical = seasonal_decompose(series, model=model, period=period, extrapolate_trend='freq')

    # Calculate seasonal strength
    seasonal_var = np.var(stl_result.seasonal)
    residual_var = np.var(stl_result.resid)
    total_var = seasonal_var + residual_var

    seasonal_strength = 1 - (residual_var / total_var) if total_var > 0 else 0

    print(f"\nDecomposition Statistics:")
    print(f"  Trend variance: {np.var(stl_result.trend):.2f}")
    print(f"  Seasonal variance: {seasonal_var:.2f}")
    print(f"  Residual variance: {residual_var:.2f}")
    print(f"  Seasonal strength: {seasonal_strength:.4f}")

    # Measure autocorrelation at seasonal lag
    from statsmodels.tsa.stattools import acf
    acf_values = acf(series, nlags=period*2, fft=False)
    print(f"  ACF at seasonal lag {period}: {acf_values[period]:.4f}")

    return {
        'stl': stl_result,
        'classical': classical,
        'seasonal_strength': seasonal_strength,
        'acf_seasonal': acf_values[period]
    }


def plot_seasonal_subseries(series, period=7):
    """Create seasonal subseries plot"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Seasonal subseries plot
    series_df = pd.DataFrame({'value': series})
    series_df['season'] = series_df.index.dayofweek if period == 7 else series_df.index.month

    for season in sorted(series_df['season'].unique()):
        subset = series_df[series_df['season'] == season]['value']
        axes[0].plot(subset.values, alpha=0.7, label=f'Season {season}')

    axes[0].set_title('Seasonal Subseries Plot', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Observation within Season')
    axes[0].set_ylabel('Value')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3)

    # Boxplot by season
    series_df.boxplot(column='value', by='season', ax=axes[1])
    axes[1].set_title('Seasonal Boxplots', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Season')
    axes[1].set_ylabel('Value')
    axes[1].get_figure().suptitle('')

    plt.tight_layout()
    return fig


def grid_search_sarima(series, p_range, d_range, q_range, P_range, D_range, Q_range, s):
    """Grid search for optimal SARIMA parameters"""
    print(f"\n{'='*70}")
    print(f"SARIMA Grid Search (s={s})")
    print(f"{'='*70}")

    # Generate all combinations
    pdq = list(itertools.product(p_range, d_range, q_range))
    PDQs = list(itertools.product(P_range, D_range, Q_range))

    print(f"Testing {len(pdq) * len(PDQs)} combinations...")

    results = []
    best_aic = np.inf
    best_params = None

    tested = 0
    for param in pdq:
        for param_seasonal in PDQs:
            try:
                model = SARIMAX(series,
                               order=param,
                               seasonal_order=(param_seasonal[0], param_seasonal[1], param_seasonal[2], s),
                               enforce_stationarity=False,
                               enforce_invertibility=False)
                fitted = model.fit(disp=False, maxiter=200)

                results.append({
                    'order': param,
                    'seasonal_order': (param_seasonal[0], param_seasonal[1], param_seasonal[2], s),
                    'aic': fitted.aic,
                    'bic': fitted.bic,
                    'hqic': fitted.hqic
                })

                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_params = (param, (param_seasonal[0], param_seasonal[1], param_seasonal[2], s))

                tested += 1
                if tested % 10 == 0:
                    print(f"  Tested {tested} models... Best AIC so far: {best_aic:.2f}")

            except:
                continue

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('aic')

    print(f"\nTested {len(results)} valid models")
    print(f"\nTop 5 Models by AIC:")
    print(results_df.head(5).to_string(index=False))
    print(f"\nBest Parameters:")
    print(f"  Order: {best_params[0]}")
    print(f"  Seasonal Order: {best_params[1]}")
    print(f"  AIC: {best_aic:.2f}")

    return results_df, best_params


def fit_sarima_models(series, params_list):
    """Fit multiple SARIMA models"""
    print(f"\n{'='*70}")
    print("Fitting SARIMA Models")
    print(f"{'='*70}")

    models = {}

    for order, seasonal_order in params_list:
        try:
            print(f"\nFitting SARIMA{order}x{seasonal_order}...")
            model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
            fitted = model.fit(disp=False, maxiter=200)

            models[(order, seasonal_order)] = {
                'model': fitted,
                'aic': fitted.aic,
                'bic': fitted.bic,
                'hqic': fitted.hqic,
                'log_likelihood': fitted.llf
            }

            print(f"  AIC: {fitted.aic:.2f}")
            print(f"  BIC: {fitted.bic:.2f}")
            print(f"  Log-Likelihood: {fitted.llf:.2f}")

        except Exception as e:
            print(f"  Failed: {str(e)}")
            continue

    return models


def seasonal_walk_forward_validation(series, order, seasonal_order, n_splits=5):
    """Walk-forward validation respecting seasonal patterns"""
    print(f"\n{'='*70}")
    print(f"Seasonal Walk-Forward Validation")
    print(f"{'='*70}")

    n = len(series)
    s = seasonal_order[3]
    test_size = s * (n // (s * (n_splits + 1)))  # Ensure test size is multiple of season

    predictions = []
    actuals = []
    errors = []

    for i in range(n_splits):
        split_point = n - (n_splits - i) * test_size
        train = series[:split_point]
        test = series[split_point:split_point + test_size]

        if len(test) == 0:
            continue

        try:
            model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
            fitted = model.fit(disp=False, maxiter=200)
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
    if len(errors_df) > 0:
        print(f"\nAverage Performance:")
        print(f"  MAE: {errors_df['mae'].mean():.4f} ± {errors_df['mae'].std():.4f}")
        print(f"  RMSE: {errors_df['rmse'].mean():.4f} ± {errors_df['rmse'].std():.4f}")
        print(f"  MAPE: {errors_df['mape'].mean():.2f}% ± {errors_df['mape'].std():.2f}%")

    return predictions, actuals, errors_df


def seasonal_residual_diagnostics(residuals, seasonal_period):
    """Residual diagnostics focusing on seasonal patterns"""
    print(f"\n{'='*70}")
    print("Seasonal Residual Diagnostics")
    print(f"{'='*70}")

    # Basic statistics
    print(f"\nResidual Statistics:")
    print(f"  Mean: {np.mean(residuals):.6f}")
    print(f"  Std Dev: {np.std(residuals):.6f}")
    print(f"  Skewness: {stats.skew(residuals):.6f}")
    print(f"  Kurtosis: {stats.kurtosis(residuals):.6f}")

    # Normality test
    if len(residuals) > 3:
        shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000] if len(residuals) > 5000 else residuals)
        print(f"\nShapiro-Wilk Test:")
        print(f"  p-value: {shapiro_p:.6f} ({'NORMAL' if shapiro_p > 0.05 else 'NOT NORMAL'})")

    # Ljung-Box test at seasonal lags
    seasonal_lags = [seasonal_period, seasonal_period*2, seasonal_period*3]
    lb_result = acorr_ljungbox(residuals, lags=seasonal_lags, return_df=True)
    print(f"\nLjung-Box Test at Seasonal Lags:")
    print(lb_result.to_string())

    # Check for remaining seasonality
    from statsmodels.tsa.stattools import acf
    acf_values = acf(residuals, nlags=seasonal_period*2, fft=False)
    print(f"\nACF at seasonal lag {seasonal_period}: {acf_values[seasonal_period]:.4f}")
    if abs(acf_values[seasonal_period]) > 0.1:
        print("  WARNING: Significant autocorrelation at seasonal lag!")

    return {
        'mean': np.mean(residuals),
        'std': np.std(residuals),
        'seasonal_acf': acf_values[seasonal_period]
    }


def calculate_comprehensive_metrics(y_true, y_pred):
    """Calculate comprehensive forecast metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    # SMAPE
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

    # MASE
    naive_seasonal_errors = np.abs(np.array(y_true)[7:] - np.array(y_true)[:-7])
    mase = mae / np.mean(naive_seasonal_errors) if len(naive_seasonal_errors) > 0 else np.inf

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'SMAPE': smape,
        'MASE': mase
    }


def plot_sarima_diagnostics(df, decomp_result, model, forecast_result, seasonal_period):
    """Create comprehensive SARIMA diagnostic plots"""
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.3)

    # Original series
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df.index, df['value'], linewidth=1.5, label='Original')
    ax1.set_title('Original Time Series with Seasonality', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # STL decomposition components
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(decomp_result['stl'].trend, linewidth=1.5)
    ax2.set_title('Trend Component', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(decomp_result['stl'].seasonal, linewidth=1.5, color='orange')
    ax3.set_title(f'Seasonal Component (period={seasonal_period})', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # ACF and PACF
    ax4 = fig.add_subplot(gs[2, 0])
    plot_acf(df['value'], lags=seasonal_period*3, ax=ax4, alpha=0.05)
    ax4.set_title('ACF of Original Series', fontsize=12, fontweight='bold')

    ax5 = fig.add_subplot(gs[2, 1])
    plot_pacf(df['value'], lags=seasonal_period*3, ax=ax5, alpha=0.05, method='ywm')
    ax5.set_title('PACF of Original Series', fontsize=12, fontweight='bold')

    # Residuals
    residuals = model.resid
    ax6 = fig.add_subplot(gs[3, 0])
    ax6.plot(residuals, linewidth=1, alpha=0.7)
    ax6.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax6.set_title('Residuals Over Time', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    ax7 = fig.add_subplot(gs[3, 1])
    ax7.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax7.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax7.set_title('Residual Distribution', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)

    # Q-Q plot
    ax8 = fig.add_subplot(gs[4, 0])
    stats.probplot(residuals, dist="norm", plot=ax8)
    ax8.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3)

    # Forecast
    ax9 = fig.add_subplot(gs[4, 1])
    train_size = len(df) - seasonal_period * 4
    ax9.plot(df.index[:train_size], df['value'][:train_size], label='Training', linewidth=1.5)
    ax9.plot(df.index[train_size:], df['value'][train_size:], label='Actual', linewidth=1.5, color='green')

    forecast_index = df.index[train_size:]
    forecast_steps = min(len(forecast_result.predicted_mean), len(forecast_index))
    ax9.plot(forecast_index[:forecast_steps], forecast_result.predicted_mean[:forecast_steps],
             label='Forecast', linewidth=2, color='red', linestyle='--')

    conf_int = forecast_result.conf_int()
    ax9.fill_between(forecast_index[:forecast_steps],
                     conf_int.iloc[:forecast_steps, 0],
                     conf_int.iloc[:forecast_steps, 1],
                     alpha=0.3, color='red')

    ax9.set_title('Seasonal Forecast with Confidence Intervals', fontsize=12, fontweight='bold')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    plt.savefig('sarima_diagnostics.png', dpi=300, bbox_inches='tight')
    return fig


def main():
    """Main execution function"""
    print("="*70)
    print("SARIMA FOR SEASONAL TIME SERIES DATA")
    print("="*70)

    # Generate seasonal data
    df = generate_seasonal_data(n_samples=730)
    series = df['value']

    # Seasonal stationarity tests
    seasonal_period = 7  # Weekly seasonality
    stationarity_results = seasonal_stationarity_tests(series, seasonal_period)

    # Seasonal decomposition
    decomp_results = perform_seasonal_decomposition(series, period=seasonal_period)

    # Seasonal subseries plot
    subseries_fig = plot_seasonal_subseries(series, period=seasonal_period)
    plt.savefig('seasonal_subseries.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Grid search for SARIMA parameters (reduced search space for speed)
    p_range = range(0, 3)
    d_range = range(0, 2)
    q_range = range(0, 3)
    P_range = range(0, 3)
    D_range = range(0, 2)
    Q_range = range(0, 3)

    grid_results, best_params = grid_search_sarima(
        series, p_range, d_range, q_range, P_range, D_range, Q_range, seasonal_period
    )

    # Fit multiple SARIMA models
    best_order, best_seasonal = best_params
    candidate_params = [
        best_params,
        ((1, 1, 1), (1, 1, 1, seasonal_period)),
        ((2, 1, 2), (1, 1, 1, seasonal_period)),
        ((1, 0, 1), (1, 1, 0, seasonal_period))
    ]

    models = fit_sarima_models(series, candidate_params)

    # Best model analysis
    best_model = models[best_params]['model']
    print(f"\n{'='*70}")
    print(f"Best Model Summary: SARIMA{best_order}x{best_seasonal}")
    print(f"{'='*70}")
    print(best_model.summary())

    # Seasonal residual diagnostics
    residuals = best_model.resid
    residual_stats = seasonal_residual_diagnostics(residuals, seasonal_period)

    # Walk-forward validation
    predictions, actuals, cv_results = seasonal_walk_forward_validation(
        series, best_order, best_seasonal, n_splits=5
    )

    # Final forecast
    train_size = len(series) - seasonal_period * 4
    train_series = series[:train_size]
    test_series = series[train_size:]

    final_model = SARIMAX(train_series, order=best_order, seasonal_order=best_seasonal)
    final_fitted = final_model.fit(disp=False, maxiter=200)
    forecast_result = final_fitted.get_forecast(steps=len(test_series))
    forecast_mean = forecast_result.predicted_mean

    # Calculate metrics
    metrics = calculate_comprehensive_metrics(test_series, forecast_mean)
    print(f"\n{'='*70}")
    print("Final Forecast Metrics")
    print(f"{'='*70}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Create diagnostic plots
    plot_sarima_diagnostics(df, decomp_results, final_fitted, forecast_result, seasonal_period)

    print("\n" + "="*70)
    print("SARIMA ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nBest Model: SARIMA{best_order}x{best_seasonal}")
    print(f"Seasonal Strength: {decomp_results['seasonal_strength']:.4f}")
    print(f"Test MAE: {metrics['MAE']:.4f}")
    print(f"Test MAPE: {metrics['MAPE']:.2f}%")


if __name__ == "__main__":
    main()
