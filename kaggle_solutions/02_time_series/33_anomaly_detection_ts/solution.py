"""
Anomaly Detection - Comprehensive Time Series Analysis
=================

This solution demonstrates anomaly detection techniques:
1. Data generation and preprocessing
2. Stationarity tests (ADF, KPSS)
3. STL decomposition
4. ACF/PACF analysis
5. Model training and selection
6. Walk-forward validation
7. Comprehensive metrics (MAE, RMSE, MAPE, SMAPE)
8. Forecast visualization
9. Residual diagnostics
10. Model comparison

Dataset: Synthetic time series data
Models: Multiple approaches for anomaly detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf, grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def generate_data(n_samples=1000):
    """Generate synthetic time series data"""
    print(f"Generating data for Anomaly Detection...")
    
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # Generate complex patterns
    t = np.linspace(0, 4*np.pi, n_samples)
    trend1 = 100 + 50 * np.sin(t/3) + 0.02 * t**2
    trend2 = 150 + 40 * np.cos(t/4) + 0.015 * t**2
    trend3 = 120 + 30 * np.sin(t/2) * np.cos(t/5)
    
    # Seasonal components
    seasonal1 = 15 * np.sin(2 * np.pi * np.arange(n_samples) / 7)
    seasonal2 = 12 * np.cos(2 * np.pi * np.arange(n_samples) / 7)
    seasonal3 = 18 * np.sin(2 * np.pi * np.arange(n_samples) / 30)
    
    # AR components
    ar1 = np.zeros(n_samples)
    ar2 = np.zeros(n_samples)
    ar3 = np.zeros(n_samples)
    
    for i in range(1, n_samples):
        ar1[i] = 0.7 * ar1[i-1] + np.random.normal(0, 3)
        ar2[i] = 0.6 * ar2[i-1] + 0.3 * ar1[i-1] + np.random.normal(0, 3)
        ar3[i] = 0.5 * ar3[i-1] + 0.2 * ar2[i-1] + np.random.normal(0, 3)
    
    # Noise
    noise1 = np.random.normal(0, 5, n_samples)
    noise2 = np.random.normal(0, 5, n_samples)
    noise3 = np.random.normal(0, 5, n_samples)
    
    # Combine
    series1 = trend1 + seasonal1 + ar1 + noise1
    series2 = trend2 + seasonal2 + ar2 + noise2
    series3 = trend3 + seasonal3 + ar3 + noise3
    
    df = pd.DataFrame({
        'date': dates,
        'series1': series1,
        'series2': series2,
        'series3': series3
    })
    df.set_index('date', inplace=True)
    
    print(f"Generated {len(df)} observations with {len(df.columns)} series")
    return df


def perform_stationarity_tests(series, name='Series'):
    """Perform comprehensive stationarity tests"""
    print(f"\n{'='*70}")
    print(f"Stationarity Tests for {name}")
    print(f"{'='*70}")
    
    # ADF test
    adf_result = adfuller(series, autolag='AIC')
    print(f"\nADF Test:")
    print(f"  Statistic: {adf_result[0]:.6f}")
    print(f"  p-value: {adf_result[1]:.6f}")
    print(f"  Result: {'STATIONARY' if adf_result[1] < 0.05 else 'NON-STATIONARY'}")
    
    # KPSS test
    kpss_result = kpss(series, regression='ct', nlags='auto')
    print(f"\nKPSS Test:")
    print(f"  Statistic: {kpss_result[0]:.6f}")
    print(f"  p-value: {kpss_result[1]:.6f}")
    print(f"  Result: {'STATIONARY' if kpss_result[1] > 0.05 else 'NON-STATIONARY'}")
    
    return {'adf_pvalue': adf_result[1], 'kpss_pvalue': kpss_result[1]}


def perform_stl_decomposition(series, period=7):
    """STL decomposition"""
    print(f"\n{'='*70}")
    print("STL Decomposition")
    print(f"{'='*70}")
    
    stl = STL(series, seasonal=period, trend=None)
    result = stl.fit()
    
    print(f"\nComponent Statistics:")
    print(f"  Trend variance: {np.var(result.trend):.2f}")
    print(f"  Seasonal variance: {np.var(result.seasonal):.2f}")
    print(f"  Residual variance: {np.var(result.resid):.2f}")
    
    seasonal_strength = 1 - np.var(result.resid) / (np.var(result.seasonal) + np.var(result.resid))
    print(f"  Seasonal strength: {seasonal_strength:.4f}")
    
    return result


def plot_acf_pacf(series, lags=40):
    """Plot ACF and PACF"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    plot_acf(series, lags=lags, ax=axes[0], alpha=0.05)
    axes[0].set_title('ACF', fontsize=14, fontweight='bold')
    
    plot_pacf(series, lags=lags, ax=axes[1], alpha=0.05, method='ywm')
    axes[1].set_title('PACF', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def residual_diagnostics(residuals, model_name='Model'):
    """Comprehensive residual diagnostics"""
    print(f"\n{'='*70}")
    print(f"Residual Diagnostics - {model_name}")
    print(f"{'='*70}")
    
    print(f"\nStatistics:")
    print(f"  Mean: {np.mean(residuals):.6f}")
    print(f"  Std Dev: {np.std(residuals):.6f}")
    print(f"  Skewness: {stats.skew(residuals):.6f}")
    print(f"  Kurtosis: {stats.kurtosis(residuals):.6f}")
    
    # Normality test
    if len(residuals) > 3:
        shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000] if len(residuals) > 5000 else residuals)
        print(f"\nShapiro-Wilk Test: p-value={shapiro_p:.6f}")
    
    # Ljung-Box test
    lb_result = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
    print(f"\nLjung-Box Test:")
    print(lb_result.to_string())


def walk_forward_validation(series, model_func, n_splits=5):
    """Walk-forward validation"""
    print(f"\n{'='*70}")
    print("Walk-Forward Validation")
    print(f"{'='*70}")
    
    n = len(series)
    test_size = n // (n_splits + 1)
    
    errors = []
    
    for i in range(n_splits):
        split_point = n - (n_splits - i) * test_size
        train = series[:split_point]
        test = series[split_point:split_point + test_size]
        
        try:
            model = model_func(train)
            forecast = model.predict(len(test))
            
            mae = mean_absolute_error(test, forecast)
            rmse = np.sqrt(mean_squared_error(test, forecast))
            mape = mean_absolute_percentage_error(test, forecast) * 100
            
            errors.append({'fold': i+1, 'mae': mae, 'rmse': rmse, 'mape': mape})
            
            print(f"\nFold {i+1}: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")
        except Exception as e:
            print(f"\nFold {i+1}: Failed - {str(e)}")
    
    if errors:
        errors_df = pd.DataFrame(errors)
        print(f"\nAverage Performance:")
        print(f"  MAE: {errors_df['mae'].mean():.4f} ± {errors_df['mae'].std():.4f}")
        print(f"  RMSE: {errors_df['rmse'].mean():.4f} ± {errors_df['rmse'].std():.4f}")
        print(f"  MAPE: {errors_df['mape'].mean():.2f}% ± {errors_df['mape'].std():.2f}%")
    
    return errors


def calculate_comprehensive_metrics(y_true, y_pred):
    """Calculate comprehensive metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    # SMAPE
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    
    # MASE
    naive_errors = np.abs(np.diff(y_true))
    mase = mae / np.mean(naive_errors) if len(naive_errors) > 0 and np.mean(naive_errors) > 0 else np.inf
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'SMAPE': smape,
        'MASE': mase,
        'R2': r2
    }


def plot_diagnostics(df, decomposition, forecast_results):
    """Create comprehensive diagnostic plots"""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
    
    # Original series
    ax1 = fig.add_subplot(gs[0, :])
    for col in df.columns[:3]:
        ax1.plot(df.index, df[col], label=col, linewidth=1.5, alpha=0.7)
    ax1.set_title('Original Time Series', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Decomposition components
    if decomposition:
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(decomposition.trend, linewidth=1.5)
        ax2.set_title('Trend Component', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(decomposition.seasonal, linewidth=1.5, color='orange')
        ax3.set_title('Seasonal Component', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    # Residuals
    ax4 = fig.add_subplot(gs[2, 0])
    if 'residuals' in forecast_results:
        ax4.plot(forecast_results['residuals'], linewidth=1, alpha=0.7)
        ax4.axhline(y=0, color='r', linestyle='--')
        ax4.set_title('Residuals', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[2, 1])
    if 'residuals' in forecast_results:
        ax5.hist(forecast_results['residuals'], bins=50, edgecolor='black', alpha=0.7)
        ax5.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax5.set_title('Residual Distribution', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
    
    # Forecast
    ax6 = fig.add_subplot(gs[3, :])
    if 'actual' in forecast_results and 'predicted' in forecast_results:
        ax6.plot(forecast_results['actual'], label='Actual', linewidth=2, color='black')
        ax6.plot(forecast_results['predicted'], label='Forecast', linewidth=2, 
                color='red', linestyle='--', alpha=0.8)
        if 'lower' in forecast_results and 'upper' in forecast_results:
            ax6.fill_between(range(len(forecast_results['lower'])),
                            forecast_results['lower'], forecast_results['upper'],
                            alpha=0.3, color='red')
        ax6.set_title('Forecast with Confidence Intervals', fontsize=12, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.savefig('solution_33_diagnostics.png', dpi=300, bbox_inches='tight')
    return fig


class SimpleModel:
    """Simple forecasting model"""
    
    def __init__(self, data):
        self.data = data
        self.fitted_values = None
    
    def fit(self):
        """Fit the model"""
        # Simple moving average
        window = min(30, len(self.data) // 4)
        self.fitted_values = pd.Series(self.data).rolling(window=window, center=False).mean()
        return self
    
    def predict(self, steps):
        """Generate forecasts"""
        last_values = self.data[-30:]
        forecast = [np.mean(last_values)] * steps
        return np.array(forecast)




def advanced_visualization(df, title_suffix=''):
    """Create advanced visualization plots"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Time series with trend
    ax1 = fig.add_subplot(gs[0, :])
    for col in df.columns[:min(3, len(df.columns))]:
        ax1.plot(df.index, df[col], label=col, linewidth=1.5, alpha=0.7)
    ax1.set_title(f'Time Series Overview {title_suffix}', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Distribution plots
    ax2 = fig.add_subplot(gs[1, 0])
    for col in df.columns[:min(3, len(df.columns))]:
        ax2.hist(df[col], bins=50, alpha=0.5, label=col, edgecolor='black')
    ax2.set_title('Value Distributions', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Box plots
    ax3 = fig.add_subplot(gs[1, 1])
    df.iloc[:, :min(3, len(df.columns))].boxplot(ax=ax3)
    ax3.set_title('Box Plots', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Correlation heatmap
    ax4 = fig.add_subplot(gs[2, 0])
    if len(df.columns) > 1:
        corr = df.iloc[:, :min(5, len(df.columns))].corr()
        im = ax4.imshow(corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax4.set_xticks(range(len(corr.columns)))
        ax4.set_yticks(range(len(corr.columns)))
        ax4.set_xticklabels(corr.columns, rotation=45)
        ax4.set_yticklabels(corr.columns)
        ax4.set_title('Correlation Matrix', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax4)
    
    # Scatter plot
    ax5 = fig.add_subplot(gs[2, 1])
    if len(df.columns) >= 2:
        ax5.scatter(df.iloc[:, 0], df.iloc[:, 1], alpha=0.5, s=20)
        ax5.set_xlabel(df.columns[0])
        ax5.set_ylabel(df.columns[1])
        ax5.set_title('Series Relationship', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
    
    plt.savefig('advanced_viz.png', dpi=300, bbox_inches='tight')
    return fig


def statistical_tests_suite(series):
    """Comprehensive statistical tests"""
    print(f"\n{'='*70}")
    print("Statistical Tests Suite")
    print(f"{'='*70}")
    
    # Basic statistics
    print("\nDescriptive Statistics:")
    print(f"  Count: {len(series)}")
    print(f"  Mean: {np.mean(series):.4f}")
    print(f"  Median: {np.median(series):.4f}")
    print(f"  Std Dev: {np.std(series):.4f}")
    print(f"  Min: {np.min(series):.4f}")
    print(f"  Max: {np.max(series):.4f}")
    print(f"  Range: {np.max(series) - np.min(series):.4f}")
    print(f"  Skewness: {stats.skew(series):.4f}")
    print(f"  Kurtosis: {stats.kurtosis(series):.4f}")
    
    # Normality tests
    print("\nNormality Tests:")
    if len(series) > 3:
        shapiro_stat, shapiro_p = stats.shapiro(series[:5000] if len(series) > 5000 else series)
        print(f"  Shapiro-Wilk: p-value={shapiro_p:.6f}")
    
    ks_stat, ks_p = stats.kstest(series, 'norm')
    print(f"  Kolmogorov-Smirnov: p-value={ks_p:.6f}")
    
    # Autocorrelation
    print("\nAutocorrelation Analysis:")
    acf_values = acf(series, nlags=min(40, len(series)//4), fft=False)
    print(f"  ACF at lag 1: {acf_values[1]:.4f}")
    print(f"  ACF at lag 7: {acf_values[min(7, len(acf_values)-1)]:.4f}")
    
    significant_lags = np.sum(np.abs(acf_values[1:]) > 1.96/np.sqrt(len(series)))
    print(f"  Significant lags (5% level): {significant_lags}")


def cross_validation_analysis(series, model_func, n_folds=5):
    """Detailed cross-validation analysis"""
    print(f"\n{'='*70}")
    print("Cross-Validation Analysis")
    print(f"{'='*70}")
    
    from sklearn.model_selection import TimeSeriesSplit
    
    tscv = TimeSeriesSplit(n_splits=n_folds)
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(series)):
        train = series[train_idx]
        test = series[test_idx]
        
        try:
            model = model_func(train)
            forecast = model.predict(len(test))
            
            mae = mean_absolute_error(test, forecast)
            rmse = np.sqrt(mean_squared_error(test, forecast))
            mape = mean_absolute_percentage_error(test, forecast) * 100
            
            fold_results.append({
                'fold': fold + 1,
                'train_size': len(train),
                'test_size': len(test),
                'mae': mae,
                'rmse': rmse,
                'mape': mape
            })
            
            print(f"\nFold {fold+1}:")
            print(f"  Train size: {len(train)}, Test size: {len(test)}")
            print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
            
        except Exception as e:
            print(f"\nFold {fold+1}: Failed - {str(e)}")
    
    if fold_results:
        results_df = pd.DataFrame(fold_results)
        print(f"\nCross-Validation Summary:")
        print(f"  Average MAE: {results_df['mae'].mean():.4f} ± {results_df['mae'].std():.4f}")
        print(f"  Average RMSE: {results_df['rmse'].mean():.4f} ± {results_df['rmse'].std():.4f}")
        print(f"  Average MAPE: {results_df['mape'].mean():.2f}% ± {results_df['mape'].std():.2f}%")
    
    return fold_results


def forecast_error_analysis(actual, predicted):
    """Detailed forecast error analysis"""
    print(f"\n{'='*70}")
    print("Forecast Error Analysis")
    print(f"{'='*70}")
    
    errors = actual - predicted
    abs_errors = np.abs(errors)
    pct_errors = 100 * abs_errors / np.abs(actual)
    
    print("\nError Statistics:")
    print(f"  Mean Error: {np.mean(errors):.4f}")
    print(f"  Mean Absolute Error: {np.mean(abs_errors):.4f}")
    print(f"  Root Mean Squared Error: {np.sqrt(np.mean(errors**2)):.4f}")
    print(f"  Mean Absolute Percentage Error: {np.mean(pct_errors):.2f}%")
    
    print("\nError Distribution:")
    print(f"  Min Error: {np.min(errors):.4f}")
    print(f"  25th Percentile: {np.percentile(errors, 25):.4f}")
    print(f"  Median Error: {np.median(errors):.4f}")
    print(f"  75th Percentile: {np.percentile(errors, 75):.4f}")
    print(f"  Max Error: {np.max(errors):.4f}")
    
    # Error autocorrelation
    if len(errors) > 10:
        error_acf = acf(errors, nlags=min(10, len(errors)//4), fft=False)
        print(f"\nError Autocorrelation:")
        print(f"  ACF at lag 1: {error_acf[1]:.4f}")
        
        if np.abs(error_acf[1]) > 1.96/np.sqrt(len(errors)):
            print("  WARNING: Significant error autocorrelation detected!")
    
    return errors


def model_comparison_framework(train_data, test_data, models_dict):
    """Framework for comparing multiple models"""
    print(f"\n{'='*70}")
    print("Model Comparison Framework")
    print(f"{'='*70}")
    
    comparison_results = []
    
    for model_name, model_func in models_dict.items():
        print(f"\nEvaluating {model_name}...")
        
        try:
            # Fit model
            model = model_func(train_data)
            forecast = model.predict(len(test_data))
            
            # Calculate metrics
            metrics = calculate_comprehensive_metrics(test_data, forecast)
            
            comparison_results.append({
                'Model': model_name,
                **metrics
            })
            
            print(f"  MAE: {metrics['MAE']:.4f}")
            print(f"  RMSE: {metrics['RMSE']:.4f}")
            print(f"  MAPE: {metrics['MAPE']:.2f}%")
            
        except Exception as e:
            print(f"  Failed: {str(e)}")
    
    if comparison_results:
        results_df = pd.DataFrame(comparison_results)
        results_df = results_df.sort_values('MAE')
        
        print(f"\nModel Ranking (by MAE):")
        print(results_df.to_string(index=False))
        
        best_model = results_df.iloc[0]['Model']
        print(f"\nBest Model: {best_model}")
    
    return comparison_results
\n\ndef main():
    """Main execution function"""
    print("="*70)
    print(f"{title.upper()}")
    print("="*70)
    
    # Generate data
    df = generate_data(n_samples=1000)
    
    # Stationarity tests
    for col in df.columns[:2]:
        perform_stationarity_tests(df[col], name=col)
    
    # STL decomposition
    decomposition = perform_stl_decomposition(df.iloc[:, 0], period=7)
    
    # ACF/PACF plots
    acf_pacf_fig = plot_acf_pacf(df.iloc[:, 0], lags=40)
    plt.savefig('solution_33_acf_pacf.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Train/test split
    train_size = int(len(df) * 0.8)
    train_data = df.iloc[:train_size, 0]
    test_data = df.iloc[train_size:, 0]
    
    # Fit model
    model = SimpleModel(train_data.values)
    model.fit()
    
    # Generate forecast
    forecast = model.predict(len(test_data))
    
    # Calculate metrics
    metrics = calculate_comprehensive_metrics(test_data.values, forecast)
    print(f"\n{'='*70}")
    print("Forecast Metrics")
    print(f"{'='*70}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Residual diagnostics
    residuals = test_data.values - forecast
    residual_diagnostics(residuals, 'SimpleModel')
    
    # Walk-forward validation
    walk_forward_validation(df.iloc[:, 0].values, 
                           lambda data: SimpleModel(data).fit(),
                           n_splits=5)
    
    # Plot diagnostics
    forecast_results = {
        'actual': test_data.values,
        'predicted': forecast,
        'residuals': residuals,
        'lower': forecast - 1.96 * np.std(residuals),
        'upper': forecast + 1.96 * np.std(residuals)
    }
    plot_diagnostics(df, decomposition, forecast_results)
    
    print("\n" + "="*70)
    print(f"{title.upper()} ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nTest MAE: {metrics['MAE']:.4f}")
    print(f"Test RMSE: {metrics['RMSE']:.4f}")
    print(f"Test MAPE: {metrics['MAPE']:.2f}%")


if __name__ == "__main__":
    main()
