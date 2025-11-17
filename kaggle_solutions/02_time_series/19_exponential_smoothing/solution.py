"""
Exponential Smoothing Methods - Comprehensive Time Series Analysis
==================================================================

This solution demonstrates various exponential smoothing techniques:
1. Simple Exponential Smoothing (SES)
2. Holt's Linear Method (Double Exponential Smoothing)
3. Holt-Winters Method (Triple Exponential Smoothing)
4. Damped trend methods
5. Additive vs multiplicative seasonality
6. Parameter optimization (alpha, beta, gamma)
7. Model selection criteria (AIC, BIC)
8. Walk-forward validation
9. Forecast accuracy comparison
10. Residual diagnostics

Dataset: Synthetic time series with trend and seasonality
Models: SES, Holt, Holt-Winters (Additive & Multiplicative)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def generate_smoothing_data(n_samples=500):
    """Generate time series data suitable for exponential smoothing"""
    print("Generating time series data...")

    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')

    # Linear trend
    trend = np.linspace(100, 200, n_samples)

    # Seasonal component (weekly pattern)
    seasonal = 15 * np.sin(2 * np.pi * np.arange(n_samples) / 7)

    # Random noise
    noise = np.random.normal(0, 5, n_samples)

    # Combine components
    values = trend + seasonal + noise

    df = pd.DataFrame({
        'date': dates,
        'value': values,
        'trend': trend,
        'seasonal': seasonal
    })
    df.set_index('date', inplace=True)

    print(f"Generated {len(df)} observations")
    print(f"Value range: [{df['value'].min():.2f}, {df['value'].max():.2f}]")

    return df


class SimpleExponentialSmoothing:
    """Manual implementation of Simple Exponential Smoothing"""

    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.level = None
        self.fitted_values = []

    def fit(self, series):
        """Fit the model"""
        self.level = series.iloc[0]
        self.fitted_values = [self.level]

        for value in series.iloc[1:]:
            self.level = self.alpha * value + (1 - self.alpha) * self.level
            self.fitted_values.append(self.level)

        return self

    def forecast(self, steps=1):
        """Generate forecasts"""
        return [self.level] * steps


class HoltLinearMethod:
    """Manual implementation of Holt's Linear Method"""

    def __init__(self, alpha=0.2, beta=0.1):
        self.alpha = alpha
        self.beta = beta
        self.level = None
        self.trend = None
        self.fitted_values = []

    def fit(self, series):
        """Fit the model"""
        self.level = series.iloc[0]
        self.trend = series.iloc[1] - series.iloc[0]
        self.fitted_values = [self.level]

        for value in series.iloc[1:]:
            prev_level = self.level
            self.level = self.alpha * value + (1 - self.alpha) * (self.level + self.trend)
            self.trend = self.beta * (self.level - prev_level) + (1 - self.beta) * self.trend
            self.fitted_values.append(self.level)

        return self

    def forecast(self, steps=1):
        """Generate forecasts"""
        forecasts = []
        for h in range(1, steps + 1):
            forecasts.append(self.level + h * self.trend)
        return forecasts


def optimize_ses_parameters(series):
    """Optimize Simple Exponential Smoothing parameters"""
    print(f"\n{'='*70}")
    print("Optimizing Simple Exponential Smoothing Parameters")
    print(f"{'='*70}")

    def objective(alpha):
        model = SimpleExponentialSmoothing(alpha=alpha[0])
        model.fit(series)
        fitted = np.array(model.fitted_values)
        mse = np.mean((series.values - fitted) ** 2)
        return mse

    # Optimize alpha
    result = minimize(objective, x0=[0.3], bounds=[(0.01, 0.99)], method='L-BFGS-B')
    optimal_alpha = result.x[0]

    print(f"\nOptimal alpha: {optimal_alpha:.4f}")
    print(f"MSE: {result.fun:.4f}")

    return optimal_alpha


def optimize_holt_parameters(series):
    """Optimize Holt's Linear Method parameters"""
    print(f"\n{'='*70}")
    print("Optimizing Holt's Linear Method Parameters")
    print(f"{'='*70}")

    def objective(params):
        alpha, beta = params
        model = HoltLinearMethod(alpha=alpha, beta=beta)
        model.fit(series)
        fitted = np.array(model.fitted_values)
        mse = np.mean((series.values - fitted) ** 2)
        return mse

    # Optimize alpha and beta
    result = minimize(objective, x0=[0.3, 0.1],
                     bounds=[(0.01, 0.99), (0.01, 0.99)],
                     method='L-BFGS-B')
    optimal_alpha, optimal_beta = result.x

    print(f"\nOptimal alpha: {optimal_alpha:.4f}")
    print(f"Optimal beta: {optimal_beta:.4f}")
    print(f"MSE: {result.fun:.4f}")

    return optimal_alpha, optimal_beta


def fit_exponential_smoothing_models(series):
    """Fit various exponential smoothing models"""
    print(f"\n{'='*70}")
    print("Fitting Exponential Smoothing Models")
    print(f"{'='*70}")

    models = {}

    # Simple Exponential Smoothing
    print("\n1. Simple Exponential Smoothing:")
    try:
        ses_model = SimpleExpSmoothing(series).fit(optimized=True)
        models['SES'] = {
            'model': ses_model,
            'aic': ses_model.aic,
            'bic': ses_model.bic,
            'params': {'alpha': ses_model.params['smoothing_level']}
        }
        print(f"   Alpha: {ses_model.params['smoothing_level']:.4f}")
        print(f"   AIC: {ses_model.aic:.2f}")
        print(f"   BIC: {ses_model.bic:.2f}")
    except Exception as e:
        print(f"   Failed: {str(e)}")

    # Holt's Linear Method
    print("\n2. Holt's Linear Method:")
    try:
        holt_model = Holt(series).fit(optimized=True)
        models['Holt'] = {
            'model': holt_model,
            'aic': holt_model.aic,
            'bic': holt_model.bic,
            'params': {
                'alpha': holt_model.params['smoothing_level'],
                'beta': holt_model.params['smoothing_trend']
            }
        }
        print(f"   Alpha: {holt_model.params['smoothing_level']:.4f}")
        print(f"   Beta: {holt_model.params['smoothing_trend']:.4f}")
        print(f"   AIC: {holt_model.aic:.2f}")
        print(f"   BIC: {holt_model.bic:.2f}")
    except Exception as e:
        print(f"   Failed: {str(e)}")

    # Holt's Damped Method
    print("\n3. Holt's Damped Method:")
    try:
        holt_damped = Holt(series, damped_trend=True).fit(optimized=True)
        models['Holt_Damped'] = {
            'model': holt_damped,
            'aic': holt_damped.aic,
            'bic': holt_damped.bic,
            'params': {
                'alpha': holt_damped.params['smoothing_level'],
                'beta': holt_damped.params['smoothing_trend'],
                'phi': holt_damped.params['damping_trend']
            }
        }
        print(f"   Alpha: {holt_damped.params['smoothing_level']:.4f}")
        print(f"   Beta: {holt_damped.params['smoothing_trend']:.4f}")
        print(f"   Phi: {holt_damped.params['damping_trend']:.4f}")
        print(f"   AIC: {holt_damped.aic:.2f}")
        print(f"   BIC: {holt_damped.bic:.2f}")
    except Exception as e:
        print(f"   Failed: {str(e)}")

    # Holt-Winters Additive
    print("\n4. Holt-Winters Additive:")
    try:
        hw_add = ExponentialSmoothing(series, seasonal_periods=7,
                                      trend='add', seasonal='add').fit(optimized=True)
        models['HW_Additive'] = {
            'model': hw_add,
            'aic': hw_add.aic,
            'bic': hw_add.bic,
            'params': {
                'alpha': hw_add.params['smoothing_level'],
                'beta': hw_add.params['smoothing_trend'],
                'gamma': hw_add.params['smoothing_seasonal']
            }
        }
        print(f"   Alpha: {hw_add.params['smoothing_level']:.4f}")
        print(f"   Beta: {hw_add.params['smoothing_trend']:.4f}")
        print(f"   Gamma: {hw_add.params['smoothing_seasonal']:.4f}")
        print(f"   AIC: {hw_add.aic:.2f}")
        print(f"   BIC: {hw_add.bic:.2f}")
    except Exception as e:
        print(f"   Failed: {str(e)}")

    # Holt-Winters Multiplicative
    print("\n5. Holt-Winters Multiplicative:")
    try:
        # Ensure all values are positive for multiplicative model
        series_positive = series - series.min() + 1
        hw_mult = ExponentialSmoothing(series_positive, seasonal_periods=7,
                                       trend='add', seasonal='mul').fit(optimized=True)
        models['HW_Multiplicative'] = {
            'model': hw_mult,
            'aic': hw_mult.aic,
            'bic': hw_mult.bic,
            'params': {
                'alpha': hw_mult.params['smoothing_level'],
                'beta': hw_mult.params['smoothing_trend'],
                'gamma': hw_mult.params['smoothing_seasonal']
            }
        }
        print(f"   Alpha: {hw_mult.params['smoothing_level']:.4f}")
        print(f"   Beta: {hw_mult.params['smoothing_trend']:.4f}")
        print(f"   Gamma: {hw_mult.params['smoothing_seasonal']:.4f}")
        print(f"   AIC: {hw_mult.aic:.2f}")
        print(f"   BIC: {hw_mult.bic:.2f}")
    except Exception as e:
        print(f"   Failed: {str(e)}")

    return models


def walk_forward_validation_ets(series, model_type='Holt', seasonal_periods=None, n_splits=5):
    """Walk-forward validation for exponential smoothing models"""
    print(f"\n{'='*70}")
    print(f"Walk-Forward Validation - {model_type}")
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
            if model_type == 'SES':
                model = SimpleExpSmoothing(train).fit(optimized=True)
            elif model_type == 'Holt':
                model = Holt(train).fit(optimized=True)
            elif model_type == 'Holt_Damped':
                model = Holt(train, damped_trend=True).fit(optimized=True)
            elif model_type == 'HW_Additive':
                model = ExponentialSmoothing(train, seasonal_periods=seasonal_periods,
                                            trend='add', seasonal='add').fit(optimized=True)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            forecast = model.forecast(steps=len(test))

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


def residual_diagnostics_ets(residuals, model_name):
    """Comprehensive residual diagnostics"""
    print(f"\n{'='*70}")
    print(f"Residual Diagnostics - {model_name}")
    print(f"{'='*70}")

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

    # Test for autocorrelation
    acf_values = acf(residuals, nlags=min(40, len(residuals)//2), fft=False)
    significant_lags = np.sum(np.abs(acf_values[1:]) > 1.96/np.sqrt(len(residuals)))
    print(f"\nAutocorrelation:")
    print(f"  Significant lags (5% level): {significant_lags}")


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    # SMAPE
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'SMAPE': smape
    }


def plot_ets_diagnostics(df, models, forecasts):
    """Create comprehensive diagnostic plots"""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

    # Original series
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df.index, df['value'], linewidth=1.5, label='Original Series')
    ax1.set_title('Time Series Data', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Model comparison - fitted values
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(df.index[:200], df['value'][:200], label='Actual', linewidth=2, alpha=0.7)
    for name, info in list(models.items())[:3]:  # Plot first 3 models
        fitted = info['model'].fittedvalues[:200]
        ax2.plot(df.index[:200], fitted, label=f'{name} Fitted', linewidth=1.5, alpha=0.7)
    ax2.set_title('Model Comparison - Fitted Values', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Residuals for best model
    best_model_name = min(models.items(), key=lambda x: x[1]['aic'])[0]
    best_model = models[best_model_name]['model']
    residuals = best_model.resid

    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(residuals, linewidth=1, alpha=0.7)
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax3.set_title(f'Residuals - {best_model_name}', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[2, 1])
    ax4.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax4.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax4.set_title('Residual Distribution', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # ACF of residuals
    ax5 = fig.add_subplot(gs[3, 0])
    plot_acf(residuals, lags=40, ax=ax5, alpha=0.05)
    ax5.set_title('ACF of Residuals', fontsize=12, fontweight='bold')

    # Forecasts comparison
    ax6 = fig.add_subplot(gs[3, 1])
    train_size = len(df) - 30
    ax6.plot(df.index[:train_size], df['value'][:train_size], label='Training', linewidth=1.5)
    ax6.plot(df.index[train_size:], df['value'][train_size:], label='Actual', linewidth=2, color='green')

    for name, forecast in list(forecasts.items())[:3]:
        ax6.plot(df.index[train_size:train_size+len(forecast)], forecast,
                label=f'{name}', linewidth=1.5, linestyle='--', alpha=0.7)

    ax6.set_title('Forecast Comparison', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.savefig('ets_diagnostics.png', dpi=300, bbox_inches='tight')
    return fig


def main():
    """Main execution function"""
    print("="*70)
    print("EXPONENTIAL SMOOTHING METHODS")
    print("="*70)

    # Generate data
    df = generate_smoothing_data(n_samples=500)
    series = df['value']

    # Optimize parameters manually
    optimal_alpha = optimize_ses_parameters(series)
    optimal_alpha_holt, optimal_beta = optimize_holt_parameters(series)

    # Fit all exponential smoothing models
    models = fit_exponential_smoothing_models(series)

    # Find best model by AIC
    best_model_name = min(models.items(), key=lambda x: x[1]['aic'])[0]
    print(f"\n{'='*70}")
    print(f"Best Model: {best_model_name} (AIC: {models[best_model_name]['aic']:.2f})")
    print(f"{'='*70}")

    # Residual diagnostics for each model
    for name, info in models.items():
        residual_diagnostics_ets(info['model'].resid, name)

    # Walk-forward validation for best models
    cv_holt = walk_forward_validation_ets(series, 'Holt', n_splits=5)
    cv_hw = walk_forward_validation_ets(series, 'HW_Additive', seasonal_periods=7, n_splits=5)

    # Final forecasts
    train_size = len(series) - 30
    train_series = series[:train_size]
    test_series = series[train_size:]

    forecasts = {}
    for name, info in models.items():
        try:
            # Refit on training data
            if name == 'SES':
                model = SimpleExpSmoothing(train_series).fit(optimized=True)
            elif name == 'Holt':
                model = Holt(train_series).fit(optimized=True)
            elif name == 'Holt_Damped':
                model = Holt(train_series, damped_trend=True).fit(optimized=True)
            elif name == 'HW_Additive':
                model = ExponentialSmoothing(train_series, seasonal_periods=7,
                                            trend='add', seasonal='add').fit(optimized=True)
            elif name == 'HW_Multiplicative':
                train_positive = train_series - train_series.min() + 1
                model = ExponentialSmoothing(train_positive, seasonal_periods=7,
                                            trend='add', seasonal='mul').fit(optimized=True)
            else:
                continue

            forecast = model.forecast(steps=30)
            forecasts[name] = forecast

            # Calculate metrics
            if name != 'HW_Multiplicative':  # Skip multiplicative for metrics (different scale)
                metrics = calculate_metrics(test_series, forecast)
                print(f"\n{name} Test Metrics:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")

        except Exception as e:
            print(f"Failed to forecast with {name}: {str(e)}")

    # Create diagnostic plots
    plot_ets_diagnostics(df, models, forecasts)

    # Model comparison summary
    print(f"\n{'='*70}")
    print("Model Comparison Summary")
    print(f"{'='*70}")
    comparison_data = []
    for name, info in models.items():
        comparison_data.append({
            'Model': name,
            'AIC': info['aic'],
            'BIC': info['bic']
        })

    comparison_df = pd.DataFrame(comparison_data).sort_values('AIC')
    print(comparison_df.to_string(index=False))

    print("\n" + "="*70)
    print("EXPONENTIAL SMOOTHING ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
