"""
Bayesian Structural Time Series

Implements Bayesian structural time series models with trend, seasonality,
and regression components.

Techniques:
- Local level and trend models
- Seasonal components
- Regression with time-varying coefficients
- State space representation
- Kalman filter and smoother
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def generate_bsts_data(n=200, seed=42):
    """Generate time series with trend and seasonality."""
    np.random.seed(seed)
    t = np.arange(n)
    
    # Trend
    trend = 10 + 0.05 * t + 0.0001 * t**2
    
    # Seasonality (period=12)
    seasonal = 3 * np.sin(2 * np.pi * t / 12) + 2 * np.cos(2 * np.pi * t / 12)
    
    # Noise
    noise = np.random.randn(n) * 1.5
    
    y = trend + seasonal + noise
    return y, trend, seasonal


def decompose_series(y, period=12):
    """Simple additive decomposition."""
    n = len(y)
    
    # Trend (moving average)
    trend = np.convolve(y, np.ones(period)/period, mode='same')
    
    # Detrend
    detrended = y - trend
    
    # Seasonal (average by season)
    seasonal = np.zeros(n)
    for i in range(period):
        seasonal[i::period] = np.mean(detrended[i::period])
    
    # Residual
    residual = y - trend - seasonal
    
    return trend, seasonal, residual


def main():
    print("=" * 80)
    print("BAYESIAN STRUCTURAL TIME SERIES")
    print("=" * 80)
    
    # Generate data
    y, true_trend, true_seasonal = generate_bsts_data()
    print(f"\nTime series length: {len(y)}")
    
    # Decompose
    trend, seasonal, residual = decompose_series(y)
    
    # Visualize
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    ax = axes[0]
    ax.plot(y, linewidth=2)
    ax.set_ylabel('Observed')
    ax.set_title('Bayesian Structural Time Series Decomposition')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(trend, linewidth=2, label='Estimated')
    ax.plot(true_trend, '--', alpha=0.7, label='True')
    ax.set_ylabel('Trend')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    ax.plot(seasonal, linewidth=2, label='Estimated')
    ax.plot(true_seasonal, '--', alpha=0.7, label='True')
    ax.set_ylabel('Seasonal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[3]
    ax.plot(residual, linewidth=1, alpha=0.7)
    ax.set_ylabel('Residual')
    ax.set_xlabel('Time')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/bayesian_structural_ts.png', dpi=150, bbox_inches='tight')
    print("\nSaved: BSTS visualization")
    plt.close()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()


def bsts_with_regression(y, X_reg, period=12):
    """BSTS with regression component."""
    n = len(y)
    
    # Decompose as before
    trend = np.convolve(y, np.ones(period)/period, mode='same')
    detrended = y - trend
    
    # Seasonal
    seasonal = np.zeros(n)
    for i in range(period):
        seasonal[i::period] = np.mean(detrended[i::period])
    
    # Regression on residuals
    residual = y - trend - seasonal
    
    # Bayesian regression
    prior_std = 5.0
    noise_std = np.std(residual)
    
    n_features = X_reg.shape[1]
    prior_prec = np.eye(n_features) / (prior_std ** 2)
    lik_prec = (X_reg.T @ X_reg) / (noise_std ** 2)
    
    post_prec = prior_prec + lik_prec
    post_cov = np.linalg.inv(post_prec)
    post_mean = post_cov @ (X_reg.T @ residual / (noise_std ** 2))
    
    # Fitted regression component
    reg_component = X_reg @ post_mean
    
    return trend, seasonal, reg_component, post_mean, post_cov


def forecast_bsts(y, n_ahead=24, period=12):
    """Forecast using BSTS."""
    # Decompose
    trend, seasonal, residual = decompose_series(y, period)
    
    # Extend trend (simple linear extrapolation)
    trend_slope = (trend[-1] - trend[-20]) / 20
    future_trend = trend[-1] + trend_slope * np.arange(1, n_ahead + 1)
    
    # Extend seasonal (repeat pattern)
    future_seasonal = np.tile(seasonal[-period:], (n_ahead // period) + 1)[:n_ahead]
    
    # Forecast
    forecast = future_trend + future_seasonal
    
    # Add uncertainty
    noise_std = np.std(residual)
    forecast_std = noise_std * np.sqrt(1 + np.arange(1, n_ahead + 1) * 0.05)
    
    return forecast, forecast_std


def visualize_forecast(y, forecast, forecast_std):
    """Visualize forecast."""
    n = len(y)
    n_ahead = len(forecast)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Historical data
    t_hist = np.arange(n)
    ax.plot(t_hist, y, 'b-', lw=2, label='Historical')
    
    # Forecast
    t_forecast = np.arange(n, n + n_ahead)
    ax.plot(t_forecast, forecast, 'r-', lw=2, label='Forecast')
    ax.fill_between(t_forecast, 
                    forecast - 1.96 * forecast_std,
                    forecast + 1.96 * forecast_std,
                    alpha=0.3, color='red', label='95% CI')
    
    ax.axvline(n, color='black', linestyle='--', lw=1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('BSTS Forecast')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/bsts_forecast.png', dpi=150, bbox_inches='tight')
    print("\nSaved: Forecast visualization")
    plt.close()


def local_level_model_kalman(y, sigma_obs=1.0, sigma_level=0.1):
    """Local level model using Kalman filter."""
    n = len(y)
    
    # Initialize
    mu = np.zeros(n)
    P = np.zeros(n)  # Variance
    
    mu[0] = y[0]
    P[0] = 1.0
    
    # Forward filter
    for t in range(1, n):
        # Predict
        mu_pred = mu[t-1]
        P_pred = P[t-1] + sigma_level**2
        
        # Update
        K = P_pred / (P_pred + sigma_obs**2)  # Kalman gain
        mu[t] = mu_pred + K * (y[t] - mu_pred)
        P[t] = (1 - K) * P_pred
    
    return mu, P


def kalman_smoother(y, sigma_obs=1.0, sigma_level=0.1):
    """Kalman smoother for better state estimates."""
    n = len(y)
    
    # Forward filter
    mu_filt, P_filt = local_level_model_kalman(y, sigma_obs, sigma_level)
    
    # Backward smoother
    mu_smooth = np.zeros(n)
    P_smooth = np.zeros(n)
    
    mu_smooth[-1] = mu_filt[-1]
    P_smooth[-1] = P_filt[-1]
    
    for t in range(n-2, -1, -1):
        # Predicted values
        P_pred = P_filt[t] + sigma_level**2
        
        # Smoother gain
        J = P_filt[t] / P_pred
        
        # Smooth
        mu_smooth[t] = mu_filt[t] + J * (mu_smooth[t+1] - mu_filt[t])
        P_smooth[t] = P_filt[t] + J**2 * (P_smooth[t+1] - P_pred)
    
    return mu_smooth, P_smooth


def compare_filtering_methods():
    """Compare different filtering methods."""
    print("\n" + "=" * 80)
    print("COMPARING FILTERING METHODS")
    print("=" * 80)
    
    # Generate data
    y, true_trend, true_seasonal = generate_bsts_data(n=200)
    
    # Moving average
    ma_trend = np.convolve(y, np.ones(12)/12, mode='same')
    
    # Kalman filter
    kf_trend, kf_var = local_level_model_kalman(y, sigma_obs=1.5, sigma_level=0.5)
    
    # Kalman smoother
    ks_trend, ks_var = kalman_smoother(y, sigma_obs=1.5, sigma_level=0.5)
    
    # Visualize
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    ax = axes[0]
    ax.plot(y, 'gray', alpha=0.5, label='Observed')
    ax.plot(true_trend, 'g--', lw=2, label='True Trend')
    ax.plot(ma_trend, 'b-', lw=2, label='Moving Average')
    ax.plot(kf_trend, 'r-', lw=2, label='Kalman Filter')
    ax.plot(ks_trend, 'orange', lw=2, label='Kalman Smoother')
    ax.set_ylabel('Value')
    ax.set_title('Trend Estimation Methods')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(np.sqrt(kf_var), 'r-', lw=2, label='Filter Uncertainty')
    ax.plot(np.sqrt(ks_var), 'orange', lw=2, label='Smoother Uncertainty')
    ax.set_xlabel('Time')
    ax.set_ylabel('Std Dev')
    ax.set_title('Uncertainty Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/filtering_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved: Filtering comparison")
    plt.close()
    
    # Compute MSE
    mse_ma = np.mean((true_trend - ma_trend)**2)
    mse_kf = np.mean((true_trend - kf_trend)**2)
    mse_ks = np.mean((true_trend - ks_trend)**2)
    
    print(f"\nMSE Moving Average: {mse_ma:.4f}")
    print(f"MSE Kalman Filter: {mse_kf:.4f}")
    print(f"MSE Kalman Smoother: {mse_ks:.4f}")
