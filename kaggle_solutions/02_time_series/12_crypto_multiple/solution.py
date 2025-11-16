#!/usr/bin/env python3
"""
Multi-Cryptocurrency Price Prediction
======================================
Predicts prices for multiple cryptocurrencies using VAR and correlation analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


def generate_crypto_data(n_days=365):
    """
    Generate synthetic multi-cryptocurrency price data.

    Includes:
    - Bitcoin (BTC) - market leader
    - Ethereum (ETH) - correlated with BTC
    - Litecoin (LTC) - follows BTC trends
    - Ripple (XRP) - partially independent
    """
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')

    # BTC: Base cryptocurrency with trend and cycles
    btc_trend = np.linspace(30000, 45000, n_days)
    btc_cycle = 5000 * np.sin(2 * np.pi * np.arange(n_days) / 90)
    btc_volatility = np.random.normal(0, 1500, n_days)
    btc_jump = np.zeros(n_days)
    # Add occasional jumps
    jump_days = np.random.choice(n_days, size=10, replace=False)
    btc_jump[jump_days] = np.random.normal(0, 3000, 10)
    btc_price = btc_trend + btc_cycle + btc_volatility + btc_jump.cumsum()

    # ETH: Highly correlated with BTC (0.8 correlation)
    eth_base = btc_price * 0.06  # Roughly 6% of BTC price
    eth_independent = 100 * np.sin(2 * np.pi * np.arange(n_days) / 60)
    eth_volatility = np.random.normal(0, 80, n_days)
    eth_price = eth_base + eth_independent + eth_volatility

    # LTC: Moderately correlated with BTC (0.6 correlation)
    ltc_base = btc_price * 0.003  # Roughly 0.3% of BTC price
    ltc_independent = 10 * np.sin(2 * np.pi * np.arange(n_days) / 45)
    ltc_volatility = np.random.normal(0, 5, n_days)
    ltc_price = ltc_base + ltc_independent + ltc_volatility

    # XRP: Less correlated with BTC (0.4 correlation)
    xrp_trend = np.linspace(0.40, 0.55, n_days)
    xrp_cycle = 0.1 * np.sin(2 * np.pi * np.arange(n_days) / 120)
    xrp_btc_influence = (btc_price - btc_price.mean()) * 0.000003
    xrp_volatility = np.random.normal(0, 0.02, n_days)
    xrp_price = xrp_trend + xrp_cycle + xrp_btc_influence + xrp_volatility

    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'btc_price': btc_price,
        'eth_price': eth_price,
        'ltc_price': ltc_price,
        'xrp_price': xrp_price
    })

    return df


def calculate_returns(df):
    """Calculate log returns for each cryptocurrency."""
    returns_df = df.copy()
    for col in ['btc_price', 'eth_price', 'ltc_price', 'xrp_price']:
        returns_df[col.replace('_price', '_return')] = np.log(df[col] / df[col].shift(1))

    return returns_df.dropna()


def check_stationarity(series, name):
    """Perform Augmented Dickey-Fuller test."""
    result = adfuller(series.dropna())
    print(f"   {name}:")
    print(f"      ADF Statistic: {result[0]:.4f}")
    print(f"      p-value: {result[1]:.4f}")
    print(f"      Stationary: {'Yes' if result[1] < 0.05 else 'No'}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("MULTI-CRYPTOCURRENCY PRICE PREDICTION")
    print("=" * 80)

    # Generate data
    print("\n1. Generating synthetic cryptocurrency data...")
    df = generate_crypto_data(n_days=365)
    print(f"   Generated {len(df)} days of data")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\n   Price ranges:")
    print(f"      BTC: ${df['btc_price'].min():.2f} - ${df['btc_price'].max():.2f}")
    print(f"      ETH: ${df['eth_price'].min():.2f} - ${df['eth_price'].max():.2f}")
    print(f"      LTC: ${df['ltc_price'].min():.2f} - ${df['ltc_price'].max():.2f}")
    print(f"      XRP: ${df['xrp_price'].min():.4f} - ${df['xrp_price'].max():.4f}")

    # Set date as index
    df.set_index('date', inplace=True)

    # Calculate returns
    print("\n2. Calculating log returns...")
    returns_df = calculate_returns(df)

    # Correlation analysis
    print("\n3. Correlation Analysis (Prices):")
    price_corr = df[['btc_price', 'eth_price', 'ltc_price', 'xrp_price']].corr()
    print(price_corr.round(3))

    print("\n   Correlation Analysis (Returns):")
    return_cols = ['btc_return', 'eth_return', 'ltc_return', 'xrp_return']
    return_corr = returns_df[return_cols].corr()
    print(return_corr.round(3))

    # Stationarity tests
    print("\n4. Stationarity Tests (Returns):")
    for col in return_cols:
        check_stationarity(returns_df[col], col)

    # Prepare data for VAR model
    print("\n5. Preparing Vector Autoregression (VAR) model...")
    var_data = returns_df[return_cols].dropna()

    # Split data
    train_size = int(len(var_data) * 0.8)
    train_data = var_data[:train_size]
    test_data = var_data[train_size:]

    print(f"   Training set: {len(train_data)} days")
    print(f"   Test set: {len(test_data)} days")

    # Fit VAR model
    print("\n6. Fitting VAR model...")
    model = VAR(train_data)

    # Select optimal lag order
    lag_order = model.select_order(maxlags=10)
    optimal_lag = lag_order.aic
    print(f"   Optimal lag order (AIC): {optimal_lag}")

    # Fit model with optimal lag
    var_model = model.fit(optimal_lag)
    print(f"   Model fitted successfully")

    # Forecast
    print("\n7. Generating forecasts...")
    forecast_steps = len(test_data)
    forecast = var_model.forecast(train_data.values[-optimal_lag:], steps=forecast_steps)

    # Create forecast DataFrame
    forecast_df = pd.DataFrame(
        forecast,
        index=test_data.index,
        columns=return_cols
    )

    # Calculate metrics for each cryptocurrency
    print("\n8. Model Evaluation (Returns):")
    for i, col in enumerate(return_cols):
        mae = mean_absolute_error(test_data[col], forecast_df[col])
        rmse = np.sqrt(mean_squared_error(test_data[col], forecast_df[col]))
        crypto_name = col.replace('_return', '').upper()
        print(f"   {crypto_name}:")
        print(f"      MAE: {mae:.6f}")
        print(f"      RMSE: {rmse:.6f}")

    # Convert returns back to prices for visualization
    print("\n9. Converting returns to price predictions...")
    last_train_prices = df.iloc[train_size - 1][['btc_price', 'eth_price', 'ltc_price', 'xrp_price']]

    predicted_prices = pd.DataFrame(index=test_data.index)
    actual_prices = df.iloc[train_size:][['btc_price', 'eth_price', 'ltc_price', 'xrp_price']]

    for i, crypto in enumerate(['btc', 'eth', 'ltc', 'xrp']):
        # Reconstruct prices from returns
        cumulative_returns = forecast_df[f'{crypto}_return'].cumsum()
        predicted_prices[f'{crypto}_price'] = last_train_prices[f'{crypto}_price'] * np.exp(cumulative_returns)

    # Price prediction metrics
    print("\n10. Price Prediction Accuracy:")
    for crypto in ['btc', 'eth', 'ltc', 'xrp']:
        price_col = f'{crypto}_price'
        mape = np.mean(np.abs((actual_prices[price_col] - predicted_prices[price_col]) / actual_prices[price_col])) * 100
        print(f"   {crypto.upper()}:")
        print(f"      MAPE: {mape:.2f}%")

    # Granger causality test
    print("\n11. Granger Causality Tests (BTC -> others):")
    for target in ['eth_return', 'ltc_return', 'xrp_return']:
        print(f"\n   BTC -> {target.replace('_return', '').upper()}:")
        try:
            test_data_gc = var_data[['btc_return', target]].dropna()
            gc_result = grangercausalitytests(test_data_gc, maxlag=5, verbose=False)
            # Get p-value for lag 1
            p_value = gc_result[1][0]['ssr_ftest'][1]
            print(f"      p-value (lag 1): {p_value:.4f}")
            print(f"      Significant: {'Yes' if p_value < 0.05 else 'No'}")
        except Exception as e:
            print(f"      Could not perform test: {str(e)}")

    # Visualization
    print("\n12. Creating visualizations...")
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    # Plot 1: All cryptocurrency prices
    ax1 = fig.add_subplot(gs[0, :])
    for col in ['btc_price', 'eth_price', 'ltc_price', 'xrp_price']:
        normalized = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        ax1.plot(df.index, normalized, label=col.replace('_price', '').upper(), linewidth=2)
    ax1.set_title('Normalized Cryptocurrency Prices', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Normalized Price')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Correlation heatmap (prices)
    ax2 = fig.add_subplot(gs[1, 0])
    sns.heatmap(price_corr, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, ax=ax2, cbar_kws={'shrink': 0.8})
    ax2.set_title('Price Correlation Matrix', fontsize=12, fontweight='bold')

    # Plot 3: Correlation heatmap (returns)
    ax3 = fig.add_subplot(gs[1, 1])
    sns.heatmap(return_corr, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, ax=ax3, cbar_kws={'shrink': 0.8})
    ax3.set_title('Return Correlation Matrix', fontsize=12, fontweight='bold')

    # Plot 4: Returns distribution
    ax4 = fig.add_subplot(gs[1, 2])
    for col in return_cols:
        ax4.hist(returns_df[col], bins=50, alpha=0.5, label=col.replace('_return', '').upper())
    ax4.set_title('Returns Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Log Return')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plots 5-8: Individual predictions
    cryptos = ['btc', 'eth', 'ltc', 'xrp']
    for idx, crypto in enumerate(cryptos):
        ax = fig.add_subplot(gs[2 + idx // 2, idx % 2])
        price_col = f'{crypto}_price'

        ax.plot(actual_prices.index, actual_prices[price_col],
                label='Actual', linewidth=2, color='blue')
        ax.plot(predicted_prices.index, predicted_prices[price_col],
                label='Predicted', linewidth=2, color='red', alpha=0.7)
        ax.set_title(f'{crypto.upper()} Price Forecast', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 9: Forecast errors
    ax9 = fig.add_subplot(gs[3, 2])
    errors = {}
    for crypto in cryptos:
        price_col = f'{crypto}_price'
        errors[crypto.upper()] = np.abs(actual_prices[price_col] - predicted_prices[price_col]).mean()

    ax9.bar(errors.keys(), errors.values())
    ax9.set_title('Mean Absolute Error by Cryptocurrency', fontsize=12, fontweight='bold')
    ax9.set_ylabel('MAE ($)')
    ax9.grid(True, alpha=0.3, axis='y')

    plt.savefig('crypto_multiple_forecast.png', dpi=300, bbox_inches='tight')
    print("   Saved: crypto_multiple_forecast.png")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
