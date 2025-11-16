#!/usr/bin/env python3
"""
Seasonal Sales Decomposition Analysis
======================================
Analyzes retail sales with strong seasonal patterns using decomposition techniques.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


def generate_seasonal_sales_data(n_years=3):
    """
    Generate synthetic retail sales data with strong seasonal patterns.

    Includes:
    - Yearly trend (growth)
    - Strong quarterly seasonality
    - Monthly effects (holiday peaks)
    - Weekly patterns
    - Random fluctuations
    """
    # Generate weekly data
    n_weeks = n_years * 52
    dates = pd.date_range(start='2021-01-01', periods=n_weeks, freq='W')

    # Base sales level
    base_sales = 100000

    # Trend component (5% annual growth)
    weekly_growth = (1.05 ** (1/52)) - 1
    trend = base_sales * np.cumprod(np.ones(n_weeks) * (1 + weekly_growth))

    # Quarterly seasonality (strong holiday season in Q4)
    week_of_year = (dates.dayofyear // 7) % 52
    quarterly_pattern = np.where(
        week_of_year < 13, -15000,  # Q1: Post-holiday slump
        np.where(week_of_year < 26, -5000,  # Q2: Spring, moderate
                 np.where(week_of_year < 39, 5000,  # Q3: Summer, above average
                          25000))  # Q4: Holiday season, peak
    )

    # Monthly effects (specific holidays)
    # Black Friday / Cyber Monday (week 47-48)
    # Back to school (week 32-35)
    # Valentine's Day (week 6-7)
    holiday_boost = np.zeros(n_weeks)
    for year in range(n_years):
        base_week = year * 52
        holiday_boost[base_week + 6:base_week + 8] += 8000  # Valentine's
        holiday_boost[base_week + 32:base_week + 36] += 12000  # Back to school
        holiday_boost[base_week + 47:base_week + 49] += 30000  # Black Friday

    # Cyclical component (economic cycles)
    cycle = 10000 * np.sin(2 * np.pi * np.arange(n_weeks) / 104)  # 2-year cycle

    # Random noise
    noise = np.random.normal(0, 5000, n_weeks)

    # Combine all components
    sales = trend + quarterly_pattern + holiday_boost + cycle + noise

    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'week_of_year': week_of_year,
        'quarter': dates.quarter,
        'month': dates.month,
        'year': dates.year
    })

    return df


def analyze_seasonality(df):
    """Perform detailed seasonality analysis."""
    results = {}

    # By quarter
    quarterly_stats = df.groupby('quarter')['sales'].agg(['mean', 'std', 'min', 'max'])
    results['quarterly'] = quarterly_stats

    # By month
    monthly_stats = df.groupby('month')['sales'].agg(['mean', 'std', 'min', 'max'])
    results['monthly'] = monthly_stats

    # Year-over-year growth
    yearly_stats = df.groupby('year')['sales'].agg(['mean', 'sum'])
    results['yearly'] = yearly_stats

    return results


def main():
    """Main execution function."""
    print("=" * 80)
    print("SEASONAL SALES DECOMPOSITION ANALYSIS")
    print("=" * 80)

    # Generate data
    print("\n1. Generating synthetic seasonal sales data...")
    df = generate_seasonal_sales_data(n_years=3)
    print(f"   Generated {len(df)} weeks of data ({len(df)//52} years)")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Sales range: ${df['sales'].min():,.2f} - ${df['sales'].max():,.2f}")
    print(f"   Mean sales: ${df['sales'].mean():,.2f}")

    # Set date as index
    df.set_index('date', inplace=True)

    # Analyze seasonality patterns
    print("\n2. Analyzing seasonal patterns...")
    seasonal_results = analyze_seasonality(df)

    print("\n   Average Sales by Quarter:")
    for quarter, row in seasonal_results['quarterly'].iterrows():
        print(f"      Q{quarter}: ${row['mean']:,.2f} (±${row['std']:,.2f})")

    print("\n   Average Sales by Month:")
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for month, row in seasonal_results['monthly'].iterrows():
        print(f"      {month_names[month-1]}: ${row['mean']:,.2f}")

    print("\n   Total Sales by Year:")
    for year, row in seasonal_results['yearly'].iterrows():
        print(f"      {int(year)}: ${row['sum']:,.2f} (avg: ${row['mean']:,.2f}/week)")

    # Classical decomposition
    print("\n3. Performing classical seasonal decomposition...")
    decomposition = seasonal_decompose(df['sales'], model='additive', period=52)

    # STL decomposition
    print("\n4. Performing STL (Seasonal-Trend decomposition using LOESS)...")
    stl = STL(df['sales'], period=52, seasonal=53)
    stl_result = stl.fit()

    # Calculate component statistics
    print("\n5. Decomposition component analysis:")
    print(f"   Trend range: ${stl_result.trend.min():,.2f} - ${stl_result.trend.max():,.2f}")
    print(f"   Seasonal range: ${stl_result.seasonal.min():,.2f} - ${stl_result.seasonal.max():,.2f}")
    print(f"   Residual std: ${stl_result.resid.std():,.2f}")

    # Strength of trend and seasonality
    var_residual = np.var(stl_result.resid)
    var_trend_resid = np.var(stl_result.trend + stl_result.resid)
    var_seasonal_resid = np.var(stl_result.seasonal + stl_result.resid)

    strength_trend = max(0, 1 - var_residual / var_trend_resid)
    strength_seasonal = max(0, 1 - var_residual / var_seasonal_resid)

    print(f"\n   Strength of trend: {strength_trend:.4f}")
    print(f"   Strength of seasonality: {strength_seasonal:.4f}")

    # Split data for forecasting
    train_size = int(len(df) * 0.85)
    train_df = df[:train_size]
    test_df = df[train_size:]

    print(f"\n6. Training forecasting models...")
    print(f"   Training set: {len(train_df)} weeks")
    print(f"   Test set: {len(test_df)} weeks")

    # Holt-Winters with multiplicative seasonality
    hw_model = ExponentialSmoothing(
        train_df['sales'],
        seasonal_periods=52,
        trend='add',
        seasonal='mul',
        damped_trend=True
    ).fit()

    # Generate forecasts
    forecast_steps = len(test_df)
    forecast_hw = hw_model.forecast(steps=forecast_steps)

    # Naive seasonal forecast (last year same week)
    naive_seasonal = []
    for i in range(forecast_steps):
        if i < 52:
            # Use same week from last year in training data
            naive_seasonal.append(train_df['sales'].iloc[-(52-i)])
        else:
            # Use previously forecasted values
            naive_seasonal.append(naive_seasonal[i-52])
    naive_seasonal = np.array(naive_seasonal)

    # Evaluate models
    print("\n7. Model Evaluation:")

    # Holt-Winters
    mae_hw = mean_absolute_error(test_df['sales'], forecast_hw)
    rmse_hw = np.sqrt(mean_squared_error(test_df['sales'], forecast_hw))
    mape_hw = np.mean(np.abs((test_df['sales'] - forecast_hw) / test_df['sales'])) * 100
    r2_hw = r2_score(test_df['sales'], forecast_hw)

    print("\n   Holt-Winters (Damped Trend + Multiplicative Seasonal):")
    print(f"      MAE: ${mae_hw:,.2f}")
    print(f"      RMSE: ${rmse_hw:,.2f}")
    print(f"      MAPE: {mape_hw:.2f}%")
    print(f"      R²: {r2_hw:.4f}")

    # Naive seasonal
    mae_naive = mean_absolute_error(test_df['sales'], naive_seasonal)
    rmse_naive = np.sqrt(mean_squared_error(test_df['sales'], naive_seasonal))
    mape_naive = np.mean(np.abs((test_df['sales'] - naive_seasonal) / test_df['sales'])) * 100
    r2_naive = r2_score(test_df['sales'], naive_seasonal)

    print("\n   Naive Seasonal (Last Year Same Week):")
    print(f"      MAE: ${mae_naive:,.2f}")
    print(f"      RMSE: ${rmse_naive:,.2f}")
    print(f"      MAPE: {mape_naive:.2f}%")
    print(f"      R²: {r2_naive:.4f}")

    # Visualization
    print("\n8. Creating comprehensive visualizations...")
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.25)

    # Plot 1: Original time series
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df.index, df['sales'], linewidth=1.5, color='darkblue')
    ax1.set_title('Weekly Sales Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Sales ($)')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(train_df.index[-1], color='red', linestyle='--', alpha=0.5, label='Train/Test Split')
    ax1.legend()

    # Plot 2: STL Trend
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(stl_result.trend.index, stl_result.trend, linewidth=2, color='darkgreen')
    ax2.set_title('Trend Component (STL)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Trend')
    ax2.grid(True, alpha=0.3)

    # Plot 3: STL Seasonal
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(stl_result.seasonal.index[:104], stl_result.seasonal[:104], linewidth=2, color='darkorange')
    ax3.set_title('Seasonal Component (First 2 Years)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Seasonal')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Residuals
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(stl_result.resid.index, stl_result.resid, linewidth=1, color='gray', alpha=0.7)
    ax4.axhline(0, color='red', linestyle='--', linewidth=1)
    ax4.set_title('Residuals (STL)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Residuals')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Residual histogram
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.hist(stl_result.resid, bins=30, color='gray', edgecolor='black', alpha=0.7)
    ax5.set_title('Residual Distribution', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Residual Value')
    ax5.set_ylabel('Frequency')
    ax5.grid(True, alpha=0.3, axis='y')

    # Plot 6: Quarterly box plot
    ax6 = fig.add_subplot(gs[3, 0])
    df.boxplot(column='sales', by='quarter', ax=ax6)
    ax6.set_title('Sales Distribution by Quarter', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Quarter')
    ax6.set_ylabel('Sales ($)')
    plt.sca(ax6)
    plt.xticks([1, 2, 3, 4], ['Q1', 'Q2', 'Q3', 'Q4'])

    # Plot 7: Monthly average
    ax7 = fig.add_subplot(gs[3, 1])
    monthly_avg = df.groupby('month')['sales'].mean()
    ax7.bar(range(1, 13), monthly_avg.values, color='steelblue')
    ax7.set_title('Average Sales by Month', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Month')
    ax7.set_ylabel('Average Sales ($)')
    ax7.set_xticks(range(1, 13))
    ax7.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax7.grid(True, alpha=0.3, axis='y')

    # Plot 8: Forecast comparison
    ax8 = fig.add_subplot(gs[4, :])
    # Show last 30 weeks of training + all test
    display_start = -30 - len(test_df)
    ax8.plot(df.index[display_start:train_size], df['sales'].iloc[display_start:train_size],
             label='Training Data', linewidth=2, color='blue')
    ax8.plot(test_df.index, test_df['sales'], label='Actual', linewidth=2, color='green')
    ax8.plot(test_df.index, forecast_hw, label='Holt-Winters Forecast',
             linewidth=2, linestyle='--', color='red')
    ax8.plot(test_df.index, naive_seasonal, label='Naive Seasonal',
             linewidth=2, linestyle=':', color='orange', alpha=0.7)
    ax8.set_title('Sales Forecast Comparison', fontsize=14, fontweight='bold')
    ax8.set_xlabel('Date')
    ax8.set_ylabel('Sales ($)')
    ax8.legend(loc='upper left')
    ax8.grid(True, alpha=0.3)
    ax8.axvline(train_df.index[-1], color='black', linestyle='--', alpha=0.3, linewidth=1)

    plt.savefig('seasonal_sales_analysis.png', dpi=300, bbox_inches='tight')
    print("   Saved: seasonal_sales_analysis.png")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
