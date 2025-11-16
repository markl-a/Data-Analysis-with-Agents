"""
Kaggle Solution: Time-Based Feature Engineering
===============================================
Demonstrates extraction of temporal features from datetime data
including cyclical encoding, lag features, and rolling statistics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def generate_retail_sales_data(n_days=730):
    """Generate synthetic retail sales data with temporal patterns."""
    start_date = pd.Timestamp('2022-01-01')
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')

    data = []
    for date in dates:
        # Seasonal patterns
        month_effect = 1000 * np.sin(2 * np.pi * date.month / 12) + 2000
        dow_effect = 500 if date.dayofweek >= 5 else 0  # Weekend boost
        holiday_effect = 1500 if date.month == 12 else 0  # December holidays

        # Trend
        days_since_start = (date - start_date).days
        trend = 50 + days_since_start * 2

        # Weekly pattern
        weekly_pattern = 200 * np.sin(2 * np.pi * date.dayofweek / 7)

        # Quarter effect
        quarter_effect = {1: -100, 2: 0, 3: 100, 4: 500}[date.quarter]

        # Base sales
        sales = (
            5000 + trend + month_effect + dow_effect + holiday_effect +
            weekly_pattern + quarter_effect +
            np.random.normal(0, 300)
        )

        # Customer count
        customers = max(50, int(sales / 50 + np.random.normal(0, 10)))

        # Temperature effect (simplified)
        temp_effect = 15 * np.sin(2 * np.pi * date.month / 12) + 15
        temperature = 60 + temp_effect + np.random.normal(0, 5)

        data.append({
            'date': date,
            'sales': max(0, sales),
            'customers': customers,
            'temperature': temperature
        })

    return pd.DataFrame(data)


def extract_basic_time_features(df):
    """Extract basic datetime components."""
    df = df.copy()

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['dayofyear'] = df['date'].dt.dayofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter

    # Boolean features
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)

    # Days since epoch (for trend)
    df['days_since_start'] = (df['date'] - df['date'].min()).dt.days

    return df


def extract_cyclical_features(df):
    """Encode cyclical time features using sine/cosine."""
    df = df.copy()

    # Month (12 months)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Day of week (7 days)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    # Day of month (approximately 30 days)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 30)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 30)

    # Day of year (365 days)
    df['doy_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)

    return df


def extract_lag_features(df, target_col='sales', lags=[1, 7, 14, 30]):
    """Create lag features."""
    df = df.copy()

    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)

    return df


def extract_rolling_features(df, target_col='sales', windows=[7, 14, 30]):
    """Create rolling window statistics."""
    df = df.copy()

    for window in windows:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
        df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
        df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()

    return df


def extract_all_time_features(df):
    """Extract all time features."""
    df = extract_basic_time_features(df)
    df = extract_cyclical_features(df)
    df = extract_lag_features(df, 'sales', [1, 7, 14, 30])
    df = extract_lag_features(df, 'customers', [1, 7])
    df = extract_rolling_features(df, 'sales', [7, 14, 30])

    return df


def evaluate_feature_set(X_train, X_test, y_train, y_test, feature_set_name):
    """Train and evaluate model with feature set."""
    # Remove NaN values
    train_mask = ~X_train.isna().any(axis=1)
    test_mask = ~X_test.isna().any(axis=1)

    X_train_clean = X_train[train_mask]
    y_train_clean = y_train[train_mask]
    X_test_clean = X_test[test_mask]
    y_test_clean = y_test[test_mask]

    if len(X_train_clean) == 0 or len(X_test_clean) == 0:
        return None

    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_clean, y_train_clean)

    y_pred = model.predict(X_test_clean)

    return {
        'feature_set': feature_set_name,
        'n_features': X_train.shape[1],
        'rmse': np.sqrt(mean_squared_error(y_test_clean, y_pred)),
        'mae': mean_absolute_error(y_test_clean, y_pred),
        'r2': r2_score(y_test_clean, y_pred),
        'model': model,
        'predictions': y_pred,
        'y_test': y_test_clean
    }


def plot_results(results, df):
    """Create comprehensive visualizations."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Performance comparison
    ax1 = fig.add_subplot(gs[0, :2])
    feature_sets = [r['feature_set'] for r in results if r is not None]
    r2_scores = [r['r2'] for r in results if r is not None]
    colors = plt.cm.RdYlGn(np.array(r2_scores) / max(r2_scores))
    bars = ax1.barh(range(len(feature_sets)), r2_scores, color=colors, alpha=0.8)
    ax1.set_yticks(range(len(feature_sets)))
    ax1.set_yticklabels(feature_sets)
    ax1.set_xlabel('R² Score', fontsize=12)
    ax1.set_title('Performance by Feature Set', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    for i, (bar, score) in enumerate(zip(bars, r2_scores)):
        ax1.text(score, i, f' {score:.4f}', va='center')

    # 2. RMSE comparison
    ax2 = fig.add_subplot(gs[0, 2])
    rmse_scores = [r['rmse'] for r in results if r is not None]
    ax2.barh(range(len(feature_sets)), rmse_scores, alpha=0.7, color='coral')
    ax2.set_yticks(range(len(feature_sets)))
    ax2.set_yticklabels(feature_sets)
    ax2.set_xlabel('RMSE', fontsize=12)
    ax2.set_title('RMSE Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    # 3. Sales over time
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(df['date'], df['sales'], alpha=0.6, linewidth=1)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel('Sales', fontsize=12)
    ax3.set_title('Sales Time Series', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Seasonal pattern (monthly)
    ax4 = fig.add_subplot(gs[2, 0])
    monthly_avg = df.groupby(df['date'].dt.month)['sales'].mean()
    ax4.plot(monthly_avg.index, monthly_avg.values, marker='o', linewidth=2, markersize=8)
    ax4.set_xlabel('Month', fontsize=12)
    ax4.set_ylabel('Average Sales', fontsize=12)
    ax4.set_title('Monthly Seasonality', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(range(1, 13))

    # 5. Weekly pattern
    ax5 = fig.add_subplot(gs[2, 1])
    dow_avg = df.groupby(df['date'].dt.dayofweek)['sales'].mean()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    ax5.bar(range(7), dow_avg.values, alpha=0.7, color='steelblue')
    ax5.set_xticks(range(7))
    ax5.set_xticklabels(days)
    ax5.set_ylabel('Average Sales', fontsize=12)
    ax5.set_title('Day of Week Pattern', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Predictions (best model)
    ax6 = fig.add_subplot(gs[2, 2])
    best_result = max([r for r in results if r is not None], key=lambda x: x['r2'])
    ax6.scatter(best_result['y_test'], best_result['predictions'], alpha=0.5, s=30)
    ax6.plot([best_result['y_test'].min(), best_result['y_test'].max()],
             [best_result['y_test'].min(), best_result['y_test'].max()], 'r--', lw=2)
    ax6.set_xlabel('Actual Sales', fontsize=12)
    ax6.set_ylabel('Predicted Sales', fontsize=12)
    ax6.set_title(f'Best Model: {best_result["feature_set"]}', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/13_feature_engineering/07_time_features/time_features_analysis.png',
                dpi=300, bbox_inches='tight')
    print("Plot saved as 'time_features_analysis.png'")
    plt.show()


def main():
    """Main execution function."""
    print("=" * 80)
    print("Time-Based Feature Engineering Example")
    print("=" * 80)

    # Generate data
    print("\n1. Generating synthetic retail sales data...")
    df = generate_retail_sales_data(n_days=730)
    print(f"   Dataset shape: {df.shape}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Average daily sales: ${df['sales'].mean():.2f}")

    # Train/test split (temporal)
    print("\n2. Temporal train/test split...")
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    print(f"   Train: {len(train_df)} days")
    print(f"   Test: {len(test_df)} days")

    results = []

    # Feature Set 1: No time features (baseline)
    print("\n3. Feature Set 1: No Time Features (Baseline)...")
    X_train = train_df[['customers', 'temperature']]
    X_test = test_df[['customers', 'temperature']]
    y_train = train_df['sales']
    y_test = test_df['sales']
    result = evaluate_feature_set(X_train, X_test, y_train, y_test, "No Time Features")
    if result:
        results.append(result)
        print(f"   R²: {result['r2']:.4f}, RMSE: {result['rmse']:.2f}")

    # Feature Set 2: Basic time features
    print("\n4. Feature Set 2: Basic Time Features...")
    train_basic = extract_basic_time_features(train_df)
    test_basic = extract_basic_time_features(test_df)
    basic_features = ['customers', 'temperature', 'month', 'dayofweek', 'quarter',
                     'is_weekend', 'is_month_end', 'days_since_start']
    result = evaluate_feature_set(train_basic[basic_features], test_basic[basic_features],
                                 y_train, y_test, "Basic Time Features")
    if result:
        results.append(result)
        print(f"   R²: {result['r2']:.4f}, RMSE: {result['rmse']:.2f}")

    # Feature Set 3: Cyclical encoding
    print("\n5. Feature Set 3: Cyclical Encoding...")
    train_cyclical = extract_cyclical_features(train_basic)
    test_cyclical = extract_cyclical_features(test_basic)
    cyclical_features = ['customers', 'temperature', 'month_sin', 'month_cos',
                        'dow_sin', 'dow_cos', 'doy_sin', 'doy_cos',
                        'is_weekend', 'days_since_start']
    result = evaluate_feature_set(train_cyclical[cyclical_features],
                                 test_cyclical[cyclical_features],
                                 y_train, y_test, "Cyclical Encoding")
    if result:
        results.append(result)
        print(f"   R²: {result['r2']:.4f}, RMSE: {result['rmse']:.2f}")

    # Feature Set 4: Lag features
    print("\n6. Feature Set 4: Lag Features...")
    train_lag = extract_lag_features(train_cyclical, 'sales', [1, 7, 14, 30])
    test_lag = extract_lag_features(test_cyclical, 'sales', [1, 7, 14, 30])
    lag_features = cyclical_features + ['sales_lag_1', 'sales_lag_7', 'sales_lag_14', 'sales_lag_30']
    result = evaluate_feature_set(train_lag[lag_features], test_lag[lag_features],
                                 y_train, y_test, "With Lag Features")
    if result:
        results.append(result)
        print(f"   R²: {result['r2']:.4f}, RMSE: {result['rmse']:.2f}")

    # Feature Set 5: Rolling statistics
    print("\n7. Feature Set 5: Rolling Statistics...")
    train_rolling = extract_rolling_features(train_lag, 'sales', [7, 14, 30])
    test_rolling = extract_rolling_features(test_lag, 'sales', [7, 14, 30])
    rolling_features = lag_features + ['sales_rolling_mean_7', 'sales_rolling_std_7',
                                       'sales_rolling_mean_14', 'sales_rolling_mean_30']
    result = evaluate_feature_set(train_rolling[rolling_features],
                                 test_rolling[rolling_features],
                                 y_train, y_test, "With Rolling Features")
    if result:
        results.append(result)
        print(f"   R²: {result['r2']:.4f}, RMSE: {result['rmse']:.2f}")

    # Summary
    print("\n8. Results Summary:")
    print("-" * 80)
    print(f"{'Feature Set':<30} {'Features':<12} {'R²':<12} {'RMSE':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['feature_set']:<30} {r['n_features']:<12} {r['r2']:<12.4f} {r['rmse']:<12.2f}")

    # Best feature set
    print("\n9. Best Feature Set:")
    best_result = max(results, key=lambda x: x['r2'])
    baseline_result = results[0]
    print(f"   Feature Set: {best_result['feature_set']}")
    print(f"   R²: {best_result['r2']:.4f}")
    print(f"   RMSE: ${best_result['rmse']:.2f}")
    print(f"\n   Improvement over baseline:")
    print(f"   R² improvement: {((best_result['r2'] - baseline_result['r2']) / baseline_result['r2'] * 100):.2f}%")
    print(f"   RMSE reduction: {((baseline_result['rmse'] - best_result['rmse']) / baseline_result['rmse'] * 100):.2f}%")

    # Feature importance
    print("\n10. Top 10 Most Important Features:")
    feature_names = rolling_features
    best_model = best_result['model']
    if hasattr(best_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        for idx, row in importance_df.head(10).iterrows():
            print(f"    {row['feature']:<30} {row['importance']:.4f}")

    # Visualizations
    print("\n11. Creating visualizations...")
    plot_results(results, df)

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
