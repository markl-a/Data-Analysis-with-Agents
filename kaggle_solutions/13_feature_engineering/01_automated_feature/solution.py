"""
Kaggle Solution: Automated Feature Generation
==============================================
Demonstrates automated feature engineering using various techniques including
polynomial features, aggregations, and mathematical transformations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def generate_sales_data(n_samples=2000):
    """
    Generate synthetic sales data suitable for automated feature engineering.
    """
    # Base features
    price = np.random.uniform(10, 200, n_samples)
    quantity = np.random.uniform(1, 100, n_samples)
    discount = np.random.uniform(0, 0.3, n_samples)
    advertising_spend = np.random.uniform(100, 10000, n_samples)
    competitor_price = price * np.random.uniform(0.8, 1.2, n_samples)
    day_of_week = np.random.randint(0, 7, n_samples)
    month = np.random.randint(1, 13, n_samples)

    # Store category
    store_category = np.random.choice(['A', 'B', 'C'], n_samples)
    category_multiplier = {'A': 1.2, 'B': 1.0, 'C': 0.8}
    category_effect = np.array([category_multiplier[c] for c in store_category])

    # Complex target with interactions
    revenue = (
        price * quantity * (1 - discount) * category_effect +
        0.1 * advertising_spend +
        50 * (price < competitor_price).astype(float) +
        30 * np.sin(month / 12 * 2 * np.pi) +
        20 * (day_of_week >= 5).astype(float) +
        np.random.normal(0, 50, n_samples)
    )

    df = pd.DataFrame({
        'price': price,
        'quantity': quantity,
        'discount': discount,
        'advertising_spend': advertising_spend,
        'competitor_price': competitor_price,
        'day_of_week': day_of_week,
        'month': month,
        'store_category': store_category,
        'revenue': revenue
    })

    return df


def create_automated_features(df):
    """
    Automatically generate features using various techniques.
    """
    df_feat = df.copy()

    # Numerical columns for feature generation
    num_cols = ['price', 'quantity', 'discount', 'advertising_spend', 'competitor_price']

    # 1. Arithmetic interactions
    df_feat['price_quantity'] = df_feat['price'] * df_feat['quantity']
    df_feat['price_discount'] = df_feat['price'] * (1 - df_feat['discount'])
    df_feat['revenue_per_unit'] = df_feat['price'] * (1 - df_feat['discount'])
    df_feat['price_diff'] = df_feat['price'] - df_feat['competitor_price']
    df_feat['price_ratio'] = df_feat['price'] / (df_feat['competitor_price'] + 1e-5)

    # 2. Statistical aggregations (rolling/grouping simulations)
    for col in num_cols:
        df_feat[f'{col}_squared'] = df_feat[col] ** 2
        df_feat[f'{col}_sqrt'] = np.sqrt(df_feat[col])
        df_feat[f'{col}_log'] = np.log1p(df_feat[col])

    # 3. Binned versions
    df_feat['price_bin'] = pd.cut(df_feat['price'], bins=5, labels=False)
    df_feat['quantity_bin'] = pd.cut(df_feat['quantity'], bins=5, labels=False)

    # 4. Time-based features
    df_feat['is_weekend'] = (df_feat['day_of_week'] >= 5).astype(int)
    df_feat['is_holiday_season'] = ((df_feat['month'] == 11) | (df_feat['month'] == 12)).astype(int)
    df_feat['month_sin'] = np.sin(df_feat['month'] / 12 * 2 * np.pi)
    df_feat['month_cos'] = np.cos(df_feat['month'] / 12 * np.pi)

    # 5. Categorical encoding
    category_dummies = pd.get_dummies(df_feat['store_category'], prefix='category')
    df_feat = pd.concat([df_feat, category_dummies], axis=1)

    # 6. Ratio and percentage features
    df_feat['discount_intensity'] = df_feat['discount'] * df_feat['price']
    df_feat['ad_spend_per_unit'] = df_feat['advertising_spend'] / (df_feat['quantity'] + 1)
    df_feat['price_relative_to_ad'] = df_feat['price'] / (df_feat['advertising_spend'] + 1)

    return df_feat


def train_and_evaluate(X_train, X_test, y_train, y_test, model_name="Model"):
    """
    Train and evaluate a model.
    """
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

    results = {
        'model': model,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred
    }

    return results


def plot_results(results_baseline, results_featured, df_test, feature_importance):
    """
    Create comprehensive visualizations.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Predictions comparison
    ax1 = axes[0, 0]
    ax1.scatter(df_test['revenue'], results_baseline['predictions'],
                alpha=0.5, label='Baseline', s=30)
    ax1.scatter(df_test['revenue'], results_featured['predictions'],
                alpha=0.5, label='With Features', s=30)
    ax1.plot([df_test['revenue'].min(), df_test['revenue'].max()],
             [df_test['revenue'].min(), df_test['revenue'].max()],
             'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Revenue', fontsize=12)
    ax1.set_ylabel('Predicted Revenue', fontsize=12)
    ax1.set_title('Predictions: Baseline vs Feature Engineered', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Residuals comparison
    ax2 = axes[0, 1]
    residuals_baseline = df_test['revenue'] - results_baseline['predictions']
    residuals_featured = df_test['revenue'] - results_featured['predictions']
    ax2.hist(residuals_baseline, bins=30, alpha=0.5, label='Baseline', edgecolor='black')
    ax2.hist(residuals_featured, bins=30, alpha=0.5, label='With Features', edgecolor='black')
    ax2.set_xlabel('Residuals', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Residual Distribution Comparison', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Feature importance
    ax3 = axes[1, 0]
    top_features = feature_importance.head(15)
    ax3.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    ax3.set_yticks(range(len(top_features)))
    ax3.set_yticklabels(top_features['feature'])
    ax3.set_xlabel('Importance', fontsize=12)
    ax3.set_title('Top 15 Most Important Features', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')

    # 4. Metrics comparison
    ax4 = axes[1, 1]
    metrics = ['RMSE', 'MAE', 'R²']
    baseline_vals = [results_baseline['rmse'], results_baseline['mae'], results_baseline['r2']]
    featured_vals = [results_featured['rmse'], results_featured['mae'], results_featured['r2']]

    # Normalize for better visualization (except R²)
    baseline_norm = [baseline_vals[0]/baseline_vals[0], baseline_vals[1]/baseline_vals[1], baseline_vals[2]]
    featured_norm = [featured_vals[0]/baseline_vals[0], featured_vals[1]/baseline_vals[1], featured_vals[2]]

    x = np.arange(len(metrics))
    width = 0.35
    ax4.bar(x - width/2, baseline_norm, width, label='Baseline', alpha=0.8)
    ax4.bar(x + width/2, featured_norm, width, label='With Features', alpha=0.8)
    ax4.set_ylabel('Normalized Score', fontsize=12)
    ax4.set_title('Model Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/13_feature_engineering/01_automated_feature/feature_engineering_results.png',
                dpi=300, bbox_inches='tight')
    print("Plot saved as 'feature_engineering_results.png'")
    plt.show()


def main():
    """
    Main execution function.
    """
    print("=" * 80)
    print("Automated Feature Engineering Example")
    print("=" * 80)

    # Generate data
    print("\n1. Generating synthetic sales data...")
    df = generate_sales_data(n_samples=2000)
    print(f"   Dataset shape: {df.shape}")
    print(f"   Features: {df.columns.tolist()}")

    # Split data
    print("\n2. Splitting data...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Baseline model (original features only)
    print("\n3. Training baseline model (original features)...")
    baseline_features = ['price', 'quantity', 'discount', 'advertising_spend',
                         'competitor_price', 'day_of_week', 'month']

    # Encode categorical for baseline
    train_baseline = train_df.copy()
    test_baseline = test_df.copy()
    train_baseline = pd.get_dummies(train_baseline, columns=['store_category'], prefix='category')
    test_baseline = pd.get_dummies(test_baseline, columns=['store_category'], prefix='category')

    baseline_cols = [col for col in train_baseline.columns if col != 'revenue']

    X_train_base = train_baseline[baseline_cols]
    X_test_base = test_baseline[baseline_cols]
    y_train = train_baseline['revenue']
    y_test = test_baseline['revenue']

    results_baseline = train_and_evaluate(X_train_base, X_test_base, y_train, y_test, "Baseline")
    print(f"   Baseline RMSE: {results_baseline['rmse']:.2f}")
    print(f"   Baseline R²: {results_baseline['r2']:.4f}")

    # Create automated features
    print("\n4. Creating automated features...")
    train_featured = create_automated_features(train_df)
    test_featured = create_automated_features(test_df)

    # Remove target and categorical columns
    feature_cols = [col for col in train_featured.columns
                   if col not in ['revenue', 'store_category']]

    X_train_feat = train_featured[feature_cols]
    X_test_feat = test_featured[feature_cols]

    print(f"   Original features: {len(baseline_cols)}")
    print(f"   After feature engineering: {len(feature_cols)}")
    print(f"   New features created: {len(feature_cols) - len(baseline_cols)}")

    # Train with engineered features
    print("\n5. Training model with engineered features...")
    results_featured = train_and_evaluate(X_train_feat, X_test_feat, y_train, y_test, "Featured")
    print(f"   Featured RMSE: {results_featured['rmse']:.2f}")
    print(f"   Featured R²: {results_featured['r2']:.4f}")

    # Calculate improvement
    print("\n6. Performance Improvement:")
    rmse_improvement = ((results_baseline['rmse'] - results_featured['rmse']) /
                        results_baseline['rmse'] * 100)
    r2_improvement = ((results_featured['r2'] - results_baseline['r2']) /
                      results_baseline['r2'] * 100)
    print(f"   RMSE improvement: {rmse_improvement:.2f}%")
    print(f"   R² improvement: {r2_improvement:.2f}%")

    # Feature importance
    print("\n7. Analyzing feature importance...")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': results_featured['model'].feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n   Top 10 Most Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:30s}: {row['importance']:.4f}")

    # Visualizations
    print("\n8. Creating visualizations...")
    plot_results(results_baseline, results_featured, test_df, feature_importance)

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
