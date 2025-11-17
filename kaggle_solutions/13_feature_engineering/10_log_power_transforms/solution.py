"""
Kaggle Solution: Log and Power Transformations
==============================================
Demonstrates various logarithmic and power transformations to handle skewed data,
normalize distributions, and improve model performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
from scipy.special import inv_boxcox
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def generate_skewed_data(n_samples=2000):
    """
    Generate synthetic data with skewed distributions.
    """
    # Highly skewed features
    income = np.random.lognormal(mean=10, sigma=1.5, size=n_samples)
    sales_volume = np.random.exponential(scale=1000, size=n_samples)
    web_traffic = np.random.gamma(shape=2, scale=500, size=n_samples)
    customer_age = np.random.gamma(shape=5, scale=8, size=n_samples) + 18

    # Moderately skewed
    purchase_amount = np.random.chisquare(df=3, size=n_samples) * 50 + 10

    # Normal-ish features
    satisfaction_score = np.random.normal(7, 2, n_samples)
    satisfaction_score = np.clip(satisfaction_score, 0, 10)

    # Target with non-linear relationships
    target = (
        0.0001 * income +
        0.002 * sales_volume +
        0.001 * web_traffic +
        5 * np.log1p(purchase_amount) +
        10 * satisfaction_score +
        0.5 * customer_age +
        np.random.normal(0, 20, n_samples)
    )

    df = pd.DataFrame({
        'income': income,
        'sales_volume': sales_volume,
        'web_traffic': web_traffic,
        'customer_age': customer_age,
        'purchase_amount': purchase_amount,
        'satisfaction_score': satisfaction_score,
        'target': target
    })

    return df


def apply_log_transforms(df, columns):
    """
    Apply various logarithmic transformations.
    """
    df_transformed = df.copy()

    for col in columns:
        # Log transformation (log1p handles zeros)
        df_transformed[f'{col}_log'] = np.log1p(df[col])

        # Log10 transformation
        df_transformed[f'{col}_log10'] = np.log10(df[col] + 1)

        # Natural log
        df_transformed[f'{col}_ln'] = np.log(df[col] + 1)

    return df_transformed


def apply_power_transforms(df, columns):
    """
    Apply various power transformations.
    """
    df_transformed = df.copy()

    for col in columns:
        # Square root
        df_transformed[f'{col}_sqrt'] = np.sqrt(df[col])

        # Cube root
        df_transformed[f'{col}_cbrt'] = np.cbrt(df[col])

        # Square
        df_transformed[f'{col}_squared'] = df[col] ** 2

        # Custom powers
        df_transformed[f'{col}_pow_0.5'] = np.power(df[col], 0.5)
        df_transformed[f'{col}_pow_0.25'] = np.power(df[col], 0.25)

    return df_transformed


def apply_boxcox_transform(df, columns):
    """
    Apply Box-Cox transformation (requires positive values).
    """
    df_transformed = df.copy()
    lambda_values = {}

    for col in columns:
        # Ensure all values are positive
        if (df[col] > 0).all():
            transformed, lambda_val = stats.boxcox(df[col])
            df_transformed[f'{col}_boxcox'] = transformed
            lambda_values[col] = lambda_val
        else:
            # Shift to make positive
            shifted = df[col] - df[col].min() + 1
            transformed, lambda_val = stats.boxcox(shifted)
            df_transformed[f'{col}_boxcox'] = transformed
            lambda_values[col] = lambda_val

    return df_transformed, lambda_values


def apply_yeo_johnson_transform(df, columns):
    """
    Apply Yeo-Johnson transformation (works with negative values).
    """
    df_transformed = df.copy()

    for col in columns:
        pt = PowerTransformer(method='yeo-johnson', standardize=True)
        df_transformed[f'{col}_yeojohnson'] = pt.fit_transform(df[[col]])

    return df_transformed


def calculate_skewness_kurtosis(df, columns):
    """
    Calculate skewness and kurtosis for columns.
    """
    stats_df = pd.DataFrame({
        'feature': columns,
        'skewness': [stats.skew(df[col]) for col in columns],
        'kurtosis': [stats.kurtosis(df[col]) for col in columns]
    })

    return stats_df


def train_and_evaluate(X_train, X_test, y_train, y_test, model_type='linear'):
    """
    Train and evaluate model.
    """
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0, random_state=42)
    elif model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    results = {
        'model': model,
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'predictions': y_pred_test
    }

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    results['cv_mean'] = cv_scores.mean()
    results['cv_std'] = cv_scores.std()

    return results


def plot_transformation_analysis(df_original, df_transformed, transform_name, columns):
    """
    Visualize the impact of transformations.
    """
    n_cols = len(columns)
    fig, axes = plt.subplots(n_cols, 3, figsize=(18, 4*n_cols))

    if n_cols == 1:
        axes = axes.reshape(1, -1)

    for idx, col in enumerate(columns):
        # Original distribution
        ax1 = axes[idx, 0]
        ax1.hist(df_original[col], bins=50, color='skyblue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel(col, fontsize=10, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax1.set_title(f'Original: {col}', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Add skewness info
        skew = stats.skew(df_original[col])
        ax1.text(0.95, 0.95, f'Skew: {skew:.2f}',
                transform=ax1.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Transformed distribution
        ax2 = axes[idx, 1]
        transformed_col = f'{col}_{transform_name}'
        if transformed_col in df_transformed.columns:
            ax2.hist(df_transformed[transformed_col], bins=50, color='lightcoral',
                    alpha=0.7, edgecolor='black')
            ax2.set_xlabel(transformed_col, fontsize=10, fontweight='bold')
            ax2.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            ax2.set_title(f'Transformed: {col}', fontsize=11, fontweight='bold')
            ax2.grid(True, alpha=0.3)

            # Add skewness info
            skew_trans = stats.skew(df_transformed[transformed_col])
            ax2.text(0.95, 0.95, f'Skew: {skew_trans:.2f}',
                    transform=ax2.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        # Q-Q plot
        ax3 = axes[idx, 2]
        if transformed_col in df_transformed.columns:
            stats.probplot(df_transformed[transformed_col], dist="norm", plot=ax3)
            ax3.set_title(f'Q-Q Plot: {col}', fontsize=11, fontweight='bold')
            ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_comprehensive_results(results_dict, skewness_comparison, y_test):
    """
    Create comprehensive visualization of all results.
    """
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Model Performance Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    models = list(results_dict.keys())
    test_r2 = [results_dict[m]['test_r2'] for m in models]
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(models)))

    bars = ax1.barh(range(len(models)), test_r2, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels(models, fontsize=9)
    ax1.set_xlabel('Test R² Score', fontsize=11, fontweight='bold')
    ax1.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}', ha='left', va='center', fontsize=8)

    # 2. RMSE Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    test_rmse = [results_dict[m]['test_rmse'] for m in models]
    train_rmse = [results_dict[m]['train_rmse'] for m in models]

    x = np.arange(len(models))
    width = 0.35
    ax2.bar(x - width/2, train_rmse, width, label='Train', alpha=0.8, color='steelblue')
    ax2.bar(x + width/2, test_rmse, width, label='Test', alpha=0.8, color='coral')
    ax2.set_xlabel('Transformation Method', fontsize=11, fontweight='bold')
    ax2.set_ylabel('RMSE', fontsize=11, fontweight='bold')
    ax2.set_title('RMSE Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Cross-Validation Scores
    ax3 = fig.add_subplot(gs[0, 2])
    cv_means = [results_dict[m]['cv_mean'] for m in models]
    cv_stds = [results_dict[m]['cv_std'] for m in models]

    ax3.errorbar(range(len(models)), cv_means, yerr=cv_stds,
                fmt='o-', linewidth=2, markersize=8, capsize=5, color='darkgreen')
    ax3.set_xlabel('Transformation Method', fontsize=11, fontweight='bold')
    ax3.set_ylabel('CV R² Score', fontsize=11, fontweight='bold')
    ax3.set_title('Cross-Validation Performance', fontsize=13, fontweight='bold')
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. Predictions vs Actual (Best Model)
    ax4 = fig.add_subplot(gs[1, 0])
    best_model_name = max(results_dict.keys(), key=lambda k: results_dict[k]['test_r2'])
    best_preds = results_dict[best_model_name]['predictions']

    ax4.scatter(y_test, best_preds, alpha=0.6, s=30, color='purple')
    ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
            'r--', lw=2, label='Perfect Prediction')
    ax4.set_xlabel('Actual Values', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Predicted Values', fontsize=11, fontweight='bold')
    ax4.set_title(f'Best Model: {best_model_name}', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Residuals Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    residuals = y_test - best_preds
    ax5.hist(residuals, bins=40, color='teal', alpha=0.7, edgecolor='black')
    ax5.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax5.set_xlabel('Residuals', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax5.set_title('Residual Distribution (Best Model)', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Skewness Reduction
    ax6 = fig.add_subplot(gs[1, 2])
    if skewness_comparison is not None:
        features = skewness_comparison['feature']
        original_skew = skewness_comparison['original_skew']
        transformed_skew = skewness_comparison['transformed_skew']

        x = np.arange(len(features))
        width = 0.35
        ax6.bar(x - width/2, np.abs(original_skew), width, label='Original',
               alpha=0.8, color='orange')
        ax6.bar(x + width/2, np.abs(transformed_skew), width, label='Transformed',
               alpha=0.8, color='green')
        ax6.set_xlabel('Feature', fontsize=11, fontweight='bold')
        ax6.set_ylabel('|Skewness|', fontsize=11, fontweight='bold')
        ax6.set_title('Skewness Reduction', fontsize=13, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(features, rotation=45, ha='right', fontsize=8)
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')

    # 7. Performance Improvement
    ax7 = fig.add_subplot(gs[2, 0])
    baseline_r2 = results_dict['Original']['test_r2']
    improvements = [(results_dict[m]['test_r2'] - baseline_r2) / baseline_r2 * 100
                   for m in models if m != 'Original']
    improved_models = [m for m in models if m != 'Original']

    colors_imp = ['green' if imp > 0 else 'red' for imp in improvements]
    ax7.barh(range(len(improvements)), improvements, color=colors_imp,
            alpha=0.8, edgecolor='black')
    ax7.set_yticks(range(len(improvements)))
    ax7.set_yticklabels(improved_models, fontsize=9)
    ax7.set_xlabel('Improvement (%)', fontsize=11, fontweight='bold')
    ax7.set_title('Performance vs Baseline', fontsize=13, fontweight='bold')
    ax7.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax7.grid(True, alpha=0.3, axis='x')

    # 8. MAE Comparison
    ax8 = fig.add_subplot(gs[2, 1])
    mae_values = [results_dict[m]['test_mae'] for m in models]
    ax8.bar(range(len(models)), mae_values, color='crimson', alpha=0.7, edgecolor='black')
    ax8.set_xlabel('Transformation Method', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Test MAE', fontsize=11, fontweight='bold')
    ax8.set_title('Mean Absolute Error Comparison', fontsize=13, fontweight='bold')
    ax8.set_xticks(range(len(models)))
    ax8.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax8.grid(True, alpha=0.3, axis='y')

    # 9. Overfitting Analysis
    ax9 = fig.add_subplot(gs[2, 2])
    train_r2_all = [results_dict[m]['train_r2'] for m in models]
    test_r2_all = [results_dict[m]['test_r2'] for m in models]
    gap = [train_r2_all[i] - test_r2_all[i] for i in range(len(models))]

    colors_gap = ['green' if g < 0.05 else 'orange' if g < 0.15 else 'red' for g in gap]
    ax9.bar(range(len(models)), gap, color=colors_gap, alpha=0.8, edgecolor='black')
    ax9.set_xlabel('Transformation Method', fontsize=11, fontweight='bold')
    ax9.set_ylabel('Train R² - Test R²', fontsize=11, fontweight='bold')
    ax9.set_title('Overfitting Analysis', fontsize=13, fontweight='bold')
    ax9.set_xticks(range(len(models)))
    ax9.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax9.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5)
    ax9.axhline(y=0.15, color='red', linestyle='--', alpha=0.5)
    ax9.grid(True, alpha=0.3, axis='y')

    plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/13_feature_engineering/10_log_power_transforms/transformation_analysis.png',
                dpi=300, bbox_inches='tight')
    print("   Comprehensive plot saved!")
    plt.show()


def main():
    """
    Main execution function.
    """
    print("=" * 90)
    print("Log and Power Transformations for Feature Engineering")
    print("=" * 90)

    # Generate skewed data
    print("\n1. Generating synthetic data with skewed distributions...")
    df = generate_skewed_data(n_samples=2000)
    print(f"   Dataset shape: {df.shape}")

    # Analyze original skewness
    print("\n2. Analyzing original data distribution...")
    feature_cols = [col for col in df.columns if col != 'target']
    stats_original = calculate_skewness_kurtosis(df, feature_cols)
    print("\n   Original Feature Statistics:")
    print(stats_original.to_string(index=False))

    # Split data
    X = df[feature_cols]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results_dict = {}

    # Baseline: Original features
    print("\n3. Training baseline model (original features)...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    results_baseline = train_and_evaluate(X_train_scaled, X_test_scaled,
                                         y_train, y_test, 'ridge')
    results_dict['Original'] = results_baseline
    print(f"   Test RMSE: {results_baseline['test_rmse']:.2f}")
    print(f"   Test R²: {results_baseline['test_r2']:.4f}")

    # Log transformations
    print("\n4. Applying log transformations...")
    skewed_cols = ['income', 'sales_volume', 'web_traffic', 'purchase_amount']

    X_train_log = apply_log_transforms(X_train, skewed_cols)
    X_test_log = apply_log_transforms(X_test, skewed_cols)

    log_cols = [col for col in X_train_log.columns if '_log' in col and '_log10' not in col and '_ln' not in col]
    X_train_log_selected = X_train_log[log_cols + ['customer_age', 'satisfaction_score']]
    X_test_log_selected = X_test_log[log_cols + ['customer_age', 'satisfaction_score']]

    scaler_log = StandardScaler()
    X_train_log_scaled = scaler_log.fit_transform(X_train_log_selected)
    X_test_log_scaled = scaler_log.transform(X_test_log_selected)

    results_log = train_and_evaluate(X_train_log_scaled, X_test_log_scaled,
                                     y_train, y_test, 'ridge')
    results_dict['Log'] = results_log
    print(f"   Test RMSE: {results_log['test_rmse']:.2f}")
    print(f"   Test R²: {results_log['test_r2']:.4f}")

    # Square root transformation
    print("\n5. Applying square root transformations...")
    X_train_sqrt = X_train.copy()
    X_test_sqrt = X_test.copy()

    for col in skewed_cols:
        X_train_sqrt[f'{col}_sqrt'] = np.sqrt(X_train[col])
        X_test_sqrt[f'{col}_sqrt'] = np.sqrt(X_test[col])

    sqrt_cols = [col for col in X_train_sqrt.columns if '_sqrt' in col]
    X_train_sqrt_selected = X_train_sqrt[sqrt_cols + ['customer_age', 'satisfaction_score']]
    X_test_sqrt_selected = X_test_sqrt[sqrt_cols + ['customer_age', 'satisfaction_score']]

    scaler_sqrt = StandardScaler()
    X_train_sqrt_scaled = scaler_sqrt.fit_transform(X_train_sqrt_selected)
    X_test_sqrt_scaled = scaler_sqrt.transform(X_test_sqrt_selected)

    results_sqrt = train_and_evaluate(X_train_sqrt_scaled, X_test_sqrt_scaled,
                                      y_train, y_test, 'ridge')
    results_dict['Sqrt'] = results_sqrt
    print(f"   Test RMSE: {results_sqrt['test_rmse']:.2f}")
    print(f"   Test R²: {results_sqrt['test_r2']:.4f}")

    # Box-Cox transformation
    print("\n6. Applying Box-Cox transformations...")
    X_train_boxcox, lambda_vals = apply_boxcox_transform(X_train, skewed_cols)
    X_test_boxcox, _ = apply_boxcox_transform(X_test, skewed_cols)

    boxcox_cols = [col for col in X_train_boxcox.columns if '_boxcox' in col]
    X_train_boxcox_selected = X_train_boxcox[boxcox_cols + ['customer_age', 'satisfaction_score']]
    X_test_boxcox_selected = X_test_boxcox[boxcox_cols + ['customer_age', 'satisfaction_score']]

    scaler_boxcox = StandardScaler()
    X_train_boxcox_scaled = scaler_boxcox.fit_transform(X_train_boxcox_selected)
    X_test_boxcox_scaled = scaler_boxcox.transform(X_test_boxcox_selected)

    results_boxcox = train_and_evaluate(X_train_boxcox_scaled, X_test_boxcox_scaled,
                                        y_train, y_test, 'ridge')
    results_dict['BoxCox'] = results_boxcox
    print(f"   Test RMSE: {results_boxcox['test_rmse']:.2f}")
    print(f"   Test R²: {results_boxcox['test_r2']:.4f}")
    print(f"   Lambda values: {lambda_vals}")

    # Yeo-Johnson transformation
    print("\n7. Applying Yeo-Johnson transformations...")
    X_train_yj = apply_yeo_johnson_transform(X_train, skewed_cols)
    X_test_yj = apply_yeo_johnson_transform(X_test, skewed_cols)

    yj_cols = [col for col in X_train_yj.columns if '_yeojohnson' in col]
    X_train_yj_selected = X_train_yj[yj_cols + ['customer_age', 'satisfaction_score']]
    X_test_yj_selected = X_test_yj[yj_cols + ['customer_age', 'satisfaction_score']]

    results_yj = train_and_evaluate(X_train_yj_selected, X_test_yj_selected,
                                    y_train, y_test, 'ridge')
    results_dict['YeoJohnson'] = results_yj
    print(f"   Test RMSE: {results_yj['test_rmse']:.2f}")
    print(f"   Test R²: {results_yj['test_r2']:.4f}")

    # Combined transformations
    print("\n8. Creating combined transformation features...")
    X_train_combined = X_train.copy()
    X_test_combined = X_test.copy()

    for col in skewed_cols:
        X_train_combined[f'{col}_log'] = np.log1p(X_train[col])
        X_test_combined[f'{col}_log'] = np.log1p(X_test[col])
        X_train_combined[f'{col}_sqrt'] = np.sqrt(X_train[col])
        X_test_combined[f'{col}_sqrt'] = np.sqrt(X_test[col])

    scaler_combined = StandardScaler()
    X_train_combined_scaled = scaler_combined.fit_transform(X_train_combined)
    X_test_combined_scaled = scaler_combined.transform(X_test_combined)

    results_combined = train_and_evaluate(X_train_combined_scaled, X_test_combined_scaled,
                                         y_train, y_test, 'ridge')
    results_dict['Combined'] = results_combined
    print(f"   Test RMSE: {results_combined['test_rmse']:.2f}")
    print(f"   Test R²: {results_combined['test_r2']:.4f}")

    # Performance summary
    print("\n9. Performance Summary:")
    print("   " + "-" * 80)
    print(f"   {'Method':<15} {'Train RMSE':<12} {'Test RMSE':<12} {'Test R²':<12} {'CV R²':<12}")
    print("   " + "-" * 80)
    for method, results in results_dict.items():
        print(f"   {method:<15} {results['train_rmse']:<12.2f} "
              f"{results['test_rmse']:<12.2f} {results['test_r2']:<12.4f} "
              f"{results['cv_mean']:<12.4f}")
    print("   " + "-" * 80)

    # Skewness comparison
    print("\n10. Skewness reduction analysis...")
    skewness_comp = pd.DataFrame({
        'feature': skewed_cols,
        'original_skew': [stats.skew(X_train[col]) for col in skewed_cols],
        'transformed_skew': [stats.skew(np.log1p(X_train[col])) for col in skewed_cols]
    })
    print(skewness_comp.to_string(index=False))

    # Visualizations
    print("\n11. Creating comprehensive visualizations...")
    plot_comprehensive_results(results_dict, skewness_comp, y_test)

    # Best transformation
    best_method = max(results_dict.items(), key=lambda x: x[1]['test_r2'])
    print(f"\n12. Best Transformation Method: {best_method[0]}")
    print(f"    Test R²: {best_method[1]['test_r2']:.4f}")
    print(f"    Improvement: {(best_method[1]['test_r2'] - results_baseline['test_r2']) / results_baseline['test_r2'] * 100:.1f}%")

    print("\n" + "=" * 90)
    print("Analysis Complete!")
    print("=" * 90)


if __name__ == "__main__":
    main()
