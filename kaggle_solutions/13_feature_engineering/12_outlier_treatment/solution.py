"""
Kaggle Solution: Outlier Detection and Treatment
================================================
Demonstrates various outlier detection methods and treatment strategies
to improve model robustness and performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def generate_data_with_outliers(n_samples=2000, outlier_fraction=0.05):
    """
    Generate synthetic data with outliers.
    """
    # Normal samples
    n_inliers = int(n_samples * (1 - outlier_fraction))
    n_outliers = n_samples - n_inliers

    # Inlier features
    x1_inliers = np.random.normal(50, 10, n_inliers)
    x2_inliers = np.random.normal(100, 20, n_inliers)
    x3_inliers = np.random.exponential(30, n_inliers)
    x4_inliers = np.random.uniform(0, 100, n_inliers)

    # Outlier features (extreme values)
    x1_outliers = np.random.normal(50, 50, n_outliers)  # Higher variance
    x2_outliers = np.random.choice([0, 200], n_outliers)  # Extreme values
    x3_outliers = np.random.exponential(200, n_outliers)  # Very skewed
    x4_outliers = np.random.uniform(-50, 200, n_outliers)  # Extended range

    # Combine
    x1 = np.concatenate([x1_inliers, x1_outliers])
    x2 = np.concatenate([x2_inliers, x2_outliers])
    x3 = np.concatenate([x3_inliers, x3_outliers])
    x4 = np.concatenate([x4_inliers, x4_outliers])

    # Target with outlier influence
    y_base = 2*x1 + 3*x2 + 0.5*x3 - 1*x4
    y = y_base + np.random.normal(0, 20, n_samples)

    # Add extreme outliers to target
    outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
    y[outlier_indices] += np.random.choice([-1, 1], n_outliers) * np.random.uniform(100, 300, n_outliers)

    df = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'x4': x4,
        'target': y
    })

    return df


def detect_outliers_zscore(data, columns, threshold=3):
    """
    Detect outliers using Z-score method.
    """
    outliers_mask = pd.Series([False] * len(data), index=data.index)

    for col in columns:
        z_scores = np.abs(stats.zscore(data[col]))
        outliers_mask |= (z_scores > threshold)

    return outliers_mask


def detect_outliers_iqr(data, columns, k=1.5):
    """
    Detect outliers using IQR method.
    """
    outliers_mask = pd.Series([False] * len(data), index=data.index)

    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - k * IQR
        upper_bound = Q3 + k * IQR

        outliers_mask |= (data[col] < lower_bound) | (data[col] > upper_bound)

    return outliers_mask


def detect_outliers_isolation_forest(data, columns, contamination=0.05):
    """
    Detect outliers using Isolation Forest.
    """
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    predictions = iso_forest.fit_predict(data[columns])
    outliers_mask = predictions == -1

    return outliers_mask


def detect_outliers_lof(data, columns, n_neighbors=20, contamination=0.05):
    """
    Detect outliers using Local Outlier Factor.
    """
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    predictions = lof.fit_predict(data[columns])
    outliers_mask = predictions == -1

    return outliers_mask


def detect_outliers_elliptic_envelope(data, columns, contamination=0.05):
    """
    Detect outliers using Elliptic Envelope (assumes Gaussian distribution).
    """
    try:
        ee = EllipticEnvelope(contamination=contamination, random_state=42)
        predictions = ee.fit_predict(data[columns])
        outliers_mask = predictions == -1
    except:
        # Fallback to IQR if Elliptic Envelope fails
        outliers_mask = detect_outliers_iqr(data, columns)

    return outliers_mask


def treat_outliers_clip(data, columns, lower_percentile=1, upper_percentile=99):
    """
    Treat outliers by clipping to percentile values.
    """
    data_treated = data.copy()

    for col in columns:
        lower = data[col].quantile(lower_percentile / 100)
        upper = data[col].quantile(upper_percentile / 100)
        data_treated[col] = data[col].clip(lower, upper)

    return data_treated


def treat_outliers_transform(data, columns):
    """
    Treat outliers using log transformation for skewed data.
    """
    data_treated = data.copy()

    for col in columns:
        data_treated[f'{col}_log'] = np.log1p(np.abs(data[col]))

    return data_treated


def treat_outliers_winsorize(data, columns, limits=(0.05, 0.05)):
    """
    Treat outliers using winsorization.
    """
    from scipy.stats.mstats import winsorize
    data_treated = data.copy()

    for col in columns:
        data_treated[col] = winsorize(data[col], limits=limits)

    return data_treated


def train_and_evaluate(X_train, X_test, y_train, y_test, model_type='ridge'):
    """
    Train and evaluate model.
    """
    if model_type == 'ridge':
        model = Ridge(alpha=1.0, random_state=42)
    elif model_type == 'huber':
        model = HuberRegressor(epsilon=1.35)
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
        'predictions': y_pred_test
    }

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    results['cv_mean'] = cv_scores.mean()
    results['cv_std'] = cv_scores.std()

    return results


def plot_outlier_analysis(df, outlier_masks_dict, results_dict):
    """
    Create comprehensive visualizations for outlier analysis.
    """
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

    # 1. Original data distribution with outliers
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(df['x1'], df['target'], alpha=0.5, s=20, color='blue')
    ax1.set_xlabel('X1', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Target', fontsize=11, fontweight='bold')
    ax1.set_title('Original Data (X1 vs Target)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 2. Outliers detected by different methods
    ax2 = fig.add_subplot(gs[0, 1])
    method_names = list(outlier_masks_dict.keys())
    outlier_counts = [outlier_masks_dict[m].sum() for m in method_names]
    colors = plt.cm.Set3(np.linspace(0, 1, len(method_names)))

    ax2.bar(range(len(method_names)), outlier_counts, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Detection Method', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Outliers', fontsize=11, fontweight='bold')
    ax2.set_title('Outliers Detected by Method', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(method_names)))
    ax2.set_xticklabels(method_names, rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    for i, count in enumerate(outlier_counts):
        ax2.text(i, count, str(count), ha='center', va='bottom', fontsize=9)

    # 3. Box plots showing outliers
    ax3 = fig.add_subplot(gs[0, 2])
    feature_cols = ['x1', 'x2', 'x3', 'x4']
    df[feature_cols].boxplot(ax=ax3)
    ax3.set_ylabel('Value', fontsize=11, fontweight='bold')
    ax3.set_title('Feature Distributions with Outliers', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Z-score outliers visualization
    ax4 = fig.add_subplot(gs[1, 0])
    if 'Z-Score' in outlier_masks_dict:
        mask = outlier_masks_dict['Z-Score']
        ax4.scatter(df.loc[~mask, 'x1'], df.loc[~mask, 'target'],
                   alpha=0.5, s=20, color='blue', label='Normal')
        ax4.scatter(df.loc[mask, 'x1'], df.loc[mask, 'target'],
                   alpha=0.7, s=40, color='red', marker='x', label='Outliers')
        ax4.set_xlabel('X1', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Target', fontsize=11, fontweight='bold')
        ax4.set_title('Z-Score Outlier Detection', fontsize=13, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # 5. IQR outliers visualization
    ax5 = fig.add_subplot(gs[1, 1])
    if 'IQR' in outlier_masks_dict:
        mask = outlier_masks_dict['IQR']
        ax5.scatter(df.loc[~mask, 'x2'], df.loc[~mask, 'target'],
                   alpha=0.5, s=20, color='blue', label='Normal')
        ax5.scatter(df.loc[mask, 'x2'], df.loc[mask, 'target'],
                   alpha=0.7, s=40, color='red', marker='x', label='Outliers')
        ax5.set_xlabel('X2', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Target', fontsize=11, fontweight='bold')
        ax5.set_title('IQR Outlier Detection', fontsize=13, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    # 6. Isolation Forest outliers
    ax6 = fig.add_subplot(gs[1, 2])
    if 'Isolation-Forest' in outlier_masks_dict:
        mask = outlier_masks_dict['Isolation-Forest']
        ax6.scatter(df.loc[~mask, 'x3'], df.loc[~mask, 'target'],
                   alpha=0.5, s=20, color='blue', label='Normal')
        ax6.scatter(df.loc[mask, 'x3'], df.loc[mask, 'target'],
                   alpha=0.7, s=40, color='red', marker='x', label='Outliers')
        ax6.set_xlabel('X3', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Target', fontsize=11, fontweight='bold')
        ax6.set_title('Isolation Forest Detection', fontsize=13, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

    # 7. Model Performance Comparison
    ax7 = fig.add_subplot(gs[2, 0])
    models = list(results_dict.keys())
    test_r2 = [results_dict[m]['test_r2'] for m in models]
    colors_perf = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(models)))

    bars = ax7.barh(range(len(models)), test_r2, color=colors_perf, alpha=0.8, edgecolor='black')
    ax7.set_yticks(range(len(models)))
    ax7.set_yticklabels(models, fontsize=9)
    ax7.set_xlabel('Test R² Score', fontsize=11, fontweight='bold')
    ax7.set_title('Model Performance by Treatment', fontsize=13, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='x')

    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax7.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}', ha='left', va='center', fontsize=8)

    # 8. RMSE Comparison
    ax8 = fig.add_subplot(gs[2, 1])
    test_rmse = [results_dict[m]['test_rmse'] for m in models]
    train_rmse = [results_dict[m]['train_rmse'] for m in models]

    x = np.arange(len(models))
    width_bar = 0.35
    ax8.bar(x - width_bar/2, train_rmse, width_bar, label='Train', alpha=0.8, color='steelblue')
    ax8.bar(x + width_bar/2, test_rmse, width_bar, label='Test', alpha=0.8, color='coral')
    ax8.set_xlabel('Treatment Method', fontsize=11, fontweight='bold')
    ax8.set_ylabel('RMSE', fontsize=11, fontweight='bold')
    ax8.set_title('RMSE Comparison', fontsize=13, fontweight='bold')
    ax8.set_xticks(x)
    ax8.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')

    # 9. Cross-Validation Scores
    ax9 = fig.add_subplot(gs[2, 2])
    cv_means = [results_dict[m]['cv_mean'] for m in models]
    cv_stds = [results_dict[m]['cv_std'] for m in models]

    ax9.errorbar(range(len(models)), cv_means, yerr=cv_stds,
                fmt='o-', linewidth=2, markersize=8, capsize=5, color='darkgreen')
    ax9.set_xlabel('Treatment Method', fontsize=11, fontweight='bold')
    ax9.set_ylabel('CV R² Score', fontsize=11, fontweight='bold')
    ax9.set_title('Cross-Validation Performance', fontsize=13, fontweight='bold')
    ax9.set_xticks(range(len(models)))
    ax9.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax9.grid(True, alpha=0.3)

    # 10. Predictions vs Actual (Best Model)
    ax10 = fig.add_subplot(gs[3, 0])
    best_model_name = max(results_dict.keys(), key=lambda k: results_dict[k]['test_r2'])
    best_preds = results_dict[best_model_name]['predictions']
    y_test_subset = df.iloc[-len(best_preds):]['target']

    ax10.scatter(y_test_subset, best_preds, alpha=0.5, s=30, color='purple')
    ax10.plot([y_test_subset.min(), y_test_subset.max()],
             [y_test_subset.min(), y_test_subset.max()],
             'r--', lw=2, label='Perfect Prediction')
    ax10.set_xlabel('Actual Values', fontsize=11, fontweight='bold')
    ax10.set_ylabel('Predicted Values', fontsize=11, fontweight='bold')
    ax10.set_title(f'Best Model: {best_model_name}', fontsize=13, fontweight='bold')
    ax10.legend()
    ax10.grid(True, alpha=0.3)

    # 11. Performance Improvement
    ax11 = fig.add_subplot(gs[3, 1])
    baseline_r2 = results_dict['No-Treatment']['test_r2']
    improvements = [(results_dict[m]['test_r2'] - baseline_r2) / baseline_r2 * 100
                   for m in models if m != 'No-Treatment']
    improved_models = [m for m in models if m != 'No-Treatment']

    colors_imp = ['green' if imp > 0 else 'red' for imp in improvements]
    ax11.barh(range(len(improvements)), improvements, color=colors_imp,
             alpha=0.8, edgecolor='black')
    ax11.set_yticks(range(len(improvements)))
    ax11.set_yticklabels(improved_models, fontsize=9)
    ax11.set_xlabel('Improvement (%)', fontsize=11, fontweight='bold')
    ax11.set_title('Performance vs No Treatment', fontsize=13, fontweight='bold')
    ax11.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax11.grid(True, alpha=0.3, axis='x')

    # 12. Outlier Overlap Between Methods
    ax12 = fig.add_subplot(gs[3, 2])
    if len(outlier_masks_dict) >= 2:
        methods = list(outlier_masks_dict.keys())[:3]  # Top 3 methods
        venn_data = []
        for method in methods:
            venn_data.append(outlier_masks_dict[method].sum())

        ax12.bar(range(len(methods)), venn_data, alpha=0.7, color='orange', edgecolor='black')
        ax12.set_xlabel('Detection Method', fontsize=11, fontweight='bold')
        ax12.set_ylabel('Outliers Count', fontsize=11, fontweight='bold')
        ax12.set_title('Outlier Detection Comparison', fontsize=13, fontweight='bold')
        ax12.set_xticks(range(len(methods)))
        ax12.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
        ax12.grid(True, alpha=0.3, axis='y')

    plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/13_feature_engineering/12_outlier_treatment/outlier_analysis.png',
                dpi=300, bbox_inches='tight')
    print("   Comprehensive plot saved!")
    plt.show()


def main():
    """
    Main execution function.
    """
    print("=" * 90)
    print("Outlier Detection and Treatment")
    print("=" * 90)

    # Generate data with outliers
    print("\n1. Generating data with outliers...")
    df = generate_data_with_outliers(n_samples=2000, outlier_fraction=0.08)
    print(f"   Dataset shape: {df.shape}")
    print(f"\n   Data summary:")
    print(df.describe())

    # Detect outliers using different methods
    feature_cols = ['x1', 'x2', 'x3', 'x4']
    outlier_masks_dict = {}

    print("\n2. Detecting outliers using various methods...")

    # Z-Score
    print("   - Z-Score method...")
    outlier_masks_dict['Z-Score'] = detect_outliers_zscore(df, feature_cols, threshold=3)
    print(f"     Outliers detected: {outlier_masks_dict['Z-Score'].sum()}")

    # IQR
    print("   - IQR method...")
    outlier_masks_dict['IQR'] = detect_outliers_iqr(df, feature_cols, k=1.5)
    print(f"     Outliers detected: {outlier_masks_dict['IQR'].sum()}")

    # Isolation Forest
    print("   - Isolation Forest...")
    outlier_masks_dict['Isolation-Forest'] = detect_outliers_isolation_forest(
        df, feature_cols, contamination=0.08
    )
    print(f"     Outliers detected: {outlier_masks_dict['Isolation-Forest'].sum()}")

    # LOF
    print("   - Local Outlier Factor...")
    outlier_masks_dict['LOF'] = detect_outliers_lof(df, feature_cols, contamination=0.08)
    print(f"     Outliers detected: {outlier_masks_dict['LOF'].sum()}")

    # Split data
    X = df[feature_cols]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results_dict = {}

    # Baseline: No treatment
    print("\n3. Training baseline (no outlier treatment)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results_baseline = train_and_evaluate(X_train_scaled, X_test_scaled,
                                         y_train, y_test, 'ridge')
    results_dict['No-Treatment'] = results_baseline
    print(f"   Test RMSE: {results_baseline['test_rmse']:.2f}")
    print(f"   Test R²: {results_baseline['test_r2']:.4f}")

    # Remove outliers (IQR method)
    print("\n4. Training with outliers removed (IQR)...")
    iqr_mask = detect_outliers_iqr(X_train.join(y_train), feature_cols)
    X_train_clean = X_train[~iqr_mask]
    y_train_clean = y_train[~iqr_mask]

    scaler_clean = StandardScaler()
    X_train_clean_scaled = scaler_clean.fit_transform(X_train_clean)
    X_test_clean_scaled = scaler_clean.transform(X_test)

    results_removed = train_and_evaluate(X_train_clean_scaled, X_test_clean_scaled,
                                        y_train_clean, y_test, 'ridge')
    results_dict['Removed-IQR'] = results_removed
    print(f"   Samples removed: {iqr_mask.sum()} ({iqr_mask.sum()/len(X_train)*100:.1f}%)")
    print(f"   Test RMSE: {results_removed['test_rmse']:.2f}")
    print(f"   Test R²: {results_removed['test_r2']:.4f}")

    # Clip outliers
    print("\n5. Training with clipped outliers...")
    X_train_clipped = treat_outliers_clip(X_train, feature_cols, 1, 99)
    X_test_clipped = treat_outliers_clip(X_test, feature_cols, 1, 99)

    scaler_clip = StandardScaler()
    X_train_clip_scaled = scaler_clip.fit_transform(X_train_clipped)
    X_test_clip_scaled = scaler_clip.transform(X_test_clipped)

    results_clipped = train_and_evaluate(X_train_clip_scaled, X_test_clip_scaled,
                                        y_train, y_test, 'ridge')
    results_dict['Clipped'] = results_clipped
    print(f"   Test RMSE: {results_clipped['test_rmse']:.2f}")
    print(f"   Test R²: {results_clipped['test_r2']:.4f}")

    # Winsorization
    print("\n6. Training with winsorized data...")
    X_train_wins = treat_outliers_winsorize(X_train, feature_cols, limits=(0.05, 0.05))
    X_test_wins = treat_outliers_winsorize(X_test, feature_cols, limits=(0.05, 0.05))

    scaler_wins = StandardScaler()
    X_train_wins_scaled = scaler_wins.fit_transform(X_train_wins)
    X_test_wins_scaled = scaler_wins.transform(X_test_wins)

    results_wins = train_and_evaluate(X_train_wins_scaled, X_test_wins_scaled,
                                     y_train, y_test, 'ridge')
    results_dict['Winsorized'] = results_wins
    print(f"   Test RMSE: {results_wins['test_rmse']:.2f}")
    print(f"   Test R²: {results_wins['test_r2']:.4f}")

    # Robust Scaler
    print("\n7. Training with Robust Scaler...")
    robust_scaler = RobustScaler()
    X_train_robust = robust_scaler.fit_transform(X_train)
    X_test_robust = robust_scaler.transform(X_test)

    results_robust = train_and_evaluate(X_train_robust, X_test_robust,
                                       y_train, y_test, 'ridge')
    results_dict['Robust-Scaler'] = results_robust
    print(f"   Test RMSE: {results_robust['test_rmse']:.2f}")
    print(f"   Test R²: {results_robust['test_r2']:.4f}")

    # Huber Regressor (robust to outliers)
    print("\n8. Training with Huber Regressor...")
    results_huber = train_and_evaluate(X_train_scaled, X_test_scaled,
                                      y_train, y_test, 'huber')
    results_dict['Huber-Regressor'] = results_huber
    print(f"   Test RMSE: {results_huber['test_rmse']:.2f}")
    print(f"   Test R²: {results_huber['test_r2']:.4f}")

    # Performance summary
    print("\n9. Performance Summary:")
    print("   " + "-" * 75)
    print(f"   {'Method':<20} {'Train RMSE':<12} {'Test RMSE':<12} {'Test R²':<12} {'CV R²':<12}")
    print("   " + "-" * 75)
    for method, results in results_dict.items():
        print(f"   {method:<20} {results['train_rmse']:<12.2f} "
              f"{results['test_rmse']:<12.2f} {results['test_r2']:<12.4f} "
              f"{results['cv_mean']:<12.4f}")
    print("   " + "-" * 75)

    # Visualizations
    print("\n10. Creating comprehensive visualizations...")
    plot_outlier_analysis(df, outlier_masks_dict, results_dict)

    # Best method
    best_method = max(results_dict.items(), key=lambda x: x[1]['test_r2'])
    print(f"\n11. Best Outlier Treatment: {best_method[0]}")
    print(f"    Test R²: {best_method[1]['test_r2']:.4f}")
    improvement = ((best_method[1]['test_r2'] - results_baseline['test_r2']) /
                   results_baseline['test_r2'] * 100)
    print(f"    Improvement: {improvement:.1f}%")

    print("\n" + "=" * 90)
    print("Analysis Complete!")
    print("=" * 90)


if __name__ == "__main__":
    main()
