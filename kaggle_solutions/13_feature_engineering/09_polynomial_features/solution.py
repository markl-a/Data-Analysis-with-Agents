"""
Kaggle Solution: Polynomial Feature Generation and Selection
=============================================================
Demonstrates polynomial feature generation, interaction terms, and intelligent
feature selection to avoid overfitting while capturing non-linear relationships.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def generate_nonlinear_data(n_samples=1500):
    """
    Generate synthetic data with non-linear relationships suitable for polynomial features.
    """
    # Independent variables
    x1 = np.random.uniform(0, 10, n_samples)
    x2 = np.random.uniform(0, 10, n_samples)
    x3 = np.random.uniform(0, 10, n_samples)
    x4 = np.random.uniform(0, 10, n_samples)
    x5 = np.random.uniform(0, 10, n_samples)

    # Target with polynomial and interaction terms
    # y = 3*x1^2 + 2*x2 + 5*x1*x2 - 0.5*x3^2 + 1.5*x4 + noise
    y = (
        3 * x1**2 +
        2 * x2 +
        5 * x1 * x2 +
        -0.5 * x3**2 +
        1.5 * x4 +
        0.3 * x1 * x3 +
        -0.2 * x2**3 +
        np.random.normal(0, 10, n_samples)
    )

    df = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'x4': x4,
        'x5': x5,  # Noise variable
        'target': y
    })

    return df


def create_polynomial_features(X, degree=2, interaction_only=False):
    """
    Create polynomial features up to specified degree.
    """
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only,
                              include_bias=False)
    X_poly = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(X.columns)

    return pd.DataFrame(X_poly, columns=feature_names, index=X.index), poly


def create_custom_interactions(df, feature_cols, max_degree=2):
    """
    Create custom polynomial and interaction features with more control.
    """
    df_new = df[feature_cols].copy()
    new_features = []

    # Second-degree polynomials
    if max_degree >= 2:
        for col in feature_cols:
            df_new[f'{col}_squared'] = df[col] ** 2
            new_features.append(f'{col}_squared')

    # Third-degree polynomials
    if max_degree >= 3:
        for col in feature_cols:
            df_new[f'{col}_cubed'] = df[col] ** 3
            new_features.append(f'{col}_cubed')

    # Two-way interactions
    for col1, col2 in combinations(feature_cols, 2):
        df_new[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        new_features.append(f'{col1}_x_{col2}')

    # Three-way interactions (if degree >= 3)
    if max_degree >= 3 and len(feature_cols) >= 3:
        for col1, col2, col3 in combinations(feature_cols[:3], 3):  # Limit for performance
            df_new[f'{col1}_x_{col2}_x_{col3}'] = df[col1] * df[col2] * df[col3]
            new_features.append(f'{col1}_x_{col2}_x_{col3}')

    return df_new, new_features


def select_best_polynomial_features(X, y, k=20, method='f_regression'):
    """
    Select top k polynomial features based on statistical tests.
    """
    if method == 'f_regression':
        selector = SelectKBest(f_regression, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)

        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()

        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features

    return X, X.columns.tolist()


def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_type='linear', alpha=1.0):
    """
    Train and evaluate different model types.
    """
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = Ridge(alpha=alpha, random_state=42)
    elif model_type == 'lasso':
        model = Lasso(alpha=alpha, random_state=42, max_iter=10000)
    elif model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

    results = {
        'model': model,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred_test
    }

    return results


def plot_polynomial_analysis(df, results_dict, feature_importance_df):
    """
    Create comprehensive visualizations for polynomial feature analysis.
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Model Performance Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    models = list(results_dict.keys())
    test_r2_scores = [results_dict[m]['test_r2'] for m in models]
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))

    bars = ax1.bar(range(len(models)), test_r2_scores, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Test R² Score', fontsize=11, fontweight='bold')
    ax1.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # 2. RMSE Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    train_rmse = [results_dict[m]['train_rmse'] for m in models]
    test_rmse = [results_dict[m]['test_rmse'] for m in models]

    x = np.arange(len(models))
    width = 0.35
    ax2.bar(x - width/2, train_rmse, width, label='Train RMSE', alpha=0.8, color='skyblue')
    ax2.bar(x + width/2, test_rmse, width, label='Test RMSE', alpha=0.8, color='coral')
    ax2.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax2.set_ylabel('RMSE', fontsize=11, fontweight='bold')
    ax2.set_title('Train vs Test RMSE', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Cross-Validation Scores
    ax3 = fig.add_subplot(gs[0, 2])
    cv_means = [results_dict[m]['cv_mean'] for m in models]
    cv_stds = [results_dict[m]['cv_std'] for m in models]

    ax3.errorbar(range(len(models)), cv_means, yerr=cv_stds,
                fmt='o-', linewidth=2, markersize=8, capsize=5, color='darkgreen')
    ax3.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax3.set_ylabel('CV R² Score', fontsize=11, fontweight='bold')
    ax3.set_title('Cross-Validation Performance', fontsize=13, fontweight='bold')
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)

    # 4. Predictions vs Actual (best model)
    ax4 = fig.add_subplot(gs[1, 0])
    best_model_name = max(results_dict.keys(), key=lambda k: results_dict[k]['test_r2'])
    best_results = results_dict[best_model_name]

    ax4.scatter(df['target'], best_results['predictions'], alpha=0.6, s=30, color='purple')
    ax4.plot([df['target'].min(), df['target'].max()],
             [df['target'].min(), df['target'].max()],
             'r--', lw=2, label='Perfect Prediction')
    ax4.set_xlabel('Actual Values', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Predicted Values', fontsize=11, fontweight='bold')
    ax4.set_title(f'Predictions vs Actual ({best_model_name})', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Residuals Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    residuals = df['target'] - best_results['predictions']
    ax5.hist(residuals, bins=40, color='steelblue', alpha=0.7, edgecolor='black')
    ax5.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax5.set_xlabel('Residuals', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax5.set_title(f'Residual Distribution ({best_model_name})', fontsize=13, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Feature Importance
    ax6 = fig.add_subplot(gs[1, 2])
    if feature_importance_df is not None and len(feature_importance_df) > 0:
        top_features = feature_importance_df.head(15)
        ax6.barh(range(len(top_features)), top_features['importance'], color='teal', alpha=0.8)
        ax6.set_yticks(range(len(top_features)))
        ax6.set_yticklabels(top_features['feature'], fontsize=9)
        ax6.set_xlabel('Importance', fontsize=11, fontweight='bold')
        ax6.set_title('Top 15 Polynomial Features', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='x')

    # 7. Polynomial Degree Impact
    ax7 = fig.add_subplot(gs[2, 0])
    degrees = [1, 2, 3]
    degree_performance = []

    # This is illustrative - using existing results
    if 'Linear' in results_dict:
        degree_performance.append(results_dict['Linear']['test_r2'])
    if 'Poly-2' in results_dict:
        degree_performance.append(results_dict['Poly-2']['test_r2'])
    if 'Poly-3' in results_dict:
        degree_performance.append(results_dict['Poly-3']['test_r2'])

    if len(degree_performance) > 0:
        ax7.plot(degrees[:len(degree_performance)], degree_performance,
                marker='o', linewidth=2, markersize=10, color='darkblue')
        ax7.set_xlabel('Polynomial Degree', fontsize=11, fontweight='bold')
        ax7.set_ylabel('Test R² Score', fontsize=11, fontweight='bold')
        ax7.set_title('Polynomial Degree Impact', fontsize=13, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        ax7.set_xticks(degrees[:len(degree_performance)])

    # 8. Feature Count Impact
    ax8 = fig.add_subplot(gs[2, 1])
    feature_counts = []
    r2_scores = []
    for model_name, results in results_dict.items():
        if hasattr(results['model'], 'n_features_in_'):
            feature_counts.append(results['model'].n_features_in_)
            r2_scores.append(results['test_r2'])

    if len(feature_counts) > 0:
        ax8.scatter(feature_counts, r2_scores, s=100, alpha=0.7, color='crimson')
        for i, name in enumerate([k for k in results_dict.keys()
                                 if hasattr(results_dict[k]['model'], 'n_features_in_')]):
            ax8.annotate(name, (feature_counts[i], r2_scores[i]),
                        fontsize=8, ha='right', va='bottom')
        ax8.set_xlabel('Number of Features', fontsize=11, fontweight='bold')
        ax8.set_ylabel('Test R² Score', fontsize=11, fontweight='bold')
        ax8.set_title('Feature Count vs Performance', fontsize=13, fontweight='bold')
        ax8.grid(True, alpha=0.3)

    # 9. Overfitting Analysis
    ax9 = fig.add_subplot(gs[2, 2])
    train_r2 = [results_dict[m]['train_r2'] for m in models]
    test_r2 = [results_dict[m]['test_r2'] for m in models]
    overfit_gap = [train_r2[i] - test_r2[i] for i in range(len(models))]

    colors_overfit = ['green' if gap < 0.1 else 'orange' if gap < 0.2 else 'red'
                      for gap in overfit_gap]
    ax9.bar(range(len(models)), overfit_gap, color=colors_overfit, alpha=0.8, edgecolor='black')
    ax9.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax9.set_ylabel('Train R² - Test R²', fontsize=11, fontweight='bold')
    ax9.set_title('Overfitting Analysis', fontsize=13, fontweight='bold')
    ax9.set_xticks(range(len(models)))
    ax9.set_xticklabels(models, rotation=45, ha='right')
    ax9.axhline(y=0.1, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Warning')
    ax9.axhline(y=0.2, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Severe')
    ax9.legend()
    ax9.grid(True, alpha=0.3, axis='y')

    plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/13_feature_engineering/09_polynomial_features/polynomial_analysis.png',
                dpi=300, bbox_inches='tight')
    print("   Plot saved as 'polynomial_analysis.png'")
    plt.show()


def main():
    """
    Main execution function.
    """
    print("=" * 90)
    print("Polynomial Feature Generation and Selection")
    print("=" * 90)

    # Generate data
    print("\n1. Generating synthetic data with non-linear relationships...")
    df = generate_nonlinear_data(n_samples=1500)
    print(f"   Dataset shape: {df.shape}")
    print(f"   Features: {df.columns.tolist()}")

    # Split data
    print("\n2. Splitting data into train/test sets...")
    feature_cols = [col for col in df.columns if col != 'target']
    X = df[feature_cols]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")

    # Standardize features
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

    results_dict = {}

    # Baseline: Linear model with original features
    print("\n3. Training baseline linear model (degree 1)...")
    results_baseline = train_and_evaluate_model(
        X_train_scaled, X_test_scaled, y_train, y_test,
        model_type='linear'
    )
    results_dict['Linear'] = results_baseline
    print(f"   Test RMSE: {results_baseline['test_rmse']:.2f}")
    print(f"   Test R²: {results_baseline['test_r2']:.4f}")

    # Polynomial features - Degree 2
    print("\n4. Creating polynomial features (degree 2)...")
    X_train_poly2, poly2 = create_polynomial_features(X_train_scaled, degree=2)
    X_test_poly2 = pd.DataFrame(
        poly2.transform(X_test_scaled),
        columns=poly2.get_feature_names_out(X_test_scaled.columns)
    )
    print(f"   Original features: {X_train_scaled.shape[1]}")
    print(f"   After polynomial (degree 2): {X_train_poly2.shape[1]}")

    results_poly2 = train_and_evaluate_model(
        X_train_poly2, X_test_poly2, y_train, y_test,
        model_type='ridge', alpha=1.0
    )
    results_dict['Poly-2'] = results_poly2
    print(f"   Test RMSE: {results_poly2['test_rmse']:.2f}")
    print(f"   Test R²: {results_poly2['test_r2']:.4f}")

    # Polynomial features - Degree 3
    print("\n5. Creating polynomial features (degree 3)...")
    X_train_poly3, poly3 = create_polynomial_features(X_train_scaled, degree=3)
    X_test_poly3 = pd.DataFrame(
        poly3.transform(X_test_scaled),
        columns=poly3.get_feature_names_out(X_test_scaled.columns)
    )
    print(f"   After polynomial (degree 3): {X_train_poly3.shape[1]}")

    results_poly3 = train_and_evaluate_model(
        X_train_poly3, X_test_poly3, y_train, y_test,
        model_type='ridge', alpha=10.0
    )
    results_dict['Poly-3'] = results_poly3
    print(f"   Test RMSE: {results_poly3['test_rmse']:.2f}")
    print(f"   Test R²: {results_poly3['test_r2']:.4f}")

    # Interaction features only
    print("\n6. Creating interaction features only (degree 2)...")
    X_train_interact, interact = create_polynomial_features(
        X_train_scaled, degree=2, interaction_only=True
    )
    X_test_interact = pd.DataFrame(
        interact.transform(X_test_scaled),
        columns=interact.get_feature_names_out(X_test_scaled.columns)
    )
    print(f"   Interaction features: {X_train_interact.shape[1]}")

    results_interact = train_and_evaluate_model(
        X_train_interact, X_test_interact, y_train, y_test,
        model_type='ridge', alpha=1.0
    )
    results_dict['Interactions'] = results_interact
    print(f"   Test RMSE: {results_interact['test_rmse']:.2f}")
    print(f"   Test R²: {results_interact['test_r2']:.4f}")

    # Feature selection on polynomial features
    print("\n7. Selecting best polynomial features...")
    X_train_selected, selected_features = select_best_polynomial_features(
        X_train_poly2, y_train, k=15
    )
    X_test_selected = X_test_poly2[selected_features]
    print(f"   Selected {len(selected_features)} features from {X_train_poly2.shape[1]}")

    results_selected = train_and_evaluate_model(
        X_train_selected, X_test_selected, y_train, y_test,
        model_type='ridge', alpha=0.1
    )
    results_dict['Poly-Selected'] = results_selected
    print(f"   Test RMSE: {results_selected['test_rmse']:.2f}")
    print(f"   Test R²: {results_selected['test_r2']:.4f}")

    # Lasso for automatic feature selection
    print("\n8. Using Lasso for automatic feature selection...")
    results_lasso = train_and_evaluate_model(
        X_train_poly2, X_test_poly2, y_train, y_test,
        model_type='lasso', alpha=0.1
    )
    results_dict['Lasso'] = results_lasso

    # Count non-zero coefficients
    non_zero_coefs = np.sum(results_lasso['model'].coef_ != 0)
    print(f"   Non-zero coefficients: {non_zero_coefs}/{len(results_lasso['model'].coef_)}")
    print(f"   Test RMSE: {results_lasso['test_rmse']:.2f}")
    print(f"   Test R²: {results_lasso['test_r2']:.4f}")

    # Performance summary
    print("\n9. Performance Summary:")
    print("   " + "-" * 70)
    print(f"   {'Model':<20} {'Test RMSE':<15} {'Test R²':<15} {'CV R²':<15}")
    print("   " + "-" * 70)
    for model_name, results in results_dict.items():
        print(f"   {model_name:<20} {results['test_rmse']:<15.2f} "
              f"{results['test_r2']:<15.4f} {results['cv_mean']:<15.4f}")
    print("   " + "-" * 70)

    # Feature importance (from Lasso)
    print("\n10. Analyzing feature importance (Lasso coefficients)...")
    feature_importance_df = pd.DataFrame({
        'feature': X_train_poly2.columns,
        'importance': np.abs(results_lasso['model'].coef_)
    }).sort_values('importance', ascending=False)

    print("\n    Top 10 Most Important Polynomial Features:")
    for idx, row in feature_importance_df.head(10).iterrows():
        print(f"    {row['feature']:30s}: {row['importance']:.4f}")

    # Visualizations
    print("\n11. Creating comprehensive visualizations...")
    plot_polynomial_analysis(
        pd.DataFrame({'target': y_test}, index=X_test.index),
        results_dict,
        feature_importance_df
    )

    # Best model recommendation
    best_model = max(results_dict.items(), key=lambda x: x[1]['test_r2'])
    print(f"\n12. Best Model: {best_model[0]}")
    print(f"    Test R²: {best_model[1]['test_r2']:.4f}")
    print(f"    Test RMSE: {best_model[1]['test_rmse']:.2f}")
    print(f"    Improvement over baseline: {(best_model[1]['test_r2'] - results_baseline['test_r2']) / results_baseline['test_r2'] * 100:.1f}%")

    print("\n" + "=" * 90)
    print("Analysis Complete!")
    print("=" * 90)


if __name__ == "__main__":
    main()
