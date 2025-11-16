"""
Kaggle Solution: Polynomial Feature Engineering
===============================================
Demonstrates polynomial feature engineering for capturing non-linear
relationships and interactions between variables.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def generate_nonlinear_data(n_samples=1500):
    """
    Generate synthetic data with non-linear relationships.
    """
    # Generate features
    x1 = np.random.uniform(-5, 5, n_samples)
    x2 = np.random.uniform(-3, 3, n_samples)
    x3 = np.random.uniform(0, 10, n_samples)

    # Create target with polynomial and interaction effects
    y = (
        3 * x1 +                          # linear term
        -2 * x2 +                         # linear term
        0.5 * x3 +                        # linear term
        0.8 * x1**2 +                     # quadratic term
        -0.3 * x2**2 +                    # quadratic term
        0.1 * x3**2 +                     # quadratic term
        1.5 * x1 * x2 +                   # interaction
        -0.5 * x1 * x3 +                  # interaction
        0.3 * x2 * x3 +                   # interaction
        0.2 * x1**2 * x2 +                # higher-order
        -0.1 * x1 * x2**2 +               # higher-order
        np.random.normal(0, 2, n_samples) # noise
    )

    df = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'y': y
    })

    return df


def create_polynomial_features_manual(df, degree=2):
    """
    Manually create polynomial features for educational purposes.
    """
    df_poly = df.copy()

    feature_names = []

    # Original features
    for col in ['x1', 'x2', 'x3']:
        feature_names.append(col)

    if degree >= 2:
        # Squared terms
        for col in ['x1', 'x2', 'x3']:
            df_poly[f'{col}^2'] = df_poly[col] ** 2
            feature_names.append(f'{col}^2')

        # Interaction terms
        df_poly['x1*x2'] = df_poly['x1'] * df_poly['x2']
        df_poly['x1*x3'] = df_poly['x1'] * df_poly['x3']
        df_poly['x2*x3'] = df_poly['x2'] * df_poly['x3']
        feature_names.extend(['x1*x2', 'x1*x3', 'x2*x3'])

    if degree >= 3:
        # Cubic terms
        for col in ['x1', 'x2', 'x3']:
            df_poly[f'{col}^3'] = df_poly[col] ** 3
            feature_names.append(f'{col}^3')

        # Higher-order interactions
        df_poly['x1^2*x2'] = df_poly['x1']**2 * df_poly['x2']
        df_poly['x1*x2^2'] = df_poly['x1'] * df_poly['x2']**2
        df_poly['x1^2*x3'] = df_poly['x1']**2 * df_poly['x3']
        df_poly['x1*x3^2'] = df_poly['x1'] * df_poly['x3']**2
        df_poly['x2^2*x3'] = df_poly['x2']**2 * df_poly['x3']
        df_poly['x2*x3^2'] = df_poly['x2'] * df_poly['x3']**2
        df_poly['x1*x2*x3'] = df_poly['x1'] * df_poly['x2'] * df_poly['x3']
        feature_names.extend(['x1^2*x2', 'x1*x2^2', 'x1^2*x3', 'x1*x3^2',
                            'x2^2*x3', 'x2*x3^2', 'x1*x2*x3'])

    return df_poly, feature_names


def train_models_with_polynomial_features(X_train, X_test, y_train, y_test, max_degree=4):
    """
    Train models with different polynomial degrees.
    """
    results = []

    for degree in range(1, max_degree + 1):
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_poly)
        X_test_scaled = scaler.transform(X_test_poly)

        # Train models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1, max_iter=10000)
        }

        for model_name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Calculate number of non-zero coefficients (for Lasso)
            if hasattr(model, 'coef_'):
                n_features_used = np.sum(model.coef_ != 0)
            else:
                n_features_used = X_train_poly.shape[1]

            results.append({
                'degree': degree,
                'model': model_name,
                'n_features': X_train_poly.shape[1],
                'n_features_used': n_features_used,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'model_obj': model,
                'predictions': y_pred
            })

    return pd.DataFrame(results)


def analyze_overfitting(X, y, max_degree=6):
    """
    Analyze overfitting with increasing polynomial degrees.
    """
    train_scores = []
    test_scores = []
    degrees = range(1, max_degree + 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for degree in degrees:
        # Create pipeline
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=1.0))
        ])

        # Fit and score
        pipeline.fit(X_train, y_train)
        train_score = pipeline.score(X_train, y_train)
        test_score = pipeline.score(X_test, y_test)

        train_scores.append(train_score)
        test_scores.append(test_score)

    return degrees, train_scores, test_scores


def plot_results(results_df, X_test, y_test, degrees, train_scores, test_scores):
    """
    Create comprehensive visualizations.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Performance by degree and model
    ax1 = fig.add_subplot(gs[0, :2])
    for model_name in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model_name]
        ax1.plot(model_data['degree'], model_data['r2'], marker='o', label=model_name, linewidth=2)
    ax1.set_xlabel('Polynomial Degree', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('Model Performance vs Polynomial Degree', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, results_df['degree'].max() + 1))

    # 2. Number of features by degree
    ax2 = fig.add_subplot(gs[0, 2])
    unique_degrees = results_df.drop_duplicates('degree')
    ax2.bar(unique_degrees['degree'], unique_degrees['n_features'], alpha=0.7, color='steelblue')
    ax2.set_xlabel('Polynomial Degree', fontsize=12)
    ax2.set_ylabel('Number of Features', fontsize=12)
    ax2.set_title('Feature Explosion', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(range(1, results_df['degree'].max() + 1))

    # 3. Overfitting analysis
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(degrees, train_scores, marker='o', label='Training Score', linewidth=2)
    ax3.plot(degrees, test_scores, marker='s', label='Test Score', linewidth=2)
    ax3.fill_between(degrees, train_scores, test_scores, alpha=0.2)
    ax3.set_xlabel('Polynomial Degree', fontsize=12)
    ax3.set_ylabel('R² Score', fontsize=12)
    ax3.set_title('Overfitting Analysis: Train vs Test Performance', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(degrees)

    # 4. RMSE comparison
    ax4 = fig.add_subplot(gs[1, 2])
    degree_2_data = results_df[results_df['degree'] == 2]
    models = degree_2_data['model'].values
    rmse_vals = degree_2_data['rmse'].values
    colors = plt.cm.RdYlGn_r(rmse_vals / rmse_vals.max())
    ax4.barh(range(len(models)), rmse_vals, color=colors, alpha=0.8)
    ax4.set_yticks(range(len(models)))
    ax4.set_yticklabels(models)
    ax4.set_xlabel('RMSE', fontsize=12)
    ax4.set_title('RMSE Comparison (Degree 2)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')

    # 5. Predictions scatter (best model)
    ax5 = fig.add_subplot(gs[2, 0])
    best_result = results_df.loc[results_df['r2'].idxmax()]
    ax5.scatter(y_test, best_result['predictions'], alpha=0.5, s=30)
    ax5.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax5.set_xlabel('Actual Values', fontsize=12)
    ax5.set_ylabel('Predicted Values', fontsize=12)
    ax5.set_title(f'Best Model: {best_result["model"]} (Degree {best_result["degree"]})',
                 fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # 6. Residuals (best model)
    ax6 = fig.add_subplot(gs[2, 1])
    residuals = y_test - best_result['predictions']
    ax6.scatter(best_result['predictions'], residuals, alpha=0.5, s=30)
    ax6.axhline(y=0, color='r', linestyle='--', lw=2)
    ax6.set_xlabel('Predicted Values', fontsize=12)
    ax6.set_ylabel('Residuals', fontsize=12)
    ax6.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # 7. Feature count vs performance efficiency
    ax7 = fig.add_subplot(gs[2, 2])
    efficiency = results_df['r2'] / np.log1p(results_df['n_features'])
    degree_2 = results_df[results_df['degree'] == 2]
    ax7.scatter(degree_2['n_features'], degree_2['r2'], s=200, alpha=0.6,
               c=range(len(degree_2)), cmap='viridis')
    for idx, row in degree_2.iterrows():
        ax7.annotate(row['model'], (row['n_features'], row['r2']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax7.set_xlabel('Number of Features', fontsize=12)
    ax7.set_ylabel('R² Score', fontsize=12)
    ax7.set_title('Features vs Performance (Degree 2)', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)

    plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/13_feature_engineering/03_polynomial_features/polynomial_features_analysis.png',
                dpi=300, bbox_inches='tight')
    print("Plot saved as 'polynomial_features_analysis.png'")
    plt.show()


def main():
    """
    Main execution function.
    """
    print("=" * 80)
    print("Polynomial Feature Engineering Example")
    print("=" * 80)

    # Generate data
    print("\n1. Generating non-linear synthetic data...")
    df = generate_nonlinear_data(n_samples=1500)
    print(f"   Dataset shape: {df.shape}")
    print(f"   Features: {df.drop('y', axis=1).columns.tolist()}")

    # Split data
    print("\n2. Splitting data...")
    X = df[['x1', 'x2', 'x3']]
    y = df['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Baseline model (linear, no polynomial features)
    print("\n3. Training baseline linear model...")
    baseline_model = LinearRegression()
    baseline_model.fit(X_train, y_train)
    y_pred_baseline = baseline_model.predict(X_test)
    baseline_r2 = r2_score(y_test, y_pred_baseline)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
    print(f"   Baseline R²: {baseline_r2:.4f}")
    print(f"   Baseline RMSE: {baseline_rmse:.2f}")

    # Train models with different polynomial degrees
    print("\n4. Training models with polynomial features...")
    results_df = train_models_with_polynomial_features(X_train, X_test, y_train, y_test, max_degree=4)

    print("\n5. Results Summary:")
    print("-" * 80)
    summary = results_df.groupby('degree').agg({
        'n_features': 'first',
        'r2': 'max',
        'rmse': 'min'
    }).round(4)
    print(summary)

    # Best model
    print("\n6. Best Model:")
    best_idx = results_df['r2'].idxmax()
    best_result = results_df.loc[best_idx]
    print(f"   Model: {best_result['model']}")
    print(f"   Degree: {best_result['degree']}")
    print(f"   Features: {best_result['n_features']}")
    print(f"   R²: {best_result['r2']:.4f}")
    print(f"   RMSE: {best_result['rmse']:.2f}")

    # Improvement
    improvement = ((best_result['r2'] - baseline_r2) / baseline_r2) * 100
    print(f"\n   Improvement over baseline: {improvement:.2f}%")

    # Overfitting analysis
    print("\n7. Analyzing overfitting with higher degrees...")
    degrees, train_scores, test_scores = analyze_overfitting(X, y, max_degree=6)

    print("\n   Degree | Train R² | Test R² | Gap")
    print("   " + "-" * 40)
    for d, tr, te in zip(degrees, train_scores, test_scores):
        gap = tr - te
        print(f"   {d:6d} | {tr:8.4f} | {te:7.4f} | {gap:.4f}")

    # Feature explosion warning
    print("\n8. Feature Explosion Analysis:")
    poly = PolynomialFeatures(degree=1, include_bias=False)
    for degree in range(1, 7):
        poly.set_params(degree=degree)
        poly.fit(X_train)
        n_features = poly.n_output_features_
        print(f"   Degree {degree}: {n_features} features")

    # Regularization comparison
    print("\n9. Regularization Effectiveness (Degree 2):")
    degree_2_results = results_df[results_df['degree'] == 2]
    for _, row in degree_2_results.iterrows():
        sparsity = (1 - row['n_features_used'] / row['n_features']) * 100
        print(f"   {row['model']:20s}: R²={row['r2']:.4f}, "
              f"Features Used={row['n_features_used']}/{row['n_features']} "
              f"(Sparsity: {sparsity:.1f}%)")

    # Visualizations
    print("\n10. Creating visualizations...")
    plot_results(results_df, X_test, y_test, degrees, train_scores, test_scores)

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
