"""
Kaggle Solution: Feature Interactions and Cross-Products
========================================================
Demonstrates creating meaningful feature interactions and cross-products
to capture complex relationships between variables.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from itertools import combinations, permutations
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)


def generate_interaction_data(n_samples=2500):
    """
    Generate data with strong interaction effects.
    """
    x1 = np.random.uniform(0, 10, n_samples)
    x2 = np.random.uniform(0, 10, n_samples)
    x3 = np.random.uniform(0, 10, n_samples)
    x4 = np.random.uniform(0, 10, n_samples)
    x5 = np.random.uniform(0, 10, n_samples)
    
    # Strong interaction effects
    target = (
        3 * x1 * x2 +  # Two-way interaction
        -2 * x2 * x3 +  # Another two-way
        5 * x1 * x2 * x3 +  # Three-way interaction
        0.5 * x1 * x4 +
        -0.3 * x3 * x5 +
        2 * x1 / (x2 + 1) +  # Ratio interaction
        1.5 * (x3 - x4) +  # Difference interaction
        np.maximum(x1, x2) * 2 +  # Max interaction
        np.random.normal(0, 5, n_samples)
    )
    
    df = pd.DataFrame({
        'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5,
        'target': target
    })
    
    return df


def create_multiplicative_interactions(df, feature_cols, max_degree=2):
    """
    Create multiplicative interaction features.
    """
    df_new = df.copy()
    
    # Two-way interactions
    for f1, f2 in combinations(feature_cols, 2):
        df_new[f'{f1}_x_{f2}'] = df[f1] * df[f2]
    
    # Three-way interactions (selective)
    if max_degree >= 3 and len(feature_cols) >= 3:
        for f1, f2, f3 in list(combinations(feature_cols, 3))[:5]:
            df_new[f'{f1}_x_{f2}_x_{f3}'] = df[f1] * df[f2] * df[f3]
    
    return df_new


def create_division_interactions(df, feature_cols):
    """
    Create ratio/division interaction features.
    """
    df_new = df.copy()
    
    for f1, f2 in combinations(feature_cols, 2):
        # Avoid division by zero
        df_new[f'{f1}_div_{f2}'] = df[f1] / (df[f2] + 1e-5)
        df_new[f'{f2}_div_{f1}'] = df[f2] / (df[f1] + 1e-5)
    
    return df_new


def create_arithmetic_interactions(df, feature_cols):
    """
    Create addition and subtraction interactions.
    """
    df_new = df.copy()
    
    for f1, f2 in combinations(feature_cols, 2):
        df_new[f'{f1}_plus_{f2}'] = df[f1] + df[f2]
        df_new[f'{f1}_minus_{f2}'] = df[f1] - df[f2]
        df_new[f'{f2}_minus_{f1}'] = df[f2] - df[f1]
    
    return df_new


def create_comparison_interactions(df, feature_cols):
    """
    Create comparison-based interactions.
    """
    df_new = df.copy()
    
    for f1, f2 in combinations(feature_cols, 2):
        df_new[f'{f1}_gt_{f2}'] = (df[f1] > df[f2]).astype(int)
        df_new[f'{f1}_max_{f2}'] = np.maximum(df[f1], df[f2])
        df_new[f'{f1}_min_{f2}'] = np.minimum(df[f1], df[f2])
    
    return df_new


def create_statistical_interactions(df, feature_cols):
    """
    Create statistical interaction features.
    """
    df_new = df.copy()
    
    # Mean and std of pairs
    for f1, f2 in combinations(feature_cols, 2):
        df_new[f'{f1}_{f2}_mean'] = (df[f1] + df[f2]) / 2
        df_new[f'{f1}_{f2}_std'] = np.sqrt(((df[f1] - df[[f1, f2]].mean(axis=1))**2 + 
                                            (df[f2] - df[[f1, f2]].mean(axis=1))**2) / 2)
    
    # Overall statistics
    df_new['features_mean'] = df[feature_cols].mean(axis=1)
    df_new['features_std'] = df[feature_cols].std(axis=1)
    df_new['features_max'] = df[feature_cols].max(axis=1)
    df_new['features_min'] = df[feature_cols].min(axis=1)
    df_new['features_range'] = df_new['features_max'] - df_new['features_min']
    
    return df_new


def select_important_interactions(X_train, y_train, X_test, threshold=0.01):
    """
    Select important interactions using Lasso.
    """
    lasso = Lasso(alpha=0.1, random_state=42)
    lasso.fit(X_train, y_train)
    
    # Get non-zero coefficients
    important_mask = np.abs(lasso.coef_) > threshold
    important_features = X_train.columns[important_mask].tolist()
    
    return X_train[important_features], X_test[important_features], important_features


def train_and_evaluate(X_train, X_test, y_train, y_test, model_type='ridge'):
    """
    Train and evaluate model.
    """
    if model_type == 'ridge':
        model = Ridge(alpha=1.0, random_state=42)
    elif model_type == 'lasso':
        model = Lasso(alpha=0.1, random_state=42)
    elif model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'gbm':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    
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


def plot_interaction_analysis(df, results_dict, interaction_importance):
    """
    Create comprehensive visualizations.
    """
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
    
    # 1. Original feature correlations
    ax1 = fig.add_subplot(gs[0, 0])
    feature_cols = ['x1', 'x2', 'x3', 'x4', 'x5']
    corr = df[feature_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax1,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    ax1.set_title('Original Feature Correlations', fontsize=13, fontweight='bold')
    
    # 2. Model Performance Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    models = list(results_dict.keys())
    test_r2 = [results_dict[m]['test_r2'] for m in models]
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    
    bars = ax2.barh(range(len(models)), test_r2, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_yticks(range(len(models)))
    ax2.set_yticklabels(models, fontsize=9)
    ax2.set_xlabel('Test R² Score', fontsize=11, fontweight='bold')
    ax2.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}', ha='left', va='center', fontsize=8)
    
    # 3. Feature Count
    ax3 = fig.add_subplot(gs[0, 2])
    feature_counts = []
    model_names = []
    for name in models:
        if hasattr(results_dict[name]['model'], 'n_features_in_'):
            feature_counts.append(results_dict[name]['model'].n_features_in_)
            model_names.append(name)
    
    if feature_counts:
        ax3.bar(range(len(model_names)), feature_counts, alpha=0.8, 
               color='coral', edgecolor='black')
        ax3.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Number of Features', fontsize=11, fontweight='bold')
        ax3.set_title('Feature Count by Model', fontsize=13, fontweight='bold')
        ax3.set_xticks(range(len(model_names)))
        ax3.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Top Interaction Importance
    ax4 = fig.add_subplot(gs[1, :2])
    if interaction_importance is not None and len(interaction_importance) > 0:
        top_n = min(20, len(interaction_importance))
        top_interactions = interaction_importance.head(top_n)
        
        colors_imp = ['red' if '_x_' in feat else 'blue' if '_div_' in feat 
                     else 'green' if '_minus_' in feat else 'orange'
                     for feat in top_interactions['feature']]
        
        ax4.barh(range(top_n), top_interactions['importance'], 
                color=colors_imp, alpha=0.8, edgecolor='black')
        ax4.set_yticks(range(top_n))
        ax4.set_yticklabels(top_interactions['feature'], fontsize=8)
        ax4.set_xlabel('Importance', fontsize=11, fontweight='bold')
        ax4.set_title('Top 20 Most Important Interactions', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
    
    # 5. Interaction Type Distribution
    ax5 = fig.add_subplot(gs[1, 2])
    if interaction_importance is not None and len(interaction_importance) > 0:
        interaction_types = []
        for feat in interaction_importance['feature']:
            if '_x_' in feat:
                interaction_types.append('Multiply')
            elif '_div_' in feat:
                interaction_types.append('Divide')
            elif '_plus_' in feat:
                interaction_types.append('Add')
            elif '_minus_' in feat:
                interaction_types.append('Subtract')
            elif '_max_' in feat or '_min_' in feat:
                interaction_types.append('MinMax')
            elif '_gt_' in feat:
                interaction_types.append('Compare')
            else:
                interaction_types.append('Other')
        
        type_counts = pd.Series(interaction_types).value_counts()
        ax5.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
               startangle=90, colors=plt.cm.Set3(range(len(type_counts))))
        ax5.set_title('Interaction Type Distribution', fontsize=13, fontweight='bold')
    
    # 6-9. Predictions and Residuals
    ax6 = fig.add_subplot(gs[2, 0])
    best_model = max(results_dict.keys(), key=lambda k: results_dict[k]['test_r2'])
    best_preds = results_dict[best_model]['predictions']
    y_test_subset = df.iloc[-len(best_preds):]['target']
    
    ax6.scatter(y_test_subset, best_preds, alpha=0.5, s=30, color='purple')
    ax6.plot([y_test_subset.min(), y_test_subset.max()],
            [y_test_subset.min(), y_test_subset.max()],
            'r--', lw=2, label='Perfect')
    ax6.set_xlabel('Actual', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Predicted', fontsize=11, fontweight='bold')
    ax6.set_title(f'Best Model: {best_model}', fontsize=13, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. RMSE Comparison
    ax7 = fig.add_subplot(gs[2, 1])
    test_rmse = [results_dict[m]['test_rmse'] for m in models]
    train_rmse = [results_dict[m]['train_rmse'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    ax7.bar(x - width/2, train_rmse, width, label='Train', alpha=0.8, color='steelblue')
    ax7.bar(x + width/2, test_rmse, width, label='Test', alpha=0.8, color='coral')
    ax7.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax7.set_ylabel('RMSE', fontsize=11, fontweight='bold')
    ax7.set_title('RMSE Comparison', fontsize=13, fontweight='bold')
    ax7.set_xticks(x)
    ax7.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. CV Scores
    ax8 = fig.add_subplot(gs[2, 2])
    cv_means = [results_dict[m]['cv_mean'] for m in models]
    cv_stds = [results_dict[m]['cv_std'] for m in models]
    
    ax8.errorbar(range(len(models)), cv_means, yerr=cv_stds,
                fmt='o-', linewidth=2, markersize=8, capsize=5, color='darkgreen')
    ax8.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax8.set_ylabel('CV R² Score', fontsize=11, fontweight='bold')
    ax8.set_title('Cross-Validation Performance', fontsize=13, fontweight='bold')
    ax8.set_xticks(range(len(models)))
    ax8.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax8.grid(True, alpha=0.3)
    
    # 9. Performance Improvement
    ax9 = fig.add_subplot(gs[3, 0])
    baseline_r2 = results_dict['Original']['test_r2']
    improvements = [(results_dict[m]['test_r2'] - baseline_r2) / baseline_r2 * 100
                   for m in models if m != 'Original']
    improved_models = [m for m in models if m != 'Original']
    
    colors_imp = ['green' if imp > 0 else 'red' for imp in improvements]
    ax9.barh(range(len(improvements)), improvements, color=colors_imp,
            alpha=0.8, edgecolor='black')
    ax9.set_yticks(range(len(improvements)))
    ax9.set_yticklabels(improved_models, fontsize=9)
    ax9.set_xlabel('Improvement (%)', fontsize=11, fontweight='bold')
    ax9.set_title('Performance vs Baseline', fontsize=13, fontweight='bold')
    ax9.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax9.grid(True, alpha=0.3, axis='x')
    
    # 10. Residuals
    ax10 = fig.add_subplot(gs[3, 1])
    residuals = y_test_subset - best_preds
    ax10.hist(residuals, bins=40, color='teal', alpha=0.7, edgecolor='black')
    ax10.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax10.set_xlabel('Residuals', fontsize=11, fontweight='bold')
    ax10.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax10.set_title('Residual Distribution', fontsize=13, fontweight='bold')
    ax10.grid(True, alpha=0.3, axis='y')
    
    # 11. Overfitting Analysis
    ax11 = fig.add_subplot(gs[3, 2])
    train_r2 = [results_dict[m]['train_r2'] for m in models]
    gap = [train_r2[i] - test_r2[i] for i in range(len(models))]
    
    colors_gap = ['green' if g < 0.05 else 'orange' if g < 0.15 else 'red' for g in gap]
    ax11.bar(range(len(models)), gap, color=colors_gap, alpha=0.8, edgecolor='black')
    ax11.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax11.set_ylabel('Train R² - Test R²', fontsize=11, fontweight='bold')
    ax11.set_title('Overfitting Analysis', fontsize=13, fontweight='bold')
    ax11.set_xticks(range(len(models)))
    ax11.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax11.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5)
    ax11.axhline(y=0.15, color='red', linestyle='--', alpha=0.5)
    ax11.grid(True, alpha=0.3, axis='y')
    
    plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/13_feature_engineering/14_feature_interactions/interaction_analysis.png',
                dpi=300, bbox_inches='tight')
    print("   Comprehensive plot saved!")
    plt.show()


def main():
    """
    Main execution function.
    """
    print("=" * 90)
    print("Feature Interactions and Cross-Products")
    print("=" * 90)
    
    # Generate data
    print("\n1. Generating data with interaction effects...")
    df = generate_interaction_data(n_samples=2500)
    print(f"   Dataset shape: {df.shape}")
    
    # Split data
    feature_cols = ['x1', 'x2', 'x3', 'x4', 'x5']
    X = df[feature_cols]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results_dict = {}
    
    # Baseline
    print("\n2. Training baseline model...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results_baseline = train_and_evaluate(X_train_scaled, X_test_scaled,
                                         y_train, y_test, 'ridge')
    results_dict['Original'] = results_baseline
    print(f"   Test R²: {results_baseline['test_r2']:.4f}")
    
    # Multiplicative interactions
    print("\n3. Creating multiplicative interactions...")
    X_train_mult = create_multiplicative_interactions(X_train, feature_cols, max_degree=2)
    X_test_mult = create_multiplicative_interactions(X_test, feature_cols, max_degree=2)
    print(f"   Features: {X_train.shape[1]} -> {X_train_mult.shape[1]}")
    
    scaler_mult = StandardScaler()
    X_train_mult_scaled = scaler_mult.fit_transform(X_train_mult)
    X_test_mult_scaled = scaler_mult.transform(X_test_mult)
    
    results_mult = train_and_evaluate(X_train_mult_scaled, X_test_mult_scaled,
                                     y_train, y_test, 'ridge')
    results_dict['Multiplicative'] = results_mult
    print(f"   Test R²: {results_mult['test_r2']:.4f}")
    
    # Division interactions
    print("\n4. Adding division interactions...")
    X_train_div = create_division_interactions(X_train, feature_cols)
    X_test_div = create_division_interactions(X_test, feature_cols)
    print(f"   Features: {X_train.shape[1]} -> {X_train_div.shape[1]}")
    
    scaler_div = StandardScaler()
    X_train_div_scaled = scaler_div.fit_transform(X_train_div)
    X_test_div_scaled = scaler_div.transform(X_test_div)
    
    results_div = train_and_evaluate(X_train_div_scaled, X_test_div_scaled,
                                    y_train, y_test, 'ridge')
    results_dict['Division'] = results_div
    print(f"   Test R²: {results_div['test_r2']:.4f}")
    
    # All interactions
    print("\n5. Creating all interaction types...")
    X_train_all = X_train.copy()
    X_test_all = X_test.copy()
    
    X_train_all = create_multiplicative_interactions(X_train_all, feature_cols)
    X_train_all = create_division_interactions(X_train_all, feature_cols)
    X_train_all = create_arithmetic_interactions(X_train_all, feature_cols)
    X_train_all = create_comparison_interactions(X_train_all, feature_cols)
    X_train_all = create_statistical_interactions(X_train_all, feature_cols)
    
    X_test_all = create_multiplicative_interactions(X_test_all, feature_cols)
    X_test_all = create_division_interactions(X_test_all, feature_cols)
    X_test_all = create_arithmetic_interactions(X_test_all, feature_cols)
    X_test_all = create_comparison_interactions(X_test_all, feature_cols)
    X_test_all = create_statistical_interactions(X_test_all, feature_cols)
    
    print(f"   Features: {X_train.shape[1]} -> {X_train_all.shape[1]}")
    
    scaler_all = StandardScaler()
    X_train_all_scaled = scaler_all.fit_transform(X_train_all)
    X_test_all_scaled = scaler_all.transform(X_test_all)
    
    results_all = train_and_evaluate(X_train_all_scaled, X_test_all_scaled,
                                    y_train, y_test, 'ridge')
    results_dict['All-Interactions'] = results_all
    print(f"   Test R²: {results_all['test_r2']:.4f}")
    
    # Feature selection
    print("\n6. Selecting important interactions...")
    X_train_sel, X_test_sel, selected_feats = select_important_interactions(
        pd.DataFrame(X_train_all_scaled, columns=X_train_all.columns),
        y_train,
        pd.DataFrame(X_test_all_scaled, columns=X_test_all.columns),
        threshold=0.01
    )
    print(f"   Selected features: {len(selected_feats)} from {X_train_all.shape[1]}")
    
    results_sel = train_and_evaluate(X_train_sel, X_test_sel,
                                    y_train, y_test, 'ridge')
    results_dict['Selected'] = results_sel
    print(f"   Test R²: {results_sel['test_r2']:.4f}")
    
    # Random Forest on interactions
    print("\n7. Training Random Forest on interactions...")
    results_rf = train_and_evaluate(X_train_all_scaled, X_test_all_scaled,
                                   y_train, y_test, 'rf')
    results_dict['RF-All'] = results_rf
    print(f"   Test R²: {results_rf['test_r2']:.4f}")
    
    # Feature importance
    print("\n8. Analyzing interaction importance...")
    if hasattr(results_rf['model'], 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': X_train_all.columns,
            'importance': results_rf['model'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n   Top 10 Most Important Interactions:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"   {row['feature']:30s}: {row['importance']:.4f}")
    else:
        importance_df = None
    
    # Summary
    print("\n9. Performance Summary:")
    print("   " + "-" * 75)
    print(f"   {'Model':<20} {'Features':<10} {'Test RMSE':<12} {'Test R²':<12} {'CV R²':<12}")
    print("   " + "-" * 75)
    for model_name, results in results_dict.items():
        n_feats = results['model'].n_features_in_ if hasattr(results['model'], 'n_features_in_') else 'N/A'
        print(f"   {model_name:<20} {n_feats:<10} {results['test_rmse']:<12.2f} "
              f"{results['test_r2']:<12.4f} {results['cv_mean']:<12.4f}")
    print("   " + "-" * 75)
    
    # Visualizations
    print("\n10. Creating comprehensive visualizations...")
    plot_interaction_analysis(df, results_dict, importance_df)
    
    # Best model
    best_model = max(results_dict.items(), key=lambda x: x[1]['test_r2'])
    print(f"\n11. Best Model: {best_model[0]}")
    print(f"    Test R²: {best_model[1]['test_r2']:.4f}")
    improvement = ((best_model[1]['test_r2'] - results_baseline['test_r2']) /
                   results_baseline['test_r2'] * 100)
    print(f"    Improvement: {improvement:.1f}%")
    
    print("\n" + "=" * 90)
    print("Analysis Complete!")
    print("=" * 90)


if __name__ == "__main__":
    main()
