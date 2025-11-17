"""
Kaggle Solution: Binning and Discretization Strategies
======================================================
Demonstrates various binning and discretization techniques to convert continuous
features into categorical bins for improved model performance and interpretability.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score,
                            classification_report, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def generate_customer_data(n_samples=2500):
    """
    Generate synthetic customer data for binning analysis.
    """
    # Customer features
    age = np.random.gamma(shape=5, scale=8, size=n_samples) + 18
    age = np.clip(age, 18, 80)

    income = np.random.lognormal(mean=10.5, sigma=0.8, size=n_samples)
    credit_score = np.random.normal(700, 100, n_samples)
    credit_score = np.clip(credit_score, 300, 850)

    years_employed = np.random.exponential(scale=5, size=n_samples)
    years_employed = np.clip(years_employed, 0, 40)

    debt_to_income = np.random.beta(a=2, b=5, size=n_samples)
    num_accounts = np.random.poisson(lam=3, size=n_samples) + 1

    # Target (regression): Customer lifetime value
    clv = (
        0.5 * income / 1000 +
        10 * (credit_score - 600) / 50 +
        5 * years_employed +
        -50 * debt_to_income +
        3 * num_accounts +
        20 * (age > 30) +
        np.random.normal(0, 20, n_samples)
    )

    # Classification target: High value customer
    high_value = (clv > np.median(clv)).astype(int)

    df = pd.DataFrame({
        'age': age,
        'income': income,
        'credit_score': credit_score,
        'years_employed': years_employed,
        'debt_to_income': debt_to_income,
        'num_accounts': num_accounts,
        'clv': clv,
        'high_value': high_value
    })

    return df


def equal_width_binning(data, column, n_bins=5):
    """
    Create equal-width bins.
    """
    binned = pd.cut(data[column], bins=n_bins, labels=False)
    bin_edges = pd.cut(data[column], bins=n_bins, retbins=True)[1]
    return binned, bin_edges


def equal_frequency_binning(data, column, n_bins=5):
    """
    Create equal-frequency (quantile) bins.
    """
    binned = pd.qcut(data[column], q=n_bins, labels=False, duplicates='drop')
    bin_edges = pd.qcut(data[column], q=n_bins, retbins=True, duplicates='drop')[1]
    return binned, bin_edges


def custom_binning(data, column, bin_edges):
    """
    Create custom bins based on domain knowledge.
    """
    binned = pd.cut(data[column], bins=bin_edges, labels=False, include_lowest=True)
    return binned


def kmeans_binning(data, column, n_bins=5):
    """
    Create bins using KBinsDiscretizer with k-means strategy.
    """
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans')
    binned = discretizer.fit_transform(data[[column]]).flatten()
    return binned, discretizer


def create_binned_features(df, feature_configs):
    """
    Create various binned features based on configurations.
    """
    df_binned = df.copy()

    for feature, config in feature_configs.items():
        if config['type'] == 'equal_width':
            binned, edges = equal_width_binning(df, feature, config['n_bins'])
            df_binned[f'{feature}_bin_width'] = binned

        elif config['type'] == 'equal_freq':
            binned, edges = equal_frequency_binning(df, feature, config['n_bins'])
            df_binned[f'{feature}_bin_freq'] = binned

        elif config['type'] == 'custom':
            binned = custom_binning(df, feature, config['edges'])
            df_binned[f'{feature}_bin_custom'] = binned

        elif config['type'] == 'kmeans':
            binned, disc = kmeans_binning(df, feature, config['n_bins'])
            df_binned[f'{feature}_bin_kmeans'] = binned

    return df_binned


def analyze_bins(data, column, binned_column):
    """
    Analyze the characteristics of bins.
    """
    df_analysis = pd.DataFrame({
        'original': data[column],
        'bin': binned_column
    })

    bin_stats = df_analysis.groupby('bin')['original'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(2)

    return bin_stats


def train_and_evaluate_regression(X_train, X_test, y_train, y_test, model_name='Ridge'):
    """
    Train and evaluate regression model.
    """
    if model_name == 'Ridge':
        model = Ridge(alpha=1.0, random_state=42)
    elif model_name == 'RF':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == 'GBM':
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


def plot_binning_analysis(df, df_binned, results_dict):
    """
    Create comprehensive visualizations for binning analysis.
    """
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

    # 1. Age distribution with different binning strategies
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df['age'], bins=50, alpha=0.6, color='skyblue', edgecolor='black', label='Original')
    ax1.set_xlabel('Age', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Original Age Distribution', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Equal-width bins
    ax2 = fig.add_subplot(gs[0, 1])
    if 'age_bin_width' in df_binned.columns:
        bin_counts = df_binned['age_bin_width'].value_counts().sort_index()
        ax2.bar(bin_counts.index, bin_counts.values, alpha=0.8, color='coral', edgecolor='black')
        ax2.set_xlabel('Bin Number', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax2.set_title('Equal-Width Binning', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

    # 3. Equal-frequency bins
    ax3 = fig.add_subplot(gs[0, 2])
    if 'age_bin_freq' in df_binned.columns:
        bin_counts = df_binned['age_bin_freq'].value_counts().sort_index()
        ax3.bar(bin_counts.index, bin_counts.values, alpha=0.8, color='lightgreen', edgecolor='black')
        ax3.set_xlabel('Bin Number', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax3.set_title('Equal-Frequency Binning', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

    # 4. Income binning comparison
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(df['income'], bins=50, alpha=0.6, color='purple', edgecolor='black')
    ax4.set_xlabel('Income', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax4.set_title('Original Income Distribution (Skewed)', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 5. Model Performance Comparison
    ax5 = fig.add_subplot(gs[1, 1])
    models = list(results_dict.keys())
    test_r2 = [results_dict[m]['test_r2'] for m in models]
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))

    bars = ax5.barh(range(len(models)), test_r2, color=colors, alpha=0.8, edgecolor='black')
    ax5.set_yticks(range(len(models)))
    ax5.set_yticklabels(models, fontsize=9)
    ax5.set_xlabel('Test R² Score', fontsize=11, fontweight='bold')
    ax5.set_title('Model Performance by Binning Strategy', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='x')

    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax5.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}', ha='left', va='center', fontsize=8)

    # 6. RMSE Comparison
    ax6 = fig.add_subplot(gs[1, 2])
    test_rmse = [results_dict[m]['test_rmse'] for m in models]
    ax6.bar(range(len(models)), test_rmse, alpha=0.8, color='crimson', edgecolor='black')
    ax6.set_xlabel('Binning Strategy', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Test RMSE', fontsize=11, fontweight='bold')
    ax6.set_title('RMSE Comparison', fontsize=13, fontweight='bold')
    ax6.set_xticks(range(len(models)))
    ax6.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax6.grid(True, alpha=0.3, axis='y')

    # 7. Credit Score bins analysis
    ax7 = fig.add_subplot(gs[2, 0])
    if 'credit_score_bin_width' in df_binned.columns:
        bin_means = df_binned.groupby('credit_score_bin_width')['clv'].mean()
        ax7.bar(bin_means.index, bin_means.values, alpha=0.8, color='teal', edgecolor='black')
        ax7.set_xlabel('Credit Score Bin', fontsize=11, fontweight='bold')
        ax7.set_ylabel('Average CLV', fontsize=11, fontweight='bold')
        ax7.set_title('CLV by Credit Score Bin', fontsize=13, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='y')

    # 8. Predictions vs Actual (Best Model)
    ax8 = fig.add_subplot(gs[2, 1])
    best_model_name = max(results_dict.keys(), key=lambda k: results_dict[k]['test_r2'])
    best_preds = results_dict[best_model_name]['predictions']
    y_test_subset = df.iloc[-len(best_preds):]['clv']

    ax8.scatter(y_test_subset, best_preds, alpha=0.5, s=30, color='darkblue')
    ax8.plot([y_test_subset.min(), y_test_subset.max()],
            [y_test_subset.min(), y_test_subset.max()],
            'r--', lw=2, label='Perfect Prediction')
    ax8.set_xlabel('Actual CLV', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Predicted CLV', fontsize=11, fontweight='bold')
    ax8.set_title(f'Best Model: {best_model_name}', fontsize=13, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # 9. Cross-Validation Scores
    ax9 = fig.add_subplot(gs[2, 2])
    cv_means = [results_dict[m]['cv_mean'] for m in models]
    cv_stds = [results_dict[m]['cv_std'] for m in models]

    ax9.errorbar(range(len(models)), cv_means, yerr=cv_stds,
                fmt='o-', linewidth=2, markersize=8, capsize=5, color='darkgreen')
    ax9.set_xlabel('Binning Strategy', fontsize=11, fontweight='bold')
    ax9.set_ylabel('CV R² Score', fontsize=11, fontweight='bold')
    ax9.set_title('Cross-Validation Performance', fontsize=13, fontweight='bold')
    ax9.set_xticks(range(len(models)))
    ax9.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax9.grid(True, alpha=0.3)

    # 10. Bin distribution for income
    ax10 = fig.add_subplot(gs[3, 0])
    if 'income_bin_freq' in df_binned.columns:
        bin_dist = df_binned.groupby('income_bin_freq')['income'].agg(['min', 'max', 'mean'])
        ax10.scatter(bin_dist.index, bin_dist['mean'], s=100, alpha=0.7, color='orange')
        ax10.errorbar(bin_dist.index,
                     bin_dist['mean'],
                     yerr=[bin_dist['mean'] - bin_dist['min'],
                           bin_dist['max'] - bin_dist['mean']],
                     fmt='none', color='orange', alpha=0.5)
        ax10.set_xlabel('Income Bin', fontsize=11, fontweight='bold')
        ax10.set_ylabel('Income Value', fontsize=11, fontweight='bold')
        ax10.set_title('Income Range by Bin (Equal-Freq)', fontsize=13, fontweight='bold')
        ax10.grid(True, alpha=0.3)

    # 11. Feature importance (if RF model exists)
    ax11 = fig.add_subplot(gs[3, 1])
    if 'Binned-Combined' in results_dict:
        model = results_dict['Binned-Combined']['model']
        if hasattr(model, 'feature_importances_'):
            feature_cols = [col for col in df_binned.columns
                          if col not in ['clv', 'high_value']][:15]
            importances = model.feature_importances_[:len(feature_cols)]
            indices = np.argsort(importances)[::-1][:10]

            ax11.barh(range(len(indices)), importances[indices], alpha=0.8, color='steelblue')
            ax11.set_yticks(range(len(indices)))
            ax11.set_yticklabels([feature_cols[i] for i in indices], fontsize=8)
            ax11.set_xlabel('Importance', fontsize=11, fontweight='bold')
            ax11.set_title('Top 10 Feature Importances', fontsize=13, fontweight='bold')
            ax11.grid(True, alpha=0.3, axis='x')

    # 12. Performance Improvement
    ax12 = fig.add_subplot(gs[3, 2])
    baseline_r2 = results_dict['Original']['test_r2']
    improvements = [(results_dict[m]['test_r2'] - baseline_r2) / baseline_r2 * 100
                   for m in models if m != 'Original']
    improved_models = [m for m in models if m != 'Original']

    colors_imp = ['green' if imp > 0 else 'red' for imp in improvements]
    ax12.barh(range(len(improvements)), improvements, color=colors_imp,
             alpha=0.8, edgecolor='black')
    ax12.set_yticks(range(len(improvements)))
    ax12.set_yticklabels(improved_models, fontsize=9)
    ax12.set_xlabel('Improvement (%)', fontsize=11, fontweight='bold')
    ax12.set_title('Performance vs Baseline', fontsize=13, fontweight='bold')
    ax12.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax12.grid(True, alpha=0.3, axis='x')

    plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/13_feature_engineering/11_binning_strategies/binning_analysis.png',
                dpi=300, bbox_inches='tight')
    print("   Comprehensive plot saved!")
    plt.show()


def main():
    """
    Main execution function.
    """
    print("=" * 90)
    print("Binning and Discretization Strategies")
    print("=" * 90)

    # Generate data
    print("\n1. Generating synthetic customer data...")
    df = generate_customer_data(n_samples=2500)
    print(f"   Dataset shape: {df.shape}")
    print(f"\n   Feature summary:")
    print(df.describe())

    # Define binning configurations
    feature_configs = {
        'age': {'type': 'equal_width', 'n_bins': 5},
        'income': {'type': 'equal_freq', 'n_bins': 5},
        'credit_score': {'type': 'custom', 'edges': [300, 550, 650, 750, 850]},
        'years_employed': {'type': 'kmeans', 'n_bins': 4},
    }

    # Create binned features
    print("\n2. Creating binned features...")
    df_binned = create_binned_features(df, feature_configs)
    print(f"   Original features: {df.shape[1]}")
    print(f"   After binning: {df_binned.shape[1]}")

    # Analyze bins
    print("\n3. Analyzing bin characteristics...")
    print("\n   Age bins (equal-width):")
    age_bin_stats = analyze_bins(df, 'age', df_binned['age_bin_width'])
    print(age_bin_stats)

    print("\n   Income bins (equal-frequency):")
    income_bin_stats = analyze_bins(df, 'income', df_binned['income_bin_freq'])
    print(income_bin_stats)

    # Split data
    target_col = 'clv'
    y = df[target_col]
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)

    results_dict = {}

    # Baseline: Original features
    print("\n4. Training baseline model (original features)...")
    X_original = df[['age', 'income', 'credit_score', 'years_employed',
                     'debt_to_income', 'num_accounts']]
    X_train_orig = X_original.loc[train_idx]
    X_test_orig = X_original.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_orig)
    X_test_scaled = scaler.transform(X_test_orig)

    results_baseline = train_and_evaluate_regression(
        X_train_scaled, X_test_scaled, y_train, y_test, 'Ridge'
    )
    results_dict['Original'] = results_baseline
    print(f"   Test RMSE: {results_baseline['test_rmse']:.2f}")
    print(f"   Test R²: {results_baseline['test_r2']:.4f}")

    # Equal-width binning
    print("\n5. Testing equal-width binning...")
    bin_cols_width = [col for col in df_binned.columns if '_bin_width' in col]
    X_binned_width = df_binned.loc[:, bin_cols_width + ['debt_to_income', 'num_accounts']]
    X_train_width = X_binned_width.loc[train_idx]
    X_test_width = X_binned_width.loc[test_idx]

    results_width = train_and_evaluate_regression(
        X_train_width, X_test_width, y_train, y_test, 'RF'
    )
    results_dict['Equal-Width'] = results_width
    print(f"   Test RMSE: {results_width['test_rmse']:.2f}")
    print(f"   Test R²: {results_width['test_r2']:.4f}")

    # Equal-frequency binning
    print("\n6. Testing equal-frequency binning...")
    bin_cols_freq = [col for col in df_binned.columns if '_bin_freq' in col]
    X_binned_freq = df_binned.loc[:, bin_cols_freq + ['debt_to_income', 'num_accounts']]
    X_train_freq = X_binned_freq.loc[train_idx]
    X_test_freq = X_binned_freq.loc[test_idx]

    results_freq = train_and_evaluate_regression(
        X_train_freq, X_test_freq, y_train, y_test, 'RF'
    )
    results_dict['Equal-Freq'] = results_freq
    print(f"   Test RMSE: {results_freq['test_rmse']:.2f}")
    print(f"   Test R²: {results_freq['test_r2']:.4f}")

    # Custom binning
    print("\n7. Testing custom binning...")
    bin_cols_custom = [col for col in df_binned.columns if '_bin_custom' in col]
    X_binned_custom = df_binned.loc[:, bin_cols_custom + ['debt_to_income', 'num_accounts']]
    X_train_custom = X_binned_custom.loc[train_idx]
    X_test_custom = X_binned_custom.loc[test_idx]

    results_custom = train_and_evaluate_regression(
        X_train_custom, X_test_custom, y_train, y_test, 'RF'
    )
    results_dict['Custom'] = results_custom
    print(f"   Test RMSE: {results_custom['test_rmse']:.2f}")
    print(f"   Test R²: {results_custom['test_r2']:.4f}")

    # K-means binning
    print("\n8. Testing K-means binning...")
    bin_cols_kmeans = [col for col in df_binned.columns if '_bin_kmeans' in col]
    X_binned_kmeans = df_binned.loc[:, bin_cols_kmeans + ['debt_to_income', 'num_accounts']]
    X_train_kmeans = X_binned_kmeans.loc[train_idx]
    X_test_kmeans = X_binned_kmeans.loc[test_idx]

    results_kmeans = train_and_evaluate_regression(
        X_train_kmeans, X_test_kmeans, y_train, y_test, 'RF'
    )
    results_dict['K-Means'] = results_kmeans
    print(f"   Test RMSE: {results_kmeans['test_rmse']:.2f}")
    print(f"   Test R²: {results_kmeans['test_r2']:.4f}")

    # Combined: Original + Binned
    print("\n9. Testing combined features (original + binned)...")
    X_combined = df_binned[[col for col in df_binned.columns
                            if col not in ['clv', 'high_value']]]
    X_train_combined = X_combined.loc[train_idx]
    X_test_combined = X_combined.loc[test_idx]

    results_combined = train_and_evaluate_regression(
        X_train_combined, X_test_combined, y_train, y_test, 'RF'
    )
    results_dict['Binned-Combined'] = results_combined
    print(f"   Test RMSE: {results_combined['test_rmse']:.2f}")
    print(f"   Test R²: {results_combined['test_r2']:.4f}")

    # Performance summary
    print("\n10. Performance Summary:")
    print("   " + "-" * 75)
    print(f"   {'Strategy':<20} {'Train RMSE':<12} {'Test RMSE':<12} {'Test R²':<12} {'CV R²':<12}")
    print("   " + "-" * 75)
    for strategy, results in results_dict.items():
        print(f"   {strategy:<20} {results['train_rmse']:<12.2f} "
              f"{results['test_rmse']:<12.2f} {results['test_r2']:<12.4f} "
              f"{results['cv_mean']:<12.4f}")
    print("   " + "-" * 75)

    # Visualizations
    print("\n11. Creating comprehensive visualizations...")
    plot_binning_analysis(df, df_binned, results_dict)

    # Best strategy
    best_strategy = max(results_dict.items(), key=lambda x: x[1]['test_r2'])
    print(f"\n12. Best Binning Strategy: {best_strategy[0]}")
    print(f"    Test R²: {best_strategy[1]['test_r2']:.4f}")
    improvement = ((best_strategy[1]['test_r2'] - results_baseline['test_r2']) /
                   results_baseline['test_r2'] * 100)
    print(f"    Improvement: {improvement:.1f}%")

    print("\n" + "=" * 90)
    print("Analysis Complete!")
    print("=" * 90)


if __name__ == "__main__":
    main()
