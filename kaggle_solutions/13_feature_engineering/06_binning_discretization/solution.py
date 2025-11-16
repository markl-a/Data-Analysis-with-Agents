"""
Kaggle Solution: Binning and Discretization
===========================================
Demonstrates binning continuous features into discrete categories
using equal-width, equal-frequency, and custom strategies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def generate_credit_data(n_samples=3000):
    """Generate synthetic credit approval data with threshold effects."""
    age = np.random.uniform(18, 75, n_samples)
    income = np.random.lognormal(10, 0.8, n_samples)
    credit_score = np.random.normal(650, 100, n_samples).clip(300, 850)
    debt_ratio = np.random.beta(2, 5, n_samples)
    years_employed = np.random.gamma(3, 2, n_samples).clip(0, 40)

    # Approval probability with clear thresholds
    approval_prob = 0.0

    # Age thresholds
    approval_prob += np.where(age < 25, -0.1,
                    np.where(age < 35, 0.0,
                    np.where(age < 55, 0.2, 0.1)))

    # Income thresholds (strong effect)
    approval_prob += np.where(income < 30000, -0.3,
                    np.where(income < 50000, -0.1,
                    np.where(income < 75000, 0.1,
                    np.where(income < 100000, 0.3, 0.5))))

    # Credit score thresholds (critical)
    approval_prob += np.where(credit_score < 580, -0.5,
                    np.where(credit_score < 670, -0.2,
                    np.where(credit_score < 740, 0.1,
                    np.where(credit_score < 800, 0.3, 0.4))))

    # Debt ratio (non-linear effect)
    approval_prob += np.where(debt_ratio < 0.2, 0.2,
                    np.where(debt_ratio < 0.4, 0.1,
                    np.where(debt_ratio < 0.6, -0.1, -0.3)))

    # Employment duration
    approval_prob += np.where(years_employed < 1, -0.2,
                    np.where(years_employed < 3, 0.0,
                    np.where(years_employed < 5, 0.1, 0.2)))

    approval_prob = 1 / (1 + np.exp(-approval_prob * 2))
    approved = (np.random.random(n_samples) < approval_prob).astype(int)

    df = pd.DataFrame({
        'age': age,
        'income': income,
        'credit_score': credit_score,
        'debt_ratio': debt_ratio,
        'years_employed': years_employed,
        'approved': approved
    })

    return df


def equal_width_binning(df, column, n_bins=5):
    """Equal-width binning."""
    df_binned = df.copy()
    df_binned[f'{column}_ew_bin'] = pd.cut(df[column], bins=n_bins, labels=False)
    return df_binned


def equal_frequency_binning(df, column, n_bins=5):
    """Equal-frequency (quantile) binning."""
    df_binned = df.copy()
    df_binned[f'{column}_ef_bin'] = pd.qcut(df[column], q=n_bins, labels=False, duplicates='drop')
    return df_binned


def custom_domain_binning(df):
    """Custom binning based on domain knowledge."""
    df_binned = df.copy()

    # Age groups
    df_binned['age_group'] = pd.cut(df['age'],
                                    bins=[0, 25, 35, 50, 65, 100],
                                    labels=['young', 'young_adult', 'middle_age', 'senior', 'elderly'])

    # Income brackets
    df_binned['income_bracket'] = pd.cut(df['income'],
                                         bins=[0, 30000, 50000, 75000, 100000, np.inf],
                                         labels=['low', 'lower_mid', 'mid', 'upper_mid', 'high'])

    # Credit score categories (industry standard)
    df_binned['credit_category'] = pd.cut(df['credit_score'],
                                          bins=[0, 580, 670, 740, 800, 900],
                                          labels=['poor', 'fair', 'good', 'very_good', 'excellent'])

    # Debt ratio buckets
    df_binned['debt_level'] = pd.cut(df['debt_ratio'],
                                     bins=[0, 0.2, 0.4, 0.6, 1.0],
                                     labels=['low', 'moderate', 'high', 'very_high'])

    # Employment stability
    df_binned['employment_stability'] = pd.cut(df['years_employed'],
                                               bins=[0, 1, 3, 5, 100],
                                               labels=['new', 'developing', 'stable', 'very_stable'])

    return df_binned


def tree_based_binning(df, column, target_col, max_bins=5):
    """Use decision tree to find optimal splits."""
    tree = DecisionTreeClassifier(max_leaf_nodes=max_bins, random_state=42)
    X = df[[column]].values
    y = df[target_col].values
    tree.fit(X, y)

    # Get bin edges from tree splits
    thresholds = tree.tree_.threshold[tree.tree_.feature != -2]
    thresholds = sorted(np.unique(thresholds))

    # Add min and max
    bins = [df[column].min() - 0.1] + list(thresholds) + [df[column].max() + 0.1]

    df_binned = df.copy()
    df_binned[f'{column}_tree_bin'] = pd.cut(df[column], bins=bins, labels=False)

    return df_binned, bins


def evaluate_binning_strategy(X_train, X_test, y_train, y_test, strategy_name):
    """Train and evaluate model with binned features."""
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    return {
        'strategy': strategy_name,
        'n_features': X_train.shape[1],
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba),
        'predictions': y_pred,
        'model': model
    }


def plot_results(results, df, bins_info):
    """Create comprehensive visualizations."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Performance comparison
    ax1 = fig.add_subplot(gs[0, :2])
    strategies = [r['strategy'] for r in results]
    aucs = [r['auc'] for r in results]
    colors = plt.cm.RdYlGn(np.array(aucs) / max(aucs))
    bars = ax1.barh(range(len(strategies)), aucs, color=colors, alpha=0.8)
    ax1.set_yticks(range(len(strategies)))
    ax1.set_yticklabels(strategies)
    ax1.set_xlabel('AUC Score', fontsize=12)
    ax1.set_title('Performance by Binning Strategy', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    for i, (bar, score) in enumerate(zip(bars, aucs)):
        ax1.text(score, i, f' {score:.4f}', va='center')

    # 2. Accuracy comparison
    ax2 = fig.add_subplot(gs[0, 2])
    accs = [r['accuracy'] for r in results]
    ax2.barh(range(len(strategies)), accs, alpha=0.7, color='steelblue')
    ax2.set_yticks(range(len(strategies)))
    ax2.set_yticklabels(strategies)
    ax2.set_xlabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    # 3. Income distribution by bins (equal-width)
    ax3 = fig.add_subplot(gs[1, 0])
    df_plot = df.copy()
    df_plot['income_bin'] = pd.cut(df_plot['income'], bins=5)
    bin_counts = df_plot['income_bin'].value_counts().sort_index()
    ax3.bar(range(len(bin_counts)), bin_counts.values, alpha=0.7, color='coral')
    ax3.set_xlabel('Bin Number', fontsize=10)
    ax3.set_ylabel('Count', fontsize=10)
    ax3.set_title('Equal-Width Bins (Income)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Income distribution by quantiles (equal-frequency)
    ax4 = fig.add_subplot(gs[1, 1])
    df_plot['income_quantile'] = pd.qcut(df_plot['income'], q=5, duplicates='drop')
    quantile_counts = df_plot['income_quantile'].value_counts().sort_index()
    ax4.bar(range(len(quantile_counts)), quantile_counts.values, alpha=0.7, color='green')
    ax4.set_xlabel('Quantile Number', fontsize=10)
    ax4.set_ylabel('Count', fontsize=10)
    ax4.set_title('Equal-Frequency Bins (Income)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Approval rate by credit score bins
    ax5 = fig.add_subplot(gs[1, 2])
    df_plot['credit_bin'] = pd.cut(df_plot['credit_score'], bins=5)
    approval_by_bin = df_plot.groupby('credit_bin')['approved'].mean()
    ax5.bar(range(len(approval_by_bin)), approval_by_bin.values, alpha=0.7, color='purple')
    ax5.set_xlabel('Credit Score Bin', fontsize=10)
    ax5.set_ylabel('Approval Rate', fontsize=10)
    ax5.set_title('Approval Rate by Credit Score Bin', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim([0, 1])

    # 6. Feature distributions
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.hist(df['income'], bins=30, alpha=0.6, color='blue', edgecolor='black')
    ax6.set_xlabel('Income', fontsize=10)
    ax6.set_ylabel('Frequency', fontsize=10)
    ax6.set_title('Income Distribution (Continuous)', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # 7. Tree-based bins visualization
    ax7 = fig.add_subplot(gs[2, 1])
    if 'income_tree_bins' in bins_info:
        bins = bins_info['income_tree_bins']
        ax7.hist(df['income'], bins=30, alpha=0.4, color='gray')
        for b in bins[1:-1]:
            ax7.axvline(b, color='red', linestyle='--', linewidth=2)
        ax7.set_xlabel('Income', fontsize=10)
        ax7.set_ylabel('Frequency', fontsize=10)
        ax7.set_title('Tree-Based Bin Boundaries', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3)

    # 8. Performance vs n_features
    ax8 = fig.add_subplot(gs[2, 2])
    n_features = [r['n_features'] for r in results]
    ax8.scatter(n_features, aucs, s=200, alpha=0.6, c=range(len(results)), cmap='viridis')
    for i, r in enumerate(results):
        ax8.annotate(r['strategy'].split()[0], (r['n_features'], r['auc']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax8.set_xlabel('Number of Features', fontsize=12)
    ax8.set_ylabel('AUC Score', fontsize=12)
    ax8.set_title('Features vs Performance', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3)

    plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/13_feature_engineering/06_binning_discretization/binning_analysis.png',
                dpi=300, bbox_inches='tight')
    print("Plot saved as 'binning_analysis.png'")
    plt.show()


def main():
    """Main execution function."""
    print("=" * 80)
    print("Binning and Discretization Example")
    print("=" * 80)

    # Generate data
    print("\n1. Generating synthetic credit approval data...")
    df = generate_credit_data(n_samples=3000)
    print(f"   Dataset shape: {df.shape}")
    print(f"   Approval rate: {df['approved'].mean():.2%}")

    # Split data
    print("\n2. Splitting data...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['approved'])

    continuous_features = ['age', 'income', 'credit_score', 'debt_ratio', 'years_employed']
    results = []
    bins_info = {}

    # Strategy 1: No binning (baseline)
    print("\n3. Strategy 1: Continuous Features (Baseline)...")
    X_train = train_df[continuous_features]
    X_test = test_df[continuous_features]
    result = evaluate_binning_strategy(X_train, X_test, train_df['approved'], test_df['approved'],
                                      "Continuous (No Binning)")
    results.append(result)
    print(f"   AUC: {result['auc']:.4f}, Accuracy: {result['accuracy']:.4f}")

    # Strategy 2: Equal-width binning
    print("\n4. Strategy 2: Equal-Width Binning...")
    train_ew = train_df.copy()
    test_ew = test_df.copy()
    for col in continuous_features:
        train_ew = equal_width_binning(train_ew, col, n_bins=5)
        test_ew = equal_width_binning(test_ew, col, n_bins=5)

    ew_features = [f'{col}_ew_bin' for col in continuous_features]
    result = evaluate_binning_strategy(train_ew[ew_features], test_ew[ew_features],
                                      train_df['approved'], test_df['approved'],
                                      "Equal-Width Binning")
    results.append(result)
    print(f"   AUC: {result['auc']:.4f}, Accuracy: {result['accuracy']:.4f}")

    # Strategy 3: Equal-frequency binning
    print("\n5. Strategy 3: Equal-Frequency Binning...")
    train_ef = train_df.copy()
    test_ef = test_df.copy()
    for col in continuous_features:
        train_ef = equal_frequency_binning(train_ef, col, n_bins=5)
        test_ef = equal_frequency_binning(test_ef, col, n_bins=5)

    ef_features = [f'{col}_ef_bin' for col in continuous_features]
    result = evaluate_binning_strategy(train_ef[ef_features], test_ef[ef_features],
                                      train_df['approved'], test_df['approved'],
                                      "Equal-Frequency Binning")
    results.append(result)
    print(f"   AUC: {result['auc']:.4f}, Accuracy: {result['accuracy']:.4f}")

    # Strategy 4: Custom domain binning
    print("\n6. Strategy 4: Custom Domain-Based Binning...")
    train_custom = custom_domain_binning(train_df)
    test_custom = custom_domain_binning(test_df)

    # One-hot encode categorical bins
    custom_features = ['age_group', 'income_bracket', 'credit_category', 'debt_level', 'employment_stability']
    train_custom_encoded = pd.get_dummies(train_custom[custom_features], drop_first=True)
    test_custom_encoded = pd.get_dummies(test_custom[custom_features], drop_first=True)

    # Ensure same columns
    missing_cols = set(train_custom_encoded.columns) - set(test_custom_encoded.columns)
    for col in missing_cols:
        test_custom_encoded[col] = 0
    test_custom_encoded = test_custom_encoded[train_custom_encoded.columns]

    result = evaluate_binning_strategy(train_custom_encoded, test_custom_encoded,
                                      train_df['approved'], test_df['approved'],
                                      "Custom Domain Binning")
    results.append(result)
    print(f"   AUC: {result['auc']:.4f}, Accuracy: {result['accuracy']:.4f}")

    # Strategy 5: Tree-based binning
    print("\n7. Strategy 5: Tree-Based Optimal Binning...")
    train_tree = train_df.copy()
    test_tree = test_df.copy()
    for col in continuous_features:
        train_tree, bins = tree_based_binning(train_tree, col, 'approved', max_bins=5)
        if col == 'income':
            bins_info['income_tree_bins'] = bins
        # Apply same bins to test
        test_tree[f'{col}_tree_bin'] = pd.cut(test_df[col], bins=bins, labels=False)

    tree_features = [f'{col}_tree_bin' for col in continuous_features]
    result = evaluate_binning_strategy(train_tree[tree_features], test_tree[tree_features],
                                      train_df['approved'], test_df['approved'],
                                      "Tree-Based Binning")
    results.append(result)
    print(f"   AUC: {result['auc']:.4f}, Accuracy: {result['accuracy']:.4f}")

    # Summary
    print("\n8. Results Summary:")
    print("-" * 80)
    print(f"{'Strategy':<30} {'Features':<12} {'AUC':<12} {'Accuracy':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['strategy']:<30} {r['n_features']:<12} {r['auc']:<12.4f} {r['accuracy']:<12.4f}")

    # Best strategy
    print("\n9. Best Strategy:")
    best_result = max(results, key=lambda x: x['auc'])
    print(f"   Strategy: {best_result['strategy']}")
    print(f"   AUC: {best_result['auc']:.4f}")
    print(f"   Features: {best_result['n_features']}")

    # Visualizations
    print("\n10. Creating visualizations...")
    plot_results(results, df, bins_info)

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
