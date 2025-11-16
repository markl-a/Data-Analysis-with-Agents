"""
Kaggle Solution: Feature Selection Methods Comparison
=====================================================
Demonstrates and compares multiple feature selection techniques including
filter, wrapper, and embedded methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel, VarianceThreshold
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def generate_customer_churn_data(n_samples=3000, n_features=30):
    """
    Generate synthetic customer churn data with relevant and irrelevant features.
    """
    # Relevant features
    tenure = np.random.uniform(0, 72, n_samples)  # months
    monthly_charges = np.random.uniform(20, 150, n_samples)
    total_charges = tenure * monthly_charges + np.random.normal(0, 100, n_samples)
    contract_type = np.random.randint(0, 3, n_samples)  # 0: month-to-month, 1: one year, 2: two year
    support_calls = np.random.poisson(3, n_samples)
    payment_failures = np.random.poisson(1, n_samples)
    usage_gb = np.random.uniform(0, 100, n_samples)
    age = np.random.uniform(18, 80, n_samples)
    satisfaction_score = np.random.uniform(1, 10, n_samples)
    num_services = np.random.randint(1, 8, n_samples)

    # Calculate churn probability based on relevant features
    churn_prob = (
        -0.05 * tenure +
        0.01 * monthly_charges +
        -0.3 * contract_type +
        0.15 * support_calls +
        0.25 * payment_failures +
        -0.2 * satisfaction_score +
        -0.05 * num_services +
        0.01 * age
    )

    # Convert to probability using sigmoid
    churn_prob = 1 / (1 + np.exp(-churn_prob / 5))

    # Add noise
    churn_prob = np.clip(churn_prob + np.random.normal(0, 0.1, n_samples), 0, 1)

    # Generate binary target
    churn = (churn_prob > 0.5).astype(int)

    # Create DataFrame with relevant features
    df = pd.DataFrame({
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'contract_type': contract_type,
        'support_calls': support_calls,
        'payment_failures': payment_failures,
        'usage_gb': usage_gb,
        'age': age,
        'satisfaction_score': satisfaction_score,
        'num_services': num_services,
    })

    # Add irrelevant/noisy features
    n_noise_features = n_features - 10
    for i in range(n_noise_features):
        df[f'noise_{i}'] = np.random.randn(n_samples)

    # Add some correlated noise features
    df['noise_corr_1'] = df['tenure'] * 0.1 + np.random.randn(n_samples) * 10
    df['noise_corr_2'] = df['age'] * 0.05 + np.random.randn(n_samples) * 5

    df['churn'] = churn

    return df


def variance_threshold_selection(X_train, X_test, threshold=0.01):
    """
    Remove features with low variance.
    """
    selector = VarianceThreshold(threshold=threshold)
    X_train_selected = selector.fit_transform(X_train)
    X_test_selected = selector.transform(X_test)

    selected_features = X_train.columns[selector.get_support()].tolist()

    return X_train_selected, X_test_selected, selected_features


def univariate_selection(X_train, X_test, y_train, k=15, score_func=f_classif):
    """
    Select K best features using statistical tests.
    """
    selector = SelectKBest(score_func=score_func, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    selected_features = X_train.columns[selector.get_support()].tolist()
    scores = pd.DataFrame({
        'feature': X_train.columns,
        'score': selector.scores_
    }).sort_values('score', ascending=False)

    return X_train_selected, X_test_selected, selected_features, scores


def rfe_selection(X_train, X_test, y_train, n_features=15):
    """
    Recursive Feature Elimination.
    """
    estimator = LogisticRegression(random_state=42, max_iter=1000)
    selector = RFE(estimator, n_features_to_select=n_features, step=1)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    selected_features = X_train.columns[selector.get_support()].tolist()
    rankings = pd.DataFrame({
        'feature': X_train.columns,
        'ranking': selector.ranking_
    }).sort_values('ranking')

    return X_train_selected, X_test_selected, selected_features, rankings


def model_based_selection(X_train, X_test, y_train, threshold='median'):
    """
    Select features based on feature importance from a model.
    """
    estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    estimator.fit(X_train, y_train)

    selector = SelectFromModel(estimator, threshold=threshold, prefit=True)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    selected_features = X_train.columns[selector.get_support()].tolist()
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': estimator.feature_importances_
    }).sort_values('importance', ascending=False)

    return X_train_selected, X_test_selected, selected_features, importance


def evaluate_model(X_train, X_test, y_train, y_test, method_name):
    """
    Train and evaluate a model.
    """
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results = {
        'method': method_name,
        'n_features': X_train.shape[1],
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'cv_score': cross_val_score(model, X_train, y_train, cv=5, scoring='f1').mean()
    }

    return results


def plot_comparison(all_results, feature_info):
    """
    Create comprehensive comparison visualizations.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Performance comparison
    ax1 = axes[0, 0]
    results_df = pd.DataFrame(all_results)
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(results_df))
    width = 0.2

    for i, metric in enumerate(metrics):
        ax1.bar(x + i*width, results_df[metric], width, label=metric.capitalize(), alpha=0.8)

    ax1.set_xlabel('Feature Selection Method', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Performance Comparison Across Methods', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(results_df['method'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Number of features vs F1 score
    ax2 = axes[0, 1]
    ax2.scatter(results_df['n_features'], results_df['f1'], s=200, alpha=0.6, c=range(len(results_df)), cmap='viridis')
    for idx, row in results_df.iterrows():
        ax2.annotate(row['method'], (row['n_features'], row['f1']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax2.set_xlabel('Number of Features', fontsize=12)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_title('Feature Count vs Performance', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Feature importance (from model-based selection)
    ax3 = axes[1, 0]
    top_features = feature_info['model_based_importance'].head(15)
    ax3.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    ax3.set_yticks(range(len(top_features)))
    ax3.set_yticklabels(top_features['feature'])
    ax3.set_xlabel('Importance', fontsize=12)
    ax3.set_title('Top 15 Features (Model-Based Selection)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')

    # 4. Method efficiency (performance vs features)
    ax4 = axes[1, 1]
    efficiency = results_df['f1'] / results_df['n_features'] * 100
    colors = plt.cm.RdYlGn(efficiency / efficiency.max())
    ax4.barh(range(len(results_df)), efficiency, color=colors, alpha=0.8)
    ax4.set_yticks(range(len(results_df)))
    ax4.set_yticklabels(results_df['method'])
    ax4.set_xlabel('Efficiency (F1 per feature × 100)', fontsize=12)
    ax4.set_title('Selection Method Efficiency', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/13_feature_engineering/02_feature_selection/feature_selection_comparison.png',
                dpi=300, bbox_inches='tight')
    print("Plot saved as 'feature_selection_comparison.png'")
    plt.show()


def main():
    """
    Main execution function.
    """
    print("=" * 80)
    print("Feature Selection Methods Comparison")
    print("=" * 80)

    # Generate data
    print("\n1. Generating synthetic customer churn data...")
    df = generate_customer_churn_data(n_samples=3000, n_features=30)
    print(f"   Dataset shape: {df.shape}")
    print(f"   Total features: {df.shape[1] - 1}")
    print(f"   Churn rate: {df['churn'].mean():.2%}")

    # Split data
    print("\n2. Splitting data...")
    X = df.drop('churn', axis=1)
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    all_results = []
    feature_info = {}

    # Baseline: All features
    print("\n3. Baseline model (all features)...")
    results = evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test, "All Features")
    all_results.append(results)
    print(f"   Features: {results['n_features']}, F1 Score: {results['f1']:.4f}")

    # Method 1: Variance Threshold
    print("\n4. Variance Threshold Selection...")
    X_train_var, X_test_var, features_var = variance_threshold_selection(
        X_train_scaled, X_test_scaled, threshold=0.01
    )
    results = evaluate_model(X_train_var, X_test_var, y_train, y_test, "Variance Threshold")
    all_results.append(results)
    print(f"   Features: {results['n_features']}, F1 Score: {results['f1']:.4f}")

    # Method 2: Univariate Selection (ANOVA F-test)
    print("\n5. Univariate Selection (F-test)...")
    X_train_uni, X_test_uni, features_uni, scores_uni = univariate_selection(
        X_train_scaled, X_test_scaled, y_train, k=15, score_func=f_classif
    )
    results = evaluate_model(X_train_uni, X_test_uni, y_train, y_test, "Univariate (F-test)")
    all_results.append(results)
    feature_info['univariate_scores'] = scores_uni
    print(f"   Features: {results['n_features']}, F1 Score: {results['f1']:.4f}")

    # Method 3: Univariate Selection (Mutual Information)
    print("\n6. Univariate Selection (Mutual Info)...")
    X_train_mi, X_test_mi, features_mi, scores_mi = univariate_selection(
        X_train_scaled, X_test_scaled, y_train, k=15, score_func=mutual_info_classif
    )
    results = evaluate_model(X_train_mi, X_test_mi, y_train, y_test, "Univariate (MI)")
    all_results.append(results)
    print(f"   Features: {results['n_features']}, F1 Score: {results['f1']:.4f}")

    # Method 4: RFE
    print("\n7. Recursive Feature Elimination...")
    X_train_rfe, X_test_rfe, features_rfe, rankings_rfe = rfe_selection(
        X_train_scaled, X_test_scaled, y_train, n_features=15
    )
    results = evaluate_model(X_train_rfe, X_test_rfe, y_train, y_test, "RFE")
    all_results.append(results)
    feature_info['rfe_rankings'] = rankings_rfe
    print(f"   Features: {results['n_features']}, F1 Score: {results['f1']:.4f}")

    # Method 5: Model-based Selection
    print("\n8. Model-Based Selection (Random Forest)...")
    X_train_model, X_test_model, features_model, importance_model = model_based_selection(
        X_train_scaled, X_test_scaled, y_train, threshold='median'
    )
    results = evaluate_model(X_train_model, X_test_model, y_train, y_test, "Model-Based")
    all_results.append(results)
    feature_info['model_based_importance'] = importance_model
    print(f"   Features: {results['n_features']}, F1 Score: {results['f1']:.4f}")

    # Summary
    print("\n9. Summary of All Methods:")
    print("-" * 80)
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))

    # Best method
    print("\n10. Best Performing Method:")
    best_idx = results_df['f1'].idxmax()
    best_method = results_df.iloc[best_idx]
    print(f"    Method: {best_method['method']}")
    print(f"    Features: {best_method['n_features']}")
    print(f"    F1 Score: {best_method['f1']:.4f}")
    print(f"    Accuracy: {best_method['accuracy']:.4f}")

    # Feature overlap analysis
    print("\n11. Feature Overlap Analysis:")
    print(f"    Univariate (F-test): {features_uni[:5]}")
    print(f"    RFE: {features_rfe[:5]}")
    print(f"    Model-Based: {features_model[:5]}")

    # Visualizations
    print("\n12. Creating visualizations...")
    plot_comparison(all_results, feature_info)

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
