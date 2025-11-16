"""
Kaggle Solution: Interaction Feature Engineering
================================================
Demonstrates creating and analyzing interaction features between
categorical and numerical variables.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def generate_insurance_data(n_samples=2000):
    """
    Generate synthetic insurance premium data with strong interactions.
    """
    # Numerical features
    age = np.random.randint(18, 70, n_samples)
    bmi = np.random.normal(28, 6, n_samples).clip(15, 50)
    children = np.random.randint(0, 6, n_samples)
    income = np.random.lognormal(10.5, 0.6, n_samples)

    # Categorical features
    smoker = np.random.choice(['yes', 'no'], n_samples, p=[0.2, 0.8])
    region = np.random.choice(['northeast', 'northwest', 'southeast', 'southwest'], n_samples)
    coverage_type = np.random.choice(['basic', 'standard', 'premium'], n_samples, p=[0.3, 0.5, 0.2])

    # Calculate premium with strong interactions
    base_premium = 5000

    # Age effect
    age_premium = age * 100

    # BMI effect (stronger for smokers)
    bmi_premium = np.where(smoker == 'yes',
                          bmi * 150,  # High BMI + smoking is expensive
                          bmi * 50)   # Lower impact for non-smokers

    # Smoker effect (increases with age)
    smoker_premium = np.where(smoker == 'yes',
                             10000 + age * 200,  # Older smokers pay more
                             0)

    # Children effect (depends on coverage)
    coverage_multiplier = {'basic': 500, 'standard': 800, 'premium': 1200}
    children_premium = np.array([coverage_multiplier[c] for c in coverage_type]) * children

    # Region effect (interacts with coverage)
    region_cost = {
        ('northeast', 'basic'): 1.0, ('northeast', 'standard'): 1.1, ('northeast', 'premium'): 1.2,
        ('northwest', 'basic'): 0.9, ('northwest', 'standard'): 1.0, ('northwest', 'premium'): 1.1,
        ('southeast', 'basic'): 1.1, ('southeast', 'standard'): 1.2, ('southeast', 'premium'): 1.3,
        ('southwest', 'basic'): 0.95, ('southwest', 'standard'): 1.05, ('southwest', 'premium'): 1.15,
    }
    region_multiplier = np.array([region_cost[(r, c)] for r, c in zip(region, coverage_type)])

    # Income interaction (higher income -> better coverage)
    income_effect = (income / 50000) * np.where(coverage_type == 'premium', 500, 0)

    # Calculate total premium
    premium = (
        base_premium +
        age_premium +
        bmi_premium +
        smoker_premium +
        children_premium +
        income_effect
    ) * region_multiplier + np.random.normal(0, 500, n_samples)

    df = pd.DataFrame({
        'age': age,
        'bmi': bmi,
        'children': children,
        'income': income,
        'smoker': smoker,
        'region': region,
        'coverage_type': coverage_type,
        'premium': premium
    })

    return df


def create_interaction_features(df, strategy='comprehensive'):
    """
    Create interaction features using different strategies.
    """
    df_feat = df.copy()

    # Encode categorical variables first
    df_feat['smoker_binary'] = (df_feat['smoker'] == 'yes').astype(int)
    region_dummies = pd.get_dummies(df_feat['region'], prefix='region')
    coverage_dummies = pd.get_dummies(df_feat['coverage_type'], prefix='coverage')
    df_feat = pd.concat([df_feat, region_dummies, coverage_dummies], axis=1)

    if strategy == 'comprehensive':
        # 1. Numerical × Numerical interactions
        df_feat['age_bmi'] = df_feat['age'] * df_feat['bmi']
        df_feat['age_income'] = df_feat['age'] * df_feat['income']
        df_feat['bmi_income'] = df_feat['bmi'] * df_feat['income']
        df_feat['age_children'] = df_feat['age'] * df_feat['children']

        # 2. Numerical × Categorical (smoker is most important)
        df_feat['age_smoker'] = df_feat['age'] * df_feat['smoker_binary']
        df_feat['bmi_smoker'] = df_feat['bmi'] * df_feat['smoker_binary']
        df_feat['income_smoker'] = df_feat['income'] * df_feat['smoker_binary']

        # 3. Coverage type interactions
        for col in coverage_dummies.columns:
            df_feat[f'children_{col}'] = df_feat['children'] * df_feat[col]
            df_feat[f'income_{col}'] = df_feat['income'] * df_feat[col]

        # 4. Region interactions
        for col in region_dummies.columns:
            df_feat[f'age_{col}'] = df_feat['age'] * df_feat[col]

        # 5. Three-way interactions (most critical ones)
        df_feat['age_bmi_smoker'] = df_feat['age'] * df_feat['bmi'] * df_feat['smoker_binary']

    elif strategy == 'domain_knowledge':
        # Only create interactions known to be important
        df_feat['bmi_smoker'] = df_feat['bmi'] * df_feat['smoker_binary']
        df_feat['age_smoker'] = df_feat['age'] * df_feat['smoker_binary']

        for col in coverage_dummies.columns:
            df_feat[f'children_{col}'] = df_feat['children'] * df_feat[col]

    elif strategy == 'automatic':
        # Create all pairwise numerical interactions
        num_cols = ['age', 'bmi', 'children', 'income']
        for col1, col2 in combinations(num_cols, 2):
            df_feat[f'{col1}_{col2}'] = df_feat[col1] * df_feat[col2]

    return df_feat


def evaluate_interaction_strategies(X_train, X_test, y_train, y_test, original_features):
    """
    Evaluate different interaction feature strategies.
    """
    results = []

    strategies = {
        'Baseline (No Interactions)': original_features,
        'Domain Knowledge': 'domain_knowledge',
        'Comprehensive': 'comprehensive',
        'Automatic Pairwise': 'automatic'
    }

    for strategy_name, strategy in strategies.items():
        if strategy_name == 'Baseline (No Interactions)':
            X_train_feat = X_train[strategy]
            X_test_feat = X_test[strategy]
        else:
            # Create features on train and test
            train_temp = X_train.copy()
            train_temp['premium'] = y_train  # Temporary
            train_feat = create_interaction_features(train_temp, strategy=strategy)
            train_feat = train_feat.drop(['premium', 'smoker', 'region', 'coverage_type'], axis=1)

            test_temp = X_test.copy()
            test_temp['premium'] = y_test  # Temporary
            test_feat = create_interaction_features(test_temp, strategy=strategy)
            test_feat = test_feat.drop(['premium', 'smoker', 'region', 'coverage_type'], axis=1)

            X_train_feat = train_feat
            X_test_feat = test_feat

        # Train model
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_feat, y_train)

        y_pred = model.predict(X_test_feat)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({
            'strategy': strategy_name,
            'n_features': X_train_feat.shape[1],
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'model': model,
            'X_test': X_test_feat,
            'predictions': y_pred
        })

    return results


def analyze_feature_importance(model, feature_names, top_n=15):
    """
    Analyze which interaction features are most important.
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Identify interaction features
    importance_df['is_interaction'] = importance_df['feature'].str.contains('_') & \
                                     ~importance_df['feature'].str.startswith(('region_', 'coverage_'))

    return importance_df


def plot_results(results, y_test):
    """
    Create comprehensive visualizations.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Performance comparison
    ax1 = axes[0, 0]
    strategies = [r['strategy'] for r in results]
    r2_scores = [r['r2'] for r in results]
    colors = plt.cm.RdYlGn(np.array(r2_scores) / max(r2_scores))
    bars = ax1.barh(range(len(strategies)), r2_scores, color=colors, alpha=0.8)
    ax1.set_yticks(range(len(strategies)))
    ax1.set_yticklabels(strategies)
    ax1.set_xlabel('R² Score', fontsize=12)
    ax1.set_title('Performance by Interaction Strategy', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    for i, (bar, score) in enumerate(zip(bars, r2_scores)):
        ax1.text(score, i, f' {score:.3f}', va='center', fontsize=10)

    # 2. RMSE comparison
    ax2 = axes[0, 1]
    rmse_scores = [r['rmse'] for r in results]
    n_features = [r['n_features'] for r in results]
    ax2.scatter(n_features, rmse_scores, s=200, alpha=0.6, c=range(len(results)), cmap='viridis')
    for i, (x, y, label) in enumerate(zip(n_features, rmse_scores, strategies)):
        ax2.annotate(label.split()[0], (x, y), xytext=(5, 5),
                    textcoords='offset points', fontsize=9)
    ax2.set_xlabel('Number of Features', fontsize=12)
    ax2.set_ylabel('RMSE', fontsize=12)
    ax2.set_title('RMSE vs Feature Count', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Feature importance (best model)
    ax3 = axes[0, 2]
    best_result = max(results, key=lambda x: x['r2'])
    feature_names = best_result['X_test'].columns.tolist()
    importance_df = analyze_feature_importance(best_result['model'], feature_names)
    top_features = importance_df.head(15)

    colors_imp = ['red' if x else 'steelblue' for x in top_features['is_interaction']]
    ax3.barh(range(len(top_features)), top_features['importance'], color=colors_imp, alpha=0.7)
    ax3.set_yticks(range(len(top_features)))
    ax3.set_yticklabels(top_features['feature'])
    ax3.set_xlabel('Importance', fontsize=12)
    ax3.set_title('Top 15 Features (Red = Interaction)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')

    # 4. Predictions scatter (best model)
    ax4 = axes[1, 0]
    ax4.scatter(y_test, best_result['predictions'], alpha=0.5, s=30)
    ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax4.set_xlabel('Actual Premium', fontsize=12)
    ax4.set_ylabel('Predicted Premium', fontsize=12)
    ax4.set_title(f'Best Model: {best_result["strategy"]}', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 5. Efficiency (R² per feature)
    ax5 = axes[1, 1]
    efficiency = [r['r2'] / r['n_features'] * 100 for r in results]
    colors_eff = plt.cm.RdYlGn(np.array(efficiency) / max(efficiency))
    ax5.barh(range(len(strategies)), efficiency, color=colors_eff, alpha=0.8)
    ax5.set_yticks(range(len(strategies)))
    ax5.set_yticklabels(strategies)
    ax5.set_xlabel('Efficiency (R² per feature × 100)', fontsize=12)
    ax5.set_title('Strategy Efficiency', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='x')

    # 6. Interaction vs non-interaction importance
    ax6 = axes[1, 2]
    interaction_imp = importance_df[importance_df['is_interaction']]['importance'].sum()
    non_interaction_imp = importance_df[~importance_df['is_interaction']]['importance'].sum()
    ax6.pie([interaction_imp, non_interaction_imp],
           labels=['Interaction Features', 'Original Features'],
           autopct='%1.1f%%', startangle=90,
           colors=['coral', 'steelblue'], explode=(0.1, 0))
    ax6.set_title('Feature Importance Distribution', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/13_feature_engineering/04_interaction_features/interaction_features_analysis.png',
                dpi=300, bbox_inches='tight')
    print("Plot saved as 'interaction_features_analysis.png'")
    plt.show()


def main():
    """
    Main execution function.
    """
    print("=" * 80)
    print("Interaction Feature Engineering Example")
    print("=" * 80)

    # Generate data
    print("\n1. Generating synthetic insurance data...")
    df = generate_insurance_data(n_samples=2000)
    print(f"   Dataset shape: {df.shape}")
    print(f"   Features: {df.drop('premium', axis=1).columns.tolist()}")

    # Data preview
    print("\n2. Data Preview:")
    print(df.head())
    print(f"\n   Premium statistics:")
    print(f"   Mean: ${df['premium'].mean():.2f}")
    print(f"   Std: ${df['premium'].std():.2f}")
    print(f"   Range: ${df['premium'].min():.2f} - ${df['premium'].max():.2f}")

    # Split data
    print("\n3. Splitting data...")
    X = df.drop('premium', axis=1)
    y = df['premium']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Original features for baseline
    original_features = ['age', 'bmi', 'children', 'income', 'smoker_binary',
                        'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest',
                        'coverage_basic', 'coverage_premium', 'coverage_standard']

    # Encode categorical for baseline
    X_train_encoded = X_train.copy()
    X_train_encoded['smoker_binary'] = (X_train_encoded['smoker'] == 'yes').astype(int)
    X_train_encoded = pd.concat([X_train_encoded,
                                 pd.get_dummies(X_train_encoded['region'], prefix='region'),
                                 pd.get_dummies(X_train_encoded['coverage_type'], prefix='coverage')],
                                axis=1)

    X_test_encoded = X_test.copy()
    X_test_encoded['smoker_binary'] = (X_test_encoded['smoker'] == 'yes').astype(int)
    X_test_encoded = pd.concat([X_test_encoded,
                                pd.get_dummies(X_test_encoded['region'], prefix='region'),
                                pd.get_dummies(X_test_encoded['coverage_type'], prefix='coverage')],
                               axis=1)

    # Evaluate interaction strategies
    print("\n4. Evaluating interaction feature strategies...")
    results = evaluate_interaction_strategies(X_train_encoded, X_test_encoded,
                                             y_train, y_test, original_features)

    # Display results
    print("\n5. Strategy Comparison:")
    print("-" * 80)
    print(f"{'Strategy':<30} {'Features':<12} {'RMSE':<12} {'R²':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['strategy']:<30} {r['n_features']:<12} {r['rmse']:<12.2f} {r['r2']:<12.4f}")

    # Best strategy
    print("\n6. Best Strategy:")
    best_result = max(results, key=lambda x: x['r2'])
    baseline_result = results[0]
    print(f"   Strategy: {best_result['strategy']}")
    print(f"   Features: {best_result['n_features']}")
    print(f"   R²: {best_result['r2']:.4f}")
    print(f"   RMSE: ${best_result['rmse']:.2f}")
    print(f"\n   Improvement over baseline:")
    print(f"   R² improvement: {((best_result['r2'] - baseline_result['r2']) / baseline_result['r2'] * 100):.2f}%")
    print(f"   RMSE reduction: {((baseline_result['rmse'] - best_result['rmse']) / baseline_result['rmse'] * 100):.2f}%")

    # Feature importance analysis
    print("\n7. Top Interaction Features:")
    feature_names = best_result['X_test'].columns.tolist()
    importance_df = analyze_feature_importance(best_result['model'], feature_names)
    interaction_features = importance_df[importance_df['is_interaction']].head(10)
    print("\n   " + "-" * 60)
    for idx, row in interaction_features.iterrows():
        print(f"   {row['feature']:<30} {row['importance']:.4f}")

    # Visualizations
    print("\n8. Creating visualizations...")
    plot_results(results, y_test)

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
