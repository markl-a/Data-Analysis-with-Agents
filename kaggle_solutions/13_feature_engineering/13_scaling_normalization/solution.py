"""
Kaggle Solution: Feature Scaling and Normalization Comparison
=============================================================
Demonstrates various scaling and normalization techniques and their impact
on different machine learning algorithms.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                   MaxAbsScaler, Normalizer, QuantileTransformer,
                                   PowerTransformer)
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)


def generate_multi_scale_data(n_samples=2000):
    """
    Generate data with features on vastly different scales.
    """
    # Features with different scales
    micro_feature = np.random.uniform(0.0001, 0.001, n_samples)  # Very small scale
    small_feature = np.random.normal(5, 2, n_samples)  # Small scale
    medium_feature = np.random.normal(100, 30, n_samples)  # Medium scale
    large_feature = np.random.normal(10000, 3000, n_samples)  # Large scale
    mega_feature = np.random.exponential(1000000, n_samples)  # Very large scale
    
    # Skewed feature
    skewed_feature = np.random.lognormal(5, 2, n_samples)
    
    # Target (all features contribute equally in reality)
    target = (
        100000 * micro_feature +
        10 * small_feature +
        0.5 * medium_feature +
        0.001 * large_feature +
        0.000001 * mega_feature +
        0.01 * skewed_feature +
        np.random.normal(0, 10, n_samples)
    )
    
    df = pd.DataFrame({
        'micro': micro_feature,
        'small': small_feature,
        'medium': medium_feature,
        'large': large_feature,
        'mega': mega_feature,
        'skewed': skewed_feature,
        'target': target
    })
    
    return df


def apply_scalers(X_train, X_test):
    """
    Apply various scaling methods.
    """
    scalers_dict = {}
    
    # Standard Scaler (z-score normalization)
    scaler = StandardScaler()
    X_train_std = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_std = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    scalers_dict['StandardScaler'] = (X_train_std, X_test_std)
    
    # Min-Max Scaler
    scaler = MinMaxScaler()
    X_train_mm = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_mm = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    scalers_dict['MinMaxScaler'] = (X_train_mm, X_test_mm)
    
    # Robust Scaler
    scaler = RobustScaler()
    X_train_rob = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_rob = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    scalers_dict['RobustScaler'] = (X_train_rob, X_test_rob)
    
    # MaxAbs Scaler
    scaler = MaxAbsScaler()
    X_train_ma = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_ma = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    scalers_dict['MaxAbsScaler'] = (X_train_ma, X_test_ma)
    
    # Quantile Transformer (Uniform)
    scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
    X_train_qt_uni = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_qt_uni = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    scalers_dict['QuantileUniform'] = (X_train_qt_uni, X_test_qt_uni)
    
    # Quantile Transformer (Normal)
    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    X_train_qt_norm = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_qt_norm = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    scalers_dict['QuantileNormal'] = (X_train_qt_norm, X_test_qt_norm)
    
    # Power Transformer (Yeo-Johnson)
    scaler = PowerTransformer(method='yeo-johnson', standardize=True)
    X_train_pt = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_pt = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    scalers_dict['PowerTransformer'] = (X_train_pt, X_test_pt)
    
    # L2 Normalizer (row-wise)
    scaler = Normalizer(norm='l2')
    X_train_norm = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_norm = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    scalers_dict['L2Normalizer'] = (X_train_norm, X_test_norm)
    
    return scalers_dict


def train_and_evaluate(X_train, X_test, y_train, y_test, model_type='ridge'):
    """
    Train and evaluate model.
    """
    if model_type == 'ridge':
        model = Ridge(alpha=1.0, random_state=42)
    elif model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'svr':
        model = SVR(kernel='rbf', C=1.0)
    elif model_type == 'knn':
        model = KNeighborsRegressor(n_neighbors=5)
    elif model_type == 'mlp':
        model = MLPRegressor(hidden_layer_sizes=(50, 30), max_iter=500, random_state=42)
    elif model_type == 'tree':
        model = DecisionTreeRegressor(random_state=42)
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


def plot_scaling_analysis(df, scalers_dict, results_by_scaler, results_by_model):
    """
    Create comprehensive visualizations.
    """
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
    
    # 1. Original feature distributions
    ax1 = fig.add_subplot(gs[0, 0])
    feature_cols = ['micro', 'small', 'medium', 'large']
    for col in feature_cols:
        ax1.hist(np.log10(df[col] + 1), bins=30, alpha=0.5, label=col)
    ax1.set_xlabel('Log10(Value + 1)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Original Feature Scales (Log)', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Feature ranges before scaling
    ax2 = fig.add_subplot(gs[0, 1])
    ranges = [df[col].max() - df[col].min() for col in feature_cols[:4]]
    ax2.bar(range(len(feature_cols[:4])), ranges, alpha=0.8, color='coral', edgecolor='black')
    ax2.set_xlabel('Feature', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Range', fontsize=11, fontweight='bold')
    ax2.set_title('Feature Ranges (Original)', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(feature_cols[:4])))
    ax2.set_xticklabels(feature_cols[:4])
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. StandardScaler distribution
    ax3 = fig.add_subplot(gs[0, 2])
    if 'StandardScaler' in scalers_dict:
        X_scaled = scalers_dict['StandardScaler'][0]
        for col in X_scaled.columns[:4]:
            ax3.hist(X_scaled[col], bins=30, alpha=0.5, label=col)
        ax3.set_xlabel('Scaled Value', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax3.set_title('After StandardScaler', fontsize=13, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Performance by Scaler (Ridge)
    ax4 = fig.add_subplot(gs[1, 0])
    if 'ridge' in results_by_scaler:
        scalers = list(results_by_scaler['ridge'].keys())
        r2_scores = [results_by_scaler['ridge'][s]['test_r2'] for s in scalers]
        colors = plt.cm.viridis(np.linspace(0, 1, len(scalers)))
        
        bars = ax4.barh(range(len(scalers)), r2_scores, color=colors, alpha=0.8, edgecolor='black')
        ax4.set_yticks(range(len(scalers)))
        ax4.set_yticklabels(scalers, fontsize=9)
        ax4.set_xlabel('Test R² Score', fontsize=11, fontweight='bold')
        ax4.set_title('Ridge: Performance by Scaler', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax4.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.4f}', ha='left', va='center', fontsize=7)
    
    # 5. Performance by Scaler (KNN)
    ax5 = fig.add_subplot(gs[1, 1])
    if 'knn' in results_by_scaler:
        scalers = list(results_by_scaler['knn'].keys())
        r2_scores = [results_by_scaler['knn'][s]['test_r2'] for s in scalers]
        colors = plt.cm.plasma(np.linspace(0, 1, len(scalers)))
        
        bars = ax5.barh(range(len(scalers)), r2_scores, color=colors, alpha=0.8, edgecolor='black')
        ax5.set_yticks(range(len(scalers)))
        ax5.set_yticklabels(scalers, fontsize=9)
        ax5.set_xlabel('Test R² Score', fontsize=11, fontweight='bold')
        ax5.set_title('KNN: Performance by Scaler', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')
    
    # 6. Performance by Scaler (MLP)
    ax6 = fig.add_subplot(gs[1, 2])
    if 'mlp' in results_by_scaler:
        scalers = list(results_by_scaler['mlp'].keys())
        r2_scores = [results_by_scaler['mlp'][s]['test_r2'] for s in scalers]
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(scalers)))
        
        bars = ax6.barh(range(len(scalers)), r2_scores, color=colors, alpha=0.8, edgecolor='black')
        ax6.set_yticks(range(len(scalers)))
        ax6.set_yticklabels(scalers, fontsize=9)
        ax6.set_xlabel('Test R² Score', fontsize=11, fontweight='bold')
        ax6.set_title('MLP: Performance by Scaler', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='x')
    
    # 7. Heatmap: Scaler vs Model Performance
    ax7 = fig.add_subplot(gs[2, :])
    model_types = list(results_by_scaler.keys())
    scaler_types = list(scalers_dict.keys())
    
    heatmap_data = np.zeros((len(scaler_types), len(model_types)))
    for i, scaler in enumerate(scaler_types):
        for j, model in enumerate(model_types):
            if scaler in results_by_scaler[model]:
                heatmap_data[i, j] = results_by_scaler[model][scaler]['test_r2']
    
    im = ax7.imshow(heatmap_data, cmap='YlGnBu', aspect='auto')
    ax7.set_xticks(range(len(model_types)))
    ax7.set_yticks(range(len(scaler_types)))
    ax7.set_xticklabels(model_types, fontsize=10)
    ax7.set_yticklabels(scaler_types, fontsize=9)
    ax7.set_xlabel('Model Type', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Scaler Type', fontsize=12, fontweight='bold')
    ax7.set_title('R² Score Heatmap: Scaler vs Model', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(scaler_types)):
        for j in range(len(model_types)):
            text = ax7.text(j, i, f'{heatmap_data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax7, label='R² Score')
    
    # 8. Best scaler for each model
    ax8 = fig.add_subplot(gs[3, 0])
    best_scalers = {}
    for model in model_types:
        best_scaler = max(results_by_scaler[model].items(),
                         key=lambda x: x[1]['test_r2'])[0]
        best_scalers[model] = best_scaler
    
    scaler_counts = pd.Series(best_scalers.values()).value_counts()
    ax8.bar(range(len(scaler_counts)), scaler_counts.values,
           alpha=0.8, color='teal', edgecolor='black')
    ax8.set_xlabel('Scaler', fontsize=11, fontweight='bold')
    ax8.set_ylabel('# of Models (Best)', fontsize=11, fontweight='bold')
    ax8.set_title('Most Effective Scalers', fontsize=13, fontweight='bold')
    ax8.set_xticks(range(len(scaler_counts)))
    ax8.set_xticklabels(scaler_counts.index, rotation=45, ha='right', fontsize=9)
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. RMSE comparison
    ax9 = fig.add_subplot(gs[3, 1])
    if 'ridge' in results_by_scaler:
        scalers = list(results_by_scaler['ridge'].keys())
        rmse_values = [results_by_scaler['ridge'][s]['test_rmse'] for s in scalers]
        ax9.bar(range(len(scalers)), rmse_values, alpha=0.8, color='crimson', edgecolor='black')
        ax9.set_xlabel('Scaler', fontsize=11, fontweight='bold')
        ax9.set_ylabel('Test RMSE', fontsize=11, fontweight='bold')
        ax9.set_title('Ridge: RMSE by Scaler', fontsize=13, fontweight='bold')
        ax9.set_xticks(range(len(scalers)))
        ax9.set_xticklabels(scalers, rotation=45, ha='right', fontsize=8)
        ax9.grid(True, alpha=0.3, axis='y')
    
    # 10. CV scores comparison
    ax10 = fig.add_subplot(gs[3, 2])
    if 'ridge' in results_by_scaler:
        scalers = list(results_by_scaler['ridge'].keys())
        cv_means = [results_by_scaler['ridge'][s]['cv_mean'] for s in scalers]
        cv_stds = [results_by_scaler['ridge'][s]['cv_std'] for s in scalers]
        
        ax10.errorbar(range(len(scalers)), cv_means, yerr=cv_stds,
                     fmt='o-', linewidth=2, markersize=8, capsize=5, color='darkgreen')
        ax10.set_xlabel('Scaler', fontsize=11, fontweight='bold')
        ax10.set_ylabel('CV R² Score', fontsize=11, fontweight='bold')
        ax10.set_title('Ridge: CV Performance', fontsize=13, fontweight='bold')
        ax10.set_xticks(range(len(scalers)))
        ax10.set_xticklabels(scalers, rotation=45, ha='right', fontsize=8)
        ax10.grid(True, alpha=0.3)
    
    plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/13_feature_engineering/13_scaling_normalization/scaling_analysis.png',
                dpi=300, bbox_inches='tight')
    print("   Comprehensive plot saved!")
    plt.show()


def main():
    """
    Main execution function.
    """
    print("=" * 90)
    print("Feature Scaling and Normalization Comparison")
    print("=" * 90)
    
    # Generate data with different scales
    print("\n1. Generating multi-scale data...")
    df = generate_data_with_outliers(n_samples=2000)
    print(f"   Dataset shape: {df.shape}")
    print(f"\n   Feature scale ranges:")
    feature_cols = [col for col in df.columns if col != 'target']
    for col in feature_cols:
        print(f"   {col:15s}: [{df[col].min():.2e}, {df[col].max():.2e}]")
    
    # Split data
    X = df[feature_cols]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Apply scalers
    print("\n2. Applying various scaling methods...")
    scalers_dict = apply_scalers(X_train, X_test)
    print(f"   Total scalers: {len(scalers_dict)}")
    
    # Test with different models
    model_types = ['ridge', 'knn', 'mlp', 'rf']
    results_by_scaler = {model: {} for model in model_types}
    results_by_model = {}
    
    print("\n3. Training models with different scalers...")
    for model_type in model_types:
        print(f"\n   {model_type.upper()} Model:")
        for scaler_name, (X_tr_scaled, X_te_scaled) in scalers_dict.items():
            results = train_and_evaluate(X_tr_scaled, X_te_scaled,
                                        y_train, y_test, model_type)
            results_by_scaler[model_type][scaler_name] = results
            print(f"      {scaler_name:20s}: R² = {results['test_r2']:.4f}, RMSE = {results['test_rmse']:.2f}")
    
    # Also test with no scaling
    print(f"\n   Testing WITHOUT scaling:")
    for model_type in model_types:
        results = train_and_evaluate(X_train, X_test, y_train, y_test, model_type)
        results_by_scaler[model_type]['NoScaling'] = results
        print(f"      {model_type.upper():20s}: R² = {results['test_r2']:.4f}, RMSE = {results['test_rmse']:.2f}")
    
    # Summary
    print("\n4. Best scaler for each model:")
    print("   " + "-" * 60)
    for model in model_types:
        best_scaler, best_result = max(results_by_scaler[model].items(),
                                       key=lambda x: x[1]['test_r2'])
        print(f"   {model.upper():10s}: {best_scaler:20s} (R² = {best_result['test_r2']:.4f})")
    print("   " + "-" * 60)
    
    # Visualizations
    print("\n5. Creating comprehensive visualizations...")
    plot_scaling_analysis(df, scalers_dict, results_by_scaler, results_by_model)
    
    print("\n6. Key Insights:")
    print("   - Distance-based models (KNN, SVM) are highly sensitive to scaling")
    print("   - Tree-based models (RF, DT) are invariant to feature scaling")
    print("   - Neural networks benefit from scaling for faster convergence")
    print("   - StandardScaler works well for normally distributed features")
    print("   - RobustScaler is better for data with outliers")
    print("   - MinMaxScaler useful when bounded range is needed")
    
    print("\n" + "=" * 90)
    print("Analysis Complete!")
    print("=" * 90)


if __name__ == "__main__":
    main()
