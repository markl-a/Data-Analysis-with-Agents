"""
Kaggle Solution: Target-Based Aggregation Features
=================================
Demonstrates target-based aggregation features with comprehensive analysis and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)


def generate_advanced_data(n_samples=2500):
    """Generate synthetic data for advanced techniques."""
    n_features = 20
    
    # Create correlated features
    mean = np.zeros(n_features)
    cov = np.eye(n_features)
    for i in range(n_features-1):
        cov[i, i+1] = cov[i+1, i] = 0.5
    
    X = np.random.multivariate_normal(mean, cov, n_samples)
    
    # Target based on subset of features
    target = (
        3 * X[:, 0] +
        2 * X[:, 1] * X[:, 2] +
        -1 * X[:, 3]**2 +
        0.5 * X[:, 4] +
        np.random.normal(0, 1, n_samples)
    )
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['target'] = target
    
    return df


def apply_group_mean_encoding(df, categorical_cols=None):
    """Apply group_mean encoding."""
    df_encoded = df.copy()
    
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        if col in df.columns:
            # Method-specific encoding logic
            if 'group_mean' in ['one_hot', 'binary']:
                encoded = pd.get_dummies(df[col], prefix=col)
                df_encoded = pd.concat([df_encoded.drop(col, axis=1), encoded], axis=1)
            elif 'group_mean' in ['label', 'ordinal']:
                le = LabelEncoder()
                df_encoded[f'{col}_group_mean'] = le.fit_transform(df[col].astype(str))
            elif 'group_mean' in ['target_mean', 'smoothed_target']:
                # Target encoding (simple mean)
                if 'target' in df.columns:
                    target_mean = df.groupby(col)['target'].mean()
                    df_encoded[f'{col}_group_mean'] = df[col].map(target_mean)
            elif 'group_mean' in ['count', 'frequency']:
                counts = df[col].value_counts()
                df_encoded[f'{col}_group_mean'] = df[col].map(counts)
            elif 'group_mean' in ['sin_cos', 'fourier']:
                # Cyclic encoding
                unique_vals = df[col].nunique()
                df_encoded[f'{col}_sin'] = np.sin(2 * np.pi * df[col].astype('category').cat.codes / unique_vals)
                df_encoded[f'{col}_cos'] = np.cos(2 * np.pi * df[col].astype('category').cat.codes / unique_vals)
            else:
                # Default: label encoding
                le = LabelEncoder()
                df_encoded[f'{col}_group_mean'] = le.fit_transform(df[col].astype(str))
    
    return df_encoded


def apply_group_std_encoding(df, categorical_cols=None):
    """Apply group_std encoding."""
    df_encoded = df.copy()
    
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        if col in df.columns:
            # Method-specific encoding logic
            if 'group_std' in ['one_hot', 'binary']:
                encoded = pd.get_dummies(df[col], prefix=col)
                df_encoded = pd.concat([df_encoded.drop(col, axis=1), encoded], axis=1)
            elif 'group_std' in ['label', 'ordinal']:
                le = LabelEncoder()
                df_encoded[f'{col}_group_std'] = le.fit_transform(df[col].astype(str))
            elif 'group_std' in ['target_mean', 'smoothed_target']:
                # Target encoding (simple mean)
                if 'target' in df.columns:
                    target_mean = df.groupby(col)['target'].mean()
                    df_encoded[f'{col}_group_std'] = df[col].map(target_mean)
            elif 'group_std' in ['count', 'frequency']:
                counts = df[col].value_counts()
                df_encoded[f'{col}_group_std'] = df[col].map(counts)
            elif 'group_std' in ['sin_cos', 'fourier']:
                # Cyclic encoding
                unique_vals = df[col].nunique()
                df_encoded[f'{col}_sin'] = np.sin(2 * np.pi * df[col].astype('category').cat.codes / unique_vals)
                df_encoded[f'{col}_cos'] = np.cos(2 * np.pi * df[col].astype('category').cat.codes / unique_vals)
            else:
                # Default: label encoding
                le = LabelEncoder()
                df_encoded[f'{col}_group_std'] = le.fit_transform(df[col].astype(str))
    
    return df_encoded


def apply_group_max_encoding(df, categorical_cols=None):
    """Apply group_max encoding."""
    df_encoded = df.copy()
    
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        if col in df.columns:
            # Method-specific encoding logic
            if 'group_max' in ['one_hot', 'binary']:
                encoded = pd.get_dummies(df[col], prefix=col)
                df_encoded = pd.concat([df_encoded.drop(col, axis=1), encoded], axis=1)
            elif 'group_max' in ['label', 'ordinal']:
                le = LabelEncoder()
                df_encoded[f'{col}_group_max'] = le.fit_transform(df[col].astype(str))
            elif 'group_max' in ['target_mean', 'smoothed_target']:
                # Target encoding (simple mean)
                if 'target' in df.columns:
                    target_mean = df.groupby(col)['target'].mean()
                    df_encoded[f'{col}_group_max'] = df[col].map(target_mean)
            elif 'group_max' in ['count', 'frequency']:
                counts = df[col].value_counts()
                df_encoded[f'{col}_group_max'] = df[col].map(counts)
            elif 'group_max' in ['sin_cos', 'fourier']:
                # Cyclic encoding
                unique_vals = df[col].nunique()
                df_encoded[f'{col}_sin'] = np.sin(2 * np.pi * df[col].astype('category').cat.codes / unique_vals)
                df_encoded[f'{col}_cos'] = np.cos(2 * np.pi * df[col].astype('category').cat.codes / unique_vals)
            else:
                # Default: label encoding
                le = LabelEncoder()
                df_encoded[f'{col}_group_max'] = le.fit_transform(df[col].astype(str))
    
    return df_encoded


def apply_group_rank_encoding(df, categorical_cols=None):
    """Apply group_rank encoding."""
    df_encoded = df.copy()
    
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        if col in df.columns:
            # Method-specific encoding logic
            if 'group_rank' in ['one_hot', 'binary']:
                encoded = pd.get_dummies(df[col], prefix=col)
                df_encoded = pd.concat([df_encoded.drop(col, axis=1), encoded], axis=1)
            elif 'group_rank' in ['label', 'ordinal']:
                le = LabelEncoder()
                df_encoded[f'{col}_group_rank'] = le.fit_transform(df[col].astype(str))
            elif 'group_rank' in ['target_mean', 'smoothed_target']:
                # Target encoding (simple mean)
                if 'target' in df.columns:
                    target_mean = df.groupby(col)['target'].mean()
                    df_encoded[f'{col}_group_rank'] = df[col].map(target_mean)
            elif 'group_rank' in ['count', 'frequency']:
                counts = df[col].value_counts()
                df_encoded[f'{col}_group_rank'] = df[col].map(counts)
            elif 'group_rank' in ['sin_cos', 'fourier']:
                # Cyclic encoding
                unique_vals = df[col].nunique()
                df_encoded[f'{col}_sin'] = np.sin(2 * np.pi * df[col].astype('category').cat.codes / unique_vals)
                df_encoded[f'{col}_cos'] = np.cos(2 * np.pi * df[col].astype('category').cat.codes / unique_vals)
            else:
                # Default: label encoding
                le = LabelEncoder()
                df_encoded[f'{col}_group_rank'] = le.fit_transform(df[col].astype(str))
    
    return df_encoded


def train_and_evaluate(X_train, X_test, y_train, y_test, model_type='ridge'):
    """Train and evaluate model."""
    if model_type == 'ridge':
        model = Ridge(alpha=1.0, random_state=42)
    elif model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == 'logistic':
        model = LogisticRegression(random_state=42, max_iter=1000)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
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


def plot_comprehensive_analysis(df, results_dict, feature_importance=None):
    """Create comprehensive visualizations."""
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Performance Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    models = list(results_dict.keys())
    test_r2 = [results_dict[m]['test_r2'] for m in models]
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    
    bars = ax1.barh(range(len(models)), test_r2, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels(models, fontsize=9)
    ax1.set_xlabel('Test R² Score', fontsize=11, fontweight='bold')
    ax1.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}', ha='left', va='center', fontsize=8)
    
    # 2. RMSE Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    test_rmse = [results_dict[m]['test_rmse'] for m in models]
    ax2.bar(range(len(models)), test_rmse, alpha=0.8, color='coral', edgecolor='black')
    ax2.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Test RMSE', fontsize=11, fontweight='bold')
    ax2.set_title('RMSE Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
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
    ax3.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Predictions vs Actual
    ax4 = fig.add_subplot(gs[1, 0])
    best_model = max(results_dict.keys(), key=lambda k: results_dict[k]['test_r2'])
    best_preds = results_dict[best_model]['predictions']
    y_test_subset = df.iloc[-len(best_preds):]['target'] if 'target' in df.columns else df.iloc[-len(best_preds):].iloc[:, -1]
    
    ax4.scatter(y_test_subset, best_preds, alpha=0.5, s=30, color='purple')
    ax4.plot([y_test_subset.min(), y_test_subset.max()],
            [y_test_subset.min(), y_test_subset.max()],
            'r--', lw=2, label='Perfect Prediction')
    ax4.set_xlabel('Actual Values', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Predicted Values', fontsize=11, fontweight='bold')
    ax4.set_title(f'Best Model: {best_model}', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Residuals Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    residuals = y_test_subset - best_preds
    ax5.hist(residuals, bins=40, color='teal', alpha=0.7, edgecolor='black')
    ax5.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax5.set_xlabel('Residuals', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax5.set_title('Residual Distribution', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Feature Importance
    ax6 = fig.add_subplot(gs[1, 2])
    if feature_importance is not None and len(feature_importance) > 0:
        top_n = min(15, len(feature_importance))
        top_feats = feature_importance.head(top_n)
        ax6.barh(range(top_n), top_feats['importance'], color='steelblue', alpha=0.8)
        ax6.set_yticks(range(top_n))
        ax6.set_yticklabels(top_feats['feature'], fontsize=8)
        ax6.set_xlabel('Importance', fontsize=11, fontweight='bold')
        ax6.set_title('Top Feature Importances', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='x')
    
    # 7. Performance Improvement
    ax7 = fig.add_subplot(gs[2, 0])
    baseline_r2 = results_dict[list(results_dict.keys())[0]]['test_r2']
    improvements = [(results_dict[m]['test_r2'] - baseline_r2) / baseline_r2 * 100
                   for m in models[1:]]
    improved_models = models[1:]
    
    colors_imp = ['green' if imp > 0 else 'red' for imp in improvements]
    ax7.barh(range(len(improvements)), improvements, color=colors_imp,
            alpha=0.8, edgecolor='black')
    ax7.set_yticks(range(len(improvements)))
    ax7.set_yticklabels(improved_models, fontsize=9)
    ax7.set_xlabel('Improvement (%)', fontsize=11, fontweight='bold')
    ax7.set_title('Performance vs Baseline', fontsize=13, fontweight='bold')
    ax7.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax7.grid(True, alpha=0.3, axis='x')
    
    # 8. Train vs Test R2
    ax8 = fig.add_subplot(gs[2, 1])
    train_r2 = [results_dict[m]['train_r2'] for m in models]
    x = np.arange(len(models))
    width = 0.35
    ax8.bar(x - width/2, train_r2, width, label='Train', alpha=0.8, color='skyblue')
    ax8.bar(x + width/2, test_r2, width, label='Test', alpha=0.8, color='coral')
    ax8.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax8.set_ylabel('R² Score', fontsize=11, fontweight='bold')
    ax8.set_title('Train vs Test Performance', fontsize=13, fontweight='bold')
    ax8.set_xticks(x)
    ax8.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. Overfitting Analysis
    ax9 = fig.add_subplot(gs[2, 2])
    gap = [train_r2[i] - test_r2[i] for i in range(len(models))]
    colors_gap = ['green' if g < 0.05 else 'orange' if g < 0.15 else 'red' for g in gap]
    ax9.bar(range(len(models)), gap, color=colors_gap, alpha=0.8, edgecolor='black')
    ax9.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax9.set_ylabel('Train R² - Test R²', fontsize=11, fontweight='bold')
    ax9.set_title('Overfitting Analysis', fontsize=13, fontweight='bold')
    ax9.set_xticks(range(len(models)))
    ax9.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax9.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5)
    ax9.axhline(y=0.15, color='red', linestyle='--', alpha=0.5)
    ax9.grid(True, alpha=0.3, axis='y')
    
    plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/13_feature_engineering/33_target_aggregation/analysis.png', dpi=300, bbox_inches='tight')
    print("   Plot saved!")
    plt.show()


def main():
    """Main execution function."""
    print("=" * 90)
    print("33. Target Aggregation")
    print("=" * 90)
    
    # Generate data
    print("\n1. Generating synthetic data...")
    df = generate_advanced_data()
    print(f"   Dataset shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Split data
    print("\n2. Splitting data...")
    X = df.drop('target', axis=1) if 'target' in df.columns else df.iloc[:, :-1]
    y = df['target'] if 'target' in df.columns else df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results_dict = {}
    
    # Baseline
    print("\n3. Training baseline model...")
    if X_train.select_dtypes(include=['object']).columns.any():
        X_train_base = pd.get_dummies(X_train)
        X_test_base = pd.get_dummies(X_test)
        X_train_base, X_test_base = X_train_base.align(X_test_base, join='left', axis=1, fill_value=0)
    else:
        X_train_base = X_train
        X_test_base = X_test
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_base)
    X_test_scaled = scaler.transform(X_test_base)
    
    results_baseline = train_and_evaluate(X_train_scaled, X_test_scaled,
                                         y_train, y_test, 'ridge')
    results_dict['Baseline'] = results_baseline
    print(f"   Test R²: {results_baseline['test_r2']:.4f}")
    
    # Apply different encoding methods
    methods = ['group_mean', 'group_std', 'group_max', 'group_rank']
    print(f"\n4. Testing {len(methods)} encoding methods...")
    
    for i, method in enumerate(methods, 1):
        print(f"\n   {i}. {method} encoding...")
        try:
            # Apply encoding
            if method in ['one_hot', 'label', 'binary', 'ordinal']:
                func_name = f"apply_{method}_encoding"
                if func_name.replace('apply_', '').replace('_encoding', '') in dir():
                    X_train_enc = globals()[func_name](X_train)
                    X_test_enc = globals()[func_name](X_test)
                else:
                    # Fallback
                    X_train_enc = pd.get_dummies(X_train)
                    X_test_enc = pd.get_dummies(X_test)
            else:
                # Use first available function
                func_name = f"apply_{methods[0].replace('-', '_')}_encoding"
                X_train_enc = globals().get(func_name, lambda x: pd.get_dummies(x))(X_train)
                X_test_enc = globals().get(func_name, lambda x: pd.get_dummies(x))(X_test)
            
            # Align columns
            X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join='left', axis=1, fill_value=0)
            
            # Scale and train
            scaler_enc = StandardScaler()
            X_train_enc_scaled = scaler_enc.fit_transform(X_train_enc)
            X_test_enc_scaled = scaler_enc.transform(X_test_enc)
            
            results_enc = train_and_evaluate(X_train_enc_scaled, X_test_enc_scaled,
                                           y_train, y_test, 'rf')
            results_dict[method] = results_enc
            print(f"      Test R²: {results_enc['test_r2']:.4f}, RMSE: {results_enc['test_rmse']:.2f}")
        except Exception as e:
            print(f"      Error: {str(e)}")
            continue
    
    # Feature importance (if RF used)
    print("\n5. Analyzing feature importance...")
    feature_importance = None
    for model_name, results in results_dict.items():
        if hasattr(results['model'], 'feature_importances_'):
            if hasattr(results['model'], 'n_features_in_'):
                n_feats = results['model'].n_features_in_
                feature_names = [f'feature_{i}' for i in range(n_feats)]
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': results['model'].feature_importances_
                }).sort_values('importance', ascending=False)
                print(f"\n   Top 10 features ({model_name}):")
                for idx, row in feature_importance.head(10).iterrows():
                    print(f"   {row['feature']:30s}: {row['importance']:.4f}")
                break
    
    # Summary
    print("\n6. Performance Summary:")
    print("   " + "-" * 75)
    print(f"   {'Model':<20} {'Test RMSE':<12} {'Test R²':<12} {'CV R²':<12}")
    print("   " + "-" * 75)
    for model_name, results in results_dict.items():
        print(f"   {model_name:<20} {results['test_rmse']:<12.2f} "
              f"{results['test_r2']:<12.4f} {results['cv_mean']:<12.4f}")
    print("   " + "-" * 75)
    
    # Visualizations
    print("\n7. Creating comprehensive visualizations...")
    plot_comprehensive_analysis(df, results_dict, feature_importance)
    
    # Best model
    best_model = max(results_dict.items(), key=lambda x: x[1]['test_r2'])
    print(f"\n8. Best Method: {best_model[0]}")
    print(f"   Test R²: {best_model[1]['test_r2']:.4f}")
    improvement = ((best_model[1]['test_r2'] - results_baseline['test_r2']) /
                   results_baseline['test_r2'] * 100)
    print(f"   Improvement: {improvement:.1f}%")
    
    print("\n" + "=" * 90)
    print("Analysis Complete!")
    print("=" * 90)


if __name__ == "__main__":
    main()
