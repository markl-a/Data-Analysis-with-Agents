"""
Kaggle Solution: Target Encoding for Categorical Features
==========================================================
Demonstrates target encoding techniques including mean encoding,
smoothing, and cross-validation to prevent overfitting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def generate_marketing_data(n_samples=5000):
    """
    Generate synthetic marketing campaign data with high-cardinality categoricals.
    """
    # High-cardinality categorical features
    n_cities = 100
    n_products = 50
    n_channels = 20

    cities = [f'City_{i:03d}' for i in range(n_cities)]
    products = [f'Product_{i:02d}' for i in range(n_products)]
    channels = [f'Channel_{i:02d}' for i in range(n_channels)]

    # Create conversion rates per category
    city_conversion = np.random.beta(2, 5, n_cities)
    product_conversion = np.random.beta(3, 4, n_products)
    channel_conversion = np.random.beta(2, 3, n_channels)

    # Generate data
    data = []
    for _ in range(n_samples):
        city_idx = np.random.randint(0, n_cities)
        product_idx = np.random.randint(0, n_products)
        channel_idx = np.random.randint(0, n_channels)

        city = cities[city_idx]
        product = products[product_idx]
        channel = channels[channel_idx]

        # Numerical features
        age = np.random.randint(18, 70)
        income = np.random.lognormal(10.5, 0.5)
        visits = np.random.poisson(5)
        time_on_site = np.random.gamma(2, 3)

        # Calculate conversion probability based on all features
        prob = (
            0.3 * city_conversion[city_idx] +
            0.3 * product_conversion[product_idx] +
            0.2 * channel_conversion[channel_idx] +
            0.0001 * income +
            0.01 * visits +
            0.02 * time_on_site -
            0.003 * age
        )

        prob = np.clip(prob, 0.05, 0.95)

        # Generate conversion
        converted = np.random.random() < prob

        data.append({
            'city': city,
            'product': product,
            'channel': channel,
            'age': age,
            'income': income,
            'visits': visits,
            'time_on_site': time_on_site,
            'converted': int(converted)
        })

    return pd.DataFrame(data)


def label_encoding(train_df, test_df, cat_columns):
    """
    Simple label encoding (baseline).
    """
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()

    for col in cat_columns:
        le = LabelEncoder()
        train_encoded[col + '_label'] = le.fit_transform(train_df[col])
        # Handle unseen categories
        test_encoded[col + '_label'] = test_df[col].map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

    return train_encoded, test_encoded


def mean_target_encoding(train_df, test_df, cat_columns, target_col):
    """
    Mean target encoding (simple but prone to overfitting).
    """
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()

    for col in cat_columns:
        # Calculate mean target per category on train
        means = train_df.groupby(col)[target_col].mean()
        global_mean = train_df[target_col].mean()

        # Apply to train and test
        train_encoded[col + '_mean_enc'] = train_df[col].map(means)
        test_encoded[col + '_mean_enc'] = test_df[col].map(means).fillna(global_mean)

    return train_encoded, test_encoded


def smoothed_target_encoding(train_df, test_df, cat_columns, target_col, smoothing=10):
    """
    Smoothed target encoding to handle low-frequency categories.
    """
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()

    global_mean = train_df[target_col].mean()

    for col in cat_columns:
        # Calculate statistics per category
        stats = train_df.groupby(col)[target_col].agg(['mean', 'count'])

        # Apply smoothing formula: (count * mean + smoothing * global_mean) / (count + smoothing)
        stats['smoothed'] = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)

        # Apply to train and test
        train_encoded[col + '_smooth_enc'] = train_df[col].map(stats['smoothed'])
        test_encoded[col + '_smooth_enc'] = test_df[col].map(stats['smoothed']).fillna(global_mean)

    return train_encoded, test_encoded


def cv_target_encoding(train_df, cat_columns, target_col, n_folds=5):
    """
    Cross-validation target encoding to prevent overfitting.
    """
    train_encoded = train_df.copy()
    global_mean = train_df[target_col].mean()

    for col in cat_columns:
        train_encoded[col + '_cv_enc'] = 0

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(train_df):
            # Calculate means on training fold
            means = train_df.iloc[train_idx].groupby(col)[target_col].mean()

            # Apply to validation fold
            train_encoded.loc[val_idx, col + '_cv_enc'] = train_df.iloc[val_idx][col].map(means).fillna(global_mean)

    return train_encoded


def evaluate_encoding_method(X_train, X_test, y_train, y_test, method_name):
    """
    Train and evaluate a model with specific encoding.
    """
    # Train model
    model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X_train, y_train)

    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)

    return {
        'method': method_name,
        'n_features': X_train.shape[1],
        'auc': auc,
        'accuracy': acc,
        'log_loss': logloss,
        'model': model
    }


def analyze_encoding_statistics(train_df, cat_col, target_col):
    """
    Analyze encoding statistics for a categorical column.
    """
    stats = train_df.groupby(cat_col).agg({
        target_col: ['mean', 'count', 'std']
    }).reset_index()
    stats.columns = ['category', 'mean', 'count', 'std']
    stats = stats.sort_values('count', ascending=False)

    return stats


def plot_results(results, train_df, stats_city):
    """
    Create comprehensive visualizations.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. AUC comparison
    ax1 = fig.add_subplot(gs[0, :2])
    methods = [r['method'] for r in results]
    aucs = [r['auc'] for r in results]
    colors = plt.cm.RdYlGn(np.array(aucs) / max(aucs))
    bars = ax1.barh(range(len(methods)), aucs, color=colors, alpha=0.8)
    ax1.set_yticks(range(len(methods)))
    ax1.set_yticklabels(methods)
    ax1.set_xlabel('AUC Score', fontsize=12)
    ax1.set_title('Performance Comparison by Encoding Method', fontsize=14, fontweight='bold')
    ax1.set_xlim([0.5, 1.0])
    ax1.grid(True, alpha=0.3, axis='x')
    for i, (bar, score) in enumerate(zip(bars, aucs)):
        ax1.text(score, i, f' {score:.4f}', va='center', fontsize=10)

    # 2. Log Loss comparison (lower is better)
    ax2 = fig.add_subplot(gs[0, 2])
    logloss_scores = [r['log_loss'] for r in results]
    colors_ll = plt.cm.RdYlGn_r(np.array(logloss_scores) / max(logloss_scores))
    ax2.barh(range(len(methods)), logloss_scores, color=colors_ll, alpha=0.8)
    ax2.set_yticks(range(len(methods)))
    ax2.set_yticklabels(methods)
    ax2.set_xlabel('Log Loss (lower is better)', fontsize=12)
    ax2.set_title('Log Loss Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    # 3. Category frequency distribution
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(stats_city['count'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax3.set_xlabel('Samples per Category', fontsize=12)
    ax3.set_ylabel('Number of Categories', fontsize=12)
    ax3.set_title('City Category Frequency Distribution', fontsize=12, fontweight='bold')
    ax3.axvline(stats_city['count'].median(), color='red', linestyle='--',
               label=f"Median: {stats_city['count'].median():.0f}")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Conversion rate by category size
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(stats_city['count'], stats_city['mean'], alpha=0.6, s=50)
    ax4.set_xlabel('Category Sample Count', fontsize=12)
    ax4.set_ylabel('Conversion Rate', fontsize=12)
    ax4.set_title('Conversion Rate vs Category Frequency', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 5. Standard deviation by count
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.scatter(stats_city['count'], stats_city['std'], alpha=0.6, s=50, color='coral')
    ax5.set_xlabel('Category Sample Count', fontsize=12)
    ax5.set_ylabel('Std Dev of Conversion', fontsize=12)
    ax5.set_title('Variance vs Category Frequency', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # 6. Top categories by conversion rate
    ax6 = fig.add_subplot(gs[2, 0])
    top_categories = stats_city.nlargest(15, 'mean')
    ax6.barh(range(len(top_categories)), top_categories['mean'], color='green', alpha=0.7)
    ax6.set_yticks(range(len(top_categories)))
    ax6.set_yticklabels(top_categories['category'])
    ax6.set_xlabel('Conversion Rate', fontsize=12)
    ax6.set_title('Top 15 Cities by Conversion Rate', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='x')

    # 7. Bottom categories by conversion rate
    ax7 = fig.add_subplot(gs[2, 1])
    bottom_categories = stats_city.nsmallest(15, 'mean')
    ax7.barh(range(len(bottom_categories)), bottom_categories['mean'], color='red', alpha=0.7)
    ax7.set_yticks(range(len(bottom_categories)))
    ax7.set_yticklabels(bottom_categories['category'])
    ax7.set_xlabel('Conversion Rate', fontsize=12)
    ax7.set_title('Bottom 15 Cities by Conversion Rate', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='x')

    # 8. Accuracy vs complexity
    ax8 = fig.add_subplot(gs[2, 2])
    accs = [r['accuracy'] for r in results]
    ax8.scatter([r['n_features'] for r in results], accs, s=200, alpha=0.6,
               c=range(len(results)), cmap='viridis')
    for i, r in enumerate(results):
        ax8.annotate(r['method'].split()[0], (r['n_features'], r['accuracy']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax8.set_xlabel('Number of Features', fontsize=12)
    ax8.set_ylabel('Accuracy', fontsize=12)
    ax8.set_title('Accuracy vs Feature Count', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3)

    plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/13_feature_engineering/05_target_encoding/target_encoding_analysis.png',
                dpi=300, bbox_inches='tight')
    print("Plot saved as 'target_encoding_analysis.png'")
    plt.show()


def main():
    """
    Main execution function.
    """
    print("=" * 80)
    print("Target Encoding for Categorical Features")
    print("=" * 80)

    # Generate data
    print("\n1. Generating synthetic marketing campaign data...")
    df = generate_marketing_data(n_samples=5000)
    print(f"   Dataset shape: {df.shape}")
    print(f"   Conversion rate: {df['converted'].mean():.2%}")

    # Categorical columns statistics
    print("\n2. Categorical Features Cardinality:")
    cat_columns = ['city', 'product', 'channel']
    for col in cat_columns:
        print(f"   {col:10s}: {df[col].nunique()} unique values")

    # Split data
    print("\n3. Splitting data...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['converted'])
    print(f"   Train size: {len(train_df)}")
    print(f"   Test size: {len(test_df)}")

    # Numerical features (baseline)
    num_features = ['age', 'income', 'visits', 'time_on_site']

    results = []

    # Method 1: Baseline (numerical features only)
    print("\n4. Method 1: Numerical Features Only (Baseline)...")
    X_train_num = train_df[num_features]
    X_test_num = test_df[num_features]
    result = evaluate_encoding_method(X_train_num, X_test_num,
                                     train_df['converted'], test_df['converted'],
                                     "Numerical Only")
    results.append(result)
    print(f"   AUC: {result['auc']:.4f}, Accuracy: {result['accuracy']:.4f}")

    # Method 2: Label Encoding
    print("\n5. Method 2: Label Encoding...")
    train_label, test_label = label_encoding(train_df, test_df, cat_columns)
    label_features = num_features + [col + '_label' for col in cat_columns]
    result = evaluate_encoding_method(train_label[label_features], test_label[label_features],
                                     train_df['converted'], test_df['converted'],
                                     "Label Encoding")
    results.append(result)
    print(f"   AUC: {result['auc']:.4f}, Accuracy: {result['accuracy']:.4f}")

    # Method 3: Mean Target Encoding
    print("\n6. Method 3: Mean Target Encoding...")
    train_mean, test_mean = mean_target_encoding(train_df, test_df, cat_columns, 'converted')
    mean_features = num_features + [col + '_mean_enc' for col in cat_columns]
    result = evaluate_encoding_method(train_mean[mean_features], test_mean[mean_features],
                                     train_df['converted'], test_df['converted'],
                                     "Mean Encoding")
    results.append(result)
    print(f"   AUC: {result['auc']:.4f}, Accuracy: {result['accuracy']:.4f}")

    # Method 4: Smoothed Target Encoding
    print("\n7. Method 4: Smoothed Target Encoding...")
    train_smooth, test_smooth = smoothed_target_encoding(train_df, test_df, cat_columns, 'converted', smoothing=10)
    smooth_features = num_features + [col + '_smooth_enc' for col in cat_columns]
    result = evaluate_encoding_method(train_smooth[smooth_features], test_smooth[smooth_features],
                                     train_df['converted'], test_df['converted'],
                                     "Smoothed Encoding")
    results.append(result)
    print(f"   AUC: {result['auc']:.4f}, Accuracy: {result['accuracy']:.4f}")

    # Method 5: CV Target Encoding
    print("\n8. Method 5: Cross-Validation Target Encoding...")
    train_cv = cv_target_encoding(train_df, cat_columns, 'converted', n_folds=5)
    # For test, use mean encoding
    train_test_mean, test_cv = mean_target_encoding(train_df, test_df, cat_columns, 'converted')
    cv_features = num_features + [col + '_cv_enc' for col in cat_columns]
    cv_test_features = num_features + [col + '_mean_enc' for col in cat_columns]
    result = evaluate_encoding_method(train_cv[cv_features], test_cv[cv_test_features],
                                     train_df['converted'], test_df['converted'],
                                     "CV Encoding")
    results.append(result)
    print(f"   AUC: {result['auc']:.4f}, Accuracy: {result['accuracy']:.4f}")

    # Summary
    print("\n9. Results Summary:")
    print("-" * 80)
    print(f"{'Method':<25} {'Features':<12} {'AUC':<12} {'Accuracy':<12} {'Log Loss':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['method']:<25} {r['n_features']:<12} {r['auc']:<12.4f} "
              f"{r['accuracy']:<12.4f} {r['log_loss']:<12.4f}")

    # Best method
    print("\n10. Best Method:")
    best_result = max(results, key=lambda x: x['auc'])
    baseline_result = results[0]
    print(f"    Method: {best_result['method']}")
    print(f"    AUC: {best_result['auc']:.4f}")
    print(f"    Improvement over baseline: {((best_result['auc'] - baseline_result['auc']) / baseline_result['auc'] * 100):.2f}%")

    # Analyze city encoding statistics
    print("\n11. Analyzing category statistics...")
    stats_city = analyze_encoding_statistics(train_df, 'city', 'converted')
    print(f"    Cities with <10 samples: {len(stats_city[stats_city['count'] < 10])}")
    print(f"    Mean samples per city: {stats_city['count'].mean():.1f}")
    print(f"    Cities with high variance (std>0.4): {len(stats_city[stats_city['std'] > 0.4])}")

    # Visualizations
    print("\n12. Creating visualizations...")
    plot_results(results, train_df, stats_city)

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
