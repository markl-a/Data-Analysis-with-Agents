"""
Transaction Pattern Anomaly Detection
Detects unusual transaction patterns in financial systems using multiple techniques
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def generate_transaction_data(n_customers=500, transactions_per_customer=100, anomaly_ratio=0.03):
    """Generate synthetic transaction data with anomalous patterns"""
    all_data = []

    for customer_id in range(n_customers):
        n_transactions = transactions_per_customer
        n_anomalies = int(n_transactions * anomaly_ratio)
        n_normal = n_transactions - n_anomalies

        # Customer spending patterns
        customer_avg_amount = np.random.lognormal(mean=np.log(100), sigma=1)
        customer_frequency = np.random.poisson(lam=5) + 1  # Transactions per day

        # Normal transactions
        normal_amounts = np.random.lognormal(mean=np.log(customer_avg_amount), sigma=0.5, size=n_normal)

        # Transaction times (business hours biased)
        normal_hours = np.random.choice(
            list(range(24)),
            size=n_normal,
            p=[0.01]*7 + [0.05]*2 + [0.08]*8 + [0.06]*5 + [0.02]*2  # Higher during day
        )
        normal_minutes = np.random.randint(0, 60, n_normal)

        # Merchant categories (normal distribution)
        normal_categories = np.random.choice(
            ['grocery', 'restaurant', 'gas', 'shopping', 'utilities'],
            size=n_normal,
            p=[0.3, 0.25, 0.15, 0.2, 0.1]
        )

        # Geographic distance from home
        normal_distance = np.random.exponential(scale=20, size=n_normal)

        # Time since last transaction
        normal_time_since_last = np.random.exponential(scale=24/customer_frequency, size=n_normal)

        # Number of transactions this merchant
        normal_merchant_count = np.random.poisson(lam=5, size=n_normal)

        # Anomalous transactions
        anomaly_types = np.random.choice(['large_amount', 'unusual_time', 'unusual_location',
                                         'rapid_sequence', 'new_merchant'], n_anomalies)

        anomaly_amounts = np.zeros(n_anomalies)
        anomaly_hours = np.zeros(n_anomalies)
        anomaly_minutes = np.zeros(n_anomalies)
        anomaly_categories = []
        anomaly_distance = np.zeros(n_anomalies)
        anomaly_time_since_last = np.zeros(n_anomalies)
        anomaly_merchant_count = np.zeros(n_anomalies)

        for i, anom_type in enumerate(anomaly_types):
            if anom_type == 'large_amount':
                # Unusually large purchase
                anomaly_amounts[i] = customer_avg_amount * np.random.uniform(10, 50)
                anomaly_hours[i] = np.random.choice(range(24))
                anomaly_minutes[i] = np.random.randint(0, 60)
                anomaly_categories.append(np.random.choice(['electronics', 'jewelry', 'travel']))
                anomaly_distance[i] = np.random.exponential(scale=25)
                anomaly_time_since_last[i] = np.random.exponential(scale=24/customer_frequency)
                anomaly_merchant_count[i] = 0  # New merchant
            elif anom_type == 'unusual_time':
                # Middle of the night transaction
                anomaly_amounts[i] = customer_avg_amount * np.random.lognormal(0, 0.5)
                anomaly_hours[i] = np.random.choice(range(2, 5))  # 2-5 AM
                anomaly_minutes[i] = np.random.randint(0, 60)
                anomaly_categories.append(np.random.choice(['online', 'gas', 'restaurant']))
                anomaly_distance[i] = np.random.exponential(scale=30)
                anomaly_time_since_last[i] = np.random.exponential(scale=24/customer_frequency)
                anomaly_merchant_count[i] = np.random.poisson(lam=2)
            elif anom_type == 'unusual_location':
                # Far from home
                anomaly_amounts[i] = customer_avg_amount * np.random.lognormal(0, 0.8)
                anomaly_hours[i] = np.random.choice(range(24))
                anomaly_minutes[i] = np.random.randint(0, 60)
                anomaly_categories.append(np.random.choice(['travel', 'hotel', 'foreign']))
                anomaly_distance[i] = np.random.uniform(200, 2000)  # Very far
                anomaly_time_since_last[i] = np.random.exponential(scale=24/customer_frequency)
                anomaly_merchant_count[i] = 0
            elif anom_type == 'rapid_sequence':
                # Multiple transactions in short time
                anomaly_amounts[i] = customer_avg_amount * np.random.lognormal(0, 0.5)
                anomaly_hours[i] = np.random.choice(range(24))
                anomaly_minutes[i] = np.random.randint(0, 60)
                anomaly_categories.append(np.random.choice(['online', 'shopping', 'electronics']))
                anomaly_distance[i] = np.random.uniform(50, 500)  # Different locations rapidly
                anomaly_time_since_last[i] = np.random.uniform(0.01, 0.5)  # Minutes apart
                anomaly_merchant_count[i] = 0
            else:  # new_merchant
                # First time at this merchant with large amount
                anomaly_amounts[i] = customer_avg_amount * np.random.uniform(3, 15)
                anomaly_hours[i] = np.random.choice(range(24))
                anomaly_minutes[i] = np.random.randint(0, 60)
                anomaly_categories.append(np.random.choice(['online', 'foreign', 'electronics']))
                anomaly_distance[i] = np.random.exponential(scale=100)
                anomaly_time_since_last[i] = np.random.exponential(scale=24/customer_frequency)
                anomaly_merchant_count[i] = 0

        # Encode categories
        category_mapping = {'grocery': 0, 'restaurant': 1, 'gas': 2, 'shopping': 3,
                           'utilities': 4, 'electronics': 5, 'jewelry': 6, 'travel': 7,
                           'online': 8, 'hotel': 9, 'foreign': 10}

        normal_category_codes = [category_mapping[cat] for cat in normal_categories]
        anomaly_category_codes = [category_mapping[cat] for cat in anomaly_categories]

        # Combine
        customer_data = pd.DataFrame({
            'customer_id': customer_id,
            'amount': np.concatenate([normal_amounts, anomaly_amounts]),
            'hour': np.concatenate([normal_hours, anomaly_hours]).astype(int),
            'minute': np.concatenate([normal_minutes, anomaly_minutes]).astype(int),
            'category_code': np.concatenate([normal_category_codes, anomaly_category_codes]),
            'distance_km': np.concatenate([normal_distance, anomaly_distance]),
            'hours_since_last': np.concatenate([normal_time_since_last, anomaly_time_since_last]),
            'merchant_transaction_count': np.concatenate([normal_merchant_count, anomaly_merchant_count]),
            'label': np.concatenate([np.zeros(n_normal), np.ones(n_anomalies)])
        })

        all_data.append(customer_data)

    # Combine all customers
    data = pd.concat(all_data, ignore_index=True)

    # Add derived features
    data['amount_log'] = np.log(data['amount'] + 1)
    data['is_weekend'] = np.random.binomial(1, 0.3, len(data))  # 30% weekend
    data['is_night'] = ((data['hour'] >= 22) | (data['hour'] <= 6)).astype(int)
    data['distance_log'] = np.log(data['distance_km'] + 1)

    # Shuffle
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    return data

def plot_transaction_patterns(data):
    """Visualize transaction patterns"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Transaction Pattern Analysis', fontsize=16)

    # Amount distribution
    ax = axes[0, 0]
    normal_amounts = data[data['label'] == 0]['amount']
    anomaly_amounts = data[data['label'] == 1]['amount']
    ax.hist(np.log(normal_amounts + 1), bins=50, alpha=0.6, label='Normal', density=True)
    ax.hist(np.log(anomaly_amounts + 1), bins=50, alpha=0.6, label='Anomaly', density=True)
    ax.set_xlabel('Log(Amount)')
    ax.set_ylabel('Density')
    ax.legend()

    # Time of day
    ax = axes[0, 1]
    ax.hist(data[data['label'] == 0]['hour'], bins=24, alpha=0.6, label='Normal', density=True)
    ax.hist(data[data['label'] == 1]['hour'], bins=24, alpha=0.6, label='Anomaly', density=True)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Density')
    ax.legend()

    # Distance from home
    ax = axes[0, 2]
    ax.hist(np.log(data[data['label'] == 0]['distance_km'] + 1), bins=50,
           alpha=0.6, label='Normal', density=True)
    ax.hist(np.log(data[data['label'] == 1]['distance_km'] + 1), bins=50,
           alpha=0.6, label='Anomaly', density=True)
    ax.set_xlabel('Log(Distance km)')
    ax.set_ylabel('Density')
    ax.legend()

    # Time since last
    ax = axes[1, 0]
    ax.hist(np.log(data[data['label'] == 0]['hours_since_last'] + 1), bins=50,
           alpha=0.6, label='Normal', density=True)
    ax.hist(np.log(data[data['label'] == 1]['hours_since_last'] + 1), bins=50,
           alpha=0.6, label='Anomaly', density=True)
    ax.set_xlabel('Log(Hours Since Last Transaction)')
    ax.set_ylabel('Density')
    ax.legend()

    # Category
    ax = axes[1, 1]
    normal_cat = data[data['label'] == 0]['category_code'].value_counts().sort_index()
    anomaly_cat = data[data['label'] == 1]['category_code'].value_counts().sort_index()
    x = range(len(normal_cat))
    width = 0.35
    ax.bar([i - width/2 for i in x], normal_cat.values, width, alpha=0.6, label='Normal')
    ax.bar([i + width/2 for i in x], anomaly_cat.values, width, alpha=0.6, label='Anomaly')
    ax.set_xlabel('Category Code')
    ax.set_ylabel('Count')
    ax.legend()

    # Merchant count
    ax = axes[1, 2]
    ax.hist(data[data['label'] == 0]['merchant_transaction_count'], bins=30,
           alpha=0.6, label='Normal', density=True)
    ax.hist(data[data['label'] == 1]['merchant_transaction_count'], bins=30,
           alpha=0.6, label='Anomaly', density=True)
    ax.set_xlabel('Merchant Transaction Count')
    ax.set_ylabel('Density')
    ax.legend()

    plt.tight_layout()
    plt.savefig('transaction_patterns.png', dpi=300, bbox_inches='tight')
    print("Saved: transaction_patterns.png")

def evaluate_detector(y_true, y_pred, model_name):
    """Evaluate anomaly detector"""
    print(f"\n{'='*60}")
    print(f"{model_name} Evaluation")
    print('='*60)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    return {'precision': precision, 'recall': recall, 'f1': f1}

def main():
    print("Transaction Pattern Anomaly Detection")
    print("="*60)

    # Generate data
    print("\nGenerating synthetic transaction data...")
    data = generate_transaction_data(n_customers=500, transactions_per_customer=100, anomaly_ratio=0.03)
    print(f"Total transactions: {len(data)}")
    print(f"Number of customers: {data['customer_id'].nunique()}")
    print(f"Anomalous transactions: {data['label'].sum():.0f} ({data['label'].mean()*100:.2f}%)")

    # Visualize
    print("\nVisualizing transaction patterns...")
    plot_transaction_patterns(data)

    # Prepare features
    feature_cols = ['amount_log', 'hour', 'category_code', 'distance_log',
                   'hours_since_last', 'merchant_transaction_count', 'is_weekend', 'is_night']

    X = data[feature_cols].values
    y_true = data['label'].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    metrics_dict = {}

    # Method 1: One-Class SVM
    print("\n" + "="*60)
    print("Training One-Class SVM...")
    ocsvm = OneClassSVM(gamma='auto', nu=0.03)
    y_pred_svm = ocsvm.fit_predict(X_scaled)
    y_pred_svm = (y_pred_svm == -1).astype(int)
    metrics_dict['One-Class SVM'] = evaluate_detector(y_true, y_pred_svm, "One-Class SVM")

    # Method 2: Local Outlier Factor
    print("\n" + "="*60)
    print("Training Local Outlier Factor...")
    lof = LocalOutlierFactor(contamination=0.03, novelty=False)
    y_pred_lof = lof.fit_predict(X_scaled)
    y_pred_lof = (y_pred_lof == -1).astype(int)
    metrics_dict['Local Outlier Factor'] = evaluate_detector(y_true, y_pred_lof, "Local Outlier Factor")

    # Method 3: Rule-based
    print("\n" + "="*60)
    print("Applying Transaction Rules...")

    # Define rules for suspicious transactions
    y_pred_rules = (
        (data['amount'] > data.groupby('customer_id')['amount'].transform('mean') * 10) |
        ((data['hour'] >= 1) & (data['hour'] <= 5)) |
        (data['distance_km'] > 500) |
        ((data['hours_since_last'] < 0.1) & (data['distance_km'] > 50)) |
        (data['merchant_transaction_count'] == 0) & (data['amount'] > 1000)
    ).astype(int)

    metrics_dict['Rule-Based'] = evaluate_detector(y_true, y_pred_rules, "Rule-Based System")

    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Detection Results Comparison', fontsize=16)

    methods = [
        ('One-Class SVM', y_pred_svm),
        ('Local Outlier Factor', y_pred_lof),
        ('Rule-Based', y_pred_rules)
    ]

    for idx, (name, y_pred) in enumerate(methods):
        ax = axes[idx // 2, idx % 2]
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax,
                   xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
        ax.set_title(f'{name}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

    # Performance comparison
    ax = axes[1, 1]
    comparison_df = pd.DataFrame(metrics_dict).T
    comparison_df.plot(kind='bar', ax=ax)
    ax.set_ylabel('Score')
    ax.set_title('Method Comparison')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(title='Metric')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('detection_results.png', dpi=300, bbox_inches='tight')
    print("\nSaved: detection_results.png")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nMethod Performance:")
    print(comparison_df.to_string())

    best_model = max(metrics_dict.items(), key=lambda x: x[1]['f1'])
    print(f"\nBest method: {best_model[0]} (F1: {best_model[1]['f1']:.4f})")

    print("\nRecommendations:")
    print("- Use ensemble of multiple methods for robust fraud detection")
    print("- Rule-based system provides interpretable alerts")
    print("- ML methods capture complex patterns")
    print("- Adjust thresholds based on customer risk profiles")
    print("- Implement real-time scoring for transaction approval")

    print("\n" + "="*60)
    print("Analysis complete! Check the generated visualizations.")
    print("="*60)

if __name__ == "__main__":
    main()
