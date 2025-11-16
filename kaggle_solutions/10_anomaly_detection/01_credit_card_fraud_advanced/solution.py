"""
Advanced Credit Card Fraud Detection
Uses multiple anomaly detection techniques to identify fraudulent transactions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def generate_credit_card_data(n_samples=10000, fraud_ratio=0.02):
    """Generate synthetic credit card transaction data with fraud"""
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud

    # Normal transactions
    normal_amounts = np.random.lognormal(mean=3.5, sigma=1.2, size=n_normal)
    normal_time = np.random.uniform(0, 86400, n_normal)
    normal_v1 = np.random.normal(0, 1, n_normal)
    normal_v2 = np.random.normal(0, 1, n_normal)
    normal_v3 = np.random.normal(0, 1, n_normal)
    normal_v4 = np.random.normal(0, 1, n_normal)
    normal_distance = np.random.exponential(scale=50, size=n_normal)
    normal_frequency = np.random.poisson(lam=3, size=n_normal)

    # Fraudulent transactions (different patterns)
    fraud_amounts = np.concatenate([
        np.random.uniform(0.01, 10, n_fraud // 2),  # Small test charges
        np.random.uniform(500, 2000, n_fraud - n_fraud // 2)  # Large fraudulent charges
    ])

    fraud_time = np.concatenate([
        np.random.uniform(0, 21600, n_fraud // 2),  # Unusual hours (midnight-6am)
        np.random.uniform(21600, 86400, n_fraud - n_fraud // 2)
    ])

    fraud_v1 = np.random.normal(2, 1.5, n_fraud)  # Different distribution
    fraud_v2 = np.random.normal(-2, 1.5, n_fraud)
    fraud_v3 = np.random.normal(3, 2, n_fraud)
    fraud_v4 = np.random.normal(-1, 1.5, n_fraud)
    fraud_distance = np.random.exponential(scale=500, size=n_fraud)  # Unusual locations
    fraud_frequency = np.random.poisson(lam=15, size=n_fraud)  # High frequency

    # Combine data
    data = pd.DataFrame({
        'Time': np.concatenate([normal_time, fraud_time]),
        'V1': np.concatenate([normal_v1, fraud_v1]),
        'V2': np.concatenate([normal_v2, fraud_v2]),
        'V3': np.concatenate([normal_v3, fraud_v3]),
        'V4': np.concatenate([normal_v4, fraud_v4]),
        'Amount': np.concatenate([normal_amounts, fraud_amounts]),
        'Distance_from_home': np.concatenate([normal_distance, fraud_distance]),
        'Transaction_frequency_24h': np.concatenate([normal_frequency, fraud_frequency]),
        'Class': np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
    })

    # Shuffle
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    return data

def plot_data_distribution(data):
    """Visualize data distribution"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Credit Card Transaction Features Distribution', fontsize=16)

    features = ['Amount', 'V1', 'V2', 'Distance_from_home', 'Transaction_frequency_24h', 'Time']

    for idx, feature in enumerate(features):
        ax = axes[idx // 3, idx % 3]

        normal_data = data[data['Class'] == 0][feature]
        fraud_data = data[data['Class'] == 1][feature]

        ax.hist(normal_data, bins=50, alpha=0.6, label='Normal', density=True)
        ax.hist(fraud_data, bins=50, alpha=0.6, label='Fraud', density=True)
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.legend()
        ax.set_title(f'{feature} Distribution')

    plt.tight_layout()
    plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
    print("Saved: data_distribution.png")

def evaluate_model(y_true, y_pred, model_name):
    """Evaluate anomaly detection model"""
    print(f"\n{'='*50}")
    print(f"{model_name} Results")
    print('='*50)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Fraud']))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    return {'precision': precision, 'recall': recall, 'f1': f1}

def plot_confusion_matrices(results, y_true):
    """Plot confusion matrices for all models"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Confusion Matrices for Different Models', fontsize=16)

    for idx, (model_name, y_pred) in enumerate(results.items()):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
        axes[idx].set_title(model_name)
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("\nSaved: confusion_matrices.png")

def plot_model_comparison(metrics_dict):
    """Compare model performance"""
    models = list(metrics_dict.keys())
    metrics = ['precision', 'recall', 'f1']

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, metric in enumerate(metrics):
        values = [metrics_dict[model][metric] for model in models]
        ax.bar(x + idx * width, values, width, label=metric.capitalize())

    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: model_comparison.png")

def main():
    print("Advanced Credit Card Fraud Detection")
    print("="*50)

    # Generate data
    print("\nGenerating synthetic credit card transaction data...")
    data = generate_credit_card_data(n_samples=10000, fraud_ratio=0.02)
    print(f"Total transactions: {len(data)}")
    print(f"Fraudulent transactions: {data['Class'].sum():.0f} ({data['Class'].mean()*100:.2f}%)")

    # Visualize distribution
    print("\nVisualizing data distribution...")
    plot_data_distribution(data)

    # Prepare features
    feature_cols = ['Time', 'V1', 'V2', 'V3', 'V4', 'Amount',
                   'Distance_from_home', 'Transaction_frequency_24h']
    X = data[feature_cols].values
    y_true = data['Class'].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Store predictions
    predictions = {}
    metrics_dict = {}

    # Model 1: Isolation Forest
    print("\n" + "="*50)
    print("Training Isolation Forest...")
    iso_forest = IsolationForest(contamination=0.02, random_state=42, n_estimators=100)
    y_pred_if = iso_forest.fit_predict(X_scaled)
    y_pred_if = (y_pred_if == -1).astype(int)  # Convert -1/1 to 0/1
    predictions['Isolation Forest'] = y_pred_if
    metrics_dict['Isolation Forest'] = evaluate_model(y_true, y_pred_if, "Isolation Forest")

    # Model 2: Local Outlier Factor
    print("\n" + "="*50)
    print("Training Local Outlier Factor...")
    lof = LocalOutlierFactor(contamination=0.02, novelty=False)
    y_pred_lof = lof.fit_predict(X_scaled)
    y_pred_lof = (y_pred_lof == -1).astype(int)
    predictions['Local Outlier Factor'] = y_pred_lof
    metrics_dict['Local Outlier Factor'] = evaluate_model(y_true, y_pred_lof, "Local Outlier Factor")

    # Model 3: One-Class SVM
    print("\n" + "="*50)
    print("Training One-Class SVM...")
    ocsvm = OneClassSVM(gamma='auto', nu=0.02)
    y_pred_svm = ocsvm.fit_predict(X_scaled)
    y_pred_svm = (y_pred_svm == -1).astype(int)
    predictions['One-Class SVM'] = y_pred_svm
    metrics_dict['One-Class SVM'] = evaluate_model(y_true, y_pred_svm, "One-Class SVM")

    # Visualize results
    print("\n" + "="*50)
    print("Generating visualizations...")
    plot_confusion_matrices(predictions, y_true)
    plot_model_comparison(metrics_dict)

    # PCA visualization of anomalies
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # True labels
    scatter = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true,
                                cmap='coolwarm', alpha=0.6, s=20)
    axes[0, 0].set_title('True Labels')
    axes[0, 0].set_xlabel('First Principal Component')
    axes[0, 0].set_ylabel('Second Principal Component')
    plt.colorbar(scatter, ax=axes[0, 0])

    # Predictions
    for idx, (model_name, y_pred) in enumerate(predictions.items()):
        ax = axes[(idx+1) // 2, (idx+1) % 2]
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred,
                           cmap='coolwarm', alpha=0.6, s=20)
        ax.set_title(f'{model_name} Predictions')
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        plt.colorbar(scatter, ax=ax)

    plt.tight_layout()
    plt.savefig('pca_visualization.png', dpi=300, bbox_inches='tight')
    print("Saved: pca_visualization.png")

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print("\nBest Model by Metric:")
    print(f"Best Precision: {max(metrics_dict.items(), key=lambda x: x[1]['precision'])[0]}")
    print(f"Best Recall: {max(metrics_dict.items(), key=lambda x: x[1]['recall'])[0]}")
    print(f"Best F1-Score: {max(metrics_dict.items(), key=lambda x: x[1]['f1'])[0]}")

    print("\n" + "="*50)
    print("Analysis complete! Check the generated visualizations.")
    print("="*50)

if __name__ == "__main__":
    main()
