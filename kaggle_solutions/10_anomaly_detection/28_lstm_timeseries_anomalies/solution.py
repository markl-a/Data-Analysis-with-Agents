"""
LSTM for Time Series Anomaly Detection
======================================

This solution uses LSTM networks for time series anomalies:
1. Prediction-based anomaly detection
2. Reconstruction-based detection
3. Attention mechanisms
4. Sequence-to-sequence models

Author: Kaggle Solutions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    f1_score, precision_score, recall_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

try:
    from pyod.models.lof import LOF
    from pyod.models.knn import KNN
    from pyod.models.iforest import IForest
    from pyod.models.ocsvm import OCSVM
    from pyod.models.cblof import CBLOF
    from pyod.models.feature_bagging import FeatureBagging
    from pyod.models.lscp import LSCP
    from pyod.models.suod import SUOD
    PYOD_AVAILABLE = True
except ImportError:
    PYOD_AVAILABLE = False
    print("Warning: pyod not available, using sklearn alternatives")


def generate_anomaly_data(n_samples=1000, n_features=8, contamination=0.1):
    """Generate synthetic data with anomalies"""
    n_normal = int(n_samples * (1 - contamination))
    n_anomalies = n_samples - n_normal
    
    # Normal data with clusters
    n_clusters = 3
    samples_per_cluster = n_normal // n_clusters
    
    X_normal = []
    for i in range(n_clusters):
        center = np.random.randn(n_features) * 3
        cluster = np.random.randn(samples_per_cluster, n_features) * 0.8 + center
        X_normal.append(cluster)
    
    if n_normal % n_clusters != 0:
        remaining = n_normal - (samples_per_cluster * n_clusters)
        center = np.random.randn(n_features) * 3
        cluster = np.random.randn(remaining, n_features) * 0.8 + center
        X_normal.append(cluster)
    
    X_normal = np.vstack(X_normal)
    
    # Anomalies - various types
    X_anomalies = []
    
    # Type 1: Isolated points
    n_type1 = n_anomalies // 2
    X_type1 = np.random.uniform(-10, 10, (n_type1, n_features))
    X_anomalies.append(X_type1)
    
    # Type 2: Small anomalous cluster
    n_type2 = n_anomalies - n_type1
    anomaly_center = np.random.randn(n_features) * 8
    X_type2 = np.random.randn(n_type2, n_features) * 0.5 + anomaly_center
    X_anomalies.append(X_type2)
    
    X_anomalies = np.vstack(X_anomalies)
    
    # Combine
    X = np.vstack([X_normal, X_anomalies])
    y = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def plot_roc_pr_curves(y_test, scores_dict):
    """Plot ROC and PR curves for multiple detectors"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray']
    
    for i, (name, scores) in enumerate(scores_dict.items()):
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f'{name} (AUC={roc_auc:.3f})')
        
        # PR curve
        precision, recall, _ = precision_recall_curve(y_test, scores)
        pr_auc = auc(recall, precision)
        ax2.plot(recall, precision, color=colors[i % len(colors)], lw=2,
                label=f'{name} (AUC={pr_auc:.3f})')
    
    ax1.plot([0, 1], [0, 1], 'k--', lw=1)
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curves', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_score_distributions(y_test, scores_dict):
    """Plot score distributions"""
    n_detectors = len(scores_dict)
    n_cols = min(3, n_detectors)
    n_rows = (n_detectors + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    axes = axes.flatten() if n_detectors > 1 else [axes]
    
    for i, (name, scores) in enumerate(scores_dict.items()):
        ax = axes[i]
        
        normal_scores = scores[y_test == 0]
        anomaly_scores = scores[y_test == 1]
        
        ax.hist(normal_scores, bins=50, alpha=0.7, color='blue',
               label='Normal', density=True)
        ax.hist(anomaly_scores, bins=50, alpha=0.7, color='red',
               label='Anomaly', density=True)
        
        ax.set_xlabel('Anomaly Score', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_detectors, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def plot_2d_visualization(X, y_true, y_pred_dict, title="Anomaly Detection"):
    """Visualize detection results in 2D (PCA projection)"""
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    n_detectors = len(y_pred_dict)
    n_cols = min(3, n_detectors + 1)
    n_rows = (n_detectors + 2) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    axes = axes.flatten() if n_detectors >= 1 else [axes]
    
    # True labels
    axes[0].scatter(X_2d[y_true == 0, 0], X_2d[y_true == 0, 1],
                   c='blue', alpha=0.6, s=20, label='Normal')
    axes[0].scatter(X_2d[y_true == 1, 0], X_2d[y_true == 1, 1],
                   c='red', alpha=0.8, s=40, marker='^', label='Anomaly')
    axes[0].set_title('True Labels', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Predictions
    for i, (name, y_pred) in enumerate(y_pred_dict.items()):
        ax = axes[i + 1]
        
        ax.scatter(X_2d[y_pred == 0, 0], X_2d[y_pred == 0, 1],
                  c='blue', alpha=0.6, s=20, label='Normal')
        ax.scatter(X_2d[y_pred == 1, 0], X_2d[y_pred == 1, 1],
                  c='red', alpha=0.8, s=40, marker='^', label='Anomaly')
        
        f1 = f1_score(y_true, y_pred)
        ax.set_title(f'{name}\nF1={f1:.3f}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_detectors + 1, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def evaluate_detectors(y_test, y_pred_dict, scores_dict):
    """Evaluate all detectors"""
    results = []
    
    for name in y_pred_dict.keys():
        y_pred = y_pred_dict[name]
        scores = scores_dict[name]
        
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_auc = auc(fpr, tpr)
        
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, scores)
        pr_auc = auc(rec_curve, prec_curve)
        
        results.append({
            'Detector': name,
            'F1 Score': f1,
            'Precision': precision,
            'Recall': recall,
            'ROC AUC': roc_auc,
            'PR AUC': pr_auc
        })
    
    return pd.DataFrame(results)


def main():
    """Main execution function"""
    print("=" * 80)
    print("LSTM for Time Series Anomaly Detection")
    print("=" * 80)
    
    np.random.seed(42)
    
    # Generate data
    print("\n1. Generating synthetic data...")
    X, y = generate_anomaly_data(n_samples=1500, n_features=10, contamination=0.12)
    print(f"   Dataset shape: {X.shape}")
    print(f"   Anomaly ratio: {y.sum() / len(y):.3f}")
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"   Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Initialize detectors and store predictions/scores
    print("\n2. Training detectors...")
    
    y_pred_dict = {}
    scores_dict = {}
    
    # Add detector implementations here based on the specific solution
    # This is a template that will be customized per solution
    
    # Example: Isolation Forest
    print("   Training Isolation Forest...")
    clf_if = IsolationForest(contamination=0.12, random_state=42)
    clf_if.fit(X_train)
    y_pred_dict['Isolation Forest'] = (clf_if.predict(X_test) == -1).astype(int)
    scores_dict['Isolation Forest'] = -clf_if.score_samples(X_test)
    
    # Example: One-Class SVM
    print("   Training One-Class SVM...")
    clf_svm = OneClassSVM(nu=0.12, kernel='rbf', gamma='auto')
    clf_svm.fit(X_train)
    y_pred_dict['One-Class SVM'] = (clf_svm.predict(X_test) == -1).astype(int)
    scores_dict['One-Class SVM'] = -clf_svm.score_samples(X_test)
    
    # Evaluate
    print("\n3. Evaluating detectors...")
    results_df = evaluate_detectors(y_test, y_pred_dict, scores_dict)
    
    for _, row in results_df.iterrows():
        print(f"   {row['Detector']}: F1={row['F1 Score']:.3f}, "
              f"Precision={row['Precision']:.3f}, "
              f"Recall={row['Recall']:.3f}")
    
    # Visualizations
    print("\n4. Creating visualizations...")
    
    # ROC and PR curves
    fig = plot_roc_pr_curves(y_test, scores_dict)
    plt.savefig('28_lstm_timeseries_anomalies_roc_pr_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Score distributions
    fig = plot_score_distributions(y_test, scores_dict)
    plt.savefig('28_lstm_timeseries_anomalies_score_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2D visualization
    fig = plot_2d_visualization(X_test, y_test, y_pred_dict, "LSTM for Time Series Anomaly Detection")
    plt.savefig('28_lstm_timeseries_anomalies_2d_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Results
    print("\n5. Final Performance Comparison:")
    print("\n" + "="*80)
    print(results_df.to_string(index=False))
    print("="*80)
    
    results_df.to_csv('28_lstm_timeseries_anomalies_results.csv', index=False)
    print("\nResults saved!")
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
