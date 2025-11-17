"""
Advanced Isolation Forest Detection
===================================

This solution implements advanced Isolation Forest methods:
1. Standard Isolation Forest
2. Extended Isolation Forest
3. Feature importance analysis
4. Ensemble of Isolation Forests with different parameters

Author: Kaggle Solutions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    f1_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree
import warnings
warnings.filterwarnings('ignore')


class ExtendedIsolationForest:
    """Extended Isolation Forest with hyperplane selection"""
    
    def __init__(self, n_estimators=100, max_samples='auto', contamination=0.1, random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.forests_ = []
        
    def fit(self, X):
        """Fit multiple Isolation Forests with different random states"""
        for i in range(self.n_estimators):
            forest = IsolationForest(
                n_estimators=1,
                max_samples=self.max_samples,
                contamination=self.contamination,
                random_state=self.random_state + i if self.random_state else None
            )
            forest.fit(X)
            self.forests_.append(forest)
        return self
    
    def decision_function(self, X):
        """Average anomaly scores from all forests"""
        scores = np.array([f.score_samples(X) for f in self.forests_])
        return -scores.mean(axis=0)  # Negative because lower score = more anomalous
    
    def predict(self, X):
        """Predict anomalies"""
        scores = self.decision_function(X)
        threshold = np.percentile(scores, (1 - self.contamination) * 100)
        return (scores > threshold).astype(int)


class AdaptiveIsolationForest:
    """Adaptive Isolation Forest with dynamic contamination"""
    
    def __init__(self, n_estimators=100, contamination_range=(0.05, 0.15), n_splits=5):
        self.n_estimators = n_estimators
        self.contamination_range = contamination_range
        self.n_splits = n_splits
        self.best_contamination_ = None
        self.forest_ = None
        
    def fit(self, X):
        """Fit with cross-validation to find best contamination"""
        contaminations = np.linspace(self.contamination_range[0], 
                                    self.contamination_range[1], 
                                    self.n_splits)
        
        best_score = -np.inf
        
        for cont in contaminations:
            forest = IsolationForest(n_estimators=self.n_estimators, contamination=cont)
            forest.fit(X)
            
            # Score based on isolation (lower = better for outliers)
            scores = forest.score_samples(X)
            score = -np.mean(scores)  # Simple scoring heuristic
            
            if score > best_score:
                best_score = score
                self.best_contamination_ = cont
        
        # Fit final model with best contamination
        self.forest_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.best_contamination_
        )
        self.forest_.fit(X)
        return self
    
    def decision_function(self, X):
        """Return anomaly scores"""
        return -self.forest_.score_samples(X)
    
    def predict(self, X):
        """Predict anomalies"""
        return (self.forest_.predict(X) == -1).astype(int)


class FeatureWeightedIsolationForest:
    """Isolation Forest with feature importance weighting"""
    
    def __init__(self, n_estimators=100, contamination=0.1):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.forest_ = None
        self.feature_importances_ = None
        
    def fit(self, X):
        """Fit and compute feature importances"""
        self.forest_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination
        )
        self.forest_.fit(X)
        
        # Compute feature importance based on usage in trees
        n_features = X.shape[1]
        self.feature_importances_ = np.zeros(n_features)
        
        for tree in self.forest_.estimators_:
            # Get feature used in each node
            tree_model = tree.tree_
            feature_used = tree_model.feature
            
            for feature_idx in feature_used:
                if feature_idx >= 0:  # Valid feature (not leaf node)
                    self.feature_importances_[feature_idx] += 1
        
        # Normalize
        if self.feature_importances_.sum() > 0:
            self.feature_importances_ /= self.feature_importances_.sum()
        
        return self
    
    def decision_function(self, X):
        """Return anomaly scores"""
        return -self.forest_.score_samples(X)
    
    def predict(self, X):
        """Predict anomalies"""
        return (self.forest_.predict(X) == -1).astype(int)


def generate_anomaly_data(n_samples=1000, n_features=10, contamination=0.1):
    """Generate synthetic data with anomalies"""
    n_normal = int(n_samples * (1 - contamination))
    n_anomalies = n_samples - n_normal
    
    # Normal data
    mean = np.zeros(n_features)
    cov = np.eye(n_features)
    X_normal = np.random.multivariate_normal(mean, cov, n_normal)
    
    # Anomalies - various types
    X_anomalies = []
    
    # Type 1: Global outliers (far from normal data)
    n_type1 = n_anomalies // 2
    X_type1 = np.random.uniform(-8, 8, (n_type1, n_features))
    X_anomalies.append(X_type1)
    
    # Type 2: Local outliers (in low-density regions)
    n_type2 = n_anomalies - n_type1
    X_type2 = np.random.multivariate_normal(mean, cov * 5, n_type2)
    X_anomalies.append(X_type2)
    
    X_anomalies = np.vstack(X_anomalies)
    
    # Combine
    X = np.vstack([X_normal, X_anomalies])
    y = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def tune_isolation_forest(X_train, y_train, X_val, y_val):
    """Tune Isolation Forest parameters"""
    n_estimators_range = [50, 100, 200]
    max_samples_range = [0.5, 0.75, 1.0, 'auto']
    contamination_range = [0.05, 0.1, 0.15, 0.2]
    
    results = []
    
    for n_est in n_estimators_range:
        for max_samp in max_samples_range:
            for cont in contamination_range:
                forest = IsolationForest(
                    n_estimators=n_est,
                    max_samples=max_samp,
                    contamination=cont,
                    random_state=42
                )
                forest.fit(X_train)
                y_pred = (forest.predict(X_val) == -1).astype(int)
                
                f1 = f1_score(y_val, y_pred)
                
                results.append({
                    'n_estimators': n_est,
                    'max_samples': str(max_samp),
                    'contamination': cont,
                    'f1_score': f1
                })
    
    return pd.DataFrame(results)


def plot_anomaly_scores_distribution(detectors, X_test, y_test, names):
    """Plot anomaly score distributions"""
    n_detectors = len(detectors)
    fig, axes = plt.subplots(1, n_detectors, figsize=(6*n_detectors, 5))
    
    if n_detectors == 1:
        axes = [axes]
    
    for ax, detector, name in zip(axes, detectors, names):
        scores = detector.decision_function(X_test)
        
        normal_scores = scores[y_test == 0]
        anomaly_scores = scores[y_test == 1]
        
        ax.hist(normal_scores, bins=50, alpha=0.7, color='blue',
               label='Normal', density=True)
        ax.hist(anomaly_scores, bins=50, alpha=0.7, color='red',
               label='Anomaly', density=True)
        
        ax.set_xlabel('Anomaly Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{name}\nScore Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_feature_importances(detector, feature_names=None):
    """Plot feature importances"""
    if not hasattr(detector, 'feature_importances_'):
        print("Detector does not have feature importances")
        return None
    
    importances = detector.feature_importances_
    n_features = len(importances)
    
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(n_features)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    indices = np.argsort(importances)[::-1]
    
    ax.bar(range(n_features), importances[indices], color='steelblue', alpha=0.7)
    ax.set_xticks(range(n_features))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Importance', fontsize=12)
    ax.set_title('Feature Importances from Isolation Forest', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_contamination_sensitivity(X_train, X_val, y_val, contamination_range):
    """Plot performance vs contamination parameter"""
    results = []
    
    for cont in contamination_range:
        forest = IsolationForest(n_estimators=100, contamination=cont, random_state=42)
        forest.fit(X_train)
        y_pred = (forest.predict(X_val) == -1).astype(int)
        
        results.append({
            'contamination': cont,
            'f1': f1_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred)
        })
    
    results_df = pd.DataFrame(results)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(results_df['contamination'], results_df['f1'], 'b-o',
           label='F1 Score', linewidth=2, markersize=8)
    ax.plot(results_df['contamination'], results_df['precision'], 'g--s',
           label='Precision', linewidth=2, markersize=6)
    ax.plot(results_df['contamination'], results_df['recall'], 'r--^',
           label='Recall', linewidth=2, markersize=6)
    
    # Mark best F1
    best_idx = results_df['f1'].idxmax()
    best_cont = results_df.loc[best_idx, 'contamination']
    best_f1 = results_df.loc[best_idx, 'f1']
    
    ax.axvline(best_cont, color='orange', linestyle=':',
              label=f'Best: {best_cont:.3f}')
    ax.scatter([best_cont], [best_f1], color='red', s=150, zorder=5, marker='*')
    
    ax.set_xlabel('Contamination Parameter', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Isolation Forest Performance vs Contamination',
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_roc_pr_curves(detectors, X_test, y_test, names):
    """Plot ROC and PR curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    
    for i, (detector, name) in enumerate(zip(detectors, names)):
        scores = detector.decision_function(X_test)
        
        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f'{name} (AUC={roc_auc:.3f})')
        
        precision, recall, _ = precision_recall_curve(y_test, scores)
        pr_auc = auc(recall, precision)
        ax2.plot(recall, precision, color=colors[i % len(colors)], lw=2,
                label=f'{name} (AUC={pr_auc:.3f})')
    
    ax1.plot([0, 1], [0, 1], 'k--', lw=1)
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curves', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def evaluate_detector(detector, X_test, y_test, name):
    """Evaluate detector"""
    y_pred = detector.predict(X_test)
    scores = detector.decision_function(X_test)
    
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_auc = auc(fpr, tpr)
    
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, scores)
    pr_auc = auc(rec_curve, prec_curve)
    
    return {
        'Detector': name,
        'F1 Score': f1,
        'Precision': precision,
        'Recall': recall,
        'ROC AUC': roc_auc,
        'PR AUC': pr_auc
    }


def main():
    """Main execution function"""
    print("=" * 80)
    print("Advanced Isolation Forest Detection")
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
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Contamination sensitivity
    print("\n2. Analyzing contamination sensitivity...")
    contamination_range = np.linspace(0.05, 0.25, 20)
    fig = plot_contamination_sensitivity(X_train, X_val, y_val, contamination_range)
    plt.savefig('iforest_contamination_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Train detectors
    print("\n3. Training Isolation Forest variants...")
    detectors = {
        'Standard IF': IsolationForest(n_estimators=100, contamination=0.12, random_state=42),
        'Extended IF': ExtendedIsolationForest(n_estimators=100, contamination=0.12, random_state=42),
        'Adaptive IF': AdaptiveIsolationForest(n_estimators=100, contamination_range=(0.08, 0.16)),
        'Weighted IF': FeatureWeightedIsolationForest(n_estimators=100, contamination=0.12),
        'IF (n=200)': IsolationForest(n_estimators=200, contamination=0.12, random_state=42)
    }
    
    for name, detector in detectors.items():
        if name == 'Standard IF':
            detector.fit(X_train)
            # Convert to our interface
            class Wrapper:
                def __init__(self, model):
                    self.model = model
                def predict(self, X):
                    return (self.model.predict(X) == -1).astype(int)
                def decision_function(self, X):
                    return -self.model.score_samples(X)
            detectors[name] = Wrapper(detector)
            detector.fit(X_train)
        elif name == 'IF (n=200)':
            detector.fit(X_train)
            class Wrapper:
                def __init__(self, model):
                    self.model = model
                def predict(self, X):
                    return (self.model.predict(X) == -1).astype(int)
                def decision_function(self, X):
                    return -self.model.score_samples(X)
            detectors[name] = Wrapper(detector)
            detector.fit(X_train)
        else:
            detector.fit(X_train)
        
        print(f"   {name} trained")
    
    # Feature importance
    print("\n4. Analyzing feature importances...")
    weighted_if = [d for k, d in detectors.items() if 'Weighted' in k][0]
    if hasattr(weighted_if, 'feature_importances_'):
        fig = plot_feature_importances(weighted_if)
        if fig:
            plt.savefig('iforest_feature_importances.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # Evaluate
    print("\n5. Evaluating detectors...")
    results = []
    for name, detector in detectors.items():
        result = evaluate_detector(detector, X_test, y_test, name)
        results.append(result)
        print(f"   {name}: F1={result['F1 Score']:.3f}, "
              f"Precision={result['Precision']:.3f}, "
              f"Recall={result['Recall']:.3f}")
    
    results_df = pd.DataFrame(results)
    
    # Visualizations
    print("\n6. Creating visualizations...")
    
    # Score distributions
    detector_list = list(detectors.values())
    names_list = list(detectors.keys())
    fig = plot_anomaly_scores_distribution(detector_list[:3], X_test, y_test, names_list[:3])
    plt.savefig('iforest_score_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC and PR curves
    fig = plot_roc_pr_curves(detector_list, X_test, y_test, names_list)
    plt.savefig('iforest_roc_pr_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Final results
    print("\n7. Final Performance Comparison:")
    print("\n" + "="*80)
    print(results_df.to_string(index=False))
    print("="*80)
    
    results_df.to_csv('iforest_detection_results.csv', index=False)
    print("\nResults saved!")
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
