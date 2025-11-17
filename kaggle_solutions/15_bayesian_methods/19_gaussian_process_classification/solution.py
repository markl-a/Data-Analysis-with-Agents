"""Bayesian Gaussian Process Classification
Implements GP classification with Laplace approximation for posterior inference."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import expit
import warnings
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)

class GPClassifier:
    def __init__(self, length_scale=1.0):
        self.length_scale = length_scale
        self.X_train = None
        self.y_train = None
        self.f_map = None
    
    def kernel(self, X1, X2):
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return np.exp(-0.5 / self.length_scale**2 * sqdist)
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        K = self.kernel(X, X)
        self.f_map = np.zeros(len(y))
        for _ in range(20):
            W = expit(self.f_map) * (1 - expit(self.f_map))
            W_sqrt = np.sqrt(W)
            L = np.linalg.cholesky(np.eye(len(X)) + W_sqrt[:,None] * K * W_sqrt[None,:])
            b = W * self.f_map + (y - expit(self.f_map))
            a = b - W_sqrt * np.linalg.solve(L.T, np.linalg.solve(L, W_sqrt * K @ b))
            self.f_map = K @ a
        return self
    
    def predict_proba(self, X_test):
        K_s = self.kernel(self.X_train, X_test)
        return expit(K_s.T @ self.f_map)

def main():
    print("="*80)
    print("GAUSSIAN PROCESS CLASSIFICATION")
    print("="*80)
    
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                               n_informative=2, random_state=42, n_clusters_per_class=1)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    gpc = GPClassifier(length_scale=1.0)
    gpc.fit(X_train, y_train)
    
    proba = gpc.predict_proba(X_test)
    pred = (proba >= 0.5).astype(int)
    
    acc = np.mean(pred == y_test)
    print(f"\nAccuracy: {acc:.3f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    ax.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], c='blue', marker='o', label='Class 0')
    ax.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], c='red', marker='s', label='Class 1')
    ax.set_title('Training Data')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/gp_classification.png', dpi=150)
    print("\nSaved: GP classification visualization")
    plt.close()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
