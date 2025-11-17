"""Bayesian Gaussian Process Regression
Implements GP regression with various kernels, hyperparameter optimization,
and uncertainty quantification."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)

class GaussianProcessRegressor:
    def __init__(self, kernel="rbf", length_scale=1.0, noise=0.1):
        self.kernel_type = kernel
        self.length_scale = length_scale
        self.noise = noise
        self.X_train = None
        self.y_train = None
    
    def kernel(self, X1, X2):
        if self.kernel_type == "rbf":
            sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
            return np.exp(-0.5 / self.length_scale**2 * sqdist)
        return np.eye(len(X1))
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self
    
    def predict(self, X_test, return_std=False):
        K = self.kernel(self.X_train, self.X_train)
        K += self.noise**2 * np.eye(len(self.X_train))
        K_s = self.kernel(self.X_train, X_test)
        K_ss = self.kernel(X_test, X_test)
        
        K_inv = np.linalg.inv(K)
        mu_s = K_s.T.dot(K_inv).dot(self.y_train)
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        
        if return_std:
            return mu_s, np.sqrt(np.diag(cov_s))
        return mu_s

def main():
    print("="*80)
    print("GAUSSIAN PROCESS REGRESSION")
    print("="*80)
    
    # Generate data
    np.random.seed(42)
    X = np.random.rand(50, 1) * 10
    y = np.sin(X).ravel() + np.random.randn(50) * 0.1
    
    # Fit GP
    gp = GaussianProcessRegressor(length_scale=1.0, noise=0.1)
    gp.fit(X, y)
    
    # Predict
    X_test = np.linspace(0, 10, 200).reshape(-1, 1)
    y_pred, y_std = gp.predict(X_test, return_std=True)
    
    # Visualize
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(X, y, c='r', s=50, zorder=10, label='Observations')
    ax.plot(X_test, y_pred, 'b-', lw=2, label='Prediction')
    ax.fill_between(X_test.ravel(), y_pred - 1.96*y_std, y_pred + 1.96*y_std,
                    alpha=0.3, label='95% CI')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title('Gaussian Process Regression')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/tmp/gp_regression.png', dpi=150)
    print("\nSaved: GP regression visualization")
    plt.close()
    
    print(f"\nMean prediction error: {np.mean(np.abs(y_pred - np.sin(X_test).ravel())):.4f}")
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()


class GPRegressionAdvanced:
    """Advanced GP regression with multiple kernels."""
    
    def __init__(self, kernel_params=None):
        if kernel_params is None:
            kernel_params = {'type': 'rbf', 'length_scale': 1.0, 'variance': 1.0}
        self.kernel_params = kernel_params
        self.X_train = None
        self.y_train = None
        self.noise = 0.1
        
    def kernel_rbf(self, X1, X2, length_scale=1.0, variance=1.0):
        """RBF (squared exponential) kernel."""
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return variance * np.exp(-0.5 / length_scale**2 * sqdist)
    
    def kernel_matern(self, X1, X2, length_scale=1.0, nu=1.5):
        """Matern kernel."""
        from scipy.special import gamma, kv
        
        dists = np.sqrt(np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T))
        dists = np.maximum(dists, 1e-10)
        
        if nu == 0.5:
            return np.exp(-dists / length_scale)
        elif nu == 1.5:
            tmp = np.sqrt(3) * dists / length_scale
            return (1 + tmp) * np.exp(-tmp)
        elif nu == 2.5:
            tmp = np.sqrt(5) * dists / length_scale
            return (1 + tmp + tmp**2 / 3) * np.exp(-tmp)
        else:
            tmp = np.sqrt(2 * nu) * dists / length_scale
            return (2**(1-nu) / gamma(nu)) * tmp**nu * kv(nu, tmp)
    
    def kernel_periodic(self, X1, X2, length_scale=1.0, period=1.0):
        """Periodic kernel."""
        dists = np.abs(X1.reshape(-1, 1) - X2.reshape(1, -1))
        return np.exp(-2 * np.sin(np.pi * dists / period)**2 / length_scale**2)
    
    def get_kernel(self, X1, X2):
        """Get kernel matrix based on kernel_params."""
        ktype = self.kernel_params.get('type', 'rbf')
        
        if ktype == 'rbf':
            return self.kernel_rbf(X1, X2, 
                                  self.kernel_params.get('length_scale', 1.0),
                                  self.kernel_params.get('variance', 1.0))
        elif ktype == 'matern':
            return self.kernel_matern(X1, X2,
                                     self.kernel_params.get('length_scale', 1.0),
                                     self.kernel_params.get('nu', 1.5))
        elif ktype == 'periodic':
            return self.kernel_periodic(X1, X2,
                                       self.kernel_params.get('length_scale', 1.0),
                                       self.kernel_params.get('period', 1.0))
        else:
            return self.kernel_rbf(X1, X2)
    
    def fit(self, X, y, noise=0.1):
        """Fit GP model."""
        self.X_train = X
        self.y_train = y
        self.noise = noise
        return self
    
    def predict(self, X_test, return_cov=False):
        """Make predictions."""
        K = self.get_kernel(self.X_train, self.X_train)
        K += self.noise**2 * np.eye(len(self.X_train))
        
        K_s = self.get_kernel(self.X_train, X_test)
        K_ss = self.get_kernel(X_test, X_test)
        
        K_inv = np.linalg.inv(K)
        mu_s = K_s.T.dot(K_inv).dot(self.y_train)
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        
        if return_cov:
            return mu_s, cov_s
        return mu_s, np.sqrt(np.diag(cov_s))
    
    def log_marginal_likelihood(self):
        """Compute log marginal likelihood for hyperparameter optimization."""
        K = self.get_kernel(self.X_train, self.X_train)
        K += self.noise**2 * np.eye(len(self.X_train))
        
        L = np.linalg.cholesky(K + 1e-6 * np.eye(len(K)))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train))
        
        lml = -0.5 * self.y_train.dot(alpha)
        lml -= np.sum(np.log(np.diag(L)))
        lml -= len(self.X_train) / 2 * np.log(2 * np.pi)
        
        return lml


def optimize_hyperparameters(X_train, y_train):
    """Optimize GP hyperparameters via marginal likelihood."""
    from scipy.optimize import minimize
    
    def objective(params):
        """Negative log marginal likelihood."""
        gp = GPRegressionAdvanced(kernel_params={
            'type': 'rbf',
            'length_scale': params[0],
            'variance': params[1]
        })
        gp.fit(X_train, y_train, noise=params[2])
        return -gp.log_marginal_likelihood()
    
    # Initial guess
    init_params = [1.0, 1.0, 0.1]
    
    # Optimize
    result = minimize(objective, init_params, method='L-BFGS-B',
                     bounds=[(0.1, 10), (0.1, 10), (0.01, 1.0)])
    
    opt_params = result.x
    print(f"\nOptimized hyperparameters:")
    print(f"  Length scale: {opt_params[0]:.3f}")
    print(f"  Variance: {opt_params[1]:.3f}")
    print(f"  Noise: {opt_params[2]:.3f}")
    
    return opt_params


def compare_kernels():
    """Compare different kernel functions."""
    print("\n" + "=" * 80)
    print("COMPARING KERNEL FUNCTIONS")
    print("=" * 80)
    
    # Generate data
    np.random.seed(42)
    X = np.random.rand(60, 1) * 10
    y = np.sin(X).ravel() + np.random.randn(60) * 0.2
    
    X_test = np.linspace(0, 10, 200).reshape(-1, 1)
    y_true = np.sin(X_test).ravel()
    
    # Different kernels
    kernels = [
        {'type': 'rbf', 'length_scale': 1.0, 'variance': 1.0},
        {'type': 'matern', 'length_scale': 1.0, 'nu': 1.5},
        {'type': 'periodic', 'length_scale': 1.0, 'period': 2*np.pi},
    ]
    
    kernel_names = ['RBF', 'Matern', 'Periodic']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (kernel, name) in enumerate(zip(kernels, kernel_names)):
        gp = GPRegressionAdvanced(kernel_params=kernel)
        gp.fit(X, y, noise=0.2)
        
        y_pred, y_std = gp.predict(X_test)
        
        ax = axes[i]
        ax.scatter(X, y, c='r', s=30, zorder=10, label='Data')
        ax.plot(X_test, y_pred, 'b-', lw=2, label='Prediction')
        ax.fill_between(X_test.ravel(), 
                        y_pred - 1.96*y_std,
                        y_pred + 1.96*y_std,
                        alpha=0.3, label='95% CI')
        ax.plot(X_test, y_true, 'g--', lw=1, alpha=0.7, label='True')
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.set_title(f'{name} Kernel')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/gp_kernels.png', dpi=150, bbox_inches='tight')
    print("\nSaved: Kernel comparison")
    plt.close()


def gp_samples_visualization():
    """Visualize samples from GP prior and posterior."""
    print("\n" + "=" * 80)
    print("GP PRIOR AND POSTERIOR SAMPLES")
    print("=" * 80)
    
    # Test points
    X_test = np.linspace(0, 10, 100).reshape(-1, 1)
    
    # Prior samples
    gp_prior = GPRegressionAdvanced(kernel_params={'type': 'rbf', 'length_scale': 1.0})
    K_prior = gp_prior.kernel_rbf(X_test, X_test)
    prior_samples = np.random.multivariate_normal(np.zeros(len(X_test)), K_prior, size=5)
    
    # Generate training data
    np.random.seed(42)
    X_train = np.array([[1], [3], [5], [6], [8]])
    y_train = np.sin(X_train).ravel() + np.random.randn(5) * 0.1
    
    # Posterior samples
    gp_post = GPRegressionAdvanced(kernel_params={'type': 'rbf', 'length_scale': 1.0})
    gp_post.fit(X_train, y_train, noise=0.1)
    
    # Sample from posterior
    mu_post, cov_post = gp_post.predict(X_test, return_cov=True)
    posterior_samples = np.random.multivariate_normal(mu_post, cov_post, size=5)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Prior samples
    ax = axes[0]
    for sample in prior_samples:
        ax.plot(X_test, sample, alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('f(X)')
    ax.set_title('Samples from GP Prior')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-3, 3])
    
    # Posterior samples
    ax = axes[1]
    for sample in posterior_samples:
        ax.plot(X_test, sample, alpha=0.6)
    ax.scatter(X_train, y_train, c='r', s=100, zorder=10, edgecolors='k', label='Data')
    ax.plot(X_test, np.sin(X_test), 'g--', lw=2, alpha=0.7, label='True Function')
    ax.set_xlabel('X')
    ax.set_ylabel('f(X)')
    ax.set_title('Samples from GP Posterior')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/gp_prior_posterior.png', dpi=150, bbox_inches='tight')
    print("\nSaved: Prior/posterior samples")
    plt.close()
