"""
Bayesian Methods Solution

Comprehensive Bayesian analysis demonstrating inference, prediction, and uncertainty quantification.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def generate_data(n_samples=500, n_features=10, noise=0.5, seed=42):
    """Generate synthetic regression data."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    beta_true = np.random.randn(n_features) * 2
    beta_true[:5] = [3, -2, 1.5, -1, 0.8]  # Make first 5 non-zero
    y = X @ beta_true + np.random.randn(n_samples) * noise
    return X, y, beta_true


def bayesian_linear_regression(X, y, prior_std=5.0, noise_std=0.5):
    """Analytical Bayesian linear regression."""
    n_features = X.shape[1]
    
    # Prior precision
    prior_prec = np.eye(n_features) / (prior_std ** 2)
    
    # Likelihood precision  
    lik_prec = (X.T @ X) / (noise_std ** 2)
    
    # Posterior
    post_prec = prior_prec + lik_prec
    post_cov = np.linalg.inv(post_prec)
    post_mean = post_cov @ (X.T @ y / (noise_std ** 2))
    
    return post_mean, post_cov


def mcmc_sampler(X, y, n_samples=2000, burn_in=500, proposal_std=0.1):
    """MCMC sampling using Metropolis-Hastings."""
    n_features = X.shape[1]
    beta = np.zeros(n_features)
    samples = []
    accepted = 0
    
    def log_posterior(b):
        resid = y - X @ b
        ll = -0.5 * np.sum(resid ** 2) / 0.25  # noise_std^2 = 0.25
        lp = -0.5 * np.sum(b ** 2) / 25  # prior_std^2 = 25
        return ll + lp
    
    lp_current = log_posterior(beta)
    
    for i in range(burn_in + n_samples):
        beta_prop = beta + np.random.randn(n_features) * proposal_std
        lp_prop = log_posterior(beta_prop)
        
        if np.log(np.random.rand()) < (lp_prop - lp_current):
            beta = beta_prop
            lp_current = lp_prop
            accepted += 1
        
        if i >= burn_in:
            samples.append(beta.copy())
    
    acc_rate = accepted / (burn_in + n_samples)
    return np.array(samples), acc_rate


def predict_with_uncertainty(X_test, samples):
    """Make predictions with uncertainty."""
    preds = X_test @ samples.T
    mean = np.mean(preds, axis=1)
    std = np.std(preds, axis=1)
    return mean, std


def evaluate_predictions(y_true, y_pred, y_std):
    """Evaluate prediction quality."""
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Coverage
    lower = y_pred - 1.96 * y_std
    upper = y_pred + 1.96 * y_std
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    
    return {'MSE': mse, 'MAE': mae, 'Coverage': coverage}


def visualize_posterior(samples, post_mean, post_cov, beta_true):
    """Visualize posterior distributions."""
    n_params = min(6, samples.shape[1])
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i in range(n_params):
        ax = axes[i]
        
        # MCMC histogram
        ax.hist(samples[:, i], bins=50, alpha=0.5, density=True,
               label='MCMC', edgecolor='black')
        
        # Analytical posterior
        x = np.linspace(samples[:, i].min(), samples[:, i].max(), 100)
        y = stats.norm.pdf(x, post_mean[i], np.sqrt(post_cov[i, i]))
        ax.plot(x, y, 'r-', lw=2, label='Analytical')
        
        # True value
        ax.axvline(beta_true[i], color='green', linestyle='--',
                  lw=2, label='True')
        
        ax.set_xlabel(f'β{i}')
        ax.set_ylabel('Density')
        ax.set_title(f'Parameter {i}')
        if i == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/bayesian_posterior.png', dpi=150, bbox_inches='tight')
    print("Saved: Posterior visualization")
    plt.close()


def visualize_predictions(y_test, y_pred, y_std):
    """Visualize predictions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot
    ax = axes[0]
    ax.scatter(y_test, y_pred, alpha=0.6, s=30)
    lims = [min(y_test.min(), y_pred.min()),
            max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', lw=2)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Predictions vs True Values')
    ax.grid(True, alpha=0.3)
    
    # Prediction intervals
    ax = axes[1]
    n_show = min(50, len(y_test))
    idx = np.arange(n_show)
    ax.scatter(idx, y_test[:n_show], label='True', alpha=0.7, s=50)
    ax.errorbar(idx, y_pred[:n_show], yerr=1.96*y_std[:n_show],
               fmt='o', alpha=0.6, label='Predicted ± 95% CI')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Value')
    ax.set_title('Predictions with Uncertainty')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/bayesian_predictions.png', dpi=150, bbox_inches='tight')
    print("Saved: Predictions visualization")
    plt.close()


def diagnostic_plots(samples):
    """Create diagnostic plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Trace plots
    ax = axes[0, 0]
    for i in range(min(3, samples.shape[1])):
        ax.plot(samples[:, i], alpha=0.7, label=f'β{i}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value')
    ax.set_title('Trace Plots')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Autocorrelation
    ax = axes[0, 1]
    for i in range(min(3, samples.shape[1])):
        acf = [1.0] + [np.corrcoef(samples[:-lag, i], samples[lag:, i])[0, 1]
                      for lag in range(1, 50)]
        ax.plot(acf, alpha=0.7, label=f'β{i}')
    ax.axhline(0, color='black', linestyle='--', lw=1)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Autocorrelation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Density
    ax = axes[1, 0]
    for i in range(min(3, samples.shape[1])):
        ax.hist(samples[:, i], bins=50, alpha=0.5, density=True, label=f'β{i}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('Marginal Posteriors')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Running mean
    ax = axes[1, 1]
    for i in range(min(3, samples.shape[1])):
        running_mean = np.cumsum(samples[:, i]) / np.arange(1, len(samples)+1)
        ax.plot(running_mean, alpha=0.7, label=f'β{i}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Running Mean')
    ax.set_title('Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/bayesian_diagnostics.png', dpi=150, bbox_inches='tight')
    print("Saved: Diagnostics visualization")
    plt.close()


def main():
    """Main execution."""
    print("=" * 80)
    print("BAYESIAN METHODS ANALYSIS")
    print("=" * 80)
    
    # Generate data
    print("\nGenerating synthetic data...")
    X, y, beta_true = generate_data(n_samples=500, n_features=10)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    
    # Analytical Bayesian inference
    print("\n" + "=" * 80)
    print("BAYESIAN INFERENCE (ANALYTICAL)")
    print("=" * 80)
    post_mean, post_cov = bayesian_linear_regression(X_train, y_train)
    print(f"\nPosterior mean (first 5): {post_mean[:5]}")
    print(f"True values (first 5): {beta_true[:5]}")
    print(f"Posterior std (first 5): {np.sqrt(np.diag(post_cov)[:5])}")
    
    # MCMC sampling
    print("\n" + "=" * 80)
    print("MCMC SAMPLING")
    print("=" * 80)
    samples, acc_rate = mcmc_sampler(X_train, y_train)
    print(f"\nAcceptance rate: {acc_rate:.3f}")
    print(f"Samples generated: {len(samples)}")
    
    # Predictions
    print("\n" + "=" * 80)
    print("PREDICTIONS")
    print("=" * 80)
    y_pred, y_std = predict_with_uncertainty(X_test, samples)
    metrics = evaluate_predictions(y_test, y_pred, y_std)
    
    print(f"\nMSE: {metrics['MSE']:.4f}")
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"95% Interval Coverage: {metrics['Coverage']:.1%}")
    
    # Visualizations
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    visualize_posterior(samples, post_mean, post_cov, beta_true)
    visualize_predictions(y_test, y_pred, y_std)
    diagnostic_plots(samples)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. Bayesian inference provides full posterior distributions")
    print("2. MCMC enables sampling from complex posteriors")
    print("3. Uncertainty quantification is natural in Bayesian framework")
    print("4. Predictions include both parameter and observation uncertainty")
    print("5. Diagnostic plots help assess convergence and mixing")


if __name__ == "__main__":
    main()
