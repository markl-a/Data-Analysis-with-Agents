"""
Bayesian Logistic Regression

This solution implements Bayesian logistic regression using various inference methods:
- Laplace approximation
- Variational inference
- MCMC sampling (Metropolis-Hastings)

Techniques:
- Multiple inference methods
- Posterior predictive distributions
- Model uncertainty quantification
- Feature importance via posterior
- ROC curves with uncertainty
- Calibration analysis

Dataset: Synthetic binary classification data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class BayesianLogisticRegression:
    """
    Bayesian Logistic Regression with multiple inference methods.
    """

    def __init__(self, prior_mean=None, prior_std=1.0, method='laplace'):
        """
        Initialize Bayesian Logistic Regression.

        Parameters:
        -----------
        prior_mean : array-like
            Prior mean for coefficients (default: zero)
        prior_std : float
            Prior standard deviation for coefficients
        method : str
            Inference method: 'laplace', 'vi', or 'mcmc'
        """
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.method = method

        # Posterior parameters
        self.posterior_mean = None
        self.posterior_cov = None
        self.samples = None

    def _log_prior(self, beta):
        """Compute log prior probability."""
        if self.prior_mean is None:
            prior_mean = np.zeros_like(beta)
        else:
            prior_mean = self.prior_mean

        return -0.5 * np.sum(((beta - prior_mean) / self.prior_std) ** 2)

    def _log_likelihood(self, beta, X, y):
        """Compute log likelihood."""
        z = X @ beta
        return np.sum(y * z - np.log(1 + np.exp(z)))

    def _log_posterior(self, beta, X, y):
        """Compute log posterior (unnormalized)."""
        return self._log_prior(beta) + self._log_likelihood(beta, X, y)

    def _negative_log_posterior(self, beta, X, y):
        """Negative log posterior for optimization."""
        return -self._log_posterior(beta, X, y)

    def _gradient_log_posterior(self, beta, X, y):
        """Gradient of log posterior."""
        # Gradient of log likelihood
        z = X @ beta
        p = expit(z)  # sigmoid
        grad_ll = X.T @ (y - p)

        # Gradient of log prior
        if self.prior_mean is None:
            prior_mean = np.zeros_like(beta)
        else:
            prior_mean = self.prior_mean

        grad_prior = -(beta - prior_mean) / (self.prior_std ** 2)

        return grad_ll + grad_prior

    def _hessian_log_posterior(self, beta, X):
        """Hessian of log posterior (for Laplace approximation)."""
        z = X @ beta
        p = expit(z)

        # Hessian of log likelihood
        W = p * (1 - p)
        H_ll = -X.T @ (W[:, np.newaxis] * X)

        # Hessian of log prior
        H_prior = -np.eye(len(beta)) / (self.prior_std ** 2)

        return H_ll + H_prior

    def fit_laplace(self, X, y):
        """
        Fit using Laplace approximation.

        Finds MAP estimate and approximates posterior as Gaussian.
        """
        n_features = X.shape[1]

        # Initialize at zero
        beta_init = np.zeros(n_features)

        # Find MAP estimate
        result = minimize(
            self._negative_log_posterior,
            beta_init,
            args=(X, y),
            jac=lambda b, X, y: -self._gradient_log_posterior(b, X, y),
            method='L-BFGS-B'
        )

        self.posterior_mean = result.x

        # Compute Hessian at MAP
        H = self._hessian_log_posterior(self.posterior_mean, X)

        # Posterior covariance is negative inverse of Hessian
        try:
            self.posterior_cov = -np.linalg.inv(H)
        except np.linalg.LinAlgError:
            # If Hessian is singular, add small diagonal term
            self.posterior_cov = -np.linalg.inv(H + np.eye(len(H)) * 1e-5)

        return self

    def fit_mcmc(self, X, y, n_samples=5000, burn_in=1000, thin=5):
        """
        Fit using Metropolis-Hastings MCMC.

        Parameters:
        -----------
        n_samples : int
            Number of MCMC samples
        burn_in : int
            Number of burn-in samples to discard
        thin : int
            Thinning interval
        """
        n_features = X.shape[1]

        # Initialize
        beta_current = np.zeros(n_features)
        log_post_current = self._log_posterior(beta_current, X, y)

        # Proposal standard deviation (tune this for acceptance rate ~0.23)
        proposal_std = 0.1

        samples = []
        accepted = 0

        total_iterations = burn_in + n_samples * thin

        for i in range(total_iterations):
            # Propose new beta
            beta_proposed = beta_current + np.random.randn(n_features) * proposal_std

            # Compute log posterior
            log_post_proposed = self._log_posterior(beta_proposed, X, y)

            # Accept/reject
            log_alpha = log_post_proposed - log_post_current

            if np.log(np.random.rand()) < log_alpha:
                beta_current = beta_proposed
                log_post_current = log_post_proposed
                accepted += 1

            # Store sample after burn-in and thinning
            if i >= burn_in and (i - burn_in) % thin == 0:
                samples.append(beta_current.copy())

        self.samples = np.array(samples)
        self.posterior_mean = np.mean(self.samples, axis=0)
        self.posterior_cov = np.cov(self.samples.T)

        acceptance_rate = accepted / total_iterations
        print(f"MCMC Acceptance Rate: {acceptance_rate:.3f}")

        return self

    def fit_variational(self, X, y, n_iterations=1000, learning_rate=0.01):
        """
        Fit using variational inference (mean-field approximation).

        Assumes posterior factorizes: q(β) = ∏ N(βᵢ | μᵢ, σᵢ²)
        """
        n_features = X.shape[1]

        # Initialize variational parameters
        mu = np.zeros(n_features)
        log_sigma = np.zeros(n_features)  # Log to ensure positivity

        # Optimize ELBO using gradient ascent
        for iteration in range(n_iterations):
            # Sample from variational distribution
            epsilon = np.random.randn(10, n_features)  # Monte Carlo samples
            sigma = np.exp(log_sigma)
            beta_samples = mu + epsilon * sigma

            # Compute ELBO gradient estimates
            grad_mu = np.zeros(n_features)
            grad_log_sigma = np.zeros(n_features)

            for beta_sample in beta_samples:
                # Gradient w.r.t. mu
                grad_mu += self._gradient_log_posterior(beta_sample, X, y)

                # Gradient w.r.t. log_sigma (using reparameterization trick)
                grad_log_sigma += (
                    self._gradient_log_posterior(beta_sample, X, y) *
                    (beta_sample - mu) / sigma
                )

            grad_mu /= len(beta_samples)
            grad_log_sigma /= len(beta_samples)

            # Add entropy gradient
            grad_log_sigma += 1  # d/d(log σ) of log σ

            # Update parameters
            mu += learning_rate * grad_mu
            log_sigma += learning_rate * grad_log_sigma

        self.posterior_mean = mu
        self.posterior_cov = np.diag(np.exp(log_sigma) ** 2)

        return self

    def fit(self, X, y):
        """Fit model using specified method."""
        if self.prior_mean is None:
            self.prior_mean = np.zeros(X.shape[1])

        if self.method == 'laplace':
            return self.fit_laplace(X, y)
        elif self.method == 'mcmc':
            return self.fit_mcmc(X, y)
        elif self.method == 'vi':
            return self.fit_variational(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def predict_proba(self, X, return_std=False, use_samples=False):
        """
        Predict class probabilities.

        Parameters:
        -----------
        X : array-like
            Features
        return_std : bool
            Whether to return standard deviation of predictions
        use_samples : bool
            Whether to use posterior samples (if available)

        Returns:
        --------
        proba : array
            Predicted probabilities
        std : array (optional)
            Standard deviation of predictions
        """
        if use_samples and self.samples is not None:
            # Use posterior samples
            probas = expit(X @ self.samples.T)
            mean_proba = np.mean(probas, axis=1)

            if return_std:
                std_proba = np.std(probas, axis=1)
                return mean_proba, std_proba
            return mean_proba
        else:
            # Use Gaussian approximation
            mean_pred = X @ self.posterior_mean

            if return_std:
                # Variance via delta method or sampling
                var_pred = np.sum(X @ self.posterior_cov * X, axis=1)

                # Monte Carlo for sigmoid transformation
                n_mc = 1000
                samples = np.random.multivariate_normal(
                    self.posterior_mean,
                    self.posterior_cov,
                    size=n_mc
                )
                probas = expit(X @ samples.T)
                std_proba = np.std(probas, axis=1)

                return expit(mean_pred), std_proba

            return expit(mean_pred)

    def predict(self, X, threshold=0.5):
        """Predict class labels."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)


def generate_classification_data(n_samples=1000, n_features=10, n_informative=5, seed=42):
    """Generate synthetic binary classification data."""
    np.random.seed(seed)

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # True coefficients (sparse)
    true_beta = np.zeros(n_features)
    true_beta[:n_informative] = np.random.randn(n_informative) * 2

    # Generate labels
    z = X @ true_beta
    p = expit(z)
    y = (np.random.rand(n_samples) < p).astype(int)

    return X, y, true_beta


def compare_inference_methods(X_train, y_train, X_test, y_test):
    """Compare different inference methods."""
    print("=" * 80)
    print("COMPARING INFERENCE METHODS")
    print("=" * 80)

    methods = {
        'Laplace': 'laplace',
        'MCMC': 'mcmc',
        'Variational': 'vi'
    }

    results = []

    for name, method in methods.items():
        print(f"\nFitting {name}...")

        model = BayesianLogisticRegression(prior_std=2.0, method=method)
        model.fit(X_train, y_train)

        # Predictions
        y_pred_proba = model.predict_proba(X_test)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Metrics
        accuracy = np.mean(y_pred == y_test)
        log_loss = -np.mean(y_test * np.log(y_pred_proba + 1e-10) +
                            (1 - y_test) * np.log(1 - y_pred_proba + 1e-10))

        # AUC
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_test, y_pred_proba)

        results.append({
            'Method': name,
            'Accuracy': accuracy,
            'Log Loss': log_loss,
            'AUC': auc,
            'Model': model
        })

    df_results = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    print(df_results[['Method', 'Accuracy', 'Log Loss', 'AUC']].to_string(index=False))

    return results


def visualize_posterior_uncertainty(models_dict, X_test, y_test, feature_names=None):
    """Visualize posterior uncertainty for different methods."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Coefficient posteriors comparison
    ax = axes[0, 0]
    for method_name, model in models_dict.items():
        if model.samples is not None:
            # Use MCMC samples
            for i in range(min(3, len(model.posterior_mean))):
                ax.hist(model.samples[:, i], bins=50, alpha=0.3,
                       label=f'{method_name} β{i}', density=True)
        else:
            # Use Gaussian approximation
            for i in range(min(3, len(model.posterior_mean))):
                x = np.linspace(
                    model.posterior_mean[i] - 3 * np.sqrt(model.posterior_cov[i, i]),
                    model.posterior_mean[i] + 3 * np.sqrt(model.posterior_cov[i, i]),
                    100
                )
                y = stats.norm.pdf(x, model.posterior_mean[i],
                                  np.sqrt(model.posterior_cov[i, i]))
                ax.plot(x, y, label=f'{method_name} β{i}')

    ax.set_xlabel('Coefficient Value')
    ax.set_ylabel('Density')
    ax.set_title('Posterior Distributions of First 3 Coefficients')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Prediction uncertainty
    ax = axes[0, 1]
    n_show = min(100, len(X_test))

    for method_name, model in models_dict.items():
        proba, std = model.predict_proba(X_test[:n_show], return_std=True,
                                         use_samples=(model.samples is not None))

        # Sort by probability for visualization
        sorted_idx = np.argsort(proba)
        ax.plot(proba[sorted_idx], label=f'{method_name} Mean', alpha=0.7)
        ax.fill_between(range(n_show),
                        (proba - std)[sorted_idx],
                        (proba + std)[sorted_idx],
                        alpha=0.2)

    ax.set_xlabel('Sample Index (sorted by probability)')
    ax.set_ylabel('Predicted Probability')
    ax.set_title('Prediction Uncertainty Across Methods')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. ROC curves with uncertainty (using first model)
    ax = axes[1, 0]
    from sklearn.metrics import roc_curve

    for method_name, model in list(models_dict.items())[:1]:  # Just one for clarity
        if model.samples is not None:
            # Multiple ROC curves from samples
            for i in range(min(100, len(model.samples))):
                probas = expit(X_test @ model.samples[i])
                fpr, tpr, _ = roc_curve(y_test, probas)
                ax.plot(fpr, tpr, 'b-', alpha=0.02)

        # Main ROC curve
        proba = model.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, 'r-', linewidth=2, label=f'{method_name} Mean')

    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve with Posterior Uncertainty')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Calibration plot
    ax = axes[1, 1]

    for method_name, model in models_dict.items():
        proba = model.predict_proba(X_test)

        # Bin predictions
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        bin_indices = np.digitize(proba, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        bin_true_probs = np.array([
            y_test[bin_indices == i].mean() if np.sum(bin_indices == i) > 0 else np.nan
            for i in range(n_bins)
        ])

        ax.plot(bin_centers, bin_true_probs, 'o-', label=method_name, alpha=0.7)

    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('True Probability')
    ax.set_title('Calibration Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/bayesian_logistic_regression_uncertainty.png', dpi=150, bbox_inches='tight')
    print("\nSaved: Posterior uncertainty visualization")
    plt.close()


def analyze_feature_importance(model, feature_names=None):
    """Analyze feature importance via posterior."""
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)

    n_features = len(model.posterior_mean)

    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(n_features)]

    # Compute posterior probability that coefficient > 0
    if model.samples is not None:
        prob_positive = np.mean(model.samples > 0, axis=0)
    else:
        prob_positive = 1 - stats.norm.cdf(
            0,
            model.posterior_mean,
            np.sqrt(np.diag(model.posterior_cov))
        )

    # Create results dataframe
    results = pd.DataFrame({
        'Feature': feature_names,
        'Posterior Mean': model.posterior_mean,
        'Posterior Std': np.sqrt(np.diag(model.posterior_cov)),
        'P(β > 0)': prob_positive,
        '|Mean|': np.abs(model.posterior_mean)
    })

    results = results.sort_values('|Mean|', ascending=False)
    print("\n", results.to_string(index=False))


def main():
    """Main execution function."""
    print("=" * 80)
    print("BAYESIAN LOGISTIC REGRESSION")
    print("=" * 80)

    # Generate data
    print("\nGenerating synthetic classification data...")
    X, y, true_beta = generate_classification_data(
        n_samples=1000,
        n_features=10,
        n_informative=5
    )

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train)} (Class 1: {np.sum(y_train)})")
    print(f"Test samples: {len(X_test)} (Class 1: {np.sum(y_test)})")
    print(f"Number of features: {X_train.shape[1]}")

    # Compare inference methods
    results = compare_inference_methods(X_train, y_train, X_test, y_test)

    # Create dict of models
    models_dict = {r['Method']: r['Model'] for r in results}

    # Visualize uncertainty
    visualize_posterior_uncertainty(models_dict, X_test, y_test)

    # Feature importance (using MCMC model)
    analyze_feature_importance(models_dict['MCMC'])

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. Multiple inference methods available for Bayesian logistic regression")
    print("2. Laplace is fast but assumes Gaussian posterior")
    print("3. MCMC is accurate but slower")
    print("4. Variational inference balances speed and accuracy")
    print("5. Uncertainty quantification for predictions and parameters")


if __name__ == "__main__":
    main()
