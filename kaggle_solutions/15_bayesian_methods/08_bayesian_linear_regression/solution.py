"""
Bayesian Linear Regression with Conjugate Priors

This solution demonstrates Bayesian linear regression using conjugate priors,
comparing it with frequentist approaches and showing posterior inference.

Techniques:
- Normal-Inverse-Gamma conjugate prior
- Posterior computation for parameters
- Credible intervals
- Predictive distributions
- Prior sensitivity analysis
- Model comparison with different priors

Dataset: Synthetic regression data with noise
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import gammaln
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class BayesianLinearRegression:
    """
    Bayesian Linear Regression with Normal-Inverse-Gamma conjugate prior.

    Prior:
        σ² ~ Inverse-Gamma(α₀, β₀)
        β | σ² ~ Normal(μ₀, σ²V₀)

    This is conjugate, so posterior has same form.
    """

    def __init__(self, prior_mean=None, prior_cov=None, alpha_0=1.0, beta_0=1.0):
        """
        Initialize with prior parameters.

        Parameters:
        -----------
        prior_mean : array-like
            Prior mean for coefficients
        prior_cov : array-like
            Prior covariance matrix for coefficients
        alpha_0 : float
            Shape parameter for Inverse-Gamma prior on variance
        beta_0 : float
            Scale parameter for Inverse-Gamma prior on variance
        """
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

        # Posterior parameters (to be computed)
        self.posterior_mean = None
        self.posterior_cov = None
        self.alpha_n = None
        self.beta_n = None

        # Data
        self.X = None
        self.y = None
        self.n_features = None

    def fit(self, X, y):
        """
        Compute posterior parameters given data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        self.X = np.array(X)
        self.y = np.array(y).ravel()
        n_samples, self.n_features = self.X.shape

        # Set default priors if not specified
        if self.prior_mean is None:
            self.prior_mean = np.zeros(self.n_features)
        if self.prior_cov is None:
            self.prior_cov = np.eye(self.n_features) * 100  # Weakly informative

        # Convert to precision matrix for computation
        prior_precision = np.linalg.inv(self.prior_cov)

        # Compute posterior parameters
        XtX = self.X.T @ self.X
        Xty = self.X.T @ self.y

        # Posterior covariance and mean for β
        posterior_precision = prior_precision + XtX
        self.posterior_cov = np.linalg.inv(posterior_precision)

        self.posterior_mean = self.posterior_cov @ (
            prior_precision @ self.prior_mean + Xty
        )

        # Posterior parameters for σ²
        self.alpha_n = self.alpha_0 + n_samples / 2

        residual = self.y - self.X @ self.posterior_mean
        prior_diff = self.posterior_mean - self.prior_mean

        self.beta_n = self.beta_0 + 0.5 * (
            self.y.T @ self.y +
            self.prior_mean.T @ prior_precision @ self.prior_mean -
            self.posterior_mean.T @ posterior_precision @ self.posterior_mean
        )

        return self

    def predict(self, X, return_std=False, return_interval=False, credible_level=0.95):
        """
        Make predictions with uncertainty quantification.

        Parameters:
        -----------
        X : array-like
            Test data
        return_std : bool
            Whether to return standard deviation
        return_interval : bool
            Whether to return credible intervals
        credible_level : float
            Credible interval level

        Returns:
        --------
        predictions : array
            Point predictions (posterior mean)
        std : array (optional)
            Predictive standard deviations
        intervals : tuple (optional)
            Lower and upper bounds of credible intervals
        """
        X = np.array(X)

        # Point prediction (posterior mean)
        y_pred = X @ self.posterior_mean

        if not return_std and not return_interval:
            return y_pred

        # Predictive variance
        # E[σ²] under posterior
        posterior_var_estimate = self.beta_n / (self.alpha_n - 1)

        # Predictive variance includes uncertainty in β and σ²
        predictive_var = posterior_var_estimate * (
            1 + np.sum(X @ self.posterior_cov * X, axis=1)
        )
        predictive_std = np.sqrt(predictive_var)

        results = [y_pred]

        if return_std:
            results.append(predictive_std)

        if return_interval:
            # Use t-distribution for credible intervals
            df = 2 * self.alpha_n
            t_value = stats.t.ppf((1 + credible_level) / 2, df)

            lower = y_pred - t_value * predictive_std
            upper = y_pred + t_value * predictive_std
            results.append((lower, upper))

        return tuple(results) if len(results) > 1 else results[0]

    def sample_posterior(self, n_samples=1000):
        """
        Sample from the posterior distribution.

        Returns:
        --------
        beta_samples : array, shape (n_samples, n_features)
            Samples of regression coefficients
        sigma2_samples : array, shape (n_samples,)
            Samples of noise variance
        """
        # Sample σ² from Inverse-Gamma
        sigma2_samples = stats.invgamma.rvs(
            self.alpha_n, scale=self.beta_n, size=n_samples
        )

        # Sample β | σ² from Normal
        beta_samples = np.zeros((n_samples, self.n_features))
        for i in range(n_samples):
            beta_samples[i] = np.random.multivariate_normal(
                self.posterior_mean,
                sigma2_samples[i] * self.posterior_cov
            )

        return beta_samples, sigma2_samples

    def compute_marginal_likelihood(self):
        """
        Compute the marginal likelihood (evidence) p(y|X).
        Useful for model comparison.
        """
        n = len(self.y)

        # Log marginal likelihood for Normal-Inverse-Gamma conjugate prior
        prior_precision = np.linalg.inv(self.prior_cov)
        posterior_precision = np.linalg.inv(self.posterior_cov)

        log_ml = (
            gammaln(self.alpha_n) - gammaln(self.alpha_0)
            - 0.5 * n * np.log(2 * np.pi)
            + 0.5 * np.linalg.slogdet(prior_precision)[1]
            - 0.5 * np.linalg.slogdet(posterior_precision)[1]
            + self.alpha_0 * np.log(self.beta_0)
            - self.alpha_n * np.log(self.beta_n)
        )

        return log_ml


def generate_synthetic_data(n_samples=200, n_features=5, noise_std=2.0, seed=42):
    """Generate synthetic regression data."""
    np.random.seed(seed)

    # True coefficients
    true_beta = np.random.randn(n_features) * 3
    true_beta[0] = 5.0  # Intercept

    # Generate features
    X = np.ones((n_samples, n_features))
    X[:, 1:] = np.random.randn(n_samples, n_features - 1)

    # Generate targets with noise
    y = X @ true_beta + np.random.randn(n_samples) * noise_std

    return X, y, true_beta


def compare_with_frequentist(X_train, y_train, X_test, y_test, true_beta):
    """Compare Bayesian and frequentist linear regression."""
    print("=" * 80)
    print("BAYESIAN VS FREQUENTIST LINEAR REGRESSION")
    print("=" * 80)

    # Frequentist (OLS)
    from sklearn.linear_model import LinearRegression
    ols = LinearRegression()
    ols.fit(X_train, y_train)

    # Bayesian with weakly informative prior
    bayes_weak = BayesianLinearRegression(
        alpha_0=1.0,
        beta_0=1.0
    )
    bayes_weak.fit(X_train, y_train)

    # Bayesian with informative prior (assume we know something)
    prior_mean = np.zeros(X_train.shape[1])
    prior_mean[0] = 5.0  # We think intercept is around 5
    bayes_info = BayesianLinearRegression(
        prior_mean=prior_mean,
        prior_cov=np.eye(X_train.shape[1]) * 10,  # More confident
        alpha_0=2.0,
        beta_0=4.0
    )
    bayes_info.fit(X_train, y_train)

    # Compare coefficients
    print("\nCoefficient Estimates:")
    print("-" * 80)
    print(f"{'Feature':<15} {'True':<12} {'OLS':<12} {'Bayes (Weak)':<15} {'Bayes (Info)':<15}")
    print("-" * 80)

    for i in range(len(true_beta)):
        print(f"β{i:<14} {true_beta[i]:<12.4f} {ols.coef_[i]:<12.4f} "
              f"{bayes_weak.posterior_mean[i]:<15.4f} {bayes_info.posterior_mean[i]:<15.4f}")

    # Predictions
    y_pred_ols = ols.predict(X_test)
    y_pred_bayes_weak = bayes_weak.predict(X_test)
    y_pred_bayes_info = bayes_info.predict(X_test)

    # Compute MSE
    mse_ols = np.mean((y_test - y_pred_ols) ** 2)
    mse_bayes_weak = np.mean((y_test - y_pred_bayes_weak) ** 2)
    mse_bayes_info = np.mean((y_test - y_pred_bayes_info) ** 2)

    print("\n" + "=" * 80)
    print("PREDICTION PERFORMANCE")
    print("=" * 80)
    print(f"OLS MSE:                  {mse_ols:.4f}")
    print(f"Bayesian (Weak) MSE:      {mse_bayes_weak:.4f}")
    print(f"Bayesian (Informative) MSE: {mse_bayes_info:.4f}")

    return bayes_weak, bayes_info


def visualize_posterior_samples(model, X_test, y_test, feature_names=None):
    """Visualize posterior samples and uncertainty."""

    # Sample from posterior
    beta_samples, sigma2_samples = model.sample_posterior(n_samples=1000)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Posterior distributions of coefficients
    ax = axes[0, 0]
    for i in range(min(5, beta_samples.shape[1])):
        label = feature_names[i] if feature_names else f'β{i}'
        ax.hist(beta_samples[:, i], bins=50, alpha=0.5, label=label, density=True)
    ax.set_xlabel('Coefficient Value')
    ax.set_ylabel('Density')
    ax.set_title('Posterior Distributions of Coefficients')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Posterior distribution of noise variance
    ax = axes[0, 1]
    ax.hist(sigma2_samples, bins=50, alpha=0.7, color='green', density=True)
    ax.axvline(np.median(sigma2_samples), color='red', linestyle='--',
               label=f'Median: {np.median(sigma2_samples):.2f}')
    ax.set_xlabel('Noise Variance (σ²)')
    ax.set_ylabel('Density')
    ax.set_title('Posterior Distribution of Noise Variance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Predictive uncertainty for a subset of test points
    ax = axes[1, 0]
    n_show = min(50, len(X_test))
    indices = np.arange(n_show)

    y_pred, pred_std, (lower, upper) = model.predict(
        X_test[:n_show],
        return_std=True,
        return_interval=True,
        credible_level=0.95
    )

    ax.scatter(indices, y_test[:n_show], alpha=0.6, label='True', s=50)
    ax.plot(indices, y_pred, 'r-', label='Prediction', linewidth=2)
    ax.fill_between(indices, lower, upper, alpha=0.3, label='95% Credible Interval')
    ax.set_xlabel('Test Sample Index')
    ax.set_ylabel('Target Value')
    ax.set_title('Predictions with Uncertainty')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Correlation between coefficients
    ax = axes[1, 1]
    # Show correlation for first few coefficients
    n_show_corr = min(5, beta_samples.shape[1])
    corr_matrix = np.corrcoef(beta_samples[:, :n_show_corr].T)

    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(n_show_corr))
    ax.set_yticks(range(n_show_corr))
    if feature_names:
        labels = [feature_names[i] for i in range(n_show_corr)]
    else:
        labels = [f'β{i}' for i in range(n_show_corr)]
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title('Posterior Correlation between Coefficients')
    plt.colorbar(im, ax=ax)

    # Add correlation values
    for i in range(n_show_corr):
        for j in range(n_show_corr):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)

    plt.tight_layout()
    plt.savefig('/tmp/bayesian_linear_regression_posterior.png', dpi=150, bbox_inches='tight')
    print("\nSaved: Posterior visualization")
    plt.close()


def prior_sensitivity_analysis(X_train, y_train, X_test, y_test):
    """Analyze sensitivity to prior choice."""
    print("\n" + "=" * 80)
    print("PRIOR SENSITIVITY ANALYSIS")
    print("=" * 80)

    # Different prior specifications
    priors = {
        'Uninformative': {'prior_cov': np.eye(X_train.shape[1]) * 1000, 'alpha_0': 0.1, 'beta_0': 0.1},
        'Weakly Informative': {'prior_cov': np.eye(X_train.shape[1]) * 100, 'alpha_0': 1.0, 'beta_0': 1.0},
        'Moderately Informative': {'prior_cov': np.eye(X_train.shape[1]) * 10, 'alpha_0': 2.0, 'beta_0': 2.0},
        'Strongly Informative': {'prior_cov': np.eye(X_train.shape[1]) * 1, 'alpha_0': 5.0, 'beta_0': 5.0},
    }

    results = []

    for name, prior_params in priors.items():
        model = BayesianLinearRegression(**prior_params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = np.mean((y_test - y_pred) ** 2)

        log_ml = model.compute_marginal_likelihood()

        results.append({
            'Prior': name,
            'MSE': mse,
            'Log Marginal Likelihood': log_ml,
            'Posterior Mean β0': model.posterior_mean[0],
            'Posterior Std β0': np.sqrt(model.posterior_cov[0, 0] * model.beta_n / (model.alpha_n - 1))
        })

    df_results = pd.DataFrame(results)
    print("\n", df_results.to_string(index=False))

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = axes[0]
    x_pos = np.arange(len(results))
    mses = [r['MSE'] for r in results]
    ax.bar(x_pos, mses, alpha=0.7, color='steelblue')
    ax.set_xlabel('Prior Type')
    ax.set_ylabel('Test MSE')
    ax.set_title('Prediction Performance vs Prior Choice')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([r['Prior'] for r in results], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    log_mls = [r['Log Marginal Likelihood'] for r in results]
    ax.bar(x_pos, log_mls, alpha=0.7, color='coral')
    ax.set_xlabel('Prior Type')
    ax.set_ylabel('Log Marginal Likelihood')
    ax.set_title('Model Evidence vs Prior Choice')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([r['Prior'] for r in results], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/tmp/bayesian_linear_regression_prior_sensitivity.png', dpi=150, bbox_inches='tight')
    print("\nSaved: Prior sensitivity analysis")
    plt.close()


def analyze_credible_intervals(model, X_test, y_test):
    """Analyze coverage of credible intervals."""
    print("\n" + "=" * 80)
    print("CREDIBLE INTERVAL ANALYSIS")
    print("=" * 80)

    levels = [0.50, 0.68, 0.90, 0.95, 0.99]

    coverage_results = []

    for level in levels:
        y_pred, (lower, upper) = model.predict(
            X_test,
            return_interval=True,
            credible_level=level
        )

        # Check coverage
        in_interval = (y_test >= lower) & (y_test <= upper)
        empirical_coverage = np.mean(in_interval)

        # Average interval width
        avg_width = np.mean(upper - lower)

        coverage_results.append({
            'Nominal Level': f'{level:.0%}',
            'Empirical Coverage': f'{empirical_coverage:.2%}',
            'Avg Interval Width': f'{avg_width:.4f}'
        })

    df_coverage = pd.DataFrame(coverage_results)
    print("\n", df_coverage.to_string(index=False))


def main():
    """Main execution function."""
    print("=" * 80)
    print("BAYESIAN LINEAR REGRESSION WITH CONJUGATE PRIORS")
    print("=" * 80)

    # Generate data
    print("\nGenerating synthetic data...")
    X, y, true_beta = generate_synthetic_data(n_samples=200, n_features=5, noise_std=2.0)

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"True noise std: 2.0")

    # Compare with frequentist
    bayes_weak, bayes_info = compare_with_frequentist(
        X_train, y_train, X_test, y_test, true_beta
    )

    # Visualize posterior
    feature_names = ['Intercept'] + [f'X{i}' for i in range(1, X_train.shape[1])]
    visualize_posterior_samples(bayes_weak, X_test, y_test, feature_names)

    # Prior sensitivity
    prior_sensitivity_analysis(X_train, y_train, X_test, y_test)

    # Credible interval analysis
    analyze_credible_intervals(bayes_weak, X_test, y_test)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. Bayesian regression provides full posterior distributions, not just point estimates")
    print("2. Conjugate priors allow closed-form posterior computation")
    print("3. Uncertainty quantification through credible intervals")
    print("4. Prior choice affects results, especially with limited data")
    print("5. Marginal likelihood enables principled model comparison")


if __name__ == "__main__":
    main()
