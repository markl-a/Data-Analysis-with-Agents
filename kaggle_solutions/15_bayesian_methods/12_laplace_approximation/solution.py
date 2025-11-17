"""
Laplace Approximation for Bayesian Inference

This solution demonstrates the Laplace approximation for approximating
posterior distributions as Gaussian around the mode (MAP estimate).

Techniques:
- Finding MAP estimates via optimization
- Computing Hessian for covariance approximation
- Comparison with exact posteriors
- Model evidence approximation
- Uncertainty quantification
- Multi-modal posterior handling
- Numerical stability considerations

Dataset: Various Bayesian models with known posteriors
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


class LaplaceApproximation:
    """
    Laplace Approximation for Bayesian posterior inference.

    Approximates posterior p(θ|D) as Gaussian N(θ_MAP, Σ) where:
    - θ_MAP is the maximum a posteriori estimate
    - Σ = [-H(θ_MAP)]^(-1) where H is the Hessian of log p(θ|D)
    """

    def __init__(self, log_posterior_fn, gradient_fn=None, hessian_fn=None):
        """
        Initialize Laplace approximation.

        Parameters:
        -----------
        log_posterior_fn : callable
            Function that computes log posterior p(θ|D)
        gradient_fn : callable, optional
            Function that computes gradient of log posterior
        hessian_fn : callable, optional
            Function that computes Hessian of log posterior
        """
        self.log_posterior_fn = log_posterior_fn
        self.gradient_fn = gradient_fn
        self.hessian_fn = hessian_fn

        # Approximation results
        self.map_estimate = None
        self.posterior_cov = None
        self.log_marginal_likelihood = None

    def fit(self, initial_params, compute_hessian=True):
        """
        Fit Laplace approximation by finding MAP and computing Hessian.

        Parameters:
        -----------
        initial_params : array
            Initial parameter values for optimization
        compute_hessian : bool
            Whether to compute Hessian (needed for uncertainty)

        Returns:
        --------
        self
        """
        # Find MAP estimate
        if self.gradient_fn is not None:
            result = minimize(
                lambda x: -self.log_posterior_fn(x),
                initial_params,
                jac=lambda x: -self.gradient_fn(x),
                method='L-BFGS-B'
            )
        else:
            result = minimize(
                lambda x: -self.log_posterior_fn(x),
                initial_params,
                method='Nelder-Mead'
            )

        self.map_estimate = result.x

        # Compute Hessian
        if compute_hessian:
            if self.hessian_fn is not None:
                hessian = self.hessian_fn(self.map_estimate)
            else:
                hessian = self._numerical_hessian(self.map_estimate)

            # Posterior covariance is negative inverse of Hessian
            try:
                self.posterior_cov = -np.linalg.inv(hessian)
            except np.linalg.LinAlgError:
                # If Hessian is singular, add small regularization
                hessian_reg = hessian - np.eye(len(hessian)) * 1e-6
                self.posterior_cov = -np.linalg.inv(hessian_reg)

            # Compute log marginal likelihood (evidence)
            self._compute_log_marginal_likelihood(hessian)

        return self

    def _numerical_hessian(self, params, eps=1e-5):
        """Compute Hessian numerically using finite differences."""
        n = len(params)
        hessian = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                # Compute second partial derivative
                params_pp = params.copy()
                params_pm = params.copy()
                params_mp = params.copy()
                params_mm = params.copy()

                params_pp[i] += eps
                params_pp[j] += eps

                params_pm[i] += eps
                params_pm[j] -= eps

                params_mp[i] -= eps
                params_mp[j] += eps

                params_mm[i] -= eps
                params_mm[j] -= eps

                hessian[i, j] = (
                    self.log_posterior_fn(params_pp) -
                    self.log_posterior_fn(params_pm) -
                    self.log_posterior_fn(params_mp) +
                    self.log_posterior_fn(params_mm)
                ) / (4 * eps * eps)

                hessian[j, i] = hessian[i, j]

        return hessian

    def _compute_log_marginal_likelihood(self, hessian):
        """
        Compute log marginal likelihood using Laplace approximation.

        log p(D) ≈ log p(D|θ_MAP) + log p(θ_MAP) + 0.5 * log det(2π Σ)
        """
        n_params = len(self.map_estimate)

        # Determinant of covariance
        sign, logdet_cov = np.linalg.slogdet(self.posterior_cov)

        self.log_marginal_likelihood = (
            self.log_posterior_fn(self.map_estimate) +
            0.5 * n_params * np.log(2 * np.pi) +
            0.5 * logdet_cov
        )

    def sample(self, n_samples=1000):
        """
        Generate samples from Laplace approximation.

        Returns:
        --------
        samples : array, shape (n_samples, n_params)
        """
        if self.posterior_cov is None:
            raise ValueError("Must call fit() with compute_hessian=True first")

        samples = np.random.multivariate_normal(
            self.map_estimate,
            self.posterior_cov,
            size=n_samples
        )

        return samples

    def predict(self, X, predictor_fn, return_std=False, n_samples=1000):
        """
        Make predictions using posterior samples.

        Parameters:
        -----------
        X : array
            Input data
        predictor_fn : callable
            Function f(params, X) that makes predictions
        return_std : bool
            Whether to return prediction uncertainty
        n_samples : int
            Number of posterior samples for prediction

        Returns:
        --------
        predictions : array
            Mean predictions
        std : array, optional
            Standard deviation of predictions
        """
        samples = self.sample(n_samples)

        # Compute predictions for each sample
        predictions_list = []
        for params in samples:
            pred = predictor_fn(params, X)
            predictions_list.append(pred)

        predictions_array = np.array(predictions_list)

        mean_pred = np.mean(predictions_array, axis=0)

        if return_std:
            std_pred = np.std(predictions_array, axis=0)
            return mean_pred, std_pred

        return mean_pred


def bayesian_logistic_regression_laplace():
    """Example: Laplace approximation for Bayesian logistic regression."""
    print("=" * 80)
    print("EXAMPLE 1: BAYESIAN LOGISTIC REGRESSION")
    print("=" * 80)

    # Generate data
    np.random.seed(42)
    n_samples = 300
    n_features = 2

    X = np.random.randn(n_samples, n_features)
    true_beta = np.array([1.5, -2.0])
    z = X @ true_beta
    p = expit(z)
    y = (np.random.rand(n_samples) < p).astype(int)

    print(f"\nData: {n_samples} samples, {n_features} features")
    print(f"True coefficients: {true_beta}")
    print(f"Class balance: {np.mean(y):.1%} positive")

    # Define log posterior
    prior_std = 3.0

    def log_posterior(beta):
        z = X @ beta
        log_lik = np.sum(y * z - np.log(1 + np.exp(np.clip(z, -500, 500))))
        log_prior = -0.5 * np.sum(beta ** 2) / (prior_std ** 2)
        return log_lik + log_prior

    def gradient(beta):
        z = X @ beta
        p = expit(z)
        grad_log_lik = X.T @ (y - p)
        grad_log_prior = -beta / (prior_std ** 2)
        return grad_log_lik + grad_log_prior

    def hessian(beta):
        z = X @ beta
        p = expit(z)
        W = p * (1 - p)
        H_log_lik = -X.T @ (W[:, np.newaxis] * X)
        H_log_prior = -np.eye(n_features) / (prior_std ** 2)
        return H_log_lik + H_log_prior

    # Fit Laplace approximation
    print("\nFitting Laplace approximation...")
    laplace = LaplaceApproximation(log_posterior, gradient, hessian)
    laplace.fit(np.zeros(n_features))

    print(f"\nMAP estimate: {laplace.map_estimate}")
    print(f"Posterior std: {np.sqrt(np.diag(laplace.posterior_cov))}")
    print(f"Log marginal likelihood: {laplace.log_marginal_likelihood:.2f}")

    # Compare with true values
    print("\n" + "=" * 80)
    print("COMPARISON WITH TRUE VALUES")
    print("=" * 80)

    results = pd.DataFrame({
        'Parameter': [f'β{i}' for i in range(n_features)],
        'True': true_beta,
        'MAP': laplace.map_estimate,
        'Posterior Std': np.sqrt(np.diag(laplace.posterior_cov))
    })

    print("\n", results.to_string(index=False))

    return laplace, X, y, true_beta


def bayesian_linear_regression_laplace():
    """Example: Laplace approximation for Bayesian linear regression."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: BAYESIAN LINEAR REGRESSION")
    print("=" * 80)

    # Generate data
    np.random.seed(42)
    n_samples = 150
    n_features = 4

    X = np.random.randn(n_samples, n_features)
    true_beta = np.array([3.0, -1.5, 0.8, -0.3])
    noise_std = 1.0
    y = X @ true_beta + np.random.randn(n_samples) * noise_std

    print(f"\nData: {n_samples} samples, {n_features} features")
    print(f"True coefficients: {true_beta}")
    print(f"Noise std: {noise_std}")

    # Define log posterior
    prior_std = 5.0

    def log_posterior(beta):
        residuals = y - X @ beta
        log_lik = -0.5 * np.sum(residuals ** 2) / (noise_std ** 2)
        log_prior = -0.5 * np.sum(beta ** 2) / (prior_std ** 2)
        return log_lik + log_prior

    def gradient(beta):
        residuals = y - X @ beta
        grad_log_lik = X.T @ residuals / (noise_std ** 2)
        grad_log_prior = -beta / (prior_std ** 2)
        return grad_log_lik + grad_log_prior

    def hessian(beta):
        H_log_lik = -X.T @ X / (noise_std ** 2)
        H_log_prior = -np.eye(n_features) / (prior_std ** 2)
        return H_log_lik + H_log_prior

    # Fit Laplace approximation
    print("\nFitting Laplace approximation...")
    laplace = LaplaceApproximation(log_posterior, gradient, hessian)
    laplace.fit(np.zeros(n_features))

    # For linear regression, we can compute exact posterior
    # (Normal-Normal conjugacy)
    prior_precision = np.eye(n_features) / (prior_std ** 2)
    posterior_precision = prior_precision + X.T @ X / (noise_std ** 2)
    posterior_cov_exact = np.linalg.inv(posterior_precision)
    posterior_mean_exact = posterior_cov_exact @ (X.T @ y / (noise_std ** 2))

    print("\n" + "=" * 80)
    print("LAPLACE VS EXACT POSTERIOR")
    print("=" * 80)

    results = pd.DataFrame({
        'Parameter': [f'β{i}' for i in range(n_features)],
        'True': true_beta,
        'Laplace Mean': laplace.map_estimate,
        'Exact Mean': posterior_mean_exact,
        'Laplace Std': np.sqrt(np.diag(laplace.posterior_cov)),
        'Exact Std': np.sqrt(np.diag(posterior_cov_exact))
    })

    print("\n", results.to_string(index=False))

    print("\nNote: For linear regression with Gaussian prior and likelihood,")
    print("Laplace approximation is exact (posterior is Gaussian)!")

    return laplace, posterior_mean_exact, posterior_cov_exact


def visualize_approximation_quality(laplace_logistic, X, y, true_beta):
    """Visualize quality of Laplace approximation."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Posterior distributions (marginals)
    ax = axes[0, 0]

    samples = laplace_logistic.sample(n_samples=5000)

    for i in range(min(2, len(true_beta))):
        # Laplace approximation
        x_range = np.linspace(
            laplace_logistic.map_estimate[i] - 4 * np.sqrt(laplace_logistic.posterior_cov[i, i]),
            laplace_logistic.map_estimate[i] + 4 * np.sqrt(laplace_logistic.posterior_cov[i, i]),
            100
        )

        laplace_density = stats.norm.pdf(
            x_range,
            laplace_logistic.map_estimate[i],
            np.sqrt(laplace_logistic.posterior_cov[i, i])
        )

        ax.plot(x_range, laplace_density, label=f'Laplace β{i}', linewidth=2)

        # Samples
        ax.hist(samples[:, i], bins=50, alpha=0.3, density=True, label=f'Samples β{i}')

        # True value
        ax.axvline(true_beta[i], color=f'C{i}', linestyle='--',
                  linewidth=2, alpha=0.7)

    ax.set_xlabel('Parameter Value')
    ax.set_ylabel('Density')
    ax.set_title('Marginal Posterior Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Joint posterior (2D)
    ax = axes[0, 1]

    # Contour plot of Laplace approximation
    from matplotlib.patches import Ellipse

    # Draw confidence ellipse
    cov = laplace_logistic.posterior_cov[:2, :2]
    mean = laplace_logistic.map_estimate[:2]

    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Angle of rotation
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    # Width and height (2 standard deviations)
    for n_std in [1, 2, 3]:
        width, height = 2 * n_std * np.sqrt(eigenvalues)

        ellipse = Ellipse(
            mean,
            width,
            height,
            angle=angle,
            facecolor='none',
            edgecolor='blue',
            linewidth=2,
            alpha=0.7,
            label=f'{n_std}σ' if n_std == 1 else None
        )
        ax.add_patch(ellipse)

    # Scatter samples
    ax.scatter(samples[:, 0], samples[:, 1], alpha=0.1, s=5, color='gray')

    # True value
    ax.plot(true_beta[0], true_beta[1], 'r*', markersize=20,
           markeredgecolor='black', markeredgewidth=2, label='True')

    # MAP
    ax.plot(mean[0], mean[1], 'bo', markersize=15,
           markeredgecolor='black', markeredgewidth=2, label='MAP')

    ax.set_xlabel('β₀')
    ax.set_ylabel('β₁')
    ax.set_title('Joint Posterior Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Prediction with uncertainty
    ax = axes[1, 0]

    if X.shape[1] == 2:
        # Create grid for visualization
        x1_range = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 50)
        x2_range = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 50)
        x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

        X_grid = np.column_stack([x1_grid.ravel(), x2_grid.ravel()])

        # Predict probability
        def predictor(params, X_pred):
            return expit(X_pred @ params)

        prob_mean, prob_std = laplace_logistic.predict(
            X_grid,
            predictor,
            return_std=True,
            n_samples=1000
        )

        # Reshape for contour
        prob_mean = prob_mean.reshape(x1_grid.shape)

        # Contour plot
        contour = ax.contourf(x1_grid, x2_grid, prob_mean, levels=20, cmap='RdYlBu_r', alpha=0.7)
        plt.colorbar(contour, ax=ax, label='P(y=1)')

        # Scatter data points
        ax.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', marker='o',
                  edgecolors='black', linewidth=0.5, s=50, alpha=0.7, label='Class 0')
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', marker='s',
                  edgecolors='black', linewidth=0.5, s=50, alpha=0.7, label='Class 1')

        ax.set_xlabel('X₀')
        ax.set_ylabel('X₁')
        ax.set_title('Predicted Probabilities')
        ax.legend()

    # 4. Convergence of optimization
    ax = axes[1, 1]

    # Run optimization with tracking
    log_posteriors = []

    def callback(x):
        log_posteriors.append(laplace_logistic.log_posterior_fn(x))

    from scipy.optimize import minimize

    result = minimize(
        lambda x: -laplace_logistic.log_posterior_fn(x),
        np.zeros(len(true_beta)),
        jac=lambda x: -laplace_logistic.gradient_fn(x),
        method='L-BFGS-B',
        callback=callback
    )

    if len(log_posteriors) > 0:
        ax.plot(log_posteriors, linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Log Posterior')
        ax.set_title('MAP Optimization Convergence')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No convergence data available',
               ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig('/tmp/laplace_approximation_quality.png', dpi=150, bbox_inches='tight')
    print("\nSaved: Approximation quality visualization")
    plt.close()


def model_comparison_via_evidence():
    """Compare models using marginal likelihood (evidence)."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON VIA LAPLACE-APPROXIMATED EVIDENCE")
    print("=" * 80)

    # Generate data
    np.random.seed(42)
    n_samples = 200

    # True model has 3 relevant features
    X_full = np.random.randn(n_samples, 6)
    true_beta = np.array([2.0, -1.5, 1.0, 0, 0, 0])
    y = X_full @ true_beta + np.random.randn(n_samples) * 0.5

    # Compare models with different numbers of features
    models = {}
    evidences = {}

    for n_features in [2, 3, 4, 5, 6]:
        X = X_full[:, :n_features]

        # Define log posterior
        prior_std = 2.0
        noise_std = 0.5

        def log_posterior(beta):
            residuals = y - X @ beta
            log_lik = -0.5 * np.sum(residuals ** 2) / (noise_std ** 2)
            log_prior = -0.5 * np.sum(beta ** 2) / (prior_std ** 2)
            return log_lik + log_prior

        # Fit Laplace approximation
        laplace = LaplaceApproximation(log_posterior)
        laplace.fit(np.zeros(n_features))

        models[n_features] = laplace
        evidences[n_features] = laplace.log_marginal_likelihood

    # Display results
    results = pd.DataFrame({
        'Num Features': list(evidences.keys()),
        'Log Evidence': list(evidences.values())
    })

    results['Relative Evidence'] = np.exp(
        results['Log Evidence'] - results['Log Evidence'].max()
    )

    print("\n", results.to_string(index=False))

    best_k = results.loc[results['Log Evidence'].idxmax(), 'Num Features']
    print(f"\nBest model (by evidence): {best_k} features")
    print(f"True number of relevant features: 3")


def main():
    """Main execution function."""
    print("=" * 80)
    print("LAPLACE APPROXIMATION FOR BAYESIAN INFERENCE")
    print("=" * 80)

    # Example 1: Logistic regression
    laplace_logistic, X, y, true_beta = bayesian_logistic_regression_laplace()

    # Example 2: Linear regression (exact vs approximate)
    laplace_linear, exact_mean, exact_cov = bayesian_linear_regression_laplace()

    # Visualize approximation quality
    visualize_approximation_quality(laplace_logistic, X, y, true_beta)

    # Model comparison
    model_comparison_via_evidence()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. Laplace approximation is fast and scales well")
    print("2. Approximation is exact for Gaussian posteriors (e.g., linear regression)")
    print("3. Quality decreases for highly non-Gaussian or multi-modal posteriors")
    print("4. Provides closed-form expression for posterior covariance")
    print("5. Can approximate marginal likelihood for model comparison")
    print("6. Requires computing Hessian at MAP (can be expensive)")


if __name__ == "__main__":
    main()
