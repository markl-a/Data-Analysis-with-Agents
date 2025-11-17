"""
Variational Inference for Mixture Models

This solution implements variational inference for Gaussian mixture models
using coordinate ascent variational inference (CAVI) and compares with EM.

Techniques:
- Mean-field variational inference
- Evidence Lower Bound (ELBO) optimization
- Coordinate ascent updates
- Comparison with Expectation-Maximization
- Convergence monitoring
- Model selection via ELBO
- Posterior visualization
- Uncertainty quantification

Dataset: Synthetic multi-modal data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import digamma, gammaln, logsumexp
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


class VariationalGaussianMixture:
    """
    Variational Inference for Gaussian Mixture Models.

    Uses mean-field approximation with conjugate priors:
    - Dirichlet prior on mixture weights
    - Normal-Wishart prior on component parameters
    """

    def __init__(self, n_components=3, max_iter=200, tol=1e-4, verbose=True):
        """
        Initialize Variational GMM.

        Parameters:
        -----------
        n_components : int
            Number of mixture components
        max_iter : int
            Maximum iterations for CAVI
        tol : float
            Convergence tolerance
        verbose : bool
            Print progress
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        # Variational parameters (to be initialized)
        self.gamma = None  # Responsibilities (N x K)
        self.alpha = None  # Dirichlet parameters for mixing weights
        self.m = None      # Mean parameters
        self.beta = None   # Precision on means
        self.W = None      # Wishart scale matrices
        self.nu = None     # Wishart degrees of freedom

        # ELBO history
        self.elbo_history = []

    def _initialize_parameters(self, X):
        """Initialize variational parameters."""
        n_samples, n_features = X.shape
        K = self.n_components

        # Initialize responsibilities randomly
        self.gamma = np.random.dirichlet(np.ones(K), size=n_samples)

        # Initialize with k-means
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # Prior hyperparameters (weakly informative)
        self.alpha_0 = np.ones(K)  # Dirichlet prior
        self.m_0 = np.mean(X, axis=0)  # Prior mean
        self.beta_0 = 0.01  # Prior precision on mean
        self.W_0 = np.eye(n_features) * 0.01  # Prior scale matrix
        self.nu_0 = n_features + 2  # Prior degrees of freedom

        # Initialize variational parameters
        self.alpha = self.alpha_0.copy()
        self.m = np.array([X[labels == k].mean(axis=0) if np.sum(labels == k) > 0
                          else self.m_0 for k in range(K)])
        self.beta = np.ones(K) * self.beta_0
        self.W = np.array([self.W_0.copy() for _ in range(K)])
        self.nu = np.ones(K) * self.nu_0

    def _update_responsibilities(self, X):
        """Update variational distribution over cluster assignments (E-step analog)."""
        n_samples, n_features = X.shape
        K = self.n_components

        log_rho = np.zeros((n_samples, K))

        for k in range(K):
            # E[log π_k]
            log_pi = digamma(self.alpha[k]) - digamma(np.sum(self.alpha))

            # E[log |Λ_k|]
            log_det_Lambda = np.sum([
                digamma((self.nu[k] + 1 - i) / 2)
                for i in range(1, n_features + 1)
            ]) + n_features * np.log(2) + np.linalg.slogdet(self.W[k])[1]

            # E[(x - μ_k)^T Λ_k (x - μ_k)]
            diff = X - self.m[k]
            W_inv = np.linalg.inv(self.W[k])

            mahalanobis = (
                n_features / self.beta[k] +
                self.nu[k] * np.sum(diff @ W_inv * diff, axis=1)
            )

            log_rho[:, k] = (
                log_pi +
                0.5 * log_det_Lambda -
                0.5 * n_features * np.log(2 * np.pi) -
                0.5 * mahalanobis
            )

        # Normalize
        log_rho_normalized = log_rho - logsumexp(log_rho, axis=1, keepdims=True)
        self.gamma = np.exp(log_rho_normalized)

    def _update_mixing_weights(self):
        """Update variational distribution over mixing weights."""
        N_k = np.sum(self.gamma, axis=0)
        self.alpha = self.alpha_0 + N_k

    def _update_component_parameters(self, X):
        """Update variational distributions over component parameters."""
        n_samples, n_features = X.shape
        K = self.n_components

        for k in range(K):
            N_k = np.sum(self.gamma[:, k])

            # Compute sufficient statistics
            x_bar_k = np.sum(self.gamma[:, k:k+1] * X, axis=0) / (N_k + 1e-10)

            S_k = np.zeros((n_features, n_features))
            for i in range(n_samples):
                diff = X[i] - x_bar_k
                S_k += self.gamma[i, k] * np.outer(diff, diff)

            # Update parameters
            self.beta[k] = self.beta_0 + N_k

            self.m[k] = (self.beta_0 * self.m_0 + N_k * x_bar_k) / self.beta[k]

            self.nu[k] = self.nu_0 + N_k

            diff_m = x_bar_k - self.m_0
            self.W[k] = np.linalg.inv(
                np.linalg.inv(self.W_0) +
                S_k +
                (self.beta_0 * N_k / self.beta[k]) * np.outer(diff_m, diff_m)
            )

    def _compute_elbo(self, X):
        """Compute Evidence Lower Bound."""
        n_samples, n_features = X.shape
        K = self.n_components

        elbo = 0.0

        # E[log p(X, Z, π, μ, Λ)]
        for k in range(K):
            N_k = np.sum(self.gamma[:, k])

            # E[log p(Z | π)]
            elbo += N_k * (digamma(self.alpha[k]) - digamma(np.sum(self.alpha)))

            # E[log p(X | Z, μ, Λ)]
            log_det_Lambda = np.sum([
                digamma((self.nu[k] + 1 - i) / 2)
                for i in range(1, n_features + 1)
            ]) + n_features * np.log(2) + np.linalg.slogdet(self.W[k])[1]

            diff = X - self.m[k]
            W_inv = np.linalg.inv(self.W[k])
            trace_term = np.sum(self.gamma[:, k:k+1] * np.sum(diff @ W_inv * diff, axis=1))

            elbo += 0.5 * N_k * (
                log_det_Lambda -
                n_features * np.log(2 * np.pi) -
                n_features / self.beta[k] -
                self.nu[k] * trace_term / N_k
            )

        # E[log p(π)]
        elbo += (
            gammaln(np.sum(self.alpha_0)) - np.sum(gammaln(self.alpha_0)) +
            np.sum((self.alpha_0 - 1) * (digamma(self.alpha) - digamma(np.sum(self.alpha))))
        )

        # E[log p(μ, Λ)] - more complex, simplified here
        # (Full implementation would include Normal-Wishart terms)

        # -E[log q(Z)]
        elbo -= np.sum(self.gamma * np.log(self.gamma + 1e-10))

        # -E[log q(π)]
        elbo -= (
            gammaln(np.sum(self.alpha)) - np.sum(gammaln(self.alpha)) +
            np.sum((self.alpha - 1) * (digamma(self.alpha) - digamma(np.sum(self.alpha))))
        )

        # -E[log q(μ, Λ)] - simplified
        # (Full implementation would include all terms)

        return elbo

    def fit(self, X):
        """
        Fit model using Coordinate Ascent Variational Inference.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        """
        X = np.array(X)
        self._initialize_parameters(X)

        self.elbo_history = []
        prev_elbo = -np.inf

        for iteration in range(self.max_iter):
            # CAVI updates
            self._update_responsibilities(X)
            self._update_mixing_weights()
            self._update_component_parameters(X)

            # Compute ELBO
            elbo = self._compute_elbo(X)
            self.elbo_history.append(elbo)

            # Check convergence
            elbo_change = elbo - prev_elbo

            if self.verbose and iteration % 20 == 0:
                print(f"Iteration {iteration:3d}: ELBO = {elbo:.2f}, "
                      f"Change = {elbo_change:.4f}")

            if abs(elbo_change) < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break

            prev_elbo = elbo

        return self

    def predict(self, X):
        """Predict cluster assignments."""
        # Update responsibilities for new data
        # (For simplicity, use fixed parameters)
        n_samples, n_features = X.shape
        K = self.n_components

        log_rho = np.zeros((n_samples, K))

        for k in range(K):
            log_pi = digamma(self.alpha[k]) - digamma(np.sum(self.alpha))

            log_det_Lambda = np.sum([
                digamma((self.nu[k] + 1 - i) / 2)
                for i in range(1, n_features + 1)
            ]) + n_features * np.log(2) + np.linalg.slogdet(self.W[k])[1]

            diff = X - self.m[k]
            W_inv = np.linalg.inv(self.W[k])
            mahalanobis = (
                n_features / self.beta[k] +
                self.nu[k] * np.sum(diff @ W_inv * diff, axis=1)
            )

            log_rho[:, k] = (
                log_pi +
                0.5 * log_det_Lambda -
                0.5 * n_features * np.log(2 * np.pi) -
                0.5 * mahalanobis
            )

        return np.argmax(log_rho, axis=1)

    def predict_proba(self, X):
        """Predict cluster probabilities."""
        n_samples, n_features = X.shape
        K = self.n_components

        log_rho = np.zeros((n_samples, K))

        for k in range(K):
            log_pi = digamma(self.alpha[k]) - digamma(np.sum(self.alpha))

            log_det_Lambda = np.sum([
                digamma((self.nu[k] + 1 - i) / 2)
                for i in range(1, n_features + 1)
            ]) + n_features * np.log(2) + np.linalg.slogdet(self.W[k])[1]

            diff = X - self.m[k]
            W_inv = np.linalg.inv(self.W[k])
            mahalanobis = (
                n_features / self.beta[k] +
                self.nu[k] * np.sum(diff @ W_inv * diff, axis=1)
            )

            log_rho[:, k] = (
                log_pi +
                0.5 * log_det_Lambda -
                0.5 * n_features * np.log(2 * np.pi) -
                0.5 * mahalanobis
            )

        log_rho_normalized = log_rho - logsumexp(log_rho, axis=1, keepdims=True)
        return np.exp(log_rho_normalized)


def generate_mixture_data(n_samples=1000, n_features=2, n_components=3, seed=42):
    """Generate synthetic data from a Gaussian mixture."""
    np.random.seed(seed)

    # True mixture weights
    true_weights = np.random.dirichlet(np.ones(n_components) * 2)

    # True component parameters
    true_means = np.random.randn(n_components, n_features) * 5
    true_covs = [np.eye(n_features) * (1 + i * 0.5) for i in range(n_components)]

    # Generate samples
    X = []
    y = []

    for _ in range(n_samples):
        # Sample component
        k = np.random.choice(n_components, p=true_weights)

        # Sample from component
        x = np.random.multivariate_normal(true_means[k], true_covs[k])

        X.append(x)
        y.append(k)

    return np.array(X), np.array(y), true_weights, true_means, true_covs


def compare_with_em(X, y_true, n_components):
    """Compare variational inference with EM algorithm."""
    print("=" * 80)
    print("VARIATIONAL INFERENCE VS EXPECTATION-MAXIMIZATION")
    print("=" * 80)

    # Variational Inference
    print("\nFitting Variational GMM...")
    vi_model = VariationalGaussianMixture(
        n_components=n_components,
        max_iter=200,
        verbose=False
    )
    vi_model.fit(X)

    # EM Algorithm (sklearn)
    print("\nFitting EM GMM...")
    from sklearn.mixture import GaussianMixture
    em_model = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        max_iter=200,
        random_state=42
    )
    em_model.fit(X)

    # Predictions
    vi_pred = vi_model.predict(X)
    em_pred = em_model.predict(X)

    # Metrics (using adjusted rand index)
    from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

    vi_ari = adjusted_rand_score(y_true, vi_pred)
    em_ari = adjusted_rand_score(y_true, em_pred)

    vi_ami = adjusted_mutual_info_score(y_true, vi_pred)
    em_ami = adjusted_mutual_info_score(y_true, em_pred)

    print("\n" + "=" * 80)
    print("CLUSTERING PERFORMANCE")
    print("=" * 80)
    print(f"Variational Inference ARI:  {vi_ari:.4f}")
    print(f"EM Algorithm ARI:           {em_ari:.4f}")
    print(f"Variational Inference AMI:  {vi_ami:.4f}")
    print(f"EM Algorithm AMI:           {em_ami:.4f}")

    # Model comparison
    vi_elbo = vi_model.elbo_history[-1]
    em_ll = em_model.score(X) * len(X)  # Log likelihood

    print("\n" + "=" * 80)
    print("MODEL EVIDENCE")
    print("=" * 80)
    print(f"VI ELBO (lower bound):      {vi_elbo:.2f}")
    print(f"EM Log Likelihood:          {em_ll:.2f}")

    return vi_model, em_model


def visualize_convergence(vi_model):
    """Visualize ELBO convergence."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(vi_model.elbo_history, linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('ELBO', fontsize=12)
    ax.set_title('Evidence Lower Bound Convergence', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Mark convergence point
    if len(vi_model.elbo_history) < vi_model.max_iter:
        ax.axvline(len(vi_model.elbo_history) - 1, color='red',
                  linestyle='--', label='Convergence')
        ax.legend()

    plt.tight_layout()
    plt.savefig('/tmp/vi_mixture_convergence.png', dpi=150, bbox_inches='tight')
    print("\nSaved: Convergence visualization")
    plt.close()


def visualize_clustering_results(X, y_true, vi_model, em_model):
    """Visualize clustering results for 2D data."""
    if X.shape[1] != 2:
        print("Skipping visualization (data not 2D)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. True clusters
    ax = axes[0, 0]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis',
                        alpha=0.6, edgecolors='k', linewidth=0.5)
    ax.set_title('True Clusters', fontsize=14)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    plt.colorbar(scatter, ax=ax, label='Cluster')

    # 2. VI clustering
    ax = axes[0, 1]
    vi_pred = vi_model.predict(X)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=vi_pred, cmap='viridis',
                        alpha=0.6, edgecolors='k', linewidth=0.5)

    # Plot means
    for k in range(vi_model.n_components):
        ax.plot(vi_model.m[k, 0], vi_model.m[k, 1], 'r*',
               markersize=20, markeredgecolor='black', markeredgewidth=2)

    ax.set_title('Variational Inference Clustering', fontsize=14)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    plt.colorbar(scatter, ax=ax, label='Cluster')

    # 3. EM clustering
    ax = axes[1, 0]
    em_pred = em_model.predict(X)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=em_pred, cmap='viridis',
                        alpha=0.6, edgecolors='k', linewidth=0.5)

    # Plot means
    for k in range(em_model.n_components):
        ax.plot(em_model.means_[k, 0], em_model.means_[k, 1], 'r*',
               markersize=20, markeredgecolor='black', markeredgewidth=2)

    ax.set_title('EM Clustering', fontsize=14)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    plt.colorbar(scatter, ax=ax, label='Cluster')

    # 4. Uncertainty (VI)
    ax = axes[1, 1]
    vi_proba = vi_model.predict_proba(X)
    uncertainty = -np.sum(vi_proba * np.log(vi_proba + 1e-10), axis=1)  # Entropy

    scatter = ax.scatter(X[:, 0], X[:, 1], c=uncertainty, cmap='Reds',
                        alpha=0.6, edgecolors='k', linewidth=0.5)
    ax.set_title('Clustering Uncertainty (VI)', fontsize=14)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    plt.colorbar(scatter, ax=ax, label='Entropy')

    plt.tight_layout()
    plt.savefig('/tmp/vi_mixture_clustering.png', dpi=150, bbox_inches='tight')
    print("Saved: Clustering visualization")
    plt.close()


def model_selection_via_elbo(X):
    """Select number of components via ELBO."""
    print("\n" + "=" * 80)
    print("MODEL SELECTION VIA ELBO")
    print("=" * 80)

    K_range = range(2, 8)
    results = []

    for K in K_range:
        print(f"\nFitting with K={K} components...")
        model = VariationalGaussianMixture(
            n_components=K,
            max_iter=200,
            verbose=False
        )
        model.fit(X)

        elbo = model.elbo_history[-1]
        results.append({
            'K': K,
            'ELBO': elbo,
            'Iterations': len(model.elbo_history)
        })

    df_results = pd.DataFrame(results)
    print("\n", df_results.to_string(index=False))

    # Find best K
    best_idx = df_results['ELBO'].idxmax()
    best_K = df_results.loc[best_idx, 'K']
    print(f"\nBest K (by ELBO): {best_K}")

    # Visualize
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(df_results['K'], df_results['ELBO'], 'o-', linewidth=2, markersize=8)
    ax.axvline(best_K, color='red', linestyle='--', label=f'Best K = {best_K}')
    ax.set_xlabel('Number of Components (K)', fontsize=12)
    ax.set_ylabel('ELBO', fontsize=12)
    ax.set_title('Model Selection via Evidence Lower Bound', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig('/tmp/vi_mixture_model_selection.png', dpi=150, bbox_inches='tight')
    print("Saved: Model selection visualization")
    plt.close()


def main():
    """Main execution function."""
    print("=" * 80)
    print("VARIATIONAL INFERENCE FOR GAUSSIAN MIXTURE MODELS")
    print("=" * 80)

    # Generate data
    print("\nGenerating synthetic mixture data...")
    X, y_true, true_weights, true_means, true_covs = generate_mixture_data(
        n_samples=1000,
        n_features=2,
        n_components=3
    )

    print(f"Samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"True components: 3")
    print(f"True weights: {true_weights}")

    # Compare VI with EM
    vi_model, em_model = compare_with_em(X, y_true, n_components=3)

    # Visualize convergence
    visualize_convergence(vi_model)

    # Visualize clustering
    visualize_clustering_results(X, y_true, vi_model, em_model)

    # Model selection
    model_selection_via_elbo(X)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. Variational inference provides scalable approximate Bayesian inference")
    print("2. ELBO serves as both optimization objective and model selection criterion")
    print("3. Mean-field approximation assumes independence between latent variables")
    print("4. VI is typically faster than MCMC but less accurate")
    print("5. Coordinate ascent guarantees ELBO increases at each iteration")


if __name__ == "__main__":
    main()
