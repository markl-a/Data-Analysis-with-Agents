"""
Hamiltonian Monte Carlo (HMC) Sampling

This solution implements Hamiltonian Monte Carlo for Bayesian inference,
demonstrating the physics-inspired approach to efficient MCMC sampling.

Techniques:
- Hamiltonian dynamics simulation
- Leapfrog integrator
- Acceptance-rejection step
- Tuning of step size and trajectory length
- Comparison with Random Walk Metropolis
- Effective sample size computation
- Convergence diagnostics
- Posterior visualization

Dataset: Bayesian linear regression and logistic regression examples
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import expit
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


class HamiltonianMonteCarlo:
    """
    Hamiltonian Monte Carlo sampler for Bayesian inference.

    HMC uses Hamiltonian dynamics to propose states far from the current state
    while maintaining high acceptance rates.
    """

    def __init__(self, log_prob_fn, grad_log_prob_fn, step_size=0.1, n_steps=10):
        """
        Initialize HMC sampler.

        Parameters:
        -----------
        log_prob_fn : callable
            Function that computes log probability of parameters
        grad_log_prob_fn : callable
            Function that computes gradient of log probability
        step_size : float
            Step size for leapfrog integrator (epsilon)
        n_steps : int
            Number of leapfrog steps (L)
        """
        self.log_prob_fn = log_prob_fn
        self.grad_log_prob_fn = grad_log_prob_fn
        self.step_size = step_size
        self.n_steps = n_steps

        # Statistics
        self.acceptance_rate = 0.0
        self.samples = None

    def leapfrog(self, theta, momentum):
        """
        Leapfrog integration for Hamiltonian dynamics.

        Parameters:
        -----------
        theta : array
            Position (parameters)
        momentum : array
            Momentum

        Returns:
        --------
        theta_new : array
            New position
        momentum_new : array
            New momentum
        """
        # Make a copy
        theta = theta.copy()
        momentum = momentum.copy()

        # Half step for momentum
        momentum += 0.5 * self.step_size * self.grad_log_prob_fn(theta)

        # Full steps for position and momentum
        for _ in range(self.n_steps - 1):
            theta += self.step_size * momentum
            momentum += self.step_size * self.grad_log_prob_fn(theta)

        # Final full step for position
        theta += self.step_size * momentum

        # Half step for momentum
        momentum += 0.5 * self.step_size * self.grad_log_prob_fn(theta)

        return theta, momentum

    def hamiltonian(self, theta, momentum):
        """
        Compute Hamiltonian (total energy).

        H(θ, p) = -log p(θ | data) + 0.5 * p^T p

        where first term is potential energy (negative log prob)
        and second term is kinetic energy.
        """
        potential_energy = -self.log_prob_fn(theta)
        kinetic_energy = 0.5 * np.sum(momentum ** 2)
        return potential_energy + kinetic_energy

    def sample(self, initial_theta, n_samples=1000, burn_in=500):
        """
        Generate samples using HMC.

        Parameters:
        -----------
        initial_theta : array
            Initial parameter values
        n_samples : int
            Number of samples to generate
        burn_in : int
            Number of burn-in samples

        Returns:
        --------
        samples : array, shape (n_samples, n_params)
            MCMC samples
        """
        n_params = len(initial_theta)
        total_samples = burn_in + n_samples

        samples = np.zeros((total_samples, n_params))
        samples[0] = initial_theta

        accepted = 0

        for i in range(1, total_samples):
            # Current state
            theta_current = samples[i - 1]

            # Sample momentum
            momentum_current = np.random.randn(n_params)

            # Compute current Hamiltonian
            H_current = self.hamiltonian(theta_current, momentum_current)

            # Leapfrog integration
            theta_proposed, momentum_proposed = self.leapfrog(
                theta_current,
                momentum_current
            )

            # Compute proposed Hamiltonian
            H_proposed = self.hamiltonian(theta_proposed, momentum_proposed)

            # Accept/reject
            log_accept_ratio = -H_proposed + H_current

            if np.log(np.random.rand()) < log_accept_ratio:
                samples[i] = theta_proposed
                accepted += 1
            else:
                samples[i] = theta_current

        self.acceptance_rate = accepted / total_samples
        self.samples = samples[burn_in:]

        return self.samples


class RandomWalkMetropolis:
    """Random Walk Metropolis-Hastings for comparison."""

    def __init__(self, log_prob_fn, proposal_std=0.1):
        self.log_prob_fn = log_prob_fn
        self.proposal_std = proposal_std
        self.acceptance_rate = 0.0
        self.samples = None

    def sample(self, initial_theta, n_samples=1000, burn_in=500):
        """Generate samples using Random Walk Metropolis."""
        n_params = len(initial_theta)
        total_samples = burn_in + n_samples

        samples = np.zeros((total_samples, n_params))
        samples[0] = initial_theta

        log_prob_current = self.log_prob_fn(initial_theta)
        accepted = 0

        for i in range(1, total_samples):
            # Propose
            theta_proposed = samples[i - 1] + np.random.randn(n_params) * self.proposal_std

            # Compute log probability
            log_prob_proposed = self.log_prob_fn(theta_proposed)

            # Accept/reject
            log_accept_ratio = log_prob_proposed - log_prob_current

            if np.log(np.random.rand()) < log_accept_ratio:
                samples[i] = theta_proposed
                log_prob_current = log_prob_proposed
                accepted += 1
            else:
                samples[i] = samples[i - 1]

        self.acceptance_rate = accepted / total_samples
        self.samples = samples[burn_in:]

        return self.samples


def bayesian_linear_regression_example():
    """Demonstrate HMC on Bayesian linear regression."""
    print("=" * 80)
    print("EXAMPLE 1: BAYESIAN LINEAR REGRESSION")
    print("=" * 80)

    # Generate data
    np.random.seed(42)
    n_samples = 100
    n_features = 3

    X = np.random.randn(n_samples, n_features)
    true_beta = np.array([2.0, -1.5, 0.5])
    noise_std = 0.5
    y = X @ true_beta + np.random.randn(n_samples) * noise_std

    print(f"\nData: {n_samples} samples, {n_features} features")
    print(f"True coefficients: {true_beta}")
    print(f"Noise std: {noise_std}")

    # Define log probability and gradient
    prior_std = 5.0

    def log_prob(beta):
        """Log posterior for linear regression."""
        # Log likelihood
        residuals = y - X @ beta
        log_lik = -0.5 * np.sum(residuals ** 2) / (noise_std ** 2)

        # Log prior (Gaussian)
        log_prior = -0.5 * np.sum(beta ** 2) / (prior_std ** 2)

        return log_lik + log_prior

    def grad_log_prob(beta):
        """Gradient of log posterior."""
        # Gradient of log likelihood
        residuals = y - X @ beta
        grad_log_lik = X.T @ residuals / (noise_std ** 2)

        # Gradient of log prior
        grad_log_prior = -beta / (prior_std ** 2)

        return grad_log_lik + grad_log_prior

    # Run HMC
    print("\nRunning HMC...")
    hmc = HamiltonianMonteCarlo(
        log_prob_fn=log_prob,
        grad_log_prob_fn=grad_log_prob,
        step_size=0.01,
        n_steps=20
    )

    initial_beta = np.zeros(n_features)
    hmc_samples = hmc.sample(initial_beta, n_samples=2000, burn_in=500)

    print(f"HMC Acceptance Rate: {hmc.acceptance_rate:.3f}")

    # Run Random Walk Metropolis for comparison
    print("\nRunning Random Walk Metropolis...")
    rwm = RandomWalkMetropolis(
        log_prob_fn=log_prob,
        proposal_std=0.1
    )

    rwm_samples = rwm.sample(initial_beta, n_samples=2000, burn_in=500)

    print(f"RWM Acceptance Rate: {rwm.acceptance_rate:.3f}")

    # Compare results
    print("\n" + "=" * 80)
    print("POSTERIOR ESTIMATES")
    print("=" * 80)

    hmc_mean = np.mean(hmc_samples, axis=0)
    hmc_std = np.std(hmc_samples, axis=0)

    rwm_mean = np.mean(rwm_samples, axis=0)
    rwm_std = np.std(rwm_samples, axis=0)

    results = pd.DataFrame({
        'Parameter': [f'β{i}' for i in range(n_features)],
        'True': true_beta,
        'HMC Mean': hmc_mean,
        'HMC Std': hmc_std,
        'RWM Mean': rwm_mean,
        'RWM Std': rwm_std
    })

    print("\n", results.to_string(index=False))

    return hmc_samples, rwm_samples, true_beta


def bayesian_logistic_regression_example():
    """Demonstrate HMC on Bayesian logistic regression."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: BAYESIAN LOGISTIC REGRESSION")
    print("=" * 80)

    # Generate data
    np.random.seed(42)
    n_samples = 200
    n_features = 2

    X = np.random.randn(n_samples, n_features)
    true_beta = np.array([1.5, -2.0])
    z = X @ true_beta
    p = expit(z)
    y = (np.random.rand(n_samples) < p).astype(int)

    print(f"\nData: {n_samples} samples, {n_features} features")
    print(f"True coefficients: {true_beta}")
    print(f"Class balance: {np.mean(y):.2%} positive")

    # Define log probability and gradient
    prior_std = 3.0

    def log_prob(beta):
        """Log posterior for logistic regression."""
        z = X @ beta
        # Log likelihood
        log_lik = np.sum(y * z - np.log(1 + np.exp(z)))

        # Log prior
        log_prior = -0.5 * np.sum(beta ** 2) / (prior_std ** 2)

        return log_lik + log_prior

    def grad_log_prob(beta):
        """Gradient of log posterior."""
        z = X @ beta
        p = expit(z)

        # Gradient of log likelihood
        grad_log_lik = X.T @ (y - p)

        # Gradient of log prior
        grad_log_prior = -beta / (prior_std ** 2)

        return grad_log_lik + grad_log_prior

    # Run HMC
    print("\nRunning HMC...")
    hmc = HamiltonianMonteCarlo(
        log_prob_fn=log_prob,
        grad_log_prob_fn=grad_log_prob,
        step_size=0.02,
        n_steps=15
    )

    initial_beta = np.zeros(n_features)
    hmc_samples = hmc.sample(initial_beta, n_samples=2000, burn_in=500)

    print(f"HMC Acceptance Rate: {hmc.acceptance_rate:.3f}")

    # Posterior estimates
    hmc_mean = np.mean(hmc_samples, axis=0)
    hmc_std = np.std(hmc_samples, axis=0)

    print("\n" + "=" * 80)
    print("POSTERIOR ESTIMATES")
    print("=" * 80)

    results = pd.DataFrame({
        'Parameter': [f'β{i}' for i in range(n_features)],
        'True': true_beta,
        'Posterior Mean': hmc_mean,
        'Posterior Std': hmc_std,
        '95% CI Lower': np.percentile(hmc_samples, 2.5, axis=0),
        '95% CI Upper': np.percentile(hmc_samples, 97.5, axis=0)
    })

    print("\n", results.to_string(index=False))

    return hmc_samples, true_beta


def compute_effective_sample_size(samples):
    """Compute effective sample size using autocorrelation."""
    n_samples, n_params = samples.shape

    ess = np.zeros(n_params)

    for i in range(n_params):
        # Compute autocorrelation
        chain = samples[:, i]
        chain_centered = chain - np.mean(chain)

        # Autocorrelation at different lags
        max_lag = min(n_samples // 2, 100)
        autocorr = np.zeros(max_lag)

        var = np.var(chain)

        for lag in range(max_lag):
            if lag == 0:
                autocorr[lag] = 1.0
            else:
                autocorr[lag] = np.mean(
                    chain_centered[:-lag] * chain_centered[lag:]
                ) / var

        # Find first negative autocorrelation
        first_negative = np.where(autocorr < 0)[0]
        if len(first_negative) > 0:
            cutoff = first_negative[0]
        else:
            cutoff = max_lag

        # Integrated autocorrelation time
        tau = 1 + 2 * np.sum(autocorr[1:cutoff])

        # Effective sample size
        ess[i] = n_samples / tau

    return ess


def visualize_sampling_comparison(hmc_samples, rwm_samples, true_params):
    """Visualize HMC vs Random Walk Metropolis."""

    n_params = hmc_samples.shape[1]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Trace plots
    for i in range(min(3, n_params)):
        ax = axes[0, i]
        ax.plot(hmc_samples[:, i], alpha=0.7, label='HMC')
        ax.plot(rwm_samples[:, i], alpha=0.7, label='RWM')
        ax.axhline(true_params[i], color='red', linestyle='--',
                  linewidth=2, label='True')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(f'β{i}')
        ax.set_title(f'Trace Plot: β{i}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 2. Posterior distributions
    for i in range(min(3, n_params)):
        ax = axes[1, i]
        ax.hist(hmc_samples[:, i], bins=50, alpha=0.5,
               label='HMC', density=True, color='blue')
        ax.hist(rwm_samples[:, i], bins=50, alpha=0.5,
               label='RWM', density=True, color='orange')
        ax.axvline(true_params[i], color='red', linestyle='--',
                  linewidth=2, label='True')
        ax.set_xlabel(f'β{i}')
        ax.set_ylabel('Density')
        ax.set_title(f'Posterior Distribution: β{i}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/hmc_sampling_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved: Sampling comparison visualization")
    plt.close()


def visualize_autocorrelation(hmc_samples, rwm_samples):
    """Visualize autocorrelation for HMC vs RWM."""

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    max_lag = 100
    n_params = min(3, hmc_samples.shape[1])

    # HMC autocorrelation
    ax = axes[0]
    for i in range(n_params):
        chain = hmc_samples[:, i]
        autocorr = np.zeros(max_lag)

        for lag in range(max_lag):
            if lag == 0:
                autocorr[lag] = 1.0
            else:
                chain_centered = chain - np.mean(chain)
                autocorr[lag] = np.corrcoef(
                    chain_centered[:-lag],
                    chain_centered[lag:]
                )[0, 1]

        ax.plot(autocorr, label=f'β{i}', alpha=0.7, linewidth=2)

    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('HMC Autocorrelation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RWM autocorrelation
    ax = axes[1]
    for i in range(n_params):
        chain = rwm_samples[:, i]
        autocorr = np.zeros(max_lag)

        for lag in range(max_lag):
            if lag == 0:
                autocorr[lag] = 1.0
            else:
                chain_centered = chain - np.mean(chain)
                autocorr[lag] = np.corrcoef(
                    chain_centered[:-lag],
                    chain_centered[lag:]
                )[0, 1]

        ax.plot(autocorr, label=f'β{i}', alpha=0.7, linewidth=2)

    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Random Walk Metropolis Autocorrelation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/hmc_autocorrelation.png', dpi=150, bbox_inches='tight')
    print("Saved: Autocorrelation visualization")
    plt.close()


def analyze_efficiency(hmc_samples, rwm_samples):
    """Analyze sampling efficiency."""
    print("\n" + "=" * 80)
    print("SAMPLING EFFICIENCY ANALYSIS")
    print("=" * 80)

    # Compute ESS
    hmc_ess = compute_effective_sample_size(hmc_samples)
    rwm_ess = compute_effective_sample_size(rwm_samples)

    results = pd.DataFrame({
        'Parameter': [f'β{i}' for i in range(len(hmc_ess))],
        'HMC ESS': hmc_ess,
        'RWM ESS': rwm_ess,
        'Efficiency Ratio': hmc_ess / rwm_ess
    })

    print("\n", results.to_string(index=False))

    print(f"\nAverage HMC ESS: {np.mean(hmc_ess):.1f}")
    print(f"Average RWM ESS: {np.mean(rwm_ess):.1f}")
    print(f"HMC is {np.mean(hmc_ess / rwm_ess):.2f}x more efficient")


def main():
    """Main execution function."""
    print("=" * 80)
    print("HAMILTONIAN MONTE CARLO SAMPLING")
    print("=" * 80)

    # Example 1: Bayesian Linear Regression
    hmc_samples_lr, rwm_samples_lr, true_beta_lr = bayesian_linear_regression_example()

    # Visualize comparison
    visualize_sampling_comparison(hmc_samples_lr, rwm_samples_lr, true_beta_lr)

    # Visualize autocorrelation
    visualize_autocorrelation(hmc_samples_lr, rwm_samples_lr)

    # Analyze efficiency
    analyze_efficiency(hmc_samples_lr, rwm_samples_lr)

    # Example 2: Bayesian Logistic Regression
    hmc_samples_logreg, true_beta_logreg = bayesian_logistic_regression_example()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. HMC uses gradient information to propose distant states efficiently")
    print("2. Leapfrog integrator simulates Hamiltonian dynamics")
    print("3. HMC typically has higher acceptance rates than Random Walk Metropolis")
    print("4. HMC produces less autocorrelated samples (higher ESS)")
    print("5. Step size and trajectory length are important tuning parameters")
    print("6. HMC is especially effective in high-dimensional spaces")


if __name__ == "__main__":
    main()
