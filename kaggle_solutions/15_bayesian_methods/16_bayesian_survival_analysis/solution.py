"""
Bayesian Survival Analysis

Implements Bayesian approaches to survival analysis including hazard modeling,
censoring handling, and credible intervals for survival functions.

Techniques:
- Weibull survival model
- Cox proportional hazards (Bayesian)
- Kaplan-Meier with uncertainty
- Censoring mechanisms
- Credible intervals for hazard rates
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


class BayesianWeibullSurvival:
    """Bayesian Weibull survival model."""
    
    def __init__(self):
        self.samples = None
    
    def fit(self, times, events, n_samples=2000):
        """Fit Weibull model using MCMC."""
        # Shape (k) and scale (lambda) parameters
        k_samples = []
        lambda_samples = []
        
        k = 1.0
        lambda_param = np.mean(times)
        
        for _ in range(n_samples):
            # Simple Metropolis-Hastings
            k_prop = abs(k + np.random.randn() * 0.1)
            lambda_prop = abs(lambda_param + np.random.randn() * 0.5)
            
            # Log likelihood
            ll_current = np.sum(events * (np.log(k) - k * np.log(lambda_param) + 
                                         (k-1) * np.log(times)) -
                               (times / lambda_param) ** k)
            
            ll_prop = np.sum(events * (np.log(k_prop) - k_prop * np.log(lambda_prop) + 
                                      (k_prop-1) * np.log(times)) -
                            (times / lambda_prop) ** k_prop)
            
            # Priors
            prior_current = stats.gamma.logpdf(k, 2, scale=1) + stats.gamma.logpdf(lambda_param, 2, scale=5)
            prior_prop = stats.gamma.logpdf(k_prop, 2, scale=1) + stats.gamma.logpdf(lambda_prop, 2, scale=5)
            
            if np.log(np.random.rand()) < (ll_prop + prior_prop - ll_current - prior_current):
                k = k_prop
                lambda_param = lambda_prop
            
            k_samples.append(k)
            lambda_samples.append(lambda_param)
        
        self.samples = {'k': np.array(k_samples), 'lambda': np.array(lambda_samples)}
        return self
    
    def survival_function(self, t):
        """Compute survival function S(t) = P(T > t)."""
        S = np.exp(-(t / self.samples['lambda'][:, np.newaxis]) ** self.samples['k'][:, np.newaxis])
        return np.mean(S, axis=0), np.std(S, axis=0)


def generate_survival_data(n=200, shape=1.5, scale=10, censoring_rate=0.3, seed=42):
    """Generate synthetic survival data."""
    np.random.seed(seed)
    
    # True event times
    true_times = np.random.weibull(shape, n) * scale
    
    # Censoring times
    censor_times = np.random.exponential(scale / censoring_rate, n)
    
    # Observed times and events
    times = np.minimum(true_times, censor_times)
    events = (true_times <= censor_times).astype(int)
    
    return times, events


def main():
    print("=" * 80)
    print("BAYESIAN SURVIVAL ANALYSIS")
    print("=" * 80)
    
    # Generate data
    times, events = generate_survival_data()
    print(f"\nSamples: {len(times)}")
    print(f"Events: {np.sum(events)} ({np.mean(events):.1%})")
    print(f"Censored: {np.sum(1-events)} ({np.mean(1-events):.1%})")
    
    # Fit model
    print("\nFitting Bayesian Weibull model...")
    model = BayesianWeibullSurvival()
    model.fit(times, events, n_samples=2000)
    
    # Results
    print(f"\nPosterior mean k: {np.mean(model.samples['k']):.3f}")
    print(f"Posterior mean λ: {np.mean(model.samples['lambda']):.3f}")
    
    # Survival function
    t_grid = np.linspace(0, np.max(times), 100)
    S_mean, S_std = model.survival_function(t_grid)
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    ax.hist(model.samples['k'], bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Shape (k)')
    ax.set_title('Posterior: Shape Parameter')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.hist(model.samples['lambda'], bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Scale (λ)')
    ax.set_title('Posterior: Scale Parameter')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(t_grid, S_mean, 'b-', linewidth=2, label='Mean')
    ax.fill_between(t_grid, S_mean - 1.96*S_std, S_mean + 1.96*S_std, 
                    alpha=0.3, label='95% CI')
    ax.set_xlabel('Time')
    ax.set_ylabel('Survival Probability')
    ax.set_title('Survival Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    event_times = times[events == 1]
    ax.hist(event_times, bins=30, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Time')
    ax.set_ylabel('Count')
    ax.set_title('Event Times Distribution')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/bayesian_survival.png', dpi=150, bbox_inches='tight')
    print("\nSaved: Survival analysis visualization")
    plt.close()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()


def kaplan_meier_estimator(times, events):
    """Compute Kaplan-Meier survival estimate."""
    # Sort by time
    sorted_idx = np.argsort(times)
    times_sorted = times[sorted_idx]
    events_sorted = events[sorted_idx]
    
    # Unique event times
    unique_times = np.unique(times_sorted[events_sorted == 1])
    
    survival_probs = []
    n_at_risk = len(times)
    
    for t in unique_times:
        # Number of events at time t
        n_events = np.sum((times_sorted == t) & (events_sorted == 1))
        
        # Update survival probability
        if n_at_risk > 0:
            surv_prob = 1 - (n_events / n_at_risk)
            survival_probs.append(surv_prob)
            
            # Update number at risk
            n_at_risk -= np.sum(times_sorted == t)
        else:
            survival_probs.append(1.0)
    
    # Cumulative product
    km_estimate = np.cumprod(survival_probs)
    
    return unique_times, km_estimate


def cox_proportional_hazards_bayesian(times, events, covariates, n_samples=1000):
    """Bayesian Cox proportional hazards model."""
    n_covariates = covariates.shape[1]
    beta = np.zeros(n_covariates)
    samples = []
    
    def partial_likelihood(b):
        """Compute Cox partial likelihood."""
        risk_scores = np.exp(covariates @ b)
        ll = 0
        
        for i in range(len(times)):
            if events[i] == 1:
                at_risk = times >= times[i]
                ll += (covariates[i] @ b) - np.log(np.sum(risk_scores[at_risk]))
        
        return ll
    
    def log_posterior(b):
        """Log posterior."""
        return partial_likelihood(b) - 0.5 * np.sum(b**2) / 4  # Prior std=2
    
    # MCMC
    lp_current = log_posterior(beta)
    accepted = 0
    
    for i in range(n_samples):
        beta_prop = beta + np.random.randn(n_covariates) * 0.1
        lp_prop = log_posterior(beta_prop)
        
        if np.log(np.random.rand()) < (lp_prop - lp_current):
            beta = beta_prop
            lp_current = lp_prop
            accepted += 1
        
        samples.append(beta.copy())
    
    print(f"Cox model acceptance rate: {accepted/n_samples:.3f}")
    return np.array(samples)


def compare_models():
    """Compare different survival models."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    # Generate data with covariates
    np.random.seed(42)
    n = 200
    
    # Covariates
    X = np.random.randn(n, 2)
    
    # True hazard depends on covariates
    beta_true = np.array([0.5, -0.3])
    baseline_scale = 10
    
    true_times = np.random.weibull(1.5, n) * baseline_scale * np.exp(-X @ beta_true)
    censor_times = np.random.exponential(15, n)
    
    times = np.minimum(true_times, censor_times)
    events = (true_times <= censor_times).astype(int)
    
    print(f"\nSamples: {n}")
    print(f"Events: {np.sum(events)} ({np.mean(events):.1%})")
    
    # Fit Cox model
    print("\nFitting Bayesian Cox model...")
    cox_samples = cox_proportional_hazards_bayesian(times, events, X, n_samples=1000)
    
    print(f"\nTrue coefficients: {beta_true}")
    print(f"Estimated coefficients: {np.mean(cox_samples, axis=0)}")
    print(f"Posterior std: {np.std(cox_samples, axis=0)}")
    
    return times, events, X, cox_samples


def visualize_cox_results(cox_samples, beta_true):
    """Visualize Cox model results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, ax in enumerate(axes):
        ax.hist(cox_samples[:, i], bins=40, alpha=0.7, edgecolor='black', density=True)
        ax.axvline(beta_true[i], color='red', linestyle='--', lw=2, label='True')
        ax.axvline(np.mean(cox_samples[:, i]), color='blue', linestyle='-', lw=2, label='Posterior Mean')
        ax.set_xlabel(f'β{i}')
        ax.set_ylabel('Density')
        ax.set_title(f'Posterior Distribution: Coefficient {i}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/bayesian_cox.png', dpi=150, bbox_inches='tight')
    print("\nSaved: Cox model visualization")
    plt.close()


def advanced_survival_analysis():
    """Additional survival analysis techniques."""
    print("\n" + "=" * 80)
    print("ADVANCED SURVIVAL ANALYSIS")
    print("=" * 80)
    
    # Generate more complex data
    np.random.seed(123)
    n = 300
    
    # Multiple risk groups
    groups = np.random.choice([0, 1, 2], n)
    scales = np.array([8, 12, 16])[groups]
    
    times = np.random.weibull(1.5, n) * scales
    censor_times = np.random.exponential(20, n)
    
    obs_times = np.minimum(times, censor_times)
    events = (times <= censor_times).astype(int)
    
    # Kaplan-Meier by group
    for g in range(3):
        mask = groups == g
        t_g, km_g = kaplan_meier_estimator(obs_times[mask], events[mask])
        print(f"\nGroup {g}: {np.sum(mask)} patients, {np.sum(events[mask])} events")
    
    # Visualize KM curves
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['blue', 'red', 'green']
    for g in range(3):
        mask = groups == g
        t_g, km_g = kaplan_meier_estimator(obs_times[mask], events[mask])
        
        # Plot step function
        ax.step(t_g, km_g, where='post', lw=2, label=f'Group {g}', color=colors[g])
        ax.scatter(t_g, km_g, s=30, color=colors[g], alpha=0.7)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Survival Probability')
    ax.set_title('Kaplan-Meier Survival Curves by Group')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig('/tmp/km_curves.png', dpi=150, bbox_inches='tight')
    print("\nSaved: Kaplan-Meier curves")
    plt.close()
