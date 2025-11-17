"""
Particle Swarm Optimization (PSO)
=================================

This solution implements Particle Swarm Optimization, a metaheuristic inspired
by the social behavior of birds flocking and fish schooling.

Mathematical Background:
-----------------------
PSO maintains a swarm of particles moving through the search space. Each particle:
- Has a position x_i (candidate solution)
- Has a velocity v_i (direction and magnitude of movement)
- Remembers its personal best position p_i
- Knows the global best position g of the swarm

Update equations:
    v_i(t+1) = w*v_i(t) + c1*r1*(p_i - x_i(t)) + c2*r2*(g - x_i(t))
    x_i(t+1) = x_i(t) + v_i(t+1)

where:
- w: inertia weight (exploration vs exploitation)
- c1, c2: cognitive and social coefficients
- r1, r2: random numbers in [0,1]

Author: Optimization Expert
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Callable, List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class ParticleSwarmOptimizer:
    """Particle Swarm Optimization implementation."""

    def __init__(self, objective: Callable, n_dim: int, bounds: List[Tuple[float, float]]):
        """
        Initialize PSO.

        Args:
            objective: Objective function to minimize
            n_dim: Number of dimensions
            bounds: List of (min, max) bounds for each dimension
        """
        self.objective = objective
        self.n_dim = n_dim
        self.bounds = np.array(bounds)
        self.history = []

    def optimize(self, n_particles: int = 30, n_iterations: int = 100,
                w: float = 0.729, c1: float = 1.49445, c2: float = 1.49445,
                adaptive_inertia: bool = False) -> Dict:
        """
        Run PSO algorithm.

        Args:
            n_particles: Number of particles in swarm
            n_iterations: Number of iterations
            w: Inertia weight (or initial weight if adaptive)
            c1: Cognitive coefficient (personal best influence)
            c2: Social coefficient (global best influence)
            adaptive_inertia: Use linearly decreasing inertia weight

        Returns:
            Optimization results
        """
        # Initialize particles
        positions = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                     (n_particles, self.n_dim))

        velocities = np.random.uniform(-1, 1, (n_particles, self.n_dim))

        # Initialize personal bests
        personal_best_positions = positions.copy()
        personal_best_scores = np.array([self.objective(p) for p in positions])

        # Initialize global best
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]

        # Main loop
        for iteration in range(n_iterations):
            # Adaptive inertia weight
            if adaptive_inertia:
                w_current = w - (w - 0.4) * iteration / n_iterations
            else:
                w_current = w

            # Update each particle
            for i in range(n_particles):
                # Random factors
                r1 = np.random.random(self.n_dim)
                r2 = np.random.random(self.n_dim)

                # Update velocity
                cognitive = c1 * r1 * (personal_best_positions[i] - positions[i])
                social = c2 * r2 * (global_best_position - positions[i])
                velocities[i] = w_current * velocities[i] + cognitive + social

                # Update position
                positions[i] += velocities[i]

                # Apply bounds
                positions[i] = np.clip(positions[i], self.bounds[:, 0], self.bounds[:, 1])

                # Evaluate
                score = self.objective(positions[i])

                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i].copy()

                    # Update global best
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = positions[i].copy()

            # Store history
            self.history.append({
                'iteration': iteration,
                'global_best_score': global_best_score,
                'mean_score': np.mean(personal_best_scores),
                'swarm_diversity': np.mean(np.std(positions, axis=0))
            })

        return {
            'best_position': global_best_position,
            'best_score': global_best_score,
            'history': self.history,
            'final_positions': positions
        }


class GPSO:
    """Guaranteed Convergence PSO variant."""

    def __init__(self, objective: Callable, n_dim: int, bounds: List[Tuple[float, float]]):
        self.objective = objective
        self.n_dim = n_dim
        self.bounds = np.array(bounds)
        self.history = []

    def optimize(self, n_particles: int = 30, n_iterations: int = 100) -> Dict:
        """Run GPSO with guaranteed convergence."""
        # Initialize particles
        positions = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                     (n_particles, self.n_dim))
        velocities = np.zeros((n_particles, self.n_dim))

        # Personal and global bests
        p_best = positions.copy()
        p_best_scores = np.array([self.objective(p) for p in positions])
        g_best_idx = np.argmin(p_best_scores)
        g_best = p_best[g_best_idx].copy()
        g_best_score = p_best_scores[g_best_idx]

        success_count = 0
        failure_count = 0
        rho = 1.0

        for iteration in range(n_iterations):
            for i in range(n_particles):
                if i == g_best_idx:
                    # Special update for global best particle
                    positions[i] = g_best + rho * np.random.normal(0, 1, self.n_dim)
                else:
                    # Standard PSO update
                    r1 = np.random.random(self.n_dim)
                    r2 = np.random.random(self.n_dim)
                    velocities[i] = (0.729 * velocities[i] +
                                   1.49445 * r1 * (p_best[i] - positions[i]) +
                                   1.49445 * r2 * (g_best - positions[i]))
                    positions[i] += velocities[i]

                # Apply bounds
                positions[i] = np.clip(positions[i], self.bounds[:, 0], self.bounds[:, 1])

                # Evaluate
                score = self.objective(positions[i])

                # Update personal best
                if score < p_best_scores[i]:
                    p_best_scores[i] = score
                    p_best[i] = positions[i].copy()

                    if score < g_best_score:
                        g_best_score = score
                        g_best = positions[i].copy()
                        g_best_idx = i
                        success_count += 1
                        failure_count = 0
                    else:
                        failure_count += 1

            # Adjust rho based on success/failure
            if success_count > 5:
                rho *= 2
                success_count = 0
            elif failure_count > 5:
                rho *= 0.5
                failure_count = 0

            self.history.append({
                'iteration': iteration,
                'global_best_score': g_best_score,
                'rho': rho
            })

        return {
            'best_position': g_best,
            'best_score': g_best_score,
            'history': self.history
        }


def benchmark_pso():
    """Benchmark PSO on standard test functions."""

    def sphere(x):
        return np.sum(x**2)

    def rastrigin(x):
        return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

    def rosenbrock(x):
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    def ackley(x):
        n = len(x)
        return (-20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n)) -
                np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e)

    functions = {
        'Sphere': (sphere, [(-10, 10)] * 5),
        'Rastrigin': (rastrigin, [(-5.12, 5.12)] * 5),
        'Rosenbrock': (rosenbrock, [(-5, 10)] * 5),
        'Ackley': (ackley, [(-32, 32)] * 5)
    }

    results = []

    for name, (func, bounds) in functions.items():
        pso = ParticleSwarmOptimizer(func, len(bounds), bounds)
        result = pso.optimize(n_particles=30, n_iterations=100)

        results.append({
            'function': name,
            'best_score': result['best_score'],
            'best_position': result['best_position'],
            'iterations': len(result['history'])
        })

        print(f"{name}:")
        print(f"  Best score: {result['best_score']:.6e}")
        print(f"  Best position: {result['best_position']}")

    return pd.DataFrame(results)


def visualize_pso_convergence():
    """Visualize PSO convergence on 2D function."""

    def rastrigin_2d(x):
        return 10 * 2 + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

    bounds = [(-5.12, 5.12), (-5.12, 5.12)]
    pso = ParticleSwarmOptimizer(rastrigin_2d, 2, bounds)

    result = pso.optimize(n_particles=20, n_iterations=50, adaptive_inertia=True)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Convergence plot
    history_df = pd.DataFrame(result['history'])
    axes[0, 0].semilogy(history_df['iteration'], history_df['global_best_score'],
                       'b-', linewidth=2, label='Global Best')
    axes[0, 0].semilogy(history_df['iteration'], history_df['mean_score'],
                       'r--', linewidth=2, label='Mean Score')
    axes[0, 0].set_xlabel('Iteration', fontsize=12)
    axes[0, 0].set_ylabel('Objective Value (log)', fontsize=12)
    axes[0, 0].set_title('PSO Convergence', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Swarm diversity
    axes[0, 1].plot(history_df['iteration'], history_df['swarm_diversity'],
                   'g-', linewidth=2)
    axes[0, 1].set_xlabel('Iteration', fontsize=12)
    axes[0, 1].set_ylabel('Swarm Diversity', fontsize=12)
    axes[0, 1].set_title('Population Diversity', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Fitness landscape
    x1 = np.linspace(-5.12, 5.12, 100)
    x2 = np.linspace(-5.12, 5.12, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = rastrigin_2d(np.array([X1[i, j], X2[i, j]]))

    contour = axes[1, 0].contourf(X1, X2, Z, levels=30, cmap='viridis', alpha=0.6)
    plt.colorbar(contour, ax=axes[1, 0])

    # Plot final swarm
    final_pos = result['final_positions']
    axes[1, 0].scatter(final_pos[:, 0], final_pos[:, 1],
                      c='red', s=100, alpha=0.7, edgecolors='black')
    axes[1, 0].plot(result['best_position'][0], result['best_position'][1],
                   'y*', markersize=20, label='Best Position')
    axes[1, 0].set_xlabel('x1', fontsize=12)
    axes[1, 0].set_ylabel('x2', fontsize=12)
    axes[1, 0].set_title('Final Swarm Configuration', fontsize=14, fontweight='bold')
    axes[1, 0].legend()

    # Parameter sensitivity
    inertia_values = np.linspace(0.4, 0.9, 10)
    best_scores = []

    for w in inertia_values:
        pso_temp = ParticleSwarmOptimizer(rastrigin_2d, 2, bounds)
        res = pso_temp.optimize(n_particles=20, n_iterations=50, w=w)
        best_scores.append(res['best_score'])

    axes[1, 1].plot(inertia_values, best_scores, 'bo-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Inertia Weight (w)', fontsize=12)
    axes[1, 1].set_ylabel('Best Score', fontsize=12)
    axes[1, 1].set_title('Parameter Sensitivity Analysis', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pso_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()


def compare_pso_variants():
    """Compare standard PSO with variants."""

    def rastrigin(x):
        return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

    bounds = [(-5.12, 5.12)] * 5

    # Standard PSO
    pso_standard = ParticleSwarmOptimizer(rastrigin, 5, bounds)
    result_standard = pso_standard.optimize(n_particles=30, n_iterations=100,
                                           adaptive_inertia=False)

    # Adaptive inertia PSO
    pso_adaptive = ParticleSwarmOptimizer(rastrigin, 5, bounds)
    result_adaptive = pso_adaptive.optimize(n_particles=30, n_iterations=100,
                                           adaptive_inertia=True)

    # GPSO
    gpso = GPSO(rastrigin, 5, bounds)
    result_gpso = gpso.optimize(n_particles=30, n_iterations=100)

    # Visualize comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    history_standard = pd.DataFrame(result_standard['history'])
    history_adaptive = pd.DataFrame(result_adaptive['history'])
    history_gpso = pd.DataFrame(result_gpso['history'])

    ax.semilogy(history_standard['iteration'], history_standard['global_best_score'],
               'b-', linewidth=2, label='Standard PSO')
    ax.semilogy(history_adaptive['iteration'], history_adaptive['global_best_score'],
               'r-', linewidth=2, label='Adaptive Inertia PSO')
    ax.semilogy(history_gpso['iteration'], history_gpso['global_best_score'],
               'g-', linewidth=2, label='GPSO')

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Best Score (log)', fontsize=12)
    ax.set_title('PSO Variants Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pso_variants_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'standard': result_standard,
        'adaptive': result_adaptive,
        'gpso': result_gpso
    }


def main():
    """Main execution function."""
    print("="*70)
    print("Particle Swarm Optimization (PSO)")
    print("="*70)

    # Example 1: Benchmark
    print("\n1. Benchmark on Test Functions")
    print("-" * 70)
    benchmark_results = benchmark_pso()

    # Example 2: Visualization
    print("\n2. PSO Convergence Visualization")
    print("-" * 70)
    visualize_pso_convergence()
    print("Convergence plot saved to 'pso_convergence.png'")

    # Example 3: Variant comparison
    print("\n3. PSO Variants Comparison")
    print("-" * 70)
    variant_results = compare_pso_variants()
    print(f"Standard PSO: {variant_results['standard']['best_score']:.6e}")
    print(f"Adaptive PSO: {variant_results['adaptive']['best_score']:.6e}")
    print(f"GPSO: {variant_results['gpso']['best_score']:.6e}")
    print("Comparison plot saved to 'pso_variants_comparison.png'")

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()


def advanced_pso_variants():
    """Demonstrate advanced PSO variants."""
    
    def ackley(x):
        n = len(x)
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(2 * np.pi * x))
        return -20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20 + np.e
    
    bounds = [(-32, 32)] * 5
    
    # Constriction PSO
    chi = 0.729
    c1 = c2 = 2.05
    
    pso_constriction = ParticleSwarmOptimizer(ackley, 5, bounds)
    result_constriction = pso_constriction.optimize(n_particles=30, n_iterations=100, 
                                                   w=chi, c1=chi*c1, c2=chi*c2)
    
    print(f"Constriction PSO best: {result_constriction['best_score']:.6e}")
    
    # Comprehensive Comparative Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Performance comparison across dimensions
    dimensions = [2, 5, 10, 20]
    scores_by_dim = []
    
    for dim in dimensions:
        bounds_dim = [(-32, 32)] * dim
        pso_dim = ParticleSwarmOptimizer(ackley, dim, bounds_dim)
        result_dim = pso_dim.optimize(n_particles=dim*10, n_iterations=100)
        scores_by_dim.append(result_dim['best_score'])
    
    axes[0, 0].semilogy(dimensions, scores_by_dim, 'bo-', linewidth=2, markersize=10)
    axes[0, 0].set_xlabel('Problem Dimension', fontsize=12)
    axes[0, 0].set_ylabel('Best Score (log)', fontsize=12)
    axes[0, 0].set_title('Scalability Analysis', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Swarm size effect
    swarm_sizes = [10, 20, 30, 50, 100]
    scores_by_swarm = []
    
    for size in swarm_sizes:
        pso_size = ParticleSwarmOptimizer(ackley, 5, bounds)
        result_size = pso_size.optimize(n_particles=size, n_iterations=100)
        scores_by_swarm.append(result_size['best_score'])
    
    axes[0, 1].semilogy(swarm_sizes, scores_by_swarm, 'ro-', linewidth=2, markersize=10)
    axes[0, 1].set_xlabel('Swarm Size', fontsize=12)
    axes[0, 1].set_ylabel('Best Score (log)', fontsize=12)
    axes[0, 1].set_title('Swarm Size Effect', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Inertia weight comparison
    weights = np.linspace(0.4, 0.9, 10)
    scores_by_weight = []
    
    for w in weights:
        pso_w = ParticleSwarmOptimizer(ackley, 5, bounds)
        result_w = pso_w.optimize(n_particles=30, n_iterations=50, w=w)
        scores_by_weight.append(result_w['best_score'])
    
    axes[1, 0].plot(weights, scores_by_weight, 'go-', linewidth=2, markersize=10)
    axes[1, 0].set_xlabel('Inertia Weight (w)', fontsize=12)
    axes[1, 0].set_ylabel('Best Score', fontsize=12)
    axes[1, 0].set_title('Inertia Weight Effect', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary statistics
    axes[1, 1].axis('off')
    summary_text = f"""
    PSO Performance Summary
    ======================
    
    Best Score: {result_constriction['best_score']:.6e}
    
    Recommended Parameters:
    - Inertia weight: 0.729
    - Cognitive coeff: 1.49445
    - Social coeff: 1.49445
    
    Scalability: Good up to ~20D
    Optimal swarm size: 10-50 particles
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                   verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('pso_advanced_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


