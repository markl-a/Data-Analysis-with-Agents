"""
Differential Evolution Optimization
===================================

This solution implements Differential Evolution (DE), a powerful evolutionary
algorithm for continuous optimization.

Mathematical Background:
-----------------------
Differential Evolution uses mutation, crossover, and selection:

1. Mutation: Create mutant vector
   v_i = x_r1 + F * (x_r2 - x_r3)

2. Crossover: Create trial vector
   u_i,j = v_i,j if rand() < CR or j = jrand
           x_i,j otherwise

3. Selection: Greedy selection
   x_i(t+1) = u_i if f(u_i) < f(x_i)
              x_i otherwise

Variants:
- DE/rand/1: v = x_r1 + F*(x_r2 - x_r3)
- DE/best/1: v = x_best + F*(x_r1 - x_r2)
- DE/rand/2: v = x_r1 + F*(x_r2 - x_r3) + F*(x_r4 - x_r5)

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


class DifferentialEvolution:
    """Differential Evolution optimizer."""

    def __init__(self, objective: Callable, bounds: List[Tuple[float, float]]):
        """
        Initialize DE.

        Args:
            objective: Objective function to minimize
            bounds: List of (min, max) bounds for each dimension
        """
        self.objective = objective
        self.bounds = np.array(bounds)
        self.n_dim = len(bounds)
        self.history = []

    def optimize(self, population_size: int = 50, max_iterations: int = 100,
                F: float = 0.8, CR: float = 0.9,
                strategy: str = 'rand1') -> Dict:
        """
        Run DE algorithm.

        Args:
            population_size: Number of individuals (usually 10*dim)
            max_iterations: Maximum iterations
            F: Differential weight (mutation factor)
            CR: Crossover probability
            strategy: 'rand1', 'best1', 'rand2', 'best2', 'currenttobest1'

        Returns:
            Optimization results
        """
        # Initialize population
        population = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            (population_size, self.n_dim)
        )

        # Evaluate initial population
        fitness = np.array([self.objective(ind) for ind in population])

        # Track best
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        for iteration in range(max_iterations):
            # For each individual
            for i in range(population_size):
                # Mutation
                mutant = self._mutate(population, fitness, i, F, strategy)

                # Crossover
                trial = self._crossover(population[i], mutant, CR)

                # Bounds enforcement
                trial = np.clip(trial, self.bounds[:, 0], self.bounds[:, 1])

                # Selection
                trial_fitness = self.objective(trial)

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update best
                    if trial_fitness < best_fitness:
                        best_solution = trial.copy()
                        best_fitness = trial_fitness

            # Store history
            self.history.append({
                'iteration': iteration,
                'best_fitness': best_fitness,
                'mean_fitness': np.mean(fitness),
                'std_fitness': np.std(fitness),
                'population_diversity': np.mean(np.std(population, axis=0))
            })

        return {
            'best_solution': best_solution,
            'best_fitness': best_fitness,
            'final_population': population,
            'history': self.history
        }

    def _mutate(self, population: np.ndarray, fitness: np.ndarray,
               current_idx: int, F: float, strategy: str) -> np.ndarray:
        """Generate mutant vector."""
        pop_size = len(population)

        if strategy == 'rand1':
            # DE/rand/1: v = x_r1 + F*(x_r2 - x_r3)
            indices = np.random.choice([i for i in range(pop_size) if i != current_idx], 3, replace=False)
            r1, r2, r3 = indices
            mutant = population[r1] + F * (population[r2] - population[r3])

        elif strategy == 'best1':
            # DE/best/1: v = x_best + F*(x_r1 - x_r2)
            best_idx = np.argmin(fitness)
            indices = np.random.choice([i for i in range(pop_size) if i != current_idx], 2, replace=False)
            r1, r2 = indices
            mutant = population[best_idx] + F * (population[r1] - population[r2])

        elif strategy == 'rand2':
            # DE/rand/2: v = x_r1 + F*(x_r2 - x_r3) + F*(x_r4 - x_r5)
            indices = np.random.choice([i for i in range(pop_size) if i != current_idx], 5, replace=False)
            r1, r2, r3, r4, r5 = indices
            mutant = population[r1] + F * (population[r2] - population[r3]) + F * (population[r4] - population[r5])

        elif strategy == 'currenttobest1':
            # DE/current-to-best/1: v = x_i + F*(x_best - x_i) + F*(x_r1 - x_r2)
            best_idx = np.argmin(fitness)
            indices = np.random.choice([i for i in range(pop_size) if i != current_idx], 2, replace=False)
            r1, r2 = indices
            mutant = population[current_idx] + F * (population[best_idx] - population[current_idx]) + F * (population[r1] - population[r2])

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return mutant

    def _crossover(self, target: np.ndarray, mutant: np.ndarray, CR: float) -> np.ndarray:
        """Perform binomial crossover."""
        trial = target.copy()
        j_rand = np.random.randint(self.n_dim)  # Ensure at least one component from mutant

        for j in range(self.n_dim):
            if np.random.random() < CR or j == j_rand:
                trial[j] = mutant[j]

        return trial


class AdaptiveDE:
    """Adaptive Differential Evolution with self-adapting parameters."""

    def __init__(self, objective: Callable, bounds: List[Tuple[float, float]]):
        self.objective = objective
        self.bounds = np.array(bounds)
        self.n_dim = len(bounds)
        self.history = []

    def optimize(self, population_size: int = 50, max_iterations: int = 100) -> Dict:
        """Run adaptive DE."""
        # Initialize population
        population = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            (population_size, self.n_dim)
        )

        # Initialize parameters for each individual
        F_values = np.random.uniform(0.5, 1.0, population_size)
        CR_values = np.random.uniform(0.0, 1.0, population_size)

        fitness = np.array([self.objective(ind) for ind in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        for iteration in range(max_iterations):
            successful_F = []
            successful_CR = []

            for i in range(population_size):
                # Mutate parameters
                F_i = np.clip(F_values[i] + 0.1 * np.random.randn(), 0.1, 1.0)
                CR_i = np.clip(CR_values[i] + 0.1 * np.random.randn(), 0.0, 1.0)

                # Generate mutant and trial
                indices = np.random.choice([j for j in range(population_size) if j != i], 3, replace=False)
                r1, r2, r3 = indices
                mutant = population[r1] + F_i * (population[r2] - population[r3])

                trial = population[i].copy()
                j_rand = np.random.randint(self.n_dim)
                for j in range(self.n_dim):
                    if np.random.random() < CR_i or j == j_rand:
                        trial[j] = mutant[j]

                trial = np.clip(trial, self.bounds[:, 0], self.bounds[:, 1])

                trial_fitness = self.objective(trial)

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Store successful parameters
                    successful_F.append(F_i)
                    successful_CR.append(CR_i)

                    # Update parameters
                    F_values[i] = F_i
                    CR_values[i] = CR_i

                    if trial_fitness < best_fitness:
                        best_solution = trial.copy()
                        best_fitness = trial_fitness

            # Update parameter distributions
            if successful_F:
                mean_F = np.mean(successful_F)
                mean_CR = np.mean(successful_CR)

                for i in range(population_size):
                    F_values[i] = 0.9 * F_values[i] + 0.1 * mean_F
                    CR_values[i] = 0.9 * CR_values[i] + 0.1 * mean_CR

            self.history.append({
                'iteration': iteration,
                'best_fitness': best_fitness,
                'mean_F': np.mean(F_values),
                'mean_CR': np.mean(CR_values)
            })

        return {
            'best_solution': best_solution,
            'best_fitness': best_fitness,
            'history': self.history
        }


def benchmark_de():
    """Benchmark DE on standard test functions."""

    def sphere(x):
        return np.sum(x**2)

    def rastrigin(x):
        return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

    def rosenbrock(x):
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    def schwefel(x):
        return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

    functions = {
        'Sphere': (sphere, [(-10, 10)] * 10),
        'Rastrigin': (rastrigin, [(-5.12, 5.12)] * 10),
        'Rosenbrock': (rosenbrock, [(-5, 10)] * 10),
        'Schwefel': (schwefel, [(-500, 500)] * 10)
    }

    results = []

    for name, (func, bounds) in functions.items():
        de = DifferentialEvolution(func, bounds)
        result = de.optimize(population_size=50, max_iterations=200)

        results.append({
            'function': name,
            'best_fitness': result['best_fitness'],
            'iterations': len(result['history'])
        })

        print(f"{name}:")
        print(f"  Best fitness: {result['best_fitness']:.6e}")

    return pd.DataFrame(results)


def compare_de_strategies():
    """Compare different DE strategies."""

    def rastrigin(x):
        return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

    bounds = [(-5.12, 5.12)] * 5

    strategies = ['rand1', 'best1', 'rand2', 'currenttobest1']
    results = {}

    for strategy in strategies:
        de = DifferentialEvolution(rastrigin, bounds)
        result = de.optimize(population_size=50, max_iterations=100, strategy=strategy)
        results[strategy] = result

    # Visualize comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    for strategy, result in results.items():
        history_df = pd.DataFrame(result['history'])
        ax.semilogy(history_df['iteration'], history_df['best_fitness'],
                   linewidth=2, label=strategy)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Best Fitness (log)', fontsize=12)
    ax.set_title('DE Strategies Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('de_strategies_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    return results


def visualize_de_convergence():
    """Visualize DE convergence."""

    def rastrigin(x):
        return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

    bounds = [(-5.12, 5.12)] * 2

    de = DifferentialEvolution(rastrigin, bounds)
    result = de.optimize(population_size=30, max_iterations=100, F=0.8, CR=0.9)

    history_df = pd.DataFrame(result['history'])

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Fitness evolution
    axes[0, 0].semilogy(history_df['iteration'], history_df['best_fitness'],
                       'b-', linewidth=2, label='Best')
    axes[0, 0].semilogy(history_df['iteration'], history_df['mean_fitness'],
                       'r--', linewidth=2, label='Mean')
    axes[0, 0].fill_between(history_df['iteration'],
                           history_df['mean_fitness'] - history_df['std_fitness'],
                           history_df['mean_fitness'] + history_df['std_fitness'],
                           alpha=0.3)
    axes[0, 0].set_xlabel('Iteration', fontsize=12)
    axes[0, 0].set_ylabel('Fitness (log)', fontsize=12)
    axes[0, 0].set_title('Fitness Evolution', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Population diversity
    axes[0, 1].plot(history_df['iteration'], history_df['population_diversity'],
                   'g-', linewidth=2)
    axes[0, 1].set_xlabel('Iteration', fontsize=12)
    axes[0, 1].set_ylabel('Diversity', fontsize=12)
    axes[0, 1].set_title('Population Diversity', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Fitness landscape with final population
    x1 = np.linspace(-5.12, 5.12, 100)
    x2 = np.linspace(-5.12, 5.12, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = rastrigin(np.array([X1[i, j], X2[i, j]]))

    contour = axes[1, 0].contourf(X1, X2, Z, levels=30, cmap='viridis', alpha=0.6)
    plt.colorbar(contour, ax=axes[1, 0])

    final_pop = result['final_population']
    axes[1, 0].scatter(final_pop[:, 0], final_pop[:, 1],
                      c='red', s=100, alpha=0.7, edgecolors='black')
    axes[1, 0].plot(result['best_solution'][0], result['best_solution'][1],
                   'y*', markersize=20, label='Best')
    axes[1, 0].set_xlabel('x1', fontsize=12)
    axes[1, 0].set_ylabel('x2', fontsize=12)
    axes[1, 0].set_title('Final Population', fontsize=14, fontweight='bold')
    axes[1, 0].legend()

    # Parameter sensitivity
    F_values = np.linspace(0.1, 2.0, 15)
    best_fitnesses = []

    for F in F_values:
        de_temp = DifferentialEvolution(rastrigin, bounds)
        res = de_temp.optimize(population_size=30, max_iterations=50, F=F, CR=0.9)
        best_fitnesses.append(res['best_fitness'])

    axes[1, 1].semilogy(F_values, best_fitnesses, 'bo-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('F (Mutation Factor)', fontsize=12)
    axes[1, 1].set_ylabel('Best Fitness (log)', fontsize=12)
    axes[1, 1].set_title('Parameter Sensitivity', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('de_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main execution function."""
    print("="*70)
    print("Differential Evolution Optimization")
    print("="*70)

    # Example 1: Benchmark
    print("\n1. Benchmark on Test Functions")
    print("-" * 70)
    benchmark_results = benchmark_de()

    # Example 2: Strategy comparison
    print("\n2. DE Strategies Comparison")
    print("-" * 70)
    strategy_results = compare_de_strategies()
    for strategy, result in strategy_results.items():
        print(f"{strategy}: {result['best_fitness']:.6e}")
    print("Comparison plot saved to 'de_strategies_comparison.png'")

    # Example 3: Convergence visualization
    print("\n3. DE Convergence Visualization")
    print("-" * 70)
    visualize_de_convergence()
    print("Convergence plot saved to 'de_convergence.png'")

    # Example 4: Adaptive DE
    print("\n4. Adaptive DE")
    print("-" * 70)

    def rastrigin(x):
        return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

    ade = AdaptiveDE(rastrigin, [(-5.12, 5.12)] * 10)
    ade_result = ade.optimize(population_size=50, max_iterations=200)
    print(f"Best fitness: {ade_result['best_fitness']:.6e}")

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
