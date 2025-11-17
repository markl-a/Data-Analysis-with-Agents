"""
Genetic Algorithm for Optimization
==================================

This solution implements comprehensive genetic algorithms (GA) for both
continuous and discrete optimization problems.

Mathematical Background:
-----------------------
Genetic Algorithms are inspired by natural evolution:
1. Population: Set of candidate solutions (chromosomes)
2. Fitness: Quality measure for each solution
3. Selection: Choose parents based on fitness
4. Crossover: Combine parents to create offspring
5. Mutation: Random changes to maintain diversity
6. Replacement: Form new generation

Key components:
- Encoding (binary, real-valued, permutation)
- Selection operators (tournament, roulette wheel, rank)
- Crossover operators (one-point, two-point, uniform, arithmetic)
- Mutation operators (bit-flip, gaussian, swap)

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


class GeneticAlgorithm:
    """
    Generic Genetic Algorithm implementation.
    """

    def __init__(self, fitness_func: Callable, n_genes: int,
                 gene_type: str = 'real', bounds: Optional[List[Tuple]] = None):
        """
        Initialize GA.

        Args:
            fitness_func: Fitness function (higher is better)
            n_genes: Number of genes (dimension)
            gene_type: 'binary', 'real', or 'permutation'
            bounds: List of (min, max) tuples for each gene (real-valued only)
        """
        self.fitness_func = fitness_func
        self.n_genes = n_genes
        self.gene_type = gene_type
        self.bounds = bounds if bounds else [(0, 1)] * n_genes
        self.history = []

    def optimize(self, population_size: int = 100, n_generations: int = 100,
                crossover_rate: float = 0.8, mutation_rate: float = 0.1,
                selection_method: str = 'tournament',
                elitism: int = 2) -> Dict:
        """
        Run genetic algorithm.

        Args:
            population_size: Number of individuals
            n_generations: Number of generations
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            selection_method: 'tournament', 'roulette', or 'rank'
            elitism: Number of best individuals to keep

        Returns:
            Optimization results
        """
        # Initialize population
        population = self._initialize_population(population_size)

        best_fitness_history = []
        mean_fitness_history = []
        diversity_history = []

        for generation in range(n_generations):
            # Evaluate fitness
            fitness = np.array([self.fitness_func(ind) for ind in population])

            # Store statistics
            best_idx = np.argmax(fitness)
            best_fitness_history.append(fitness[best_idx])
            mean_fitness_history.append(np.mean(fitness))
            diversity_history.append(self._calculate_diversity(population))

            self.history.append({
                'generation': generation,
                'best_fitness': fitness[best_idx],
                'mean_fitness': np.mean(fitness),
                'worst_fitness': np.min(fitness),
                'diversity': diversity_history[-1]
            })

            # Elitism: keep best individuals
            elite_indices = np.argsort(fitness)[-elitism:]
            elite = [population[i].copy() for i in elite_indices]

            # Create new population
            new_population = elite.copy()

            while len(new_population) < population_size:
                # Selection
                parent1 = self._select(population, fitness, selection_method)
                parent2 = self._select(population, fitness, selection_method)

                # Crossover
                if np.random.random() < crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Mutation
                if np.random.random() < mutation_rate:
                    child1 = self._mutate(child1)
                if np.random.random() < mutation_rate:
                    child2 = self._mutate(child2)

                new_population.extend([child1, child2])

            population = new_population[:population_size]

        # Final evaluation
        fitness = np.array([self.fitness_func(ind) for ind in population])
        best_idx = np.argmax(fitness)

        return {
            'best_solution': population[best_idx],
            'best_fitness': fitness[best_idx],
            'history': self.history,
            'final_population': population
        }

    def _initialize_population(self, size: int) -> List[np.ndarray]:
        """Initialize random population."""
        population = []

        for _ in range(size):
            if self.gene_type == 'binary':
                individual = np.random.randint(0, 2, self.n_genes)
            elif self.gene_type == 'real':
                individual = np.array([
                    np.random.uniform(low, high)
                    for low, high in self.bounds
                ])
            elif self.gene_type == 'permutation':
                individual = np.random.permutation(self.n_genes)
            else:
                raise ValueError(f"Unknown gene type: {self.gene_type}")

            population.append(individual)

        return population

    def _select(self, population: List[np.ndarray], fitness: np.ndarray,
               method: str) -> np.ndarray:
        """Select individual based on fitness."""
        if method == 'tournament':
            tournament_size = 3
            indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = fitness[indices]
            winner_idx = indices[np.argmax(tournament_fitness)]
            return population[winner_idx].copy()

        elif method == 'roulette':
            # Ensure non-negative fitness
            min_fitness = np.min(fitness)
            adjusted_fitness = fitness - min_fitness + 1e-10
            probabilities = adjusted_fitness / np.sum(adjusted_fitness)
            idx = np.random.choice(len(population), p=probabilities)
            return population[idx].copy()

        elif method == 'rank':
            ranks = np.argsort(np.argsort(fitness)) + 1
            probabilities = ranks / np.sum(ranks)
            idx = np.random.choice(len(population), p=probabilities)
            return population[idx].copy()

        else:
            raise ValueError(f"Unknown selection method: {method}")

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform crossover."""
        if self.gene_type == 'binary' or self.gene_type == 'real':
            # Uniform crossover
            mask = np.random.random(self.n_genes) < 0.5
            child1 = np.where(mask, parent1, parent2)
            child2 = np.where(mask, parent2, parent1)

        elif self.gene_type == 'permutation':
            # Order crossover (OX)
            child1, child2 = self._order_crossover(parent1, parent2)

        else:
            child1, child2 = parent1.copy(), parent2.copy()

        return child1, child2

    def _order_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Order crossover for permutations."""
        n = len(parent1)
        point1, point2 = sorted(np.random.choice(n, 2, replace=False))

        # Create children
        child1 = -np.ones(n, dtype=int)
        child2 = -np.ones(n, dtype=int)

        # Copy segment
        child1[point1:point2] = parent1[point1:point2]
        child2[point1:point2] = parent2[point1:point2]

        # Fill remaining positions
        self._fill_child(child1, parent2, point2)
        self._fill_child(child2, parent1, point2)

        return child1, child2

    def _fill_child(self, child: np.ndarray, parent: np.ndarray, start_pos: int):
        """Helper for order crossover."""
        n = len(child)
        pos = start_pos
        for val in np.roll(parent, -start_pos):
            if val not in child:
                child[pos % n] = val
                pos += 1

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Perform mutation."""
        mutated = individual.copy()

        if self.gene_type == 'binary':
            # Bit flip mutation
            idx = np.random.randint(0, self.n_genes)
            mutated[idx] = 1 - mutated[idx]

        elif self.gene_type == 'real':
            # Gaussian mutation
            idx = np.random.randint(0, self.n_genes)
            sigma = (self.bounds[idx][1] - self.bounds[idx][0]) * 0.1
            mutated[idx] += np.random.normal(0, sigma)
            mutated[idx] = np.clip(mutated[idx], self.bounds[idx][0], self.bounds[idx][1])

        elif self.gene_type == 'permutation':
            # Swap mutation
            idx1, idx2 = np.random.choice(self.n_genes, 2, replace=False)
            mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]

        return mutated

    def _calculate_diversity(self, population: List[np.ndarray]) -> float:
        """Calculate population diversity."""
        if self.gene_type == 'permutation':
            return 1.0  # Simplified for permutations

        pop_array = np.array(population)
        return np.mean(np.std(pop_array, axis=0))


def rastrigin_function(x: np.ndarray) -> float:
    """
    Rastrigin function (minimization problem, convert to maximization).
    Global minimum at x = 0 with f(0) = 0
    """
    n = len(x)
    A = 10
    return -(A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x)))


def rosenbrock_function(x: np.ndarray) -> float:
    """Rosenbrock function (minimization, convert to maximization)."""
    return -np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def sphere_function(x: np.ndarray) -> float:
    """Sphere function (minimization, convert to maximization)."""
    return -np.sum(x**2)


def benchmark_ga_on_functions():
    """Benchmark GA on standard test functions."""
    test_functions = {
        'Rastrigin': (rastrigin_function, [(-5.12, 5.12)] * 5),
        'Rosenbrock': (rosenbrock_function, [(-5, 10)] * 5),
        'Sphere': (sphere_function, [(-10, 10)] * 5)
    }

    results = []

    for func_name, (func, bounds) in test_functions.items():
        print(f"\nOptimizing {func_name} function...")

        ga = GeneticAlgorithm(func, n_genes=len(bounds),
                            gene_type='real', bounds=bounds)

        result = ga.optimize(population_size=100, n_generations=200,
                           crossover_rate=0.8, mutation_rate=0.1)

        results.append({
            'function': func_name,
            'best_fitness': result['best_fitness'],
            'best_solution': result['best_solution'],
            'generations': len(result['history'])
        })

        print(f"  Best fitness: {result['best_fitness']:.6f}")
        print(f"  Best solution: {result['best_solution']}")

    return pd.DataFrame(results), test_functions


def visualize_ga_convergence():
    """Visualize GA convergence."""
    # Use Rastrigin function
    ga = GeneticAlgorithm(rastrigin_function, n_genes=2,
                        gene_type='real', bounds=[(-5.12, 5.12)] * 2)

    result = ga.optimize(population_size=50, n_generations=100)

    history_df = pd.DataFrame(result['history'])

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Fitness over generations
    axes[0, 0].plot(history_df['generation'], -history_df['best_fitness'], 'b-', linewidth=2, label='Best')
    axes[0, 0].plot(history_df['generation'], -history_df['mean_fitness'], 'g-', linewidth=2, label='Mean')
    axes[0, 0].plot(history_df['generation'], -history_df['worst_fitness'], 'r-', linewidth=2, label='Worst')
    axes[0, 0].set_xlabel('Generation', fontsize=12)
    axes[0, 0].set_ylabel('Fitness Value', fontsize=12)
    axes[0, 0].set_title('Fitness Evolution', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Diversity
    axes[0, 1].plot(history_df['generation'], history_df['diversity'], 'purple', linewidth=2)
    axes[0, 1].set_xlabel('Generation', fontsize=12)
    axes[0, 1].set_ylabel('Population Diversity', fontsize=12)
    axes[0, 1].set_title('Population Diversity Over Time', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Fitness landscape with final population
    x1 = np.linspace(-5.12, 5.12, 100)
    x2 = np.linspace(-5.12, 5.12, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = -rastrigin_function(np.array([X1[i, j], X2[i, j]]))

    contour = axes[1, 0].contourf(X1, X2, Z, levels=20, cmap='viridis', alpha=0.6)
    plt.colorbar(contour, ax=axes[1, 0])

    # Plot final population
    final_pop = np.array(result['final_population'])
    axes[1, 0].scatter(final_pop[:, 0], final_pop[:, 1], c='red', s=50, alpha=0.7, edgecolors='black')
    axes[1, 0].plot(result['best_solution'][0], result['best_solution'][1],
                   'r*', markersize=20, label='Best Solution')
    axes[1, 0].set_xlabel('x1', fontsize=12)
    axes[1, 0].set_ylabel('x2', fontsize=12)
    axes[1, 0].set_title('Final Population on Fitness Landscape', fontsize=14, fontweight='bold')
    axes[1, 0].legend()

    # Convergence rate (log scale)
    improvement = -history_df['best_fitness'] - (-history_df['best_fitness'].iloc[-1])
    axes[1, 1].semilogy(history_df['generation'], improvement + 1e-10, 'b-', linewidth=2)
    axes[1, 1].set_xlabel('Generation', fontsize=12)
    axes[1, 1].set_ylabel('Distance to Final Best (log)', fontsize=12)
    axes[1, 1].set_title('Convergence Rate', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('genetic_algorithm_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()


def compare_selection_methods():
    """Compare different selection methods."""
    methods = ['tournament', 'roulette', 'rank']
    results = []

    for method in methods:
        ga = GeneticAlgorithm(rastrigin_function, n_genes=5,
                            gene_type='real', bounds=[(-5.12, 5.12)] * 5)

        result = ga.optimize(population_size=50, n_generations=100,
                           selection_method=method)

        results.append({
            'method': method,
            'best_fitness': result['best_fitness'],
            'history': result['history']
        })

    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for res in results:
        history_df = pd.DataFrame(res['history'])
        axes[0].plot(history_df['generation'], -history_df['best_fitness'],
                    linewidth=2, label=res['method'])
        axes[1].plot(history_df['generation'], history_df['diversity'],
                    linewidth=2, label=res['method'])

    axes[0].set_xlabel('Generation', fontsize=12)
    axes[0].set_ylabel('Best Fitness', fontsize=12)
    axes[0].set_title('Best Fitness by Selection Method', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Generation', fontsize=12)
    axes[1].set_ylabel('Diversity', fontsize=12)
    axes[1].set_title('Population Diversity by Selection Method', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ga_selection_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    return results


def tsp_with_ga():
    """Solve Traveling Salesman Problem with GA."""
    np.random.seed(42)

    # Generate random cities
    n_cities = 20
    cities = np.random.rand(n_cities, 2) * 100

    # Define distance matrix
    def distance(i, j):
        return np.linalg.norm(cities[i] - cities[j])

    def tour_length(tour):
        """Calculate total tour length (minimize, so negate)."""
        length = sum(distance(tour[i], tour[i+1]) for i in range(len(tour)-1))
        length += distance(tour[-1], tour[0])  # Return to start
        return -length  # Negative for maximization

    # Run GA with permutation encoding
    ga = GeneticAlgorithm(tour_length, n_genes=n_cities, gene_type='permutation')

    result = ga.optimize(population_size=100, n_generations=500,
                        crossover_rate=0.9, mutation_rate=0.2)

    best_tour = result['best_solution']
    best_length = -result['best_fitness']

    print(f"\nTSP Results:")
    print(f"  Best tour length: {best_length:.2f}")
    print(f"  Best tour: {best_tour}")

    # Visualize tour
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Best tour
    tour_cities = cities[best_tour]
    axes[0].plot(tour_cities[:, 0], tour_cities[:, 1], 'b-', linewidth=2, alpha=0.6)
    axes[0].plot([tour_cities[-1, 0], tour_cities[0, 0]],
                [tour_cities[-1, 1], tour_cities[0, 1]], 'b-', linewidth=2, alpha=0.6)
    axes[0].scatter(cities[:, 0], cities[:, 1], c='red', s=100, zorder=5, edgecolors='black')
    axes[0].set_xlabel('X', fontsize=12)
    axes[0].set_ylabel('Y', fontsize=12)
    axes[0].set_title(f'Best Tour (Length: {best_length:.2f})', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Convergence
    history_df = pd.DataFrame(result['history'])
    axes[1].plot(history_df['generation'], -history_df['best_fitness'], linewidth=2)
    axes[1].set_xlabel('Generation', fontsize=12)
    axes[1].set_ylabel('Tour Length', fontsize=12)
    axes[1].set_title('GA Convergence for TSP', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ga_tsp_solution.png', dpi=300, bbox_inches='tight')
    plt.close()

    return result


def main():
    """Main execution function."""
    print("="*70)
    print("Genetic Algorithm for Optimization")
    print("="*70)

    # Example 1: Benchmark on test functions
    print("\n1. Benchmark on Standard Test Functions")
    print("-" * 70)
    benchmark_results, test_funcs = benchmark_ga_on_functions()

    # Example 2: Convergence visualization
    print("\n2. Convergence Visualization")
    print("-" * 70)
    visualize_ga_convergence()
    print("Convergence plots saved to 'genetic_algorithm_convergence.png'")

    # Example 3: Selection methods comparison
    print("\n3. Selection Methods Comparison")
    print("-" * 70)
    selection_results = compare_selection_methods()
    print("Selection comparison saved to 'ga_selection_comparison.png'")

    # Example 4: TSP
    print("\n4. Traveling Salesman Problem")
    print("-" * 70)
    tsp_result = tsp_with_ga()
    print("TSP solution saved to 'ga_tsp_solution.png'")

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
