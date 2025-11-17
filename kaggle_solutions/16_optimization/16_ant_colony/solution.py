"""
Ant Colony Optimization (ACO)
=============================

This solution implements Ant Colony Optimization, a probabilistic metaheuristic
inspired by the foraging behavior of ants.

Mathematical Background:
-----------------------
ACO uses artificial ants to construct solutions:
1. Ants build solutions incrementally
2. Pheromone trails τ guide construction (learned information)
3. Heuristic information η provides problem-specific guidance
4. Probability of choosing component j from i:
   p_ij = (τ_ij^α * η_ij^β) / Σ (τ_ik^α * η_ik^β)

Update rules:
- Pheromone evaporation: τ_ij ← (1-ρ) * τ_ij
- Pheromone deposit: τ_ij ← τ_ij + Δτ_ij (from ants)

Variants:
- Ant System (AS)
- Max-Min Ant System (MMAS)
- Ant Colony System (ACS)

Author: Optimization Expert
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class AntColonyOptimizer:
    """Ant Colony Optimization for TSP and combinatorial problems."""

    def __init__(self, distance_matrix: np.ndarray):
        """
        Initialize ACO.

        Args:
            distance_matrix: Matrix of distances between nodes
        """
        self.distances = distance_matrix
        self.n_nodes = len(distance_matrix)
        self.pheromones = np.ones((self.n_nodes, self.n_nodes))
        self.history = []

    def optimize(self, n_ants: int = 20, n_iterations: int = 100,
                alpha: float = 1.0, beta: float = 2.0,
                evaporation_rate: float = 0.1,
                q: float = 1.0) -> Dict:
        """
        Run ACO algorithm.

        Args:
            n_ants: Number of ants
            n_iterations: Number of iterations
            alpha: Pheromone importance
            beta: Heuristic importance
            evaporation_rate: Pheromone evaporation rate (ρ)
            q: Pheromone deposit factor

        Returns:
            Optimization results
        """
        best_tour = None
        best_length = np.inf

        for iteration in range(n_iterations):
            tours = []
            lengths = []

            # Each ant constructs a tour
            for ant in range(n_ants):
                tour = self._construct_tour(alpha, beta)
                length = self._tour_length(tour)

                tours.append(tour)
                lengths.append(length)

                # Update best
                if length < best_length:
                    best_length = length
                    best_tour = tour.copy()

            # Pheromone evaporation
            self.pheromones *= (1 - evaporation_rate)

            # Pheromone deposit
            for tour, length in zip(tours, lengths):
                deposit = q / length
                for i in range(len(tour) - 1):
                    self.pheromones[tour[i], tour[i+1]] += deposit
                    self.pheromones[tour[i+1], tour[i]] += deposit

                # Close the tour
                self.pheromones[tour[-1], tour[0]] += deposit
                self.pheromones[tour[0], tour[-1]] += deposit

            # Store history
            self.history.append({
                'iteration': iteration,
                'best_length': best_length,
                'mean_length': np.mean(lengths),
                'std_length': np.std(lengths)
            })

        return {
            'best_tour': best_tour,
            'best_length': best_length,
            'history': self.history
        }

    def _construct_tour(self, alpha: float, beta: float) -> List[int]:
        """Construct a tour using ACO probability rules."""
        tour = []
        unvisited = set(range(self.n_nodes))

        # Start from random node
        current = np.random.randint(self.n_nodes)
        tour.append(current)
        unvisited.remove(current)

        while unvisited:
            # Calculate probabilities
            probabilities = []
            for node in unvisited:
                pheromone = self.pheromones[current, node] ** alpha
                heuristic = (1.0 / self.distances[current, node]) ** beta if self.distances[current, node] > 0 else 0
                probabilities.append(pheromone * heuristic)

            probabilities = np.array(probabilities)
            probabilities /= probabilities.sum()

            # Select next node
            next_node = np.random.choice(list(unvisited), p=probabilities)

            tour.append(next_node)
            unvisited.remove(next_node)
            current = next_node

        return tour

    def _tour_length(self, tour: List[int]) -> float:
        """Calculate total tour length."""
        length = sum(self.distances[tour[i], tour[i+1]] for i in range(len(tour)-1))
        length += self.distances[tour[-1], tour[0]]  # Return to start
        return length


class MaxMinAntSystem:
    """Max-Min Ant System variant with bounds on pheromone values."""

    def __init__(self, distance_matrix: np.ndarray):
        self.distances = distance_matrix
        self.n_nodes = len(distance_matrix)
        self.pheromones = None
        self.tau_min = 0.01
        self.tau_max = 10.0
        self.history = []

    def optimize(self, n_ants: int = 20, n_iterations: int = 100) -> Dict:
        """Run MMAS."""
        # Initialize pheromones to tau_max
        self.pheromones = np.ones((self.n_nodes, self.n_nodes)) * self.tau_max

        best_tour = None
        best_length = np.inf
        iteration_best_tour = None
        iteration_best_length = np.inf

        for iteration in range(n_iterations):
            tours = []
            lengths = []

            for ant in range(n_ants):
                tour = self._construct_tour()
                length = self._tour_length(tour)

                tours.append(tour)
                lengths.append(length)

                if length < best_length:
                    best_length = length
                    best_tour = tour.copy()

                if length < iteration_best_length:
                    iteration_best_length = length
                    iteration_best_tour = tour.copy()

            # Pheromone update (only best ant)
            self.pheromones *= 0.98  # Evaporation

            deposit = 1.0 / best_length
            for i in range(len(best_tour) - 1):
                self.pheromones[best_tour[i], best_tour[i+1]] += deposit
                self.pheromones[best_tour[i+1], best_tour[i]] += deposit

            # Apply bounds
            self.pheromones = np.clip(self.pheromones, self.tau_min, self.tau_max)

            self.history.append({
                'iteration': iteration,
                'best_length': best_length,
                'mean_length': np.mean(lengths)
            })

            iteration_best_length = np.inf

        return {
            'best_tour': best_tour,
            'best_length': best_length,
            'history': self.history
        }

    def _construct_tour(self) -> List[int]:
        """Construct tour using pheromones."""
        tour = []
        unvisited = set(range(self.n_nodes))

        current = np.random.randint(self.n_nodes)
        tour.append(current)
        unvisited.remove(current)

        while unvisited:
            probabilities = []
            for node in unvisited:
                pheromone = self.pheromones[current, node]
                heuristic = 1.0 / self.distances[current, node] if self.distances[current, node] > 0 else 0
                probabilities.append(pheromone * heuristic ** 2)

            probabilities = np.array(probabilities)
            probabilities /= probabilities.sum()

            next_node = np.random.choice(list(unvisited), p=probabilities)
            tour.append(next_node)
            unvisited.remove(next_node)
            current = next_node

        return tour

    def _tour_length(self, tour: List[int]) -> float:
        """Calculate tour length."""
        length = sum(self.distances[tour[i], tour[i+1]] for i in range(len(tour)-1))
        length += self.distances[tour[-1], tour[0]]
        return length


def demonstrate_aco_tsp():
    """Demonstrate ACO on TSP."""
    np.random.seed(42)

    # Generate random cities
    n_cities = 30
    cities = np.random.rand(n_cities, 2) * 100

    # Compute distance matrix
    distances = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(n_cities):
            distances[i, j] = np.linalg.norm(cities[i] - cities[j])

    # Run ACO
    aco = AntColonyOptimizer(distances)
    result = aco.optimize(n_ants=30, n_iterations=200)

    print(f"Best tour length: {result['best_length']:.2f}")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Best tour
    tour = result['best_tour']
    tour_cities = cities[tour]

    axes[0].plot(tour_cities[:, 0], tour_cities[:, 1], 'b-', linewidth=2, alpha=0.6)
    axes[0].plot([tour_cities[-1, 0], tour_cities[0, 0]],
                [tour_cities[-1, 1], tour_cities[0, 1]], 'b-', linewidth=2, alpha=0.6)
    axes[0].scatter(cities[:, 0], cities[:, 1], c='red', s=100, zorder=5, edgecolors='black')
    axes[0].set_xlabel('X', fontsize=12)
    axes[0].set_ylabel('Y', fontsize=12)
    axes[0].set_title(f'ACO Best Tour (Length: {result["best_length"]:.2f})',
                     fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Convergence
    history_df = pd.DataFrame(result['history'])
    axes[1].plot(history_df['iteration'], history_df['best_length'], 'b-', linewidth=2, label='Best')
    axes[1].plot(history_df['iteration'], history_df['mean_length'], 'r--', linewidth=2, label='Mean')
    axes[1].fill_between(history_df['iteration'],
                         history_df['mean_length'] - history_df['std_length'],
                         history_df['mean_length'] + history_df['std_length'],
                         alpha=0.3)
    axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('Tour Length', fontsize=12)
    axes[1].set_title('ACO Convergence', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('aco_tsp_solution.png', dpi=300, bbox_inches='tight')
    plt.close()

    return result


def compare_aco_variants():
    """Compare ACO variants."""
    np.random.seed(42)

    n_cities = 20
    cities = np.random.rand(n_cities, 2) * 100

    distances = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(n_cities):
            distances[i, j] = np.linalg.norm(cities[i] - cities[j])

    # Standard ACO
    aco = AntColonyOptimizer(distances)
    result_aco = aco.optimize(n_ants=20, n_iterations=100)

    # MMAS
    mmas = MaxMinAntSystem(distances)
    result_mmas = mmas.optimize(n_ants=20, n_iterations=100)

    # Visualize comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    history_aco = pd.DataFrame(result_aco['history'])
    history_mmas = pd.DataFrame(result_mmas['history'])

    ax.plot(history_aco['iteration'], history_aco['best_length'],
           'b-', linewidth=2, label='Standard ACO')
    ax.plot(history_mmas['iteration'], history_mmas['best_length'],
           'r-', linewidth=2, label='Max-Min AS')

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Best Tour Length', fontsize=12)
    ax.set_title('ACO Variants Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('aco_variants_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Standard ACO: {result_aco['best_length']:.2f}")
    print(f"Max-Min AS: {result_mmas['best_length']:.2f}")

    return result_aco, result_mmas


def main():
    """Main execution function."""
    print("="*70)
    print("Ant Colony Optimization (ACO)")
    print("="*70)

    # Example 1: TSP with ACO
    print("\n1. Traveling Salesman Problem with ACO")
    print("-" * 70)
    tsp_result = demonstrate_aco_tsp()

    # Example 2: ACO variants comparison
    print("\n2. ACO Variants Comparison")
    print("-" * 70)
    result_aco, result_mmas = compare_aco_variants()

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()


def advanced_aco_applications():
    """Advanced ACO applications and analysis."""
    np.random.seed(42)
    
    # Vehicle Routing Problem simulation
    n_customers = 15
    depot = np.array([50, 50])
    customers = np.random.rand(n_customers, 2) * 100
    all_locations = np.vstack([depot, customers])
    
    # Calculate distance matrix
    n_locations = len(all_locations)
    distances = np.zeros((n_locations, n_locations))
    for i in range(n_locations):
        for j in range(n_locations):
            distances[i, j] = np.linalg.norm(all_locations[i] - all_locations[j])
    
    # Run ACO
    aco_vrp = AntColonyOptimizer(distances)
    result_vrp = aco_vrp.optimize(n_ants=20, n_iterations=150)
    
    print(f"\nVehicle Routing Problem:")
    print(f"  Best route length: {result_vrp['best_length']:.2f}")
    
    # Parameter sensitivity analysis
    alphas = np.linspace(0.5, 2.0, 8)
    betas = np.linspace(1.0, 5.0, 8)
    
    sensitivity_results = []
    
    for alpha in alphas:
        for beta in betas:
            aco_param = AntColonyOptimizer(distances[:10, :10])
            result_param = aco_param.optimize(n_ants=15, n_iterations=50, 
                                             alpha=alpha, beta=beta)
            sensitivity_results.append({
                'alpha': alpha,
                'beta': beta,
                'best_length': result_param['best_length']
            })
    
    sensitivity_df = pd.DataFrame(sensitivity_results)
    
    # Create heatmap
    pivot_table = sensitivity_df.pivot(index='beta', columns='alpha', values='best_length')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Heatmap
    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[0, 0])
    axes[0, 0].set_title('Parameter Sensitivity (α, β)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Alpha (α)', fontsize=12)
    axes[0, 0].set_ylabel('Beta (β)', fontsize=12)
    
    # VRP route visualization
    tour = result_vrp['best_tour']
    tour_locs = all_locations[tour]
    
    axes[0, 1].plot(tour_locs[:, 0], tour_locs[:, 1], 'b-', linewidth=2, alpha=0.6)
    axes[0, 1].plot([tour_locs[-1, 0], tour_locs[0, 0]],
                   [tour_locs[-1, 1], tour_locs[0, 1]], 'b-', linewidth=2, alpha=0.6)
    axes[0, 1].scatter(customers[:, 0], customers[:, 1], c='red', s=100, 
                      zorder=5, edgecolors='black', label='Customers')
    axes[0, 1].scatter(depot[0], depot[1], c='green', s=200, marker='s',
                      zorder=5, edgecolors='black', label='Depot')
    axes[0, 1].set_xlabel('X', fontsize=12)
    axes[0, 1].set_ylabel('Y', fontsize=12)
    axes[0, 1].set_title('Vehicle Routing Solution', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Alpha effect
    alpha_effect = sensitivity_df[sensitivity_df['beta'] == 3.0].groupby('alpha')['best_length'].mean()
    axes[1, 0].plot(alpha_effect.index, alpha_effect.values, 'ro-', linewidth=2, markersize=10)
    axes[1, 0].set_xlabel('Alpha (Pheromone Weight)', fontsize=12)
    axes[1, 0].set_ylabel('Average Best Length', fontsize=12)
    axes[1, 0].set_title('Alpha Parameter Effect', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Beta effect
    beta_effect = sensitivity_df[sensitivity_df['alpha'] == 1.0].groupby('beta')['best_length'].mean()
    axes[1, 1].plot(beta_effect.index, beta_effect.values, 'bo-', linewidth=2, markersize=10)
    axes[1, 1].set_xlabel('Beta (Heuristic Weight)', fontsize=12)
    axes[1, 1].set_ylabel('Average Best Length', fontsize=12)
    axes[1, 1].set_title('Beta Parameter Effect', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('aco_advanced_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


