"""
Simulated Annealing Optimization
================================

This solution implements simulated annealing, a probabilistic metaheuristic
inspired by the annealing process in metallurgy.

Mathematical Background:
-----------------------
Simulated Annealing accepts worse solutions with probability:
    P(accept) = exp(-ΔE / T)

where:
- ΔE = E_new - E_current (energy difference)
- T = current temperature

The algorithm:
1. Start with high temperature T0
2. At each iteration:
   - Generate neighbor solution
   - Accept if better, or with probability exp(-ΔE/T) if worse
3. Gradually decrease temperature (cooling schedule)
4. Stop when temperature is low enough

Cooling schedules:
- Linear: T(t) = T0 - α*t
- Geometric: T(t) = T0 * α^t
- Logarithmic: T(t) = T0 / log(1 + t)

Author: Optimization Expert
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Callable, Tuple, Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')


class SimulatedAnnealing:
    """Simulated Annealing optimizer."""

    def __init__(self, objective: Callable, bounds: List[Tuple[float, float]],
                 neighbor_func: Optional[Callable] = None):
        """
        Initialize SA optimizer.

        Args:
            objective: Objective function to minimize
            bounds: List of (min, max) bounds for each dimension
            neighbor_func: Function to generate neighbor (if None, use Gaussian)
        """
        self.objective = objective
        self.bounds = np.array(bounds)
        self.n_dim = len(bounds)
        self.neighbor_func = neighbor_func or self._gaussian_neighbor
        self.history = []

    def optimize(self, initial_temp: float = 100.0, final_temp: float = 0.01,
                cooling_schedule: str = 'geometric', alpha: float = 0.95,
                max_iterations: int = 1000, max_no_improve: int = 100) -> Dict:
        """
        Run simulated annealing.

        Args:
            initial_temp: Initial temperature
            final_temp: Final temperature (stopping criterion)
            cooling_schedule: 'geometric', 'linear', or 'logarithmic'
            alpha: Cooling rate parameter
            max_iterations: Maximum iterations
            max_no_improve: Stop if no improvement for this many iterations

        Returns:
            Optimization results
        """
        # Initialize
        current = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        current_energy = self.objective(current)

        best = current.copy()
        best_energy = current_energy

        temp = initial_temp
        iteration = 0
        no_improve_count = 0

        while temp > final_temp and iteration < max_iterations and no_improve_count < max_no_improve:
            # Generate neighbor
            candidate = self.neighbor_func(current, temp)
            candidate = np.clip(candidate, self.bounds[:, 0], self.bounds[:, 1])

            # Evaluate
            candidate_energy = self.objective(candidate)

            # Accept or reject
            delta_e = candidate_energy - current_energy

            if delta_e < 0 or np.random.random() < np.exp(-delta_e / temp):
                current = candidate
                current_energy = candidate_energy

                # Update best
                if current_energy < best_energy:
                    best = current.copy()
                    best_energy = current_energy
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1

            # Update temperature
            if cooling_schedule == 'geometric':
                temp *= alpha
            elif cooling_schedule == 'linear':
                temp -= alpha
            elif cooling_schedule == 'logarithmic':
                temp = initial_temp / np.log(2 + iteration)

            # Store history
            self.history.append({
                'iteration': iteration,
                'temperature': temp,
                'current_energy': current_energy,
                'best_energy': best_energy,
                'accepted': delta_e < 0 or np.random.random() < np.exp(-delta_e / temp)
            })

            iteration += 1

        return {
            'best_solution': best,
            'best_energy': best_energy,
            'iterations': iteration,
            'history': self.history
        }

    def _gaussian_neighbor(self, current: np.ndarray, temp: float) -> np.ndarray:
        """Generate neighbor using Gaussian perturbation."""
        sigma = temp / 100.0 * (self.bounds[:, 1] - self.bounds[:, 0])
        return current + np.random.normal(0, sigma, self.n_dim)


class AdaptiveSimulatedAnnealing:
    """Adaptive simulated annealing with automatic parameter tuning."""

    def __init__(self, objective: Callable, bounds: List[Tuple[float, float]]):
        self.objective = objective
        self.bounds = np.array(bounds)
        self.n_dim = len(bounds)
        self.history = []

    def optimize(self, max_iterations: int = 1000) -> Dict:
        """Run adaptive SA."""
        current = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        current_energy = self.objective(current)

        best = current.copy()
        best_energy = current_energy

        # Adaptive parameters
        temp = 100.0
        acceptance_rate = []
        window_size = 50

        for iteration in range(max_iterations):
            # Adaptive neighbor generation
            sigma = temp / 100.0 * (self.bounds[:, 1] - self.bounds[:, 0])
            candidate = current + np.random.normal(0, sigma, self.n_dim)
            candidate = np.clip(candidate, self.bounds[:, 0], self.bounds[:, 1])

            candidate_energy = self.objective(candidate)
            delta_e = candidate_energy - current_energy

            # Acceptance criterion
            if delta_e < 0 or np.random.random() < np.exp(-delta_e / temp):
                current = candidate
                current_energy = candidate_energy
                accepted = True

                if current_energy < best_energy:
                    best = current.copy()
                    best_energy = current_energy
            else:
                accepted = False

            acceptance_rate.append(accepted)

            # Adaptive cooling
            if len(acceptance_rate) >= window_size:
                recent_acceptance = np.mean(acceptance_rate[-window_size:])

                if recent_acceptance > 0.6:
                    temp *= 0.99  # Cool faster
                elif recent_acceptance < 0.2:
                    temp *= 1.01  # Heat up
                else:
                    temp *= 0.95  # Normal cooling

            self.history.append({
                'iteration': iteration,
                'temperature': temp,
                'best_energy': best_energy
            })

        return {
            'best_solution': best,
            'best_energy': best_energy,
            'iterations': len(self.history),
            'history': self.history
        }


def tsp_with_sa(cities: np.ndarray) -> Dict:
    """Solve TSP using simulated annealing."""
    n_cities = len(cities)

    def tour_length(tour):
        length = sum(np.linalg.norm(cities[tour[i]] - cities[tour[i+1]])
                    for i in range(len(tour)-1))
        length += np.linalg.norm(cities[tour[-1]] - cities[tour[0]])
        return length

    def neighbor(tour, temp):
        """Generate neighbor by 2-opt swap."""
        new_tour = tour.copy()
        i, j = sorted(np.random.choice(n_cities, 2, replace=False))
        new_tour[i:j+1] = new_tour[i:j+1][::-1]
        return new_tour

    # Initial random tour
    initial_tour = np.arange(n_cities)
    current = initial_tour.copy()
    current_energy = tour_length(current)

    best = current.copy()
    best_energy = current_energy

    temp = 100.0
    alpha = 0.995
    history = []

    for iteration in range(10000):
        candidate = neighbor(current, temp)
        candidate_energy = tour_length(candidate)

        delta_e = candidate_energy - current_energy

        if delta_e < 0 or np.random.random() < np.exp(-delta_e / temp):
            current = candidate
            current_energy = candidate_energy

            if current_energy < best_energy:
                best = current.copy()
                best_energy = current_energy

        temp *= alpha

        history.append({
            'iteration': iteration,
            'temperature': temp,
            'best_energy': best_energy
        })

        if temp < 0.01:
            break

    return {
        'best_tour': best,
        'best_length': best_energy,
        'history': history
    }


def benchmark_cooling_schedules():
    """Compare different cooling schedules."""

    def rastrigin(x):
        return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

    bounds = [(-5.12, 5.12)] * 5

    schedules = ['geometric', 'linear', 'logarithmic']
    results = {}

    for schedule in schedules:
        sa = SimulatedAnnealing(rastrigin, bounds)
        result = sa.optimize(initial_temp=100, final_temp=0.01,
                           cooling_schedule=schedule, max_iterations=1000)
        results[schedule] = result

    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for schedule, result in results.items():
        history_df = pd.DataFrame(result['history'])
        axes[0].semilogy(history_df['iteration'], history_df['best_energy'],
                        linewidth=2, label=schedule)
        axes[1].semilogy(history_df['iteration'], history_df['temperature'],
                        linewidth=2, label=schedule)

    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel('Best Energy (log)', fontsize=12)
    axes[0].set_title('Convergence by Cooling Schedule', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('Temperature (log)', fontsize=12)
    axes[1].set_title('Temperature Decay', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sa_cooling_schedules.png', dpi=300, bbox_inches='tight')
    plt.close()

    return results


def visualize_sa_convergence():
    """Visualize SA convergence."""

    def rastrigin(x):
        return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

    bounds = [(-5.12, 5.12)] * 2

    sa = SimulatedAnnealing(rastrigin, bounds)
    result = sa.optimize(initial_temp=100, final_temp=0.01,
                        cooling_schedule='geometric', alpha=0.95,
                        max_iterations=1000)

    history_df = pd.DataFrame(result['history'])

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Energy over time
    axes[0, 0].plot(history_df['iteration'], history_df['current_energy'],
                   'b-', alpha=0.5, linewidth=1, label='Current')
    axes[0, 0].plot(history_df['iteration'], history_df['best_energy'],
                   'r-', linewidth=2, label='Best')
    axes[0, 0].set_xlabel('Iteration', fontsize=12)
    axes[0, 0].set_ylabel('Energy', fontsize=12)
    axes[0, 0].set_title('Energy Evolution', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Temperature decay
    axes[0, 1].semilogy(history_df['iteration'], history_df['temperature'],
                       'g-', linewidth=2)
    axes[0, 1].set_xlabel('Iteration', fontsize=12)
    axes[0, 1].set_ylabel('Temperature (log)', fontsize=12)
    axes[0, 1].set_title('Temperature Schedule', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Acceptance ratio over time
    window = 50
    acceptance_ratio = history_df['accepted'].rolling(window=window).mean()
    axes[1, 0].plot(history_df['iteration'], acceptance_ratio, 'purple', linewidth=2)
    axes[1, 0].set_xlabel('Iteration', fontsize=12)
    axes[1, 0].set_ylabel('Acceptance Ratio', fontsize=12)
    axes[1, 0].set_title(f'Acceptance Ratio (window={window})', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Temperature vs acceptance
    temp_bins = np.linspace(history_df['temperature'].min(),
                           history_df['temperature'].max(), 20)
    acceptance_by_temp = []
    temp_centers = []

    for i in range(len(temp_bins)-1):
        mask = (history_df['temperature'] >= temp_bins[i]) & (history_df['temperature'] < temp_bins[i+1])
        if mask.sum() > 0:
            acceptance_by_temp.append(history_df[mask]['accepted'].mean())
            temp_centers.append((temp_bins[i] + temp_bins[i+1]) / 2)

    axes[1, 1].semilogx(temp_centers, acceptance_by_temp, 'o-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Temperature (log)', fontsize=12)
    axes[1, 1].set_ylabel('Acceptance Rate', fontsize=12)
    axes[1, 1].set_title('Acceptance vs Temperature', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sa_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()


def demonstrate_tsp():
    """Demonstrate TSP solution with SA."""
    np.random.seed(42)

    # Generate random cities
    n_cities = 30
    cities = np.random.rand(n_cities, 2) * 100

    result = tsp_with_sa(cities)

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
    axes[0].set_title(f'Best Tour (Length: {result["best_length"]:.2f})',
                     fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Convergence
    history_df = pd.DataFrame(result['history'])
    axes[1].plot(history_df['iteration'], history_df['best_energy'], linewidth=2)
    axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('Tour Length', fontsize=12)
    axes[1].set_title('SA Convergence for TSP', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sa_tsp_solution.png', dpi=300, bbox_inches='tight')
    plt.close()

    return result


def main():
    """Main execution function."""
    print("="*70)
    print("Simulated Annealing Optimization")
    print("="*70)

    # Example 1: Cooling schedules comparison
    print("\n1. Cooling Schedules Comparison")
    print("-" * 70)
    schedule_results = benchmark_cooling_schedules()
    for schedule, result in schedule_results.items():
        print(f"{schedule}: Best energy = {result['best_energy']:.6e}")
    print("Comparison plot saved to 'sa_cooling_schedules.png'")

    # Example 2: Convergence visualization
    print("\n2. SA Convergence Visualization")
    print("-" * 70)
    visualize_sa_convergence()
    print("Convergence plot saved to 'sa_convergence.png'")

    # Example 3: TSP
    print("\n3. Traveling Salesman Problem")
    print("-" * 70)
    tsp_result = demonstrate_tsp()
    print(f"Best tour length: {tsp_result['best_length']:.2f}")
    print("TSP solution saved to 'sa_tsp_solution.png'")

    # Example 4: Adaptive SA
    print("\n4. Adaptive Simulated Annealing")
    print("-" * 70)

    def rastrigin(x):
        return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

    asa = AdaptiveSimulatedAnnealing(rastrigin, [(-5.12, 5.12)] * 5)
    asa_result = asa.optimize(max_iterations=1000)
    print(f"Best energy: {asa_result['best_energy']:.6e}")
    print(f"Iterations: {asa_result['iterations']}")

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
