"""
Traveling Salesman Problem (TSP) Solutions
===========================================

This example demonstrates multiple approaches to solving the classic Traveling
Salesman Problem, where we need to find the shortest route visiting all cities
exactly once and returning to the start.

Problem: Given N cities and distances between them, find the shortest tour
that visits each city exactly once and returns to the starting city.

Author: Kaggle Solutions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
from typing import Tuple, List, Dict
import itertools
import time
import warnings
warnings.filterwarnings('ignore')


class TSPSolver:
    """
    Comprehensive TSP solver with multiple algorithms.
    """

    def __init__(self, n_cities=15, seed=42):
        """
        Initialize the TSP solver.

        Args:
            n_cities: Number of cities
            seed: Random seed for reproducibility
        """
        self.n_cities = n_cities
        self.seed = seed
        np.random.seed(seed)
        self.results = {}

    def generate_cities(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate random city locations and compute distance matrix.

        Returns:
            Tuple of (city_coords, distance_matrix)
        """
        # Generate random city coordinates
        cities = np.random.rand(self.n_cities, 2) * 100

        # Compute Euclidean distance matrix
        dist_matrix = squareform(pdist(cities, metric='euclidean'))

        return cities, dist_matrix

    def tour_length(self, tour: List[int], dist_matrix: np.ndarray) -> float:
        """
        Calculate the total length of a tour.

        Args:
            tour: List of city indices representing the tour
            dist_matrix: Distance matrix between cities

        Returns:
            Total tour length
        """
        length = 0
        for i in range(len(tour)):
            length += dist_matrix[tour[i], tour[(i + 1) % len(tour)]]
        return length

    def solve_brute_force(self, dist_matrix: np.ndarray) -> Dict:
        """
        Solve TSP using brute force (only for small instances).

        Args:
            dist_matrix: Distance matrix between cities

        Returns:
            Dictionary with solution results
        """
        print("\n" + "="*60)
        print("Method 1: Brute Force (Exact)")
        print("="*60)

        n = len(dist_matrix)

        if n > 10:
            print(f"Skipping brute force for {n} cities (too many permutations: {np.math.factorial(n-1)})")
            return {'method': 'Brute Force', 'success': False, 'message': 'Too many cities'}

        start_time = time.time()

        # Fix first city, permute others
        best_tour = None
        best_length = np.inf

        # Generate all permutations starting from city 0
        cities = list(range(1, n))
        permutations_checked = 0

        for perm in itertools.permutations(cities):
            tour = [0] + list(perm)
            length = self.tour_length(tour, dist_matrix)
            permutations_checked += 1

            if length < best_length:
                best_length = length
                best_tour = tour

        elapsed_time = time.time() - start_time

        solution = {
            'method': 'Brute Force',
            'tour': best_tour,
            'length': best_length,
            'success': True,
            'time': elapsed_time,
            'permutations_checked': permutations_checked
        }

        print(f"Permutations checked: {permutations_checked:,}")
        print(f"Best tour: {best_tour}")
        print(f"Tour length: {best_length:.2f}")
        print(f"Time: {elapsed_time:.4f} seconds")

        return solution

    def solve_nearest_neighbor(self, dist_matrix: np.ndarray, start_city=0) -> Dict:
        """
        Solve TSP using nearest neighbor heuristic.

        Args:
            dist_matrix: Distance matrix between cities
            start_city: Starting city index

        Returns:
            Dictionary with solution results
        """
        print("\n" + "="*60)
        print("Method 2: Nearest Neighbor Heuristic")
        print("="*60)

        start_time = time.time()
        n = len(dist_matrix)

        unvisited = set(range(n))
        current = start_city
        tour = [current]
        unvisited.remove(current)

        print(f"\nStarting from city {start_city}")
        print("Greedy selection:")

        while unvisited:
            # Find nearest unvisited city
            nearest = min(unvisited, key=lambda city: dist_matrix[current, city])
            print(f"  From city {current} -> city {nearest} (distance: {dist_matrix[current, nearest]:.2f})")
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest

        length = self.tour_length(tour, dist_matrix)
        elapsed_time = time.time() - start_time

        solution = {
            'method': 'Nearest Neighbor',
            'tour': tour,
            'length': length,
            'success': True,
            'time': elapsed_time
        }

        print(f"\nFinal tour: {tour}")
        print(f"Tour length: {length:.2f}")
        print(f"Time: {elapsed_time:.4f} seconds")

        return solution

    def solve_2opt(self, dist_matrix: np.ndarray, initial_tour=None) -> Dict:
        """
        Solve TSP using 2-opt local search improvement.

        Args:
            dist_matrix: Distance matrix between cities
            initial_tour: Initial tour (uses nearest neighbor if None)

        Returns:
            Dictionary with solution results
        """
        print("\n" + "="*60)
        print("Method 3: 2-Opt Local Search")
        print("="*60)

        start_time = time.time()

        # Get initial tour if not provided
        if initial_tour is None:
            nn_solution = self.solve_nearest_neighbor(dist_matrix)
            tour = nn_solution['tour'].copy()
            print(f"\nStarting with nearest neighbor tour: length = {nn_solution['length']:.2f}")
        else:
            tour = initial_tour.copy()

        n = len(tour)
        improved = True
        iteration = 0

        print("\n2-Opt improvements:")

        while improved:
            improved = False
            iteration += 1

            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    # Try reversing segment [i:j]
                    new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]

                    new_length = self.tour_length(new_tour, dist_matrix)
                    current_length = self.tour_length(tour, dist_matrix)

                    if new_length < current_length:
                        tour = new_tour
                        improved = True
                        print(f"  Iteration {iteration}: Improved to {new_length:.2f} (reversed segment [{i}:{j}])")
                        break
                if improved:
                    break

        length = self.tour_length(tour, dist_matrix)
        elapsed_time = time.time() - start_time

        solution = {
            'method': '2-Opt',
            'tour': tour,
            'length': length,
            'success': True,
            'time': elapsed_time,
            'iterations': iteration
        }

        print(f"\nFinal tour: {tour}")
        print(f"Tour length: {length:.2f}")
        print(f"Iterations: {iteration}")
        print(f"Time: {elapsed_time:.4f} seconds")

        return solution

    def solve_simulated_annealing(self, dist_matrix: np.ndarray,
                                  initial_temp=1000, cooling_rate=0.995,
                                  max_iterations=10000) -> Dict:
        """
        Solve TSP using simulated annealing.

        Args:
            dist_matrix: Distance matrix between cities
            initial_temp: Starting temperature
            cooling_rate: Temperature reduction factor
            max_iterations: Maximum number of iterations

        Returns:
            Dictionary with solution results
        """
        print("\n" + "="*60)
        print("Method 4: Simulated Annealing")
        print("="*60)

        start_time = time.time()
        n = len(dist_matrix)

        # Start with random tour
        current_tour = list(range(n))
        np.random.shuffle(current_tour)
        current_length = self.tour_length(current_tour, dist_matrix)

        best_tour = current_tour.copy()
        best_length = current_length

        temp = initial_temp
        history = []

        print(f"Initial temperature: {initial_temp}")
        print(f"Cooling rate: {cooling_rate}")
        print(f"Max iterations: {max_iterations}\n")

        for iteration in range(max_iterations):
            # Generate neighbor by swapping two cities
            i, j = np.random.randint(0, n, 2)
            new_tour = current_tour.copy()
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]

            new_length = self.tour_length(new_tour, dist_matrix)
            delta = new_length - current_length

            # Accept if better, or with probability based on temperature
            if delta < 0 or np.random.random() < np.exp(-delta / temp):
                current_tour = new_tour
                current_length = new_length

                if current_length < best_length:
                    best_tour = current_tour.copy()
                    best_length = current_length

            # Cool down
            temp *= cooling_rate

            # Record progress
            if iteration % 1000 == 0:
                history.append({'iteration': iteration, 'length': best_length, 'temp': temp})
                print(f"  Iteration {iteration}: Best = {best_length:.2f}, Temp = {temp:.2f}")

        elapsed_time = time.time() - start_time

        solution = {
            'method': 'Simulated Annealing',
            'tour': best_tour,
            'length': best_length,
            'success': True,
            'time': elapsed_time,
            'history': history
        }

        print(f"\nFinal tour: {best_tour}")
        print(f"Tour length: {best_length:.2f}")
        print(f"Time: {elapsed_time:.4f} seconds")

        return solution

    def solve_all_methods(self, cities: np.ndarray, dist_matrix: np.ndarray):
        """Solve the TSP using all available methods."""
        print("\nSOLVING TRAVELING SALESMAN PROBLEM")
        print("="*60)
        print(f"Number of cities: {self.n_cities}")
        print(f"Search space: {np.math.factorial(self.n_cities-1):,} possible tours")
        print("="*60)

        # Solve with different methods
        if self.n_cities <= 10:
            self.results['brute_force'] = self.solve_brute_force(dist_matrix)

        self.results['nearest_neighbor'] = self.solve_nearest_neighbor(dist_matrix)
        self.results['2opt'] = self.solve_2opt(dist_matrix)
        self.results['simulated_annealing'] = self.solve_simulated_annealing(dist_matrix)

        return self.results

    def visualize_solutions(self, cities: np.ndarray, dist_matrix: np.ndarray):
        """
        Visualize all solutions.

        Args:
            cities: City coordinates
            dist_matrix: Distance matrix
        """
        # Filter successful results
        successful_results = {k: v for k, v in self.results.items() if v['success']}

        n_methods = len(successful_results)
        n_cols = min(2, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
        if n_methods == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_methods > 1 else [axes]

        for idx, (method_name, result) in enumerate(successful_results.items()):
            ax = axes[idx]

            tour = result['tour']

            # Plot cities
            ax.scatter(cities[:, 0], cities[:, 1], c='red', s=200, zorder=3,
                      edgecolors='black', linewidths=2)

            # Label cities
            for i, (x, y) in enumerate(cities):
                ax.annotate(str(i), xy=(x, y), fontsize=10, ha='center', va='center',
                           color='white', fontweight='bold')

            # Plot tour
            for i in range(len(tour)):
                start = cities[tour[i]]
                end = cities[tour[(i + 1) % len(tour)]]
                ax.plot([start[0], end[0]], [start[1], end[1]],
                       'b-', linewidth=2, alpha=0.6, zorder=1)
                ax.arrow(start[0], start[1],
                        (end[0] - start[0]) * 0.9, (end[1] - start[1]) * 0.9,
                        head_width=3, head_length=2, fc='blue', ec='blue',
                        alpha=0.4, zorder=2)

            # Highlight start city
            start = cities[tour[0]]
            ax.scatter([start[0]], [start[1]], c='green', s=400, marker='*',
                      zorder=4, edgecolors='black', linewidths=2,
                      label='Start')

            ax.set_xlabel('X Coordinate', fontsize=12)
            ax.set_ylabel('Y Coordinate', fontsize=12)
            ax.set_title(f'{result["method"]}\nLength: {result["length"]:.2f}, Time: {result["time"]:.4f}s',
                        fontsize=14, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')

        # Hide extra subplots
        for idx in range(n_methods, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/16_optimization/03_tsp/tsp_tours.png',
                    dpi=300, bbox_inches='tight')
        print("\nTour visualization saved to: tsp_tours.png")
        plt.show()

    def visualize_comparison(self):
        """Visualize comparison of methods."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Filter successful results
        successful_results = {k: v for k, v in self.results.items() if v['success']}

        # Plot 1: Tour length comparison
        ax = axes[0]
        methods = [v['method'] for v in successful_results.values()]
        lengths = [v['length'] for v in successful_results.values()]
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12'][:len(methods)]

        bars = ax.bar(methods, lengths, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

        # Add value labels
        for bar, length in zip(bars, lengths):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{length:.2f}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

        # Mark best solution
        best_idx = np.argmin(lengths)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(4)

        ax.set_ylabel('Tour Length', fontsize=12)
        ax.set_title('Tour Length Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 2: Computation time comparison
        ax = axes[1]
        times = [v['time'] for v in successful_results.values()]

        bars = ax.bar(methods, times, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

        for bar, t in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{t:.4f}s',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_ylabel('Computation Time (seconds)', fontsize=12)
        ax.set_title('Computation Time Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/16_optimization/03_tsp/tsp_comparison.png',
                    dpi=300, bbox_inches='tight')
        print("Comparison visualization saved to: tsp_comparison.png")
        plt.show()

    def visualize_sa_convergence(self):
        """Visualize simulated annealing convergence."""
        if 'simulated_annealing' not in self.results:
            return

        sa_result = self.results['simulated_annealing']
        if 'history' not in sa_result:
            return

        history = sa_result['history']
        if not history:
            return

        df_history = pd.DataFrame(history)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Best solution over iterations
        ax = axes[0]
        ax.plot(df_history['iteration'], df_history['length'],
               'b-', linewidth=2, label='Best Tour Length')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Tour Length', fontsize=12)
        ax.set_title('Simulated Annealing Convergence', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Plot 2: Temperature decay
        ax = axes[1]
        ax.plot(df_history['iteration'], df_history['temp'],
               'r-', linewidth=2, label='Temperature')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Temperature', fontsize=12)
        ax.set_title('Temperature Decay', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/16_optimization/03_tsp/tsp_sa_convergence.png',
                    dpi=300, bbox_inches='tight')
        print("SA convergence visualization saved to: tsp_sa_convergence.png")
        plt.show()


def main():
    """Main execution function."""
    print("="*60)
    print("TRAVELING SALESMAN PROBLEM")
    print("="*60)

    # Create solver (use fewer cities to allow brute force comparison)
    solver = TSPSolver(n_cities=10)

    # Generate problem
    cities, dist_matrix = solver.generate_cities()

    # Solve using all methods
    results = solver.solve_all_methods(cities, dist_matrix)

    # Compare results
    print("\n" + "="*60)
    print("Comparison of Methods")
    print("="*60)

    comparison_data = []
    for method_name, result in results.items():
        if result['success']:
            comparison_data.append({
                'Method': result['method'],
                'Tour Length': f"{result['length']:.2f}",
                'Time (s)': f"{result['time']:.4f}",
                'Status': 'Success'
            })
        else:
            comparison_data.append({
                'Method': result.get('method', method_name),
                'Tour Length': '-',
                'Time (s)': '-',
                'Status': 'Skipped/Failed'
            })

    df_comparison = pd.DataFrame(comparison_data)
    print("\n", df_comparison.to_string(index=False))

    # Find best solution
    successful = {k: v for k, v in results.items() if v['success']}
    best_method = min(successful.items(), key=lambda x: x[1]['length'])
    print(f"\nBest solution: {best_method[1]['method']} with length {best_method[1]['length']:.2f}")

    # Visualize
    solver.visualize_solutions(cities, dist_matrix)
    solver.visualize_comparison()
    solver.visualize_sa_convergence()

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
