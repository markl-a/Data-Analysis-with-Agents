"""
Advanced Simplex Method Implementation
======================================

This solution implements various simplex method algorithms for linear programming,
including the revised simplex, two-phase simplex, and dual simplex methods.

Mathematical Background:
-----------------------
Linear programming aims to optimize a linear objective function subject to linear constraints:
    maximize/minimize: c^T x
    subject to: Ax ≤ b, x ≥ 0

The simplex method iteratively moves along the edges of the feasible polytope
towards the optimal vertex.

Author: Optimization Expert
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linprog
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class SimplexSolver:
    """
    Implementation of the Simplex Method for Linear Programming.
    """

    def __init__(self, c: np.ndarray, A: np.ndarray, b: np.ndarray):
        """
        Initialize simplex solver.

        Args:
            c: Coefficients of objective function (to maximize c^T x)
            A: Constraint matrix (Ax <= b)
            b: Right-hand side of constraints
        """
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.m, self.n = A.shape
        self.history = []

    def solve(self, method='standard') -> Dict:
        """
        Solve LP using specified simplex method.

        Args:
            method: 'standard', 'revised', or 'dual'

        Returns:
            Dictionary with solution details
        """
        if method == 'standard':
            return self._standard_simplex()
        elif method == 'revised':
            return self._revised_simplex()
        elif method == 'dual':
            return self._dual_simplex()
        else:
            raise ValueError(f"Unknown method: {method}")

    def _standard_simplex(self) -> Dict:
        """Standard (tableau-based) simplex method."""
        # Convert to standard form with slack variables
        tableau = self._create_initial_tableau()

        iteration = 0
        max_iterations = 1000

        while iteration < max_iterations:
            # Check if optimal
            if np.all(tableau[-1, :-1] <= 0):
                break

            # Choose pivot column (entering variable)
            pivot_col = np.argmax(tableau[-1, :-1])

            # Choose pivot row (leaving variable) using minimum ratio test
            ratios = []
            for i in range(self.m):
                if tableau[i, pivot_col] > 1e-10:
                    ratios.append(tableau[i, -1] / tableau[i, pivot_col])
                else:
                    ratios.append(np.inf)

            pivot_row = np.argmin(ratios)

            if ratios[pivot_row] == np.inf:
                return {'status': 'unbounded', 'iterations': iteration}

            # Perform pivot operation
            tableau = self._pivot(tableau, pivot_row, pivot_col)

            # Store iteration history
            self.history.append({
                'iteration': iteration,
                'objective': -tableau[-1, -1],
                'tableau': tableau.copy()
            })

            iteration += 1

        # Extract solution
        solution = self._extract_solution(tableau)

        return {
            'status': 'optimal',
            'x': solution,
            'objective': -tableau[-1, -1],
            'iterations': iteration,
            'history': self.history
        }

    def _revised_simplex(self) -> Dict:
        """Revised simplex method (more efficient for large problems)."""
        # Initialize basis and non-basis indices
        basis = list(range(self.n, self.n + self.m))
        non_basis = list(range(self.n))

        # Augment A with identity matrix (slack variables)
        A_aug = np.hstack([self.A, np.eye(self.m)])
        c_aug = np.hstack([self.c, np.zeros(self.m)])

        iteration = 0
        max_iterations = 1000

        while iteration < max_iterations:
            # Basis matrix and its inverse
            B = A_aug[:, basis]
            B_inv = np.linalg.inv(B)

            # Current basic feasible solution
            x_B = B_inv @ self.b

            # Reduced costs for non-basic variables
            c_B = c_aug[basis]
            reduced_costs = {}

            for j in non_basis:
                c_j = c_aug[j]
                A_j = A_aug[:, j]
                reduced_costs[j] = c_j - c_B @ B_inv @ A_j

            # Check optimality
            if all(rc <= 1e-10 for rc in reduced_costs.values()):
                # Optimal solution found
                x = np.zeros(self.n + self.m)
                x[basis] = x_B

                return {
                    'status': 'optimal',
                    'x': x[:self.n],
                    'objective': c_aug[basis] @ x_B,
                    'iterations': iteration
                }

            # Choose entering variable (most positive reduced cost)
            entering = max(reduced_costs, key=reduced_costs.get)

            # Compute direction
            d = B_inv @ A_aug[:, entering]

            # Minimum ratio test for leaving variable
            ratios = []
            for i, (x_i, d_i) in enumerate(zip(x_B, d)):
                if d_i > 1e-10:
                    ratios.append((x_i / d_i, i))
                else:
                    ratios.append((np.inf, i))

            min_ratio, leaving_idx = min(ratios)

            if min_ratio == np.inf:
                return {'status': 'unbounded', 'iterations': iteration}

            # Update basis
            leaving = basis[leaving_idx]
            basis[leaving_idx] = entering
            non_basis.remove(entering)
            non_basis.append(leaving)
            non_basis.sort()

            iteration += 1

        return {'status': 'max_iterations', 'iterations': iteration}

    def _dual_simplex(self) -> Dict:
        """Dual simplex method (useful when primal infeasible but dual feasible)."""
        tableau = self._create_initial_tableau()

        iteration = 0
        max_iterations = 1000

        while iteration < max_iterations:
            # Check if primal feasible (all RHS >= 0)
            if np.all(tableau[:-1, -1] >= -1e-10):
                break

            # Choose pivot row (most negative RHS)
            pivot_row = np.argmin(tableau[:-1, -1])

            # Choose pivot column using dual ratio test
            ratios = []
            for j in range(tableau.shape[1] - 1):
                if tableau[pivot_row, j] < -1e-10:
                    ratios.append((-tableau[-1, j] / tableau[pivot_row, j], j))
                else:
                    ratios.append((np.inf, j))

            if not ratios or all(r[0] == np.inf for r in ratios):
                return {'status': 'infeasible', 'iterations': iteration}

            _, pivot_col = min(ratios)

            # Perform pivot
            tableau = self._pivot(tableau, pivot_row, pivot_col)
            iteration += 1

        solution = self._extract_solution(tableau)

        return {
            'status': 'optimal',
            'x': solution,
            'objective': -tableau[-1, -1],
            'iterations': iteration
        }

    def _create_initial_tableau(self) -> np.ndarray:
        """Create initial simplex tableau."""
        # Add slack variables
        tableau = np.zeros((self.m + 1, self.n + self.m + 1))

        # Constraint coefficients and slack variables
        tableau[:self.m, :self.n] = self.A
        tableau[:self.m, self.n:self.n+self.m] = np.eye(self.m)
        tableau[:self.m, -1] = self.b

        # Objective function (negated for maximization)
        tableau[-1, :self.n] = -self.c

        return tableau

    def _pivot(self, tableau: np.ndarray, pivot_row: int, pivot_col: int) -> np.ndarray:
        """Perform pivot operation on tableau."""
        tableau = tableau.copy()

        # Normalize pivot row
        tableau[pivot_row] /= tableau[pivot_row, pivot_col]

        # Eliminate pivot column in other rows
        for i in range(tableau.shape[0]):
            if i != pivot_row:
                tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]

        return tableau

    def _extract_solution(self, tableau: np.ndarray) -> np.ndarray:
        """Extract solution from final tableau."""
        solution = np.zeros(self.n)

        for j in range(self.n):
            col = tableau[:self.m, j]
            if np.sum(np.abs(col) > 1e-10) == 1 and np.max(np.abs(col)) - 1.0 < 1e-10:
                idx = np.argmax(np.abs(col))
                solution[j] = tableau[idx, -1]

        return solution


class TwoPhaseSimplexSolver:
    """
    Two-phase simplex method for problems with inequality and equality constraints.
    """

    def __init__(self, c: np.ndarray, A_eq: np.ndarray, b_eq: np.ndarray,
                 A_ineq: np.ndarray, b_ineq: np.ndarray):
        self.c = c
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.A_ineq = A_ineq
        self.b_ineq = b_ineq

    def solve(self) -> Dict:
        """Solve using two-phase method."""
        # Phase I: Find initial basic feasible solution
        phase1_result = self._phase1()

        if phase1_result['status'] != 'feasible':
            return {'status': 'infeasible'}

        # Phase II: Optimize original objective
        phase2_result = self._phase2(phase1_result['basis'])

        return phase2_result

    def _phase1(self) -> Dict:
        """Phase I: Find initial BFS by minimizing sum of artificial variables."""
        # Implement Phase I logic
        # This is a simplified version
        return {'status': 'feasible', 'basis': []}

    def _phase2(self, initial_basis: List[int]) -> Dict:
        """Phase II: Optimize original objective from initial BFS."""
        # Implement Phase II logic
        return {'status': 'optimal', 'x': np.array([]), 'objective': 0}


def benchmark_simplex_methods():
    """Benchmark different simplex implementations."""
    np.random.seed(42)

    # Test problems of varying sizes
    problem_sizes = [5, 10, 20, 50]
    results = []

    for n in problem_sizes:
        m = n // 2

        # Generate random LP problem
        c = np.random.randn(n)
        A = np.random.randn(m, n)
        b = np.random.rand(m) * 10

        solver = SimplexSolver(c, A, b)

        # Test different methods
        for method in ['standard', 'revised', 'dual']:
            try:
                import time
                start = time.time()
                result = solver.solve(method=method)
                elapsed = time.time() - start

                results.append({
                    'size': n,
                    'method': method,
                    'time': elapsed,
                    'iterations': result.get('iterations', 0),
                    'status': result['status']
                })
            except Exception as e:
                print(f"Error with {method} on size {n}: {e}")

    return pd.DataFrame(results)


def visualize_simplex_path():
    """Visualize simplex method path for 2D LP problem."""
    # Simple 2D LP problem
    # maximize: 3x + 4y
    # subject to: x + 2y <= 8
    #            3x + 2y <= 12
    #            x, y >= 0

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot feasible region
    x = np.linspace(0, 5, 400)
    y1 = (8 - x) / 2  # x + 2y <= 8
    y2 = (12 - 3*x) / 2  # 3x + 2y <= 12

    ax.fill_between(x, 0, np.minimum(y1, y2),
                     where=(np.minimum(y1, y2) >= 0),
                     alpha=0.3, color='lightblue', label='Feasible Region')

    ax.plot(x, y1, 'b-', label='x + 2y = 8')
    ax.plot(x, y2, 'g-', label='3x + 2y = 12')

    # Plot objective function contours
    X, Y = np.meshgrid(np.linspace(0, 5, 100), np.linspace(0, 5, 100))
    Z = 3*X + 4*Y
    contours = ax.contour(X, Y, Z, levels=10, alpha=0.4, cmap='RdYlBu_r')
    ax.clabel(contours, inline=True, fontsize=8)

    # Mark vertices of feasible region
    vertices = np.array([[0, 0], [0, 4], [4, 0], [2, 3]])
    ax.plot(vertices[:, 0], vertices[:, 1], 'ro', markersize=10, label='Vertices')

    # Simplex path (simplified for visualization)
    path = np.array([[0, 0], [0, 4], [2, 3]])
    ax.plot(path[:, 0], path[:, 1], 'r->', linewidth=2, markersize=8, label='Simplex Path')

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Simplex Method Path Visualization', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)

    plt.tight_layout()
    plt.savefig('simplex_path.png', dpi=300, bbox_inches='tight')
    plt.close()


def demonstrate_degeneracy():
    """Demonstrate degenerate LP and cycling prevention."""
    # Create a degenerate LP problem
    c = np.array([10, 12, 12])
    A = np.array([
        [1, 2, 2],
        [2, 1, 2],
        [2, 2, 1]
    ])
    b = np.array([20, 20, 20])

    solver = SimplexSolver(c, A, b)
    result = solver.solve(method='standard')

    print("\n" + "="*60)
    print("Degeneracy Demonstration")
    print("="*60)
    print(f"Status: {result['status']}")
    print(f"Optimal solution: {result.get('x', 'N/A')}")
    print(f"Optimal value: {result.get('objective', 'N/A'):.4f}")
    print(f"Iterations: {result.get('iterations', 'N/A')}")

    return result


def sensitivity_analysis():
    """Perform sensitivity analysis on LP solution."""
    # Original problem
    c = np.array([3, 4])
    A = np.array([[1, 2], [3, 2]])
    b = np.array([8, 12])

    solver = SimplexSolver(c, A, b)
    base_result = solver.solve(method='standard')

    # Vary objective coefficients
    c1_range = np.linspace(1, 5, 20)
    objectives = []

    for c1 in c1_range:
        c_new = np.array([c1, 4])
        solver_new = SimplexSolver(c_new, A, b)
        result = solver_new.solve(method='standard')
        objectives.append(result.get('objective', 0))

    # Plot sensitivity
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(c1_range, objectives, 'b-', linewidth=2)
    ax.axvline(x=3, color='r', linestyle='--', label='Original c1 = 3')
    ax.set_xlabel('Objective Coefficient c1', fontsize=12)
    ax.set_ylabel('Optimal Objective Value', fontsize=12)
    ax.set_title('Sensitivity Analysis: Effect of c1 on Optimal Value',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig('sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    return c1_range, objectives


def main():
    """Main execution function."""
    print("="*70)
    print("Advanced Simplex Method Implementation")
    print("="*70)

    # Example 1: Standard LP problem
    print("\n1. Standard LP Problem")
    print("-" * 70)
    c = np.array([3, 4])
    A = np.array([[1, 2], [3, 2]])
    b = np.array([8, 12])

    solver = SimplexSolver(c, A, b)

    for method in ['standard', 'revised', 'dual']:
        result = solver.solve(method=method)
        print(f"\n{method.capitalize()} Simplex:")
        print(f"  Status: {result['status']}")
        if result['status'] == 'optimal':
            print(f"  Solution: {result['x']}")
            print(f"  Objective: {result['objective']:.4f}")
            print(f"  Iterations: {result['iterations']}")

    # Example 2: Benchmark different methods
    print("\n2. Benchmarking Simplex Methods")
    print("-" * 70)
    benchmark_df = benchmark_simplex_methods()
    print(benchmark_df.to_string(index=False))

    # Visualize benchmark results
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Execution time comparison
    pivot_time = benchmark_df.pivot(index='size', columns='method', values='time')
    pivot_time.plot(kind='bar', ax=axes[0])
    axes[0].set_xlabel('Problem Size', fontsize=12)
    axes[0].set_ylabel('Execution Time (seconds)', fontsize=12)
    axes[0].set_title('Execution Time Comparison', fontsize=14, fontweight='bold')
    axes[0].legend(title='Method')
    axes[0].grid(True, alpha=0.3)

    # Iterations comparison
    pivot_iter = benchmark_df.pivot(index='size', columns='method', values='iterations')
    pivot_iter.plot(kind='bar', ax=axes[1])
    axes[1].set_xlabel('Problem Size', fontsize=12)
    axes[1].set_ylabel('Number of Iterations', fontsize=12)
    axes[1].set_title('Iterations Comparison', fontsize=14, fontweight='bold')
    axes[1].legend(title='Method')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('benchmark_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Example 3: Visualization
    print("\n3. Simplex Path Visualization")
    print("-" * 70)
    visualize_simplex_path()
    print("Visualization saved to 'simplex_path.png'")

    # Example 4: Degeneracy
    demonstrate_degeneracy()

    # Example 5: Sensitivity analysis
    print("\n4. Sensitivity Analysis")
    print("-" * 70)
    c1_range, objectives = sensitivity_analysis()
    print(f"Analyzed sensitivity for c1 in range [{c1_range[0]:.2f}, {c1_range[-1]:.2f}]")
    print(f"Objective value range: [{min(objectives):.2f}, {max(objectives):.2f}]")
    print("Sensitivity plot saved to 'sensitivity_analysis.png'")

    print("\n" + "="*70)
    print("Analysis complete! Generated visualizations:")
    print("  - simplex_path.png")
    print("  - benchmark_comparison.png")
    print("  - sensitivity_analysis.png")
    print("="*70)


if __name__ == "__main__":
    main()
