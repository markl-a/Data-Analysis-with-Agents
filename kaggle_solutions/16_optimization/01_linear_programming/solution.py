"""
Linear Programming Solutions
============================

This example demonstrates various linear programming (LP) techniques for optimization.
We'll solve a production planning problem using different approaches.

Problem: A company produces two products (A and B) with limited resources:
- Product A: profit $30, requires 2 units of material, 3 hours of labor
- Product B: profit $40, requires 3 units of material, 2 hours of labor
- Constraints: 120 units of material, 100 hours of labor available
- Goal: Maximize profit

Author: Kaggle Solutions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog, minimize
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class LinearProgrammingSolver:
    """
    Comprehensive Linear Programming solver with multiple methods.
    """

    def __init__(self):
        """Initialize the LP solver."""
        self.results = {}

    def setup_production_problem(self) -> Dict:
        """
        Set up the production planning problem.

        Maximize: 30*x1 + 40*x2
        Subject to:
            2*x1 + 3*x2 <= 120  (material constraint)
            3*x1 + 2*x2 <= 100  (labor constraint)
            x1, x2 >= 0         (non-negativity)
        """
        problem = {
            'c': np.array([-30, -40]),  # Negative because linprog minimizes
            'A_ub': np.array([
                [2, 3],  # Material constraint
                [3, 2]   # Labor constraint
            ]),
            'b_ub': np.array([120, 100]),
            'bounds': [(0, None), (0, None)],
            'products': ['Product A', 'Product B'],
            'constraints': ['Material', 'Labor']
        }
        return problem

    def solve_scipy_linprog(self, problem: Dict) -> Dict:
        """
        Solve using scipy.optimize.linprog (Simplex method).

        Args:
            problem: Problem definition dictionary

        Returns:
            Dictionary with solution results
        """
        print("\n" + "="*60)
        print("Method 1: SciPy linprog (Simplex Method)")
        print("="*60)

        result = linprog(
            c=problem['c'],
            A_ub=problem['A_ub'],
            b_ub=problem['b_ub'],
            bounds=problem['bounds'],
            method='highs'
        )

        solution = {
            'method': 'SciPy linprog',
            'x': result.x,
            'optimal_value': -result.fun,  # Negate back to maximization
            'success': result.success,
            'message': result.message,
            'iterations': result.nit if hasattr(result, 'nit') else None
        }

        print(f"Status: {result.message}")
        print(f"Optimal Solution: Product A = {result.x[0]:.2f}, Product B = {result.x[1]:.2f}")
        print(f"Maximum Profit: ${-result.fun:.2f}")

        return solution

    def solve_graphical_method(self, problem: Dict) -> Dict:
        """
        Solve using graphical method (visualizing feasible region).

        Args:
            problem: Problem definition dictionary

        Returns:
            Dictionary with solution results
        """
        print("\n" + "="*60)
        print("Method 2: Graphical Method")
        print("="*60)

        # Find corner points of feasible region
        corner_points = [
            (0, 0),  # Origin
            (0, min(problem['b_ub'][0]/problem['A_ub'][0,1],
                    problem['b_ub'][1]/problem['A_ub'][1,1])),  # x1=0 intercept
            (min(problem['b_ub'][0]/problem['A_ub'][0,0],
                 problem['b_ub'][1]/problem['A_ub'][1,0]), 0),  # x2=0 intercept
        ]

        # Find intersection of two constraints
        A = problem['A_ub']
        b = problem['b_ub']
        try:
            x_intersect = np.linalg.solve(A, b)
            if all(x_intersect >= 0):
                corner_points.append(tuple(x_intersect))
        except np.linalg.LinAlgError:
            pass

        # Evaluate objective function at each corner point
        c = -problem['c']  # Convert back to maximization
        best_value = -np.inf
        best_point = None

        print("\nEvaluating corner points:")
        for point in corner_points:
            if point[0] >= 0 and point[1] >= 0:
                # Check if point satisfies constraints
                if all(A @ point <= b + 1e-6):
                    value = c @ point
                    print(f"  Point ({point[0]:.2f}, {point[1]:.2f}): Profit = ${value:.2f}")
                    if value > best_value:
                        best_value = value
                        best_point = point

        solution = {
            'method': 'Graphical Method',
            'x': np.array(best_point),
            'optimal_value': best_value,
            'success': True,
            'message': 'Optimal solution found',
            'corner_points': corner_points
        }

        print(f"\nOptimal Solution: Product A = {best_point[0]:.2f}, Product B = {best_point[1]:.2f}")
        print(f"Maximum Profit: ${best_value:.2f}")

        return solution

    def solve_custom_simplex(self, problem: Dict) -> Dict:
        """
        Solve using custom implementation of Simplex algorithm.

        Args:
            problem: Problem definition dictionary

        Returns:
            Dictionary with solution results
        """
        print("\n" + "="*60)
        print("Method 3: Custom Simplex Implementation")
        print("="*60)

        # Convert to standard form: maximize c^T x
        # subject to Ax = b, x >= 0
        c = -problem['c']  # Maximization coefficients
        A = problem['A_ub']
        b = problem['b_ub']

        # Add slack variables
        n_vars = len(c)
        n_constraints = len(b)

        # Create tableau
        tableau = np.zeros((n_constraints + 1, n_vars + n_constraints + 1))

        # Fill constraint rows
        tableau[:n_constraints, :n_vars] = A
        tableau[:n_constraints, n_vars:n_vars+n_constraints] = np.eye(n_constraints)
        tableau[:n_constraints, -1] = b

        # Fill objective row (bottom row)
        tableau[-1, :n_vars] = -c

        iteration = 0
        max_iterations = 100

        print("\nSimplex iterations:")
        while iteration < max_iterations:
            # Check if optimal
            if all(tableau[-1, :-1] >= 0):
                break

            # Choose pivot column (most negative in objective row)
            pivot_col = np.argmin(tableau[-1, :-1])

            # Choose pivot row (minimum ratio test)
            ratios = []
            for i in range(n_constraints):
                if tableau[i, pivot_col] > 0:
                    ratios.append(tableau[i, -1] / tableau[i, pivot_col])
                else:
                    ratios.append(np.inf)

            pivot_row = np.argmin(ratios)

            if ratios[pivot_row] == np.inf:
                print("Problem is unbounded!")
                break

            # Perform pivot operation
            pivot_element = tableau[pivot_row, pivot_col]
            tableau[pivot_row, :] /= pivot_element

            for i in range(n_constraints + 1):
                if i != pivot_row:
                    tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

            iteration += 1
            print(f"  Iteration {iteration}: Objective = ${tableau[-1, -1]:.2f}")

        # Extract solution
        x = np.zeros(n_vars)
        for j in range(n_vars):
            col = tableau[:n_constraints, j]
            if np.sum(col == 1) == 1 and np.sum(col == 0) == n_constraints - 1:
                row = np.where(col == 1)[0][0]
                x[j] = tableau[row, -1]

        solution = {
            'method': 'Custom Simplex',
            'x': x,
            'optimal_value': tableau[-1, -1],
            'success': True,
            'message': 'Optimal solution found',
            'iterations': iteration
        }

        print(f"\nFinal Solution: Product A = {x[0]:.2f}, Product B = {x[1]:.2f}")
        print(f"Maximum Profit: ${tableau[-1, -1]:.2f}")
        print(f"Iterations: {iteration}")

        return solution

    def solve_all_methods(self):
        """Solve the problem using all available methods."""
        problem = self.setup_production_problem()

        print("\nSOLVING LINEAR PROGRAMMING PROBLEM")
        print("="*60)
        print("Production Planning Problem:")
        print("  Product A: Profit $30, Material: 2 units, Labor: 3 hours")
        print("  Product B: Profit $40, Material: 3 units, Labor: 2 hours")
        print("  Available: Material 120 units, Labor 100 hours")
        print("="*60)

        # Solve with different methods
        self.results['scipy'] = self.solve_scipy_linprog(problem)
        self.results['graphical'] = self.solve_graphical_method(problem)
        self.results['simplex'] = self.solve_custom_simplex(problem)

        return self.results

    def visualize_feasible_region(self, problem: Dict, solution: Dict):
        """
        Visualize the feasible region and optimal solution.

        Args:
            problem: Problem definition dictionary
            solution: Solution dictionary
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Left plot: Feasible region
        ax = axes[0]
        x1 = np.linspace(0, 60, 300)

        # Constraint lines
        # Material: 2*x1 + 3*x2 <= 120
        x2_material = (problem['b_ub'][0] - problem['A_ub'][0,0] * x1) / problem['A_ub'][0,1]
        # Labor: 3*x1 + 2*x2 <= 100
        x2_labor = (problem['b_ub'][1] - problem['A_ub'][1,0] * x1) / problem['A_ub'][1,1]

        ax.plot(x1, x2_material, 'b-', label='Material Constraint', linewidth=2)
        ax.plot(x1, x2_labor, 'r-', label='Labor Constraint', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

        # Shade feasible region
        x1_fill = np.linspace(0, 50, 300)
        x2_fill_material = (problem['b_ub'][0] - problem['A_ub'][0,0] * x1_fill) / problem['A_ub'][0,1]
        x2_fill_labor = (problem['b_ub'][1] - problem['A_ub'][1,0] * x1_fill) / problem['A_ub'][1,1]
        x2_fill = np.minimum(x2_fill_material, x2_fill_labor)
        x2_fill = np.maximum(x2_fill, 0)

        ax.fill_between(x1_fill, 0, x2_fill, alpha=0.3, color='green', label='Feasible Region')

        # Plot optimal solution
        opt_x = solution['x']
        ax.plot(opt_x[0], opt_x[1], 'r*', markersize=20, label='Optimal Solution')
        ax.annotate(f'Optimal: ({opt_x[0]:.1f}, {opt_x[1]:.1f})\nProfit: ${solution["optimal_value"]:.2f}',
                   xy=(opt_x[0], opt_x[1]), xytext=(opt_x[0]+5, opt_x[1]+5),
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        # Plot isoprofit lines
        c = -problem['c']
        for profit in [0, 500, 1000, 1500]:
            x2_profit = (profit - c[0] * x1) / c[1]
            ax.plot(x1, x2_profit, 'g--', alpha=0.3, linewidth=1)

        ax.set_xlim(0, 50)
        ax.set_ylim(0, 50)
        ax.set_xlabel('Product A (units)', fontsize=12)
        ax.set_ylabel('Product B (units)', fontsize=12)
        ax.set_title('Feasible Region and Optimal Solution', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Right plot: Constraint utilization
        ax = axes[1]

        A = problem['A_ub']
        b = problem['b_ub']
        used = A @ opt_x
        utilization = (used / b) * 100

        constraints = problem['constraints']
        colors = ['blue', 'red']

        bars = ax.barh(constraints, utilization, color=colors, alpha=0.7)
        ax.axvline(x=100, color='k', linestyle='--', linewidth=2, label='100% Utilization')

        for i, (bar, util, u, cap) in enumerate(zip(bars, utilization, used, b)):
            ax.text(util + 2, i, f'{util:.1f}%\n({u:.1f}/{cap})',
                   va='center', fontsize=10, fontweight='bold')

        ax.set_xlim(0, 110)
        ax.set_xlabel('Utilization (%)', fontsize=12)
        ax.set_title('Resource Utilization at Optimal Solution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/16_optimization/01_linear_programming/lp_visualization.png',
                    dpi=300, bbox_inches='tight')
        print("\nVisualization saved to: lp_visualization.png")
        plt.show()

    def sensitivity_analysis(self, problem: Dict):
        """
        Perform sensitivity analysis on the solution.

        Args:
            problem: Problem definition dictionary
        """
        print("\n" + "="*60)
        print("Sensitivity Analysis")
        print("="*60)

        c_original = -problem['c']
        A = problem['A_ub']
        b = problem['b_ub']

        # Vary profit of Product A
        print("\nVarying Product A profit:")
        profits_A = np.linspace(10, 60, 11)
        optimal_values = []

        for profit_A in profits_A:
            c = np.array([-profit_A, -40])
            result = linprog(c=c, A_ub=A, b_ub=b, bounds=problem['bounds'], method='highs')
            optimal_values.append(-result.fun)
            print(f"  Profit A = ${profit_A:.0f}: Optimal Total Profit = ${-result.fun:.2f}")

        # Visualize sensitivity
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        ax = axes[0]
        ax.plot(profits_A, optimal_values, 'b-o', linewidth=2, markersize=8)
        ax.axvline(x=30, color='r', linestyle='--', label='Original Profit A = $30')
        ax.set_xlabel('Product A Profit ($)', fontsize=12)
        ax.set_ylabel('Optimal Total Profit ($)', fontsize=12)
        ax.set_title('Sensitivity to Product A Profit', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Vary material availability
        print("\nVarying Material availability:")
        materials = np.linspace(60, 180, 11)
        optimal_values_material = []

        ax = axes[1]
        for material in materials:
            b_new = np.array([material, 100])
            result = linprog(c=problem['c'], A_ub=A, b_ub=b_new, bounds=problem['bounds'], method='highs')
            optimal_values_material.append(-result.fun)
            print(f"  Material = {material:.0f}: Optimal Total Profit = ${-result.fun:.2f}")

        ax.plot(materials, optimal_values_material, 'g-o', linewidth=2, markersize=8)
        ax.axvline(x=120, color='r', linestyle='--', label='Original Material = 120')
        ax.set_xlabel('Material Available (units)', fontsize=12)
        ax.set_ylabel('Optimal Total Profit ($)', fontsize=12)
        ax.set_title('Sensitivity to Material Availability', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/16_optimization/01_linear_programming/lp_sensitivity.png',
                    dpi=300, bbox_inches='tight')
        print("\nSensitivity analysis saved to: lp_sensitivity.png")
        plt.show()


def main():
    """Main execution function."""
    print("="*60)
    print("LINEAR PROGRAMMING OPTIMIZATION")
    print("="*60)

    # Create solver
    solver = LinearProgrammingSolver()

    # Solve using all methods
    results = solver.solve_all_methods()

    # Compare results
    print("\n" + "="*60)
    print("Comparison of Methods")
    print("="*60)

    comparison_data = []
    for method_name, result in results.items():
        comparison_data.append({
            'Method': result['method'],
            'Product A': f"{result['x'][0]:.2f}",
            'Product B': f"{result['x'][1]:.2f}",
            'Profit': f"${result['optimal_value']:.2f}",
            'Status': 'Success' if result['success'] else 'Failed'
        })

    df_comparison = pd.DataFrame(comparison_data)
    print("\n", df_comparison.to_string(index=False))

    # Get problem and best solution for visualization
    problem = solver.setup_production_problem()
    best_solution = results['scipy']

    # Visualize
    solver.visualize_feasible_region(problem, best_solution)

    # Sensitivity analysis
    solver.sensitivity_analysis(problem)

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
