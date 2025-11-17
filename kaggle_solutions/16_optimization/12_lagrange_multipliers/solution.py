"""
Constrained Optimization with Lagrange Multipliers
==================================================

This solution implements constrained optimization using Lagrange multipliers
and KKT conditions for solving equality and inequality constrained problems.

Mathematical Background:
-----------------------
For constrained optimization:
    minimize: f(x)
    subject to: h_i(x) = 0 (equality constraints)
                g_j(x) <= 0 (inequality constraints)

The Lagrangian is:
    L(x, λ, μ) = f(x) + Σ λ_i h_i(x) + Σ μ_j g_j(x)

KKT conditions (necessary for optimality):
1. Stationarity: ∇f(x*) + Σ λ_i ∇h_i(x*) + Σ μ_j ∇g_j(x*) = 0
2. Primal feasibility: h_i(x*) = 0, g_j(x*) <= 0
3. Dual feasibility: μ_j >= 0
4. Complementary slackness: μ_j g_j(x*) = 0

Author: Optimization Expert
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize, NonlinearConstraint
from scipy.linalg import solve
from typing import Callable, List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class LagrangianOptimizer:
    """
    Optimizer using Lagrange multipliers for constrained problems.
    """

    def __init__(self, objective: Callable, gradient: Callable,
                 equality_constraints: Optional[List[Callable]] = None,
                 inequality_constraints: Optional[List[Callable]] = None):
        """
        Initialize Lagrangian optimizer.

        Args:
            objective: Objective function f(x)
            gradient: Gradient of objective ∇f(x)
            equality_constraints: List of equality constraint functions h_i(x)
            inequality_constraints: List of inequality constraint functions g_j(x)
        """
        self.objective = objective
        self.gradient = gradient
        self.eq_constraints = equality_constraints or []
        self.ineq_constraints = inequality_constraints or []
        self.history = []

    def solve_penalty_method(self, x0: np.ndarray, penalty: float = 1.0,
                            penalty_increase: float = 10.0,
                            max_outer: int = 20, max_inner: int = 100) -> Dict:
        """
        Solve using penalty method (quadratic penalty for constraints).

        Args:
            x0: Initial point
            penalty: Initial penalty parameter
            penalty_increase: Factor to increase penalty
            max_outer: Maximum outer iterations
            max_inner: Maximum inner iterations per penalty

        Returns:
            Optimization result
        """
        x = x0.copy()
        rho = penalty

        for outer_iter in range(max_outer):
            # Define penalized objective
            def penalized_objective(x):
                obj = self.objective(x)

                # Add equality constraint penalties
                for h in self.eq_constraints:
                    obj += rho * h(x)**2 / 2

                # Add inequality constraint penalties
                for g in self.ineq_constraints:
                    violation = max(0, g(x))
                    obj += rho * violation**2 / 2

                return obj

            # Minimize penalized objective
            result = minimize(penalized_objective, x, method='BFGS',
                            options={'maxiter': max_inner})

            x = result.x

            # Check convergence
            eq_violation = sum(abs(h(x)) for h in self.eq_constraints)
            ineq_violation = sum(max(0, g(x)) for g in self.ineq_constraints)

            self.history.append({
                'outer_iteration': outer_iter,
                'penalty': rho,
                'objective': self.objective(x),
                'eq_violation': eq_violation,
                'ineq_violation': ineq_violation
            })

            if eq_violation < 1e-6 and ineq_violation < 1e-6:
                break

            # Increase penalty
            rho *= penalty_increase

        return {
            'x': x,
            'objective': self.objective(x),
            'iterations': outer_iter + 1,
            'history': self.history
        }

    def solve_augmented_lagrangian(self, x0: np.ndarray,
                                  penalty: float = 1.0,
                                  max_outer: int = 20,
                                  max_inner: int = 100) -> Dict:
        """
        Solve using augmented Lagrangian method (method of multipliers).

        Args:
            x0: Initial point
            penalty: Penalty parameter
            max_outer: Maximum outer iterations
            max_inner: Maximum inner iterations

        Returns:
            Optimization result
        """
        x = x0.copy()
        rho = penalty

        # Initialize multipliers
        lambda_eq = np.zeros(len(self.eq_constraints))
        mu_ineq = np.zeros(len(self.ineq_constraints))

        self.history = []

        for outer_iter in range(max_outer):
            # Define augmented Lagrangian
            def augmented_lagrangian(x):
                obj = self.objective(x)

                # Equality constraints
                for i, h in enumerate(self.eq_constraints):
                    h_val = h(x)
                    obj += lambda_eq[i] * h_val + rho * h_val**2 / 2

                # Inequality constraints
                for i, g in enumerate(self.ineq_constraints):
                    g_val = g(x)
                    obj += mu_ineq[i] * g_val + rho * max(0, g_val)**2 / 2

                return obj

            # Minimize augmented Lagrangian
            result = minimize(augmented_lagrangian, x, method='BFGS',
                            options={'maxiter': max_inner})

            x = result.x

            # Update multipliers
            for i, h in enumerate(self.eq_constraints):
                lambda_eq[i] += rho * h(x)

            for i, g in enumerate(self.ineq_constraints):
                mu_ineq[i] = max(0, mu_ineq[i] + rho * g(x))

            # Check convergence
            eq_violation = sum(abs(h(x)) for h in self.eq_constraints)
            ineq_violation = sum(max(0, g(x)) for g in self.ineq_constraints)

            self.history.append({
                'outer_iteration': outer_iter,
                'objective': self.objective(x),
                'eq_violation': eq_violation,
                'ineq_violation': ineq_violation,
                'lambda_norm': np.linalg.norm(lambda_eq),
                'mu_norm': np.linalg.norm(mu_ineq)
            })

            if eq_violation < 1e-6 and ineq_violation < 1e-6:
                break

        return {
            'x': x,
            'objective': self.objective(x),
            'lambda': lambda_eq,
            'mu': mu_ineq,
            'iterations': outer_iter + 1,
            'history': self.history
        }


def solve_kkt_system(Q: np.ndarray, c: np.ndarray, A: np.ndarray, b: np.ndarray) -> Dict:
    """
    Solve QP with equality constraints using KKT conditions.

    Problem:
        minimize: (1/2) x^T Q x + c^T x
        subject to: Ax = b

    KKT system:
        [Q   A^T] [x]   = [-c]
        [A    0 ] [λ]     [ b]
    """
    n = len(c)
    m = len(b)

    # Build KKT matrix
    KKT = np.block([
        [Q, A.T],
        [A, np.zeros((m, m))]
    ])

    rhs = np.hstack([-c, b])

    # Solve KKT system
    solution = solve(KKT, rhs)

    x = solution[:n]
    lambda_mult = solution[n:]

    objective = 0.5 * x @ Q @ x + c @ x

    return {
        'x': x,
        'lambda': lambda_mult,
        'objective': objective
    }


def lagrangian_dual_problem():
    """
    Demonstrate Lagrangian duality.

    Primal: minimize f(x) subject to h(x) = 0
    Dual: maximize g(λ) where g(λ) = inf_x L(x,λ)
    """
    # Simple example: minimize x^2 + y^2 subject to x + y = 1

    def primal_objective(x):
        return x[0]**2 + x[1]**2

    def constraint(x):
        return x[0] + x[1] - 1

    # Solve primal using Lagrange multipliers
    # Analytical solution: x* = y* = 0.5, λ* = 1

    # Lagrangian: L(x,y,λ) = x^2 + y^2 + λ(x + y - 1)
    # Stationarity: 2x + λ = 0, 2y + λ = 0
    # Constraint: x + y = 1
    # Solution: x = y = 0.5, λ = -1

    x_analytical = np.array([0.5, 0.5])
    lambda_analytical = -1.0

    print("Analytical solution:")
    print(f"  x* = {x_analytical}")
    print(f"  λ* = {lambda_analytical}")
    print(f"  f(x*) = {primal_objective(x_analytical):.4f}")

    # Verify with numerical solution
    result = minimize(primal_objective, [0, 0],
                     constraints={'type': 'eq', 'fun': constraint})

    print("\nNumerical solution:")
    print(f"  x* = {result.x}")
    print(f"  f(x*) = {result.fun:.4f}")

    return {
        'analytical': {'x': x_analytical, 'lambda': lambda_analytical},
        'numerical': {'x': result.x, 'objective': result.fun}
    }


def visualize_constrained_optimization():
    """Visualize constrained optimization with equality constraint."""
    # Problem: minimize x^2 + 4y^2 subject to x + y = 1

    # Create mesh grid
    x = np.linspace(-1, 2, 200)
    y = np.linspace(-1, 2, 200)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + 4*Y**2

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left plot: contours and constraint
    contours = axes[0].contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    axes[0].clabel(contours, inline=True, fontsize=8)

    # Plot constraint x + y = 1
    x_constraint = np.linspace(-1, 2, 100)
    y_constraint = 1 - x_constraint
    axes[0].plot(x_constraint, y_constraint, 'r-', linewidth=3, label='x + y = 1')

    # Mark optimum
    x_opt = np.array([0.8, 0.2])
    axes[0].plot(x_opt[0], x_opt[1], 'r*', markersize=20, label='Optimum')

    # Plot unconstrained optimum
    axes[0].plot(0, 0, 'b*', markersize=20, label='Unconstrained Opt')

    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('y', fontsize=12)
    axes[0].set_title('Constrained Optimization', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-1, 2)
    axes[0].set_ylim(-1, 2)

    # Right plot: Lagrangian landscape
    lambda_vals = np.linspace(-3, 1, 100)
    L_vals = []

    for lam in lambda_vals:
        # For fixed λ, minimize L(x,y,λ) = x^2 + 4y^2 + λ(x+y-1)
        # Optimal: x = -λ/2, y = -λ/4
        x_lam = -lam / 2
        y_lam = -lam / 4
        L_vals.append(x_lam**2 + 4*y_lam**2 + lam*(x_lam + y_lam - 1))

    axes[1].plot(lambda_vals, L_vals, 'b-', linewidth=2)
    axes[1].axvline(x=-1.6, color='r', linestyle='--', linewidth=2, label='Optimal λ')
    axes[1].set_xlabel('λ (Lagrange Multiplier)', fontsize=12)
    axes[1].set_ylabel('Dual Function g(λ)', fontsize=12)
    axes[1].set_title('Lagrangian Dual Function', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lagrange_multipliers.png', dpi=300, bbox_inches='tight')
    plt.close()


def inequality_constraint_example():
    """Example with inequality constraints and KKT conditions."""
    # Problem: minimize x^2 + y^2 subject to x + y >= 1, x >= 0, y >= 0

    def objective(x):
        return x[0]**2 + x[1]**2

    def gradient(x):
        return 2 * x

    # Inequality constraints (in form g(x) <= 0)
    constraints = [
        lambda x: -(x[0] + x[1] - 1),  # x + y >= 1
        lambda x: -x[0],  # x >= 0
        lambda x: -x[1]   # y >= 0
    ]

    optimizer = LagrangianOptimizer(objective, gradient,
                                   inequality_constraints=constraints)

    x0 = np.array([2.0, 2.0])
    result = optimizer.solve_augmented_lagrangian(x0)

    print("\nInequality Constrained Optimization:")
    print(f"  Solution: {result['x']}")
    print(f"  Objective: {result['objective']:.4f}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Multipliers μ: {result['mu']}")

    # Check KKT conditions
    x_opt = result['x']
    print("\nKKT Conditions Check:")
    print(f"  1. Stationarity: ∇f + Σ μ_i ∇g_i = {gradient(x_opt) + sum(result['mu'][i] * np.array([1, 1]) if i == 0 else result['mu'][i] * np.array([-1, 0]) if i == 1 else result['mu'][i] * np.array([0, -1]) for i in range(3))}")
    print(f"  2. Primal feasibility: g_i(x*) <= 0")
    for i, g in enumerate(constraints):
        print(f"     g_{i}(x*) = {g(x_opt):.6f}")
    print(f"  3. Dual feasibility: μ_i >= 0")
    print(f"     μ = {result['mu']}")
    print(f"  4. Complementary slackness: μ_i g_i(x*) = 0")
    for i, g in enumerate(constraints):
        print(f"     μ_{i} * g_{i}(x*) = {result['mu'][i] * g(x_opt):.6f}")

    return result


def penalty_method_comparison():
    """Compare penalty method and augmented Lagrangian."""
    # Simple problem: minimize (x-2)^2 + (y-2)^2 subject to x + y = 2

    def objective(x):
        return (x[0] - 2)**2 + (x[1] - 2)**2

    def gradient(x):
        return np.array([2*(x[0] - 2), 2*(x[1] - 2)])

    constraint = lambda x: x[0] + x[1] - 2

    optimizer = LagrangianOptimizer(objective, gradient,
                                   equality_constraints=[constraint])

    x0 = np.array([0.0, 0.0])

    # Penalty method
    result_penalty = optimizer.solve_penalty_method(x0)

    # Augmented Lagrangian
    optimizer2 = LagrangianOptimizer(objective, gradient,
                                    equality_constraints=[constraint])
    result_al = optimizer2.solve_augmented_lagrangian(x0)

    # Visualize convergence
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Penalty method
    history_penalty = pd.DataFrame(result_penalty['history'])
    axes[0].semilogy(history_penalty['outer_iteration'], history_penalty['eq_violation'],
                     'b-o', linewidth=2, markersize=8, label='Constraint Violation')
    axes[0].set_xlabel('Outer Iteration', fontsize=12)
    axes[0].set_ylabel('Constraint Violation (log)', fontsize=12)
    axes[0].set_title('Penalty Method', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Augmented Lagrangian
    history_al = pd.DataFrame(result_al['history'])
    axes[1].semilogy(history_al['outer_iteration'], history_al['eq_violation'],
                    'r-o', linewidth=2, markersize=8, label='Constraint Violation')
    axes[1].plot(history_al['outer_iteration'], history_al['lambda_norm'],
                's-', linewidth=2, markersize=8, label='Multiplier Norm')
    axes[1].set_xlabel('Outer Iteration', fontsize=12)
    axes[1].set_ylabel('Value (log)', fontsize=12)
    axes[1].set_title('Augmented Lagrangian', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('penalty_vs_augmented.png', dpi=300, bbox_inches='tight')
    plt.close()

    return result_penalty, result_al


def main():
    """Main execution function."""
    print("="*70)
    print("Constrained Optimization with Lagrange Multipliers")
    print("="*70)

    # Example 1: Lagrangian duality
    print("\n1. Lagrangian Duality")
    print("-" * 70)
    dual_result = lagrangian_dual_problem()

    # Example 2: KKT system
    print("\n2. Solving QP with KKT Conditions")
    print("-" * 70)
    Q = np.array([[2, 0], [0, 2]])
    c = np.array([1, 1])
    A = np.array([[1, 1]])
    b = np.array([1])

    kkt_result = solve_kkt_system(Q, c, A, b)
    print(f"Solution x = {kkt_result['x']}")
    print(f"Multiplier λ = {kkt_result['lambda']}")
    print(f"Objective = {kkt_result['objective']:.4f}")

    # Example 3: Visualization
    print("\n3. Constrained Optimization Visualization")
    print("-" * 70)
    visualize_constrained_optimization()
    print("Visualization saved to 'lagrange_multipliers.png'")

    # Example 4: Inequality constraints
    print("\n4. Inequality Constraints with KKT")
    print("-" * 70)
    ineq_result = inequality_constraint_example()

    # Example 5: Method comparison
    print("\n5. Penalty vs Augmented Lagrangian")
    print("-" * 70)
    penalty_res, al_res = penalty_method_comparison()
    print(f"\nPenalty Method:")
    print(f"  Solution: {penalty_res['x']}")
    print(f"  Iterations: {penalty_res['iterations']}")
    print(f"\nAugmented Lagrangian:")
    print(f"  Solution: {al_res['x']}")
    print(f"  Iterations: {al_res['iterations']}")
    print("\nComparison plot saved to 'penalty_vs_augmented.png'")

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
