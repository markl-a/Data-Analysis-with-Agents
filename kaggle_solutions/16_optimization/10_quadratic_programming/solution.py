"""
Quadratic Programming Implementation
====================================

This solution implements various quadratic programming (QP) algorithms
for solving optimization problems with quadratic objectives.

Mathematical Background:
-----------------------
Quadratic Programming solves problems of the form:
    minimize: (1/2) x^T Q x + c^T x
    subject to: Ax <= b, Aeq x = beq, lb <= x <= ub

where Q is a positive semi-definite matrix (convex QP) or indefinite (non-convex QP).

Applications:
- Portfolio optimization (Markowitz model)
- Support Vector Machines (SVM)
- Model Predictive Control (MPC)
- Least squares regression

Author: Optimization Expert
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.linalg import cholesky, cho_solve, cho_factor
from typing import Tuple, Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')


class QuadraticProgrammingSolver:
    """
    Solver for Quadratic Programming problems.
    """

    def __init__(self, Q: np.ndarray, c: np.ndarray,
                 A: Optional[np.ndarray] = None,
                 b: Optional[np.ndarray] = None,
                 Aeq: Optional[np.ndarray] = None,
                 beq: Optional[np.ndarray] = None):
        """
        Initialize QP solver.

        Args:
            Q: Quadratic coefficient matrix (n x n)
            c: Linear coefficient vector (n,)
            A: Inequality constraint matrix (m x n), Ax <= b
            b: Inequality RHS (m,)
            Aeq: Equality constraint matrix (p x n), Aeq x = beq
            beq: Equality RHS (p,)
        """
        self.Q = np.array(Q, dtype=float)
        self.c = np.array(c, dtype=float)
        self.n = len(c)

        self.A = A if A is not None else np.array([]).reshape(0, self.n)
        self.b = b if b is not None else np.array([])
        self.Aeq = Aeq if Aeq is not None else np.array([]).reshape(0, self.n)
        self.beq = beq if beq is not None else np.array([])

        self.iteration_history = []

    def solve_active_set(self, x0: Optional[np.ndarray] = None) -> Dict:
        """
        Solve QP using active set method.

        Args:
            x0: Initial point

        Returns:
            Dictionary with solution details
        """
        if x0 is None:
            x0 = np.zeros(self.n)

        x = x0.copy()
        working_set = set()  # Indices of active inequality constraints

        max_iterations = 1000
        tolerance = 1e-6

        for iteration in range(max_iterations):
            # Compute gradient at current point
            grad = self.Q @ x + self.c

            # Build KKT system with active constraints
            if len(working_set) > 0 or len(self.Aeq) > 0:
                # Combine equality and active inequality constraints
                active_A = []
                if len(self.Aeq) > 0:
                    active_A.append(self.Aeq)
                if len(working_set) > 0:
                    active_A.append(self.A[list(working_set)])

                A_active = np.vstack(active_A) if active_A else np.array([]).reshape(0, self.n)

                # Solve KKT system for search direction
                if A_active.shape[0] > 0:
                    # Build KKT matrix
                    m_active = A_active.shape[0]
                    KKT = np.block([
                        [self.Q, A_active.T],
                        [A_active, np.zeros((m_active, m_active))]
                    ])

                    rhs = np.hstack([-grad, np.zeros(m_active)])

                    try:
                        solution = np.linalg.solve(KKT, rhs)
                        p = solution[:self.n]  # Search direction
                        lambda_active = solution[self.n:]  # Lagrange multipliers
                    except np.linalg.LinAlgError:
                        break
                else:
                    p = -grad
                    lambda_active = np.array([])
            else:
                p = -grad
                lambda_active = np.array([])

            # Check for convergence
            if np.linalg.norm(p) < tolerance:
                # Check if all multipliers are non-negative
                if len(working_set) > 0:
                    # Find most negative multiplier
                    min_lambda_idx = np.argmin(lambda_active[len(self.beq):])
                    min_lambda = lambda_active[len(self.beq) + min_lambda_idx]

                    if min_lambda < -tolerance:
                        # Remove constraint from working set
                        working_set_list = sorted(list(working_set))
                        working_set.remove(working_set_list[min_lambda_idx])
                        continue

                # Optimal solution found
                break

            # Line search: find maximum step size
            alpha = 1.0

            # Check inequality constraints
            for i in range(len(self.b)):
                if i not in working_set:
                    denom = self.A[i] @ p
                    if denom > tolerance:
                        alpha_i = (self.b[i] - self.A[i] @ x) / denom
                        if alpha_i < alpha:
                            alpha = alpha_i
                            blocking_constraint = i

            # Update solution
            x_new = x + alpha * p

            # Add blocking constraint to working set
            if alpha < 1.0:
                working_set.add(blocking_constraint)

            x = x_new

            # Store history
            self.iteration_history.append({
                'iteration': iteration,
                'x': x.copy(),
                'objective': 0.5 * x @ self.Q @ x + self.c @ x,
                'active_set_size': len(working_set)
            })

        objective = 0.5 * x @ self.Q @ x + self.c @ x

        return {
            'x': x,
            'objective': objective,
            'iterations': iteration + 1,
            'history': self.iteration_history
        }

    def solve_interior_point(self) -> Dict:
        """Solve QP using interior point method."""

        def objective(x):
            return 0.5 * x @ self.Q @ x + self.c @ x

        def gradient(x):
            return self.Q @ x + self.c

        # Set up constraints
        constraints = []

        if len(self.beq) > 0:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: self.Aeq @ x - self.beq,
                'jac': lambda x: self.Aeq
            })

        if len(self.b) > 0:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: self.b - self.A @ x,
                'jac': lambda x: -self.A
            })

        # Solve using interior point
        result = minimize(objective, np.zeros(self.n), method='trust-constr',
                         jac=gradient, constraints=constraints,
                         options={'verbose': 0})

        return {
            'x': result.x,
            'objective': result.fun,
            'success': result.success,
            'iterations': result.nit
        }

    def solve_conjugate_gradient(self) -> Dict:
        """
        Solve unconstrained QP using conjugate gradient method.
        Only works when there are no constraints.
        """
        if len(self.A) > 0 or len(self.Aeq) > 0:
            raise ValueError("Conjugate gradient only for unconstrained QP")

        x = np.zeros(self.n)
        r = self.Q @ x + self.c  # Gradient
        p = -r  # Search direction

        max_iterations = self.n * 10
        tolerance = 1e-10

        for iteration in range(max_iterations):
            # Check convergence
            if np.linalg.norm(r) < tolerance:
                break

            # Compute step size
            Qp = self.Q @ p
            alpha = (r @ r) / (p @ Qp + 1e-10)

            # Update solution
            x = x + alpha * p

            # Update residual
            r_new = r + alpha * Qp

            # Compute beta (Polak-Ribiere formula)
            beta = (r_new @ r_new) / (r @ r + 1e-10)

            # Update search direction
            p = -r_new + beta * p

            r = r_new

            # Store history
            self.iteration_history.append({
                'iteration': iteration,
                'x': x.copy(),
                'objective': 0.5 * x @ self.Q @ x + self.c @ x,
                'gradient_norm': np.linalg.norm(r)
            })

        objective = 0.5 * x @ self.Q @ x + self.c @ x

        return {
            'x': x,
            'objective': objective,
            'iterations': iteration + 1,
            'history': self.iteration_history
        }


class PortfolioOptimizer:
    """
    Portfolio optimization using quadratic programming (Markowitz model).
    """

    def __init__(self, returns: np.ndarray, cov_matrix: np.ndarray):
        """
        Initialize portfolio optimizer.

        Args:
            returns: Expected returns for each asset
            cov_matrix: Covariance matrix of returns
        """
        self.mu = returns
        self.Sigma = cov_matrix
        self.n_assets = len(returns)

    def optimize(self, target_return: Optional[float] = None,
                risk_aversion: float = 1.0) -> Dict:
        """
        Optimize portfolio allocation.

        Args:
            target_return: Target expected return (if None, use mean-variance)
            risk_aversion: Risk aversion parameter (lambda)

        Returns:
            Optimal portfolio weights and statistics
        """
        if target_return is not None:
            # Minimize variance for target return
            Q = 2 * self.Sigma
            c = np.zeros(self.n_assets)

            # Constraints: sum(w) = 1, mu^T w = target_return, w >= 0
            Aeq = np.vstack([
                np.ones(self.n_assets),
                self.mu
            ])
            beq = np.array([1.0, target_return])

            solver = QuadraticProgrammingSolver(Q, c, Aeq=Aeq, beq=beq)
            result = solver.solve_interior_point()

        else:
            # Mean-variance optimization: min (1/2) w^T Sigma w - lambda * mu^T w
            Q = 2 * self.Sigma
            c = -risk_aversion * self.mu

            # Constraint: sum(w) = 1, w >= 0
            Aeq = np.ones((1, self.n_assets))
            beq = np.array([1.0])

            solver = QuadraticProgrammingSolver(Q, c, Aeq=Aeq, beq=beq)
            result = solver.solve_interior_point()

        weights = result['x']

        # Compute portfolio statistics
        portfolio_return = weights @ self.mu
        portfolio_risk = np.sqrt(weights @ self.Sigma @ weights)
        sharpe_ratio = portfolio_return / (portfolio_risk + 1e-10)

        return {
            'weights': weights,
            'expected_return': portfolio_return,
            'risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio
        }

    def efficient_frontier(self, n_points: int = 50) -> pd.DataFrame:
        """Compute efficient frontier."""
        min_return = np.min(self.mu)
        max_return = np.max(self.mu)

        target_returns = np.linspace(min_return, max_return, n_points)
        results = []

        for target_return in target_returns:
            try:
                result = self.optimize(target_return=target_return)
                results.append({
                    'return': result['expected_return'],
                    'risk': result['risk'],
                    'sharpe': result['sharpe_ratio']
                })
            except:
                pass

        return pd.DataFrame(results)


def demonstrate_qp_methods():
    """Demonstrate different QP solving methods."""
    # Create a simple QP problem
    Q = np.array([[2, 0], [0, 2]])
    c = np.array([-2, -5])
    A = np.array([[1, 1], [-1, 0], [0, -1]])
    b = np.array([3, 0, 0])

    print("Problem formulation:")
    print(f"  minimize: (1/2) x^T Q x + c^T x")
    print(f"  subject to: Ax <= b")
    print(f"\nQ = {Q}")
    print(f"c = {c}")
    print(f"A = {A}")
    print(f"b = {b}")

    results = {}

    # Active set method
    print("\n1. Active Set Method:")
    solver1 = QuadraticProgrammingSolver(Q, c, A, b)
    result1 = solver1.solve_active_set()
    results['active_set'] = result1
    print(f"   Solution: {result1['x']}")
    print(f"   Objective: {result1['objective']:.4f}")
    print(f"   Iterations: {result1['iterations']}")

    # Interior point method
    print("\n2. Interior Point Method:")
    solver2 = QuadraticProgrammingSolver(Q, c, A, b)
    result2 = solver2.solve_interior_point()
    results['interior_point'] = result2
    print(f"   Solution: {result2['x']}")
    print(f"   Objective: {result2['objective']:.4f}")
    print(f"   Iterations: {result2['iterations']}")

    # Unconstrained CG method
    print("\n3. Conjugate Gradient (unconstrained):")
    solver3 = QuadraticProgrammingSolver(Q, c)
    result3 = solver3.solve_conjugate_gradient()
    results['conjugate_gradient'] = result3
    print(f"   Solution: {result3['x']}")
    print(f"   Objective: {result3['objective']:.4f}")
    print(f"   Iterations: {result3['iterations']}")

    return results


def visualize_qp_solution():
    """Visualize QP solution landscape."""
    Q = np.array([[2, 0], [0, 2]])
    c = np.array([-2, -5])
    A = np.array([[1, 1], [-1, 0], [0, -1]])
    b = np.array([3, 0, 0])

    # Create mesh grid
    x1 = np.linspace(-0.5, 3.5, 200)
    x2 = np.linspace(-0.5, 3.5, 200)
    X1, X2 = np.meshgrid(x1, x2)

    # Compute objective function
    Z = 0.5 * (Q[0,0] * X1**2 + Q[1,1] * X2**2) + c[0] * X1 + c[1] * X2

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot contours
    contours = ax.contour(X1, X2, Z, levels=20, cmap='viridis', alpha=0.6)
    ax.clabel(contours, inline=True, fontsize=8)

    # Plot constraints
    x1_range = np.linspace(-0.5, 3.5, 100)

    # x1 + x2 <= 3
    x2_constraint1 = 3 - x1_range
    ax.fill_between(x1_range, -0.5, x2_constraint1,
                     where=(x2_constraint1 >= -0.5),
                     alpha=0.2, color='blue', label='x1 + x2 <= 3')

    # x1 >= 0
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='x1 >= 0')

    # x2 >= 0
    ax.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='x2 >= 0')

    # Solve and plot solution
    solver = QuadraticProgrammingSolver(Q, c, A, b)
    result = solver.solve_active_set()

    ax.plot(result['x'][0], result['x'][1], 'r*', markersize=20,
            label=f"Optimal: ({result['x'][0]:.2f}, {result['x'][1]:.2f})")

    # Plot iteration history if available
    if result['history']:
        x_history = np.array([h['x'] for h in result['history']])
        ax.plot(x_history[:, 0], x_history[:, 1], 'ro-', alpha=0.5, linewidth=2,
                markersize=6, label='Iteration path')

    ax.set_xlabel('x1', fontsize=12)
    ax.set_ylabel('x2', fontsize=12)
    ax.set_title('Quadratic Programming Solution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)

    plt.tight_layout()
    plt.savefig('qp_solution.png', dpi=300, bbox_inches='tight')
    plt.close()


def portfolio_optimization_example():
    """Demonstrate portfolio optimization."""
    np.random.seed(42)

    # Generate random asset returns and covariances
    n_assets = 5
    mu = np.random.uniform(0.05, 0.15, n_assets)  # Expected returns 5-15%

    # Generate positive definite covariance matrix
    A = np.random.randn(n_assets, n_assets)
    Sigma = A @ A.T / n_assets * 0.01  # Scale to reasonable variance

    optimizer = PortfolioOptimizer(mu, Sigma)

    # Compute efficient frontier
    frontier = optimizer.efficient_frontier(n_points=100)

    # Find optimal portfolio for different risk aversions
    portfolios = []
    for risk_aversion in [0.5, 1.0, 2.0, 5.0]:
        result = optimizer.optimize(risk_aversion=risk_aversion)
        portfolios.append({
            'risk_aversion': risk_aversion,
            'return': result['expected_return'],
            'risk': result['risk'],
            'sharpe': result['sharpe_ratio']
        })

    portfolios_df = pd.DataFrame(portfolios)

    # Visualize efficient frontier
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Efficient frontier
    axes[0].plot(frontier['risk'], frontier['return'], 'b-', linewidth=2,
                 label='Efficient Frontier')
    axes[0].scatter(portfolios_df['risk'], portfolios_df['return'],
                   c=portfolios_df['risk_aversion'], cmap='RdYlGn_r',
                   s=100, edgecolors='black', linewidth=2,
                   label='Optimal Portfolios')

    # Plot individual assets
    asset_risks = np.sqrt(np.diag(Sigma))
    axes[0].scatter(asset_risks, mu, marker='s', s=100, c='red',
                   edgecolors='black', linewidth=2, label='Individual Assets')

    axes[0].set_xlabel('Portfolio Risk (Std Dev)', fontsize=12)
    axes[0].set_ylabel('Expected Return', fontsize=12)
    axes[0].set_title('Efficient Frontier', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Sharpe ratio
    axes[1].plot(frontier['risk'], frontier['sharpe'], 'g-', linewidth=2)
    axes[1].scatter(portfolios_df['risk'], portfolios_df['sharpe'],
                   c=portfolios_df['risk_aversion'], cmap='RdYlGn_r',
                   s=100, edgecolors='black', linewidth=2)
    axes[1].set_xlabel('Portfolio Risk (Std Dev)', fontsize=12)
    axes[1].set_ylabel('Sharpe Ratio', fontsize=12)
    axes[1].set_title('Sharpe Ratio vs Risk', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('portfolio_optimization.png', dpi=300, bbox_inches='tight')
    plt.close()

    return frontier, portfolios_df


def main():
    """Main execution function."""
    print("="*70)
    print("Quadratic Programming Implementation")
    print("="*70)

    # Example 1: QP methods comparison
    print("\nExample 1: QP Methods Comparison")
    print("-" * 70)
    results = demonstrate_qp_methods()

    # Example 2: Visualization
    print("\nExample 2: QP Solution Visualization")
    print("-" * 70)
    visualize_qp_solution()
    print("Visualization saved to 'qp_solution.png'")

    # Example 3: Portfolio optimization
    print("\nExample 3: Portfolio Optimization")
    print("-" * 70)
    frontier, portfolios = portfolio_optimization_example()
    print("\nOptimal Portfolios for Different Risk Aversions:")
    print(portfolios.to_string(index=False))
    print("\nPortfolio optimization plots saved to 'portfolio_optimization.png'")

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
