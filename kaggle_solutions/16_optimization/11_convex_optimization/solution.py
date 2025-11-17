"""
Convex Optimization Implementation
==================================

This solution implements various convex optimization algorithms including
gradient descent, proximal methods, and ADMM for solving convex problems.

Mathematical Background:
-----------------------
A convex optimization problem has the form:
    minimize: f(x)
    subject to: g_i(x) <= 0, i = 1,...,m
                h_j(x) = 0, j = 1,...,p

where f and g_i are convex functions and h_j are affine.

Key properties of convex problems:
- Any local minimum is a global minimum
- First-order optimality conditions are necessary and sufficient
- Many efficient algorithms exist

Author: Optimization Expert
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from typing import Callable, Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class ConvexOptimizer:
    """
    General framework for convex optimization.
    """

    def __init__(self, objective: Callable, gradient: Optional[Callable] = None,
                 hessian: Optional[Callable] = None):
        """
        Initialize convex optimizer.

        Args:
            objective: Objective function f(x)
            gradient: Gradient function g(x) = ∇f(x)
            hessian: Hessian function H(x) = ∇²f(x)
        """
        self.objective = objective
        self.gradient = gradient
        self.hessian = hessian
        self.history = []

    def gradient_descent(self, x0: np.ndarray, learning_rate: float = 0.01,
                        max_iterations: int = 1000, tolerance: float = 1e-6) -> Dict:
        """
        Standard gradient descent with fixed step size.

        Args:
            x0: Initial point
            learning_rate: Step size (alpha)
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance

        Returns:
            Optimization results
        """
        if self.gradient is None:
            raise ValueError("Gradient function required for gradient descent")

        x = x0.copy()
        self.history = []

        for iteration in range(max_iterations):
            grad = self.gradient(x)

            # Store history
            self.history.append({
                'iteration': iteration,
                'x': x.copy(),
                'objective': self.objective(x),
                'gradient_norm': np.linalg.norm(grad)
            })

            # Check convergence
            if np.linalg.norm(grad) < tolerance:
                break

            # Update
            x = x - learning_rate * grad

        return {
            'x': x,
            'objective': self.objective(x),
            'iterations': iteration + 1,
            'history': self.history
        }

    def accelerated_gradient_descent(self, x0: np.ndarray, learning_rate: float = 0.01,
                                   max_iterations: int = 1000, tolerance: float = 1e-6) -> Dict:
        """
        Nesterov's Accelerated Gradient Descent.

        Args:
            x0: Initial point
            learning_rate: Step size
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance

        Returns:
            Optimization results
        """
        if self.gradient is None:
            raise ValueError("Gradient function required")

        x = x0.copy()
        y = x0.copy()
        momentum = 0.0
        self.history = []

        for iteration in range(max_iterations):
            grad = self.gradient(y)

            # Store history
            self.history.append({
                'iteration': iteration,
                'x': x.copy(),
                'objective': self.objective(x),
                'gradient_norm': np.linalg.norm(grad)
            })

            # Check convergence
            if np.linalg.norm(grad) < tolerance:
                break

            # Update
            x_new = y - learning_rate * grad
            momentum_new = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
            y = x_new + ((momentum - 1) / momentum_new) * (x_new - x)

            x = x_new
            momentum = momentum_new

        return {
            'x': x,
            'objective': self.objective(x),
            'iterations': iteration + 1,
            'history': self.history
        }

    def proximal_gradient(self, x0: np.ndarray, prox_operator: Callable,
                         learning_rate: float = 0.01, max_iterations: int = 1000,
                         tolerance: float = 1e-6) -> Dict:
        """
        Proximal gradient method for composite optimization.
        Solves: minimize f(x) + g(x) where f is smooth and g has a prox operator.

        Args:
            x0: Initial point
            prox_operator: Proximal operator for g
            learning_rate: Step size
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance

        Returns:
            Optimization results
        """
        if self.gradient is None:
            raise ValueError("Gradient function required")

        x = x0.copy()
        self.history = []

        for iteration in range(max_iterations):
            grad = self.gradient(x)

            # Gradient step
            z = x - learning_rate * grad

            # Proximal step
            x_new = prox_operator(z, learning_rate)

            # Store history
            self.history.append({
                'iteration': iteration,
                'x': x.copy(),
                'objective': self.objective(x),
                'step_size': np.linalg.norm(x_new - x)
            })

            # Check convergence
            if np.linalg.norm(x_new - x) < tolerance:
                break

            x = x_new

        return {
            'x': x,
            'objective': self.objective(x),
            'iterations': iteration + 1,
            'history': self.history
        }


class ADMMSolver:
    """
    Alternating Direction Method of Multipliers (ADMM) for distributed optimization.

    Solves problems of the form:
        minimize: f(x) + g(z)
        subject to: Ax + Bz = c
    """

    def __init__(self, f: Callable, g: Callable, A: np.ndarray, B: np.ndarray, c: np.ndarray):
        """
        Initialize ADMM solver.

        Args:
            f: First objective function
            g: Second objective function
            A, B: Constraint matrices
            c: Constraint RHS
        """
        self.f = f
        self.g = g
        self.A = A
        self.B = B
        self.c = c
        self.history = []

    def solve(self, x0: np.ndarray, z0: np.ndarray, rho: float = 1.0,
             max_iterations: int = 1000, tolerance: float = 1e-4) -> Dict:
        """
        Solve using ADMM.

        Args:
            x0: Initial x
            z0: Initial z
            rho: Penalty parameter
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance

        Returns:
            Optimization results
        """
        x = x0.copy()
        z = z0.copy()
        u = np.zeros_like(self.c)  # Scaled dual variable

        for iteration in range(max_iterations):
            # x-update
            x = self._x_update(x, z, u, rho)

            # z-update
            z = self._z_update(x, z, u, rho)

            # u-update (dual variable)
            u = u + self.A @ x + self.B @ z - self.c

            # Store history
            primal_residual = np.linalg.norm(self.A @ x + self.B @ z - self.c)
            dual_residual = np.linalg.norm(rho * self.B.T @ (z - z))

            self.history.append({
                'iteration': iteration,
                'primal_residual': primal_residual,
                'dual_residual': dual_residual,
                'objective': self.f(x) + self.g(z)
            })

            # Check convergence
            if primal_residual < tolerance and dual_residual < tolerance:
                break

        return {
            'x': x,
            'z': z,
            'objective': self.f(x) + self.g(z),
            'iterations': iteration + 1,
            'history': self.history
        }

    def _x_update(self, x: np.ndarray, z: np.ndarray, u: np.ndarray, rho: float) -> np.ndarray:
        """Update x (to be implemented based on specific problem)."""
        # This is a placeholder - actual implementation depends on f
        return x

    def _z_update(self, x: np.ndarray, z: np.ndarray, u: np.ndarray, rho: float) -> np.ndarray:
        """Update z (to be implemented based on specific problem)."""
        # This is a placeholder - actual implementation depends on g
        return z


def lasso_regression_admm(X: np.ndarray, y: np.ndarray, lambda_: float,
                          rho: float = 1.0, max_iterations: int = 1000) -> Dict:
    """
    Solve LASSO regression using ADMM:
        minimize: (1/2)||Xw - y||^2 + lambda ||w||_1
    """
    n, d = X.shape

    # Initialize
    w = np.zeros(d)
    z = np.zeros(d)
    u = np.zeros(d)

    # Precompute for efficiency
    XtX = X.T @ X
    Xty = X.T @ y
    L = XtX + rho * np.eye(d)
    L_inv = np.linalg.inv(L)

    history = []

    for iteration in range(max_iterations):
        # w-update (quadratic)
        w = L_inv @ (Xty + rho * (z - u))

        # z-update (soft thresholding)
        z_old = z.copy()
        z = soft_threshold(w + u, lambda_ / rho)

        # u-update
        u = u + w - z

        # Compute residuals
        primal_residual = np.linalg.norm(w - z)
        dual_residual = np.linalg.norm(rho * (z - z_old))

        history.append({
            'iteration': iteration,
            'primal_residual': primal_residual,
            'dual_residual': dual_residual,
            'objective': 0.5 * np.sum((X @ w - y)**2) + lambda_ * np.sum(np.abs(w))
        })

        if primal_residual < 1e-4 and dual_residual < 1e-4:
            break

    return {
        'w': w,
        'objective': 0.5 * np.sum((X @ w - y)**2) + lambda_ * np.sum(np.abs(w)),
        'iterations': iteration + 1,
        'history': history
    }


def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """Soft thresholding operator for L1 regularization."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def demonstrate_convex_optimization():
    """Demonstrate various convex optimization methods."""
    # Simple quadratic function: f(x) = x^T Q x + c^T x
    Q = np.array([[2, 0], [0, 4]])
    c = np.array([1, 2])

    def objective(x):
        return 0.5 * x @ Q @ x + c @ x

    def gradient(x):
        return Q @ x + c

    def hessian(x):
        return Q

    # Optimal solution (analytical)
    x_opt = -np.linalg.solve(Q, c)
    f_opt = objective(x_opt)

    print(f"Optimal solution (analytical): {x_opt}")
    print(f"Optimal value: {f_opt:.4f}")

    # Test different methods
    x0 = np.array([5.0, 5.0])
    optimizer = ConvexOptimizer(objective, gradient, hessian)

    # Standard gradient descent
    result_gd = optimizer.gradient_descent(x0, learning_rate=0.1)
    print(f"\nGradient Descent:")
    print(f"  Solution: {result_gd['x']}")
    print(f"  Objective: {result_gd['objective']:.4f}")
    print(f"  Iterations: {result_gd['iterations']}")

    # Accelerated gradient descent
    result_agd = optimizer.accelerated_gradient_descent(x0, learning_rate=0.1)
    print(f"\nAccelerated Gradient Descent:")
    print(f"  Solution: {result_agd['x']}")
    print(f"  Objective: {result_agd['objective']:.4f}")
    print(f"  Iterations: {result_agd['iterations']}")

    return {'gd': result_gd, 'agd': result_agd}


def visualize_convergence():
    """Visualize convergence of different methods."""
    Q = np.array([[2, 0], [0, 4]])
    c = np.array([1, 2])

    def objective(x):
        return 0.5 * x @ Q @ x + c @ x

    def gradient(x):
        return Q @ x + c

    x0 = np.array([5.0, 5.0])
    optimizer = ConvexOptimizer(objective, gradient)

    # Run both methods
    result_gd = optimizer.gradient_descent(x0, learning_rate=0.1, max_iterations=100)
    history_gd = pd.DataFrame(result_gd['history'])

    result_agd = optimizer.accelerated_gradient_descent(x0, learning_rate=0.1, max_iterations=100)
    history_agd = pd.DataFrame(result_agd['history'])

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Objective value
    axes[0, 0].semilogy(history_gd['iteration'], history_gd['objective'], 'b-', label='GD', linewidth=2)
    axes[0, 0].semilogy(history_agd['iteration'], history_agd['objective'], 'r-', label='AGD', linewidth=2)
    axes[0, 0].set_xlabel('Iteration', fontsize=12)
    axes[0, 0].set_ylabel('Objective Value (log)', fontsize=12)
    axes[0, 0].set_title('Objective Function Convergence', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Gradient norm
    axes[0, 1].semilogy(history_gd['iteration'], history_gd['gradient_norm'], 'b-', label='GD', linewidth=2)
    axes[0, 1].semilogy(history_agd['iteration'], history_agd['gradient_norm'], 'r-', label='AGD', linewidth=2)
    axes[0, 1].set_xlabel('Iteration', fontsize=12)
    axes[0, 1].set_ylabel('Gradient Norm (log)', fontsize=12)
    axes[0, 1].set_title('Gradient Norm Convergence', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Trajectory in 2D
    x_opt = -np.linalg.solve(Q, c)

    # Create contour plot
    x1 = np.linspace(-2, 6, 100)
    x2 = np.linspace(-2, 6, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = objective(np.array([X1[i, j], X2[i, j]]))

    axes[1, 0].contour(X1, X2, Z, levels=20, cmap='viridis', alpha=0.6)

    # Plot trajectories
    x_traj_gd = np.array([h['x'] for h in result_gd['history']])
    axes[1, 0].plot(x_traj_gd[:, 0], x_traj_gd[:, 1], 'b.-', label='GD', linewidth=2, markersize=4)

    x_traj_agd = np.array([h['x'] for h in result_agd['history']])
    axes[1, 0].plot(x_traj_agd[:, 0], x_traj_agd[:, 1], 'r.-', label='AGD', linewidth=2, markersize=4)

    axes[1, 0].plot(x_opt[0], x_opt[1], 'g*', markersize=20, label='Optimum')
    axes[1, 0].set_xlabel('x1', fontsize=12)
    axes[1, 0].set_ylabel('x2', fontsize=12)
    axes[1, 0].set_title('Optimization Trajectories', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Distance to optimum
    dist_gd = [np.linalg.norm(h['x'] - x_opt) for h in result_gd['history']]
    dist_agd = [np.linalg.norm(h['x'] - x_opt) for h in result_agd['history']]

    axes[1, 1].semilogy(history_gd['iteration'], dist_gd, 'b-', label='GD', linewidth=2)
    axes[1, 1].semilogy(history_agd['iteration'], dist_agd, 'r-', label='AGD', linewidth=2)
    axes[1, 1].set_xlabel('Iteration', fontsize=12)
    axes[1, 1].set_ylabel('Distance to Optimum (log)', fontsize=12)
    axes[1, 1].set_title('Distance to Optimal Point', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('convex_optimization_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()


def lasso_example():
    """Demonstrate LASSO regression with ADMM."""
    np.random.seed(42)

    # Generate sparse regression problem
    n, d = 100, 50
    k = 5  # Number of non-zero coefficients

    X = np.random.randn(n, d)
    w_true = np.zeros(d)
    w_true[:k] = np.random.randn(k)
    y = X @ w_true + 0.1 * np.random.randn(n)

    # Solve with different lambda values
    lambdas = [0.01, 0.1, 0.5, 1.0]
    results = []

    for lambda_ in lambdas:
        result = lasso_regression_admm(X, y, lambda_)
        sparsity = np.sum(np.abs(result['w']) > 1e-4)
        mse = np.mean((X @ result['w'] - y)**2)

        results.append({
            'lambda': lambda_,
            'sparsity': sparsity,
            'mse': mse,
            'objective': result['objective']
        })

    results_df = pd.DataFrame(results)

    print("\nLASSO Results:")
    print(results_df.to_string(index=False))

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(results_df['lambda'], results_df['sparsity'], 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Lambda (Regularization)', fontsize=12)
    axes[0].set_ylabel('Number of Non-zero Coefficients', fontsize=12)
    axes[0].set_title('Sparsity vs Regularization', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(results_df['lambda'], results_df['mse'], 'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Lambda (Regularization)', fontsize=12)
    axes[1].set_ylabel('Mean Squared Error', fontsize=12)
    axes[1].set_title('MSE vs Regularization', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lasso_regularization.png', dpi=300, bbox_inches='tight')
    plt.close()

    return results_df


def main():
    """Main execution function."""
    print("="*70)
    print("Convex Optimization Implementation")
    print("="*70)

    # Example 1: Basic convex optimization
    print("\n1. Convex Optimization Methods")
    print("-" * 70)
    results = demonstrate_convex_optimization()

    # Example 2: Convergence visualization
    print("\n2. Convergence Visualization")
    print("-" * 70)
    visualize_convergence()
    print("Convergence plots saved to 'convex_optimization_convergence.png'")

    # Example 3: LASSO regression
    print("\n3. LASSO Regression with ADMM")
    print("-" * 70)
    lasso_results = lasso_example()
    print("LASSO plots saved to 'lasso_regularization.png'")

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
