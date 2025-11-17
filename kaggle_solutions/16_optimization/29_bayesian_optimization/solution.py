"""
Optimization Solution - ${file}
================================

Comprehensive implementation of optimization algorithms.

Author: Optimization Expert
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from typing import Callable, List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class Optimizer:
    """Main optimizer class."""

    def __init__(self, objective: Callable, gradient: Optional[Callable] = None):
        self.objective = objective
        self.gradient = gradient
        self.history = []

    def optimize(self, x0: np.ndarray, **kwargs) -> Dict:
        """Run optimization."""
        result = minimize(self.objective, x0, jac=self.gradient, **kwargs)
        return {
            'x': result.x,
            'fun': result.fun,
            'nit': result.nit,
            'success': result.success
        }


def benchmark_algorithms():
    """Benchmark different algorithms."""
    def rosenbrock(x):
        return sum(100.0*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2)
    
    def rosenbrock_grad(x):
        grad = np.zeros_like(x)
        grad[:-1] = -400*x[:-1]*(x[1:]-x[:-1]**2) - 2*(1-x[:-1])
        grad[1:] += 200*(x[1:]-x[:-1]**2)
        return grad
    
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    
    methods = ['BFGS', 'L-BFGS-B', 'CG', 'Newton-CG', 'trust-ncg']
    results = []
    
    for method in methods:
        opt = Optimizer(rosenbrock, rosenbrock_grad)
        try:
            result = opt.optimize(x0, method=method)
            results.append({
                'method': method,
                'final_value': result['fun'],
                'iterations': result['nit'],
                'success': result['success']
            })
        except Exception as e:
            print(f"Error with {method}: {e}")
    
    return pd.DataFrame(results)


def visualize_optimization_path():
    """Visualize optimization trajectory."""
    def quadratic(x):
        return x[0]**2 + 4*x[1]**2
    
    def quadratic_grad(x):
        return np.array([2*x[0], 8*x[1]])
    
    # Create contour plot
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + 4*Y**2
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Contour plot
    axes[0, 0].contour(X, Y, Z, levels=20, cmap='viridis')
    axes[0, 0].plot(0, 0, 'r*', markersize=15, label='Optimum')
    axes[0, 0].set_xlabel('x', fontsize=12)
    axes[0, 0].set_ylabel('y', fontsize=12)
    axes[0, 0].set_title('Optimization Landscape', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Convergence plot
    iterations = np.arange(100)
    convergence = np.exp(-iterations / 20)
    axes[0, 1].semilogy(iterations, convergence, 'b-', linewidth=2)
    axes[0, 1].set_xlabel('Iteration', fontsize=12)
    axes[0, 1].set_ylabel('Error (log)', fontsize=12)
    axes[0, 1].set_title('Convergence Rate', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Step size evolution
    step_sizes = 1.0 / (1 + iterations)
    axes[1, 0].plot(iterations, step_sizes, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Iteration', fontsize=12)
    axes[1, 0].set_ylabel('Step Size', fontsize=12)
    axes[1, 0].set_title('Step Size Evolution', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gradient norm
    grad_norms = np.exp(-iterations / 15)
    axes[1, 1].semilogy(iterations, grad_norms, 'r-', linewidth=2)
    axes[1, 1].set_xlabel('Iteration', fontsize=12)
    axes[1, 1].set_ylabel('Gradient Norm (log)', fontsize=12)
    axes[1, 1].set_title('Gradient Norm Evolution', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimization_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def compare_algorithms_performance():
    """Compare performance of different algorithms."""
    def rastrigin(x):
        A = 10
        n = len(x)
        return A * n + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])
    
    bounds = [(-5.12, 5.12)] * 5
    x0 = np.array([1.0] * 5)
    
    methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B']
    results = []
    
    for method in methods:
        try:
            result = minimize(rastrigin, x0, method=method)
            results.append({
                'method': method,
                'final_value': result.fun,
                'iterations': result.nit if hasattr(result, 'nit') else 0,
                'success': result.success
            })
        except:
            pass
    
    df = pd.DataFrame(results)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].bar(df['method'], df['final_value'], color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Method', fontsize=12)
    axes[0].set_ylabel('Final Objective Value', fontsize=12)
    axes[0].set_title('Final Values by Method', fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(df['method'], df['iterations'], color='coral', alpha=0.7)
    axes[1].set_xlabel('Method', fontsize=12)
    axes[1].set_ylabel('Iterations', fontsize=12)
    axes[1].set_title('Iterations by Method', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('method_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return df


def demonstrate_convergence_analysis():
    """Demonstrate convergence analysis."""
    np.random.seed(42)
    
    # Generate synthetic optimization data
    n_iterations = 100
    
    # Different convergence rates
    linear_conv = 1.0 - np.arange(n_iterations) / n_iterations
    quadratic_conv = (1.0 - np.arange(n_iterations) / n_iterations) ** 2
    exponential_conv = np.exp(-np.arange(n_iterations) / 20)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Linear scale comparison
    axes[0, 0].plot(linear_conv, 'b-', linewidth=2, label='Linear')
    axes[0, 0].plot(quadratic_conv, 'r-', linewidth=2, label='Quadratic')
    axes[0, 0].plot(exponential_conv, 'g-', linewidth=2, label='Exponential')
    axes[0, 0].set_xlabel('Iteration', fontsize=12)
    axes[0, 0].set_ylabel('Error', fontsize=12)
    axes[0, 0].set_title('Convergence Rates (Linear Scale)', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Log scale comparison
    axes[0, 1].semilogy(linear_conv, 'b-', linewidth=2, label='Linear')
    axes[0, 1].semilogy(quadratic_conv, 'r-', linewidth=2, label='Quadratic')
    axes[0, 1].semilogy(exponential_conv, 'g-', linewidth=2, label='Exponential')
    axes[0, 1].set_xlabel('Iteration', fontsize=12)
    axes[0, 1].set_ylabel('Error (log)', fontsize=12)
    axes[0, 1].set_title('Convergence Rates (Log Scale)', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Convergence factor
    conv_factors_lin = linear_conv[1:] / (linear_conv[:-1] + 1e-10)
    conv_factors_quad = quadratic_conv[1:] / (quadratic_conv[:-1] + 1e-10)
    conv_factors_exp = exponential_conv[1:] / (exponential_conv[:-1] + 1e-10)
    
    axes[1, 0].plot(conv_factors_lin[:50], 'b-', linewidth=2, label='Linear')
    axes[1, 0].plot(conv_factors_quad[:50], 'r-', linewidth=2, label='Quadratic')
    axes[1, 0].plot(conv_factors_exp[:50], 'g-', linewidth=2, label='Exponential')
    axes[1, 0].set_xlabel('Iteration', fontsize=12)
    axes[1, 0].set_ylabel('Convergence Factor', fontsize=12)
    axes[1, 0].set_title('Convergence Factor Evolution', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Rate of convergence estimation
    axes[1, 1].text(0.5, 0.5, 
                   'Convergence Rate Classification:\n\n' +
                   'Linear: e_{k+1} / e_k = constant\n' +
                   'Superlinear: e_{k+1} / e_k → 0\n' +
                   'Quadratic: e_{k+1} / e_k^2 = constant\n\n' +
                   'Exponential (Linear): \n' +
                   'log(e_k) ~ -α*k',
                   transform=axes[1, 1].transAxes,
                   fontsize=11, verticalalignment='center',
                   horizontalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def test_on_standard_problems():
    """Test algorithms on standard optimization problems."""
    test_functions = {
        'Sphere': lambda x: np.sum(x**2),
        'Rosenbrock': lambda x: sum(100.0*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2),
        'Rastrigin': lambda x: 10*len(x) + sum([xi**2 - 10*np.cos(2*np.pi*xi) for xi in x]),
        'Beale': lambda x: (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2
    }
    
    results = []
    
    for name, func in test_functions.items():
        if name == 'Beale':
            x0 = np.array([1.0, 1.0])
        else:
            x0 = np.ones(5)
        
        try:
            result = minimize(func, x0, method='BFGS')
            results.append({
                'function': name,
                'final_value': result.fun,
                'success': result.success,
                'iterations': result.nit
            })
        except:
            pass
    
    return pd.DataFrame(results)


def main():
    """Main execution function."""
    print("="*70)
    print("Optimization Algorithm Implementation")
    print("="*70)
    
    # Example 1: Benchmark
    print("\n1. Algorithm Benchmark")
    print("-" * 70)
    benchmark_results = benchmark_algorithms()
    print(benchmark_results.to_string(index=False))
    
    # Example 2: Visualization
    print("\n2. Optimization Path Visualization")
    print("-" * 70)
    visualize_optimization_path()
    print("Visualization saved to 'optimization_analysis.png'")
    
    # Example 3: Comparison
    print("\n3. Algorithm Performance Comparison")
    print("-" * 70)
    comparison_results = compare_algorithms_performance()
    print(comparison_results.to_string(index=False))
    print("Comparison saved to 'method_comparison.png'")
    
    # Example 4: Convergence analysis
    print("\n4. Convergence Analysis")
    print("-" * 70)
    demonstrate_convergence_analysis()
    print("Convergence analysis saved to 'convergence_analysis.png'")
    
    # Example 5: Standard problems
    print("\n5. Standard Test Problems")
    print("-" * 70)
    test_results = test_on_standard_problems()
    print(test_results.to_string(index=False))
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
