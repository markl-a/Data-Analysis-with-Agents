"""
Gradient Descent Variants Comparison
====================================

This solution implements and compares various gradient descent algorithms
including momentum, NAG, AdaGrad, RMSprop, and Adam.

Mathematical Background:
-----------------------
Gradient Descent updates: x_{t+1} = x_t - η∇f(x_t)

Variants:
1. Momentum: v_t = γv_{t-1} + η∇f(x_t), x_t+1 = x_t - v_t
2. Nesterov (NAG): v_t = γv_{t-1} + η∇f(x_t - γv_{t-1})
3. AdaGrad: Adaptive learning rates for each parameter
4. RMSprop: Exponential moving average of squared gradients
5. Adam: Combines momentum and RMSprop

Author: Optimization Expert
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Callable, Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class GradientDescentOptimizer:
    """Base gradient descent optimizer."""

    def __init__(self, objective: Callable, gradient: Callable, 
                 bounds: Optional[List[Tuple[float, float]]] = None):
        self.objective = objective
        self.gradient = gradient
        self.bounds = bounds
        self.history = []

    def vanilla_gd(self, x0: np.ndarray, learning_rate: float = 0.01,
                  max_iterations: int = 1000, tolerance: float = 1e-6) -> Dict:
        """Vanilla gradient descent."""
        x = x0.copy()
        
        for iteration in range(max_iterations):
            grad = self.gradient(x)
            
            self.history.append({
                'iteration': iteration,
                'x': x.copy(),
                'objective': self.objective(x),
                'gradient_norm': np.linalg.norm(grad)
            })
            
            if np.linalg.norm(grad) < tolerance:
                break
            
            x = x - learning_rate * grad
            
            if self.bounds is not None:
                x = np.clip(x, [b[0] for b in self.bounds], [b[1] for b in self.bounds])
        
        return {
            'x': x,
            'objective': self.objective(x),
            'iterations': iteration + 1,
            'history': self.history
        }

    def momentum_gd(self, x0: np.ndarray, learning_rate: float = 0.01,
                   momentum: float = 0.9, max_iterations: int = 1000,
                   tolerance: float = 1e-6) -> Dict:
        """Gradient descent with momentum."""
        x = x0.copy()
        v = np.zeros_like(x)  # Velocity
        self.history = []
        
        for iteration in range(max_iterations):
            grad = self.gradient(x)
            
            self.history.append({
                'iteration': iteration,
                'x': x.copy(),
                'objective': self.objective(x),
                'gradient_norm': np.linalg.norm(grad),
                'velocity_norm': np.linalg.norm(v)
            })
            
            if np.linalg.norm(grad) < tolerance:
                break
            
            v = momentum * v - learning_rate * grad
            x = x + v
            
            if self.bounds is not None:
                x = np.clip(x, [b[0] for b in self.bounds], [b[1] for b in self.bounds])
        
        return {
            'x': x,
            'objective': self.objective(x),
            'iterations': iteration + 1,
            'history': self.history
        }

    def nesterov_gd(self, x0: np.ndarray, learning_rate: float = 0.01,
                   momentum: float = 0.9, max_iterations: int = 1000,
                   tolerance: float = 1e-6) -> Dict:
        """Nesterov accelerated gradient descent."""
        x = x0.copy()
        v = np.zeros_like(x)
        self.history = []
        
        for iteration in range(max_iterations):
            # Look-ahead gradient
            x_lookahead = x + momentum * v
            grad = self.gradient(x_lookahead)
            
            self.history.append({
                'iteration': iteration,
                'x': x.copy(),
                'objective': self.objective(x),
                'gradient_norm': np.linalg.norm(grad)
            })
            
            if np.linalg.norm(grad) < tolerance:
                break
            
            v = momentum * v - learning_rate * grad
            x = x + v
            
            if self.bounds is not None:
                x = np.clip(x, [b[0] for b in self.bounds], [b[1] for b in self.bounds])
        
        return {
            'x': x,
            'objective': self.objective(x),
            'iterations': iteration + 1,
            'history': self.history
        }

    def adagrad(self, x0: np.ndarray, learning_rate: float = 0.1,
               max_iterations: int = 1000, tolerance: float = 1e-6,
               epsilon: float = 1e-8) -> Dict:
        """AdaGrad optimizer."""
        x = x0.copy()
        sum_squared_gradients = np.zeros_like(x)
        self.history = []
        
        for iteration in range(max_iterations):
            grad = self.gradient(x)
            
            self.history.append({
                'iteration': iteration,
                'x': x.copy(),
                'objective': self.objective(x),
                'gradient_norm': np.linalg.norm(grad)
            })
            
            if np.linalg.norm(grad) < tolerance:
                break
            
            sum_squared_gradients += grad ** 2
            adapted_lr = learning_rate / (np.sqrt(sum_squared_gradients) + epsilon)
            x = x - adapted_lr * grad
            
            if self.bounds is not None:
                x = np.clip(x, [b[0] for b in self.bounds], [b[1] for b in self.bounds])
        
        return {
            'x': x,
            'objective': self.objective(x),
            'iterations': iteration + 1,
            'history': self.history
        }

    def rmsprop(self, x0: np.ndarray, learning_rate: float = 0.01,
               decay_rate: float = 0.9, max_iterations: int = 1000,
               tolerance: float = 1e-6, epsilon: float = 1e-8) -> Dict:
        """RMSprop optimizer."""
        x = x0.copy()
        squared_gradients = np.zeros_like(x)
        self.history = []
        
        for iteration in range(max_iterations):
            grad = self.gradient(x)
            
            self.history.append({
                'iteration': iteration,
                'x': x.copy(),
                'objective': self.objective(x),
                'gradient_norm': np.linalg.norm(grad)
            })
            
            if np.linalg.norm(grad) < tolerance:
                break
            
            squared_gradients = decay_rate * squared_gradients + (1 - decay_rate) * grad ** 2
            adapted_lr = learning_rate / (np.sqrt(squared_gradients) + epsilon)
            x = x - adapted_lr * grad
            
            if self.bounds is not None:
                x = np.clip(x, [b[0] for b in self.bounds], [b[1] for b in self.bounds])
        
        return {
            'x': x,
            'objective': self.objective(x),
            'iterations': iteration + 1,
            'history': self.history
        }

    def adam(self, x0: np.ndarray, learning_rate: float = 0.01,
            beta1: float = 0.9, beta2: float = 0.999,
            max_iterations: int = 1000, tolerance: float = 1e-6,
            epsilon: float = 1e-8) -> Dict:
        """Adam optimizer."""
        x = x0.copy()
        m = np.zeros_like(x)  # First moment
        v = np.zeros_like(x)  # Second moment
        self.history = []
        
        for iteration in range(max_iterations):
            grad = self.gradient(x)
            
            self.history.append({
                'iteration': iteration,
                'x': x.copy(),
                'objective': self.objective(x),
                'gradient_norm': np.linalg.norm(grad)
            })
            
            if np.linalg.norm(grad) < tolerance:
                break
            
            # Update biased first moment estimate
            m = beta1 * m + (1 - beta1) * grad
            
            # Update biased second moment estimate
            v = beta2 * v + (1 - beta2) * grad ** 2
            
            # Bias correction
            m_hat = m / (1 - beta1 ** (iteration + 1))
            v_hat = v / (1 - beta2 ** (iteration + 1))
            
            # Update parameters
            x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            
            if self.bounds is not None:
                x = np.clip(x, [b[0] for b in self.bounds], [b[1] for b in self.bounds])
        
        return {
            'x': x,
            'objective': self.objective(x),
            'iterations': iteration + 1,
            'history': self.history
        }


def compare_optimizers():
    """Compare all gradient descent variants."""
    
    # Test on Rosenbrock function
    def rosenbrock(x):
        return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    
    def rosenbrock_grad(x):
        grad = np.zeros_like(x)
        grad[:-1] = -400 * x[:-1] * (x[1:] - x[:-1]**2) - 2 * (1 - x[:-1])
        grad[1:] += 200 * (x[1:] - x[:-1]**2)
        return grad
    
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    
    optimizer = GradientDescentOptimizer(rosenbrock, rosenbrock_grad)
    
    results = {}
    
    # Vanilla GD
    optimizer.history = []
    results['Vanilla GD'] = optimizer.vanilla_gd(x0, learning_rate=0.001, max_iterations=1000)
    
    # Momentum
    optimizer.history = []
    results['Momentum'] = optimizer.momentum_gd(x0, learning_rate=0.001, momentum=0.9, max_iterations=1000)
    
    # Nesterov
    optimizer.history = []
    results['Nesterov'] = optimizer.nesterov_gd(x0, learning_rate=0.001, momentum=0.9, max_iterations=1000)
    
    # AdaGrad
    optimizer.history = []
    results['AdaGrad'] = optimizer.adagrad(x0, learning_rate=0.1, max_iterations=1000)
    
    # RMSprop
    optimizer.history = []
    results['RMSprop'] = optimizer.rmsprop(x0, learning_rate=0.01, max_iterations=1000)
    
    # Adam
    optimizer.history = []
    results['Adam'] = optimizer.adam(x0, learning_rate=0.01, max_iterations=1000)
    
    # Visualize comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Objective value convergence
    for name, result in results.items():
        history_df = pd.DataFrame(result['history'])
        axes[0, 0].semilogy(history_df['iteration'], history_df['objective'],
                           linewidth=2, label=name)
    
    axes[0, 0].set_xlabel('Iteration', fontsize=12)
    axes[0, 0].set_ylabel('Objective Value (log)', fontsize=12)
    axes[0, 0].set_title('Convergence Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gradient norm
    for name, result in results.items():
        history_df = pd.DataFrame(result['history'])
        axes[0, 1].semilogy(history_df['iteration'], history_df['gradient_norm'],
                           linewidth=2, label=name)
    
    axes[0, 1].set_xlabel('Iteration', fontsize=12)
    axes[0, 1].set_ylabel('Gradient Norm (log)', fontsize=12)
    axes[0, 1].set_title('Gradient Norm Evolution', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Final results bar chart
    methods = list(results.keys())
    final_values = [results[m]['objective'] for m in methods]
    iterations = [results[m]['iterations'] for m in methods]
    
    axes[1, 0].bar(methods, final_values, color='steelblue', alpha=0.7)
    axes[1, 0].set_ylabel('Final Objective Value', fontsize=12)
    axes[1, 0].set_title('Final Objective Values', fontsize=14, fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Iterations bar chart
    axes[1, 1].bar(methods, iterations, color='coral', alpha=0.7)
    axes[1, 1].set_ylabel('Iterations', fontsize=12)
    axes[1, 1].set_title('Iterations to Convergence', fontsize=14, fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('gradient_descent_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return results


def learning_rate_sensitivity():
    """Analyze learning rate sensitivity."""
    
    def quadratic(x):
        return x[0]**2 + 4*x[1]**2
    
    def quadratic_grad(x):
        return np.array([2*x[0], 8*x[1]])
    
    x0 = np.array([5.0, 5.0])
    learning_rates = [0.001, 0.01, 0.1, 0.2, 0.5]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, lr in enumerate(learning_rates):
        optimizer = GradientDescentOptimizer(quadratic, quadratic_grad)
        result = optimizer.vanilla_gd(x0, learning_rate=lr, max_iterations=50)
        
        history_df = pd.DataFrame(result['history'])
        
        axes[idx].semilogy(history_df['iteration'], history_df['objective'],
                          'b-', linewidth=2)
        axes[idx].set_xlabel('Iteration', fontsize=10)
        axes[idx].set_ylabel('Objective (log)', fontsize=10)
        axes[idx].set_title(f'Learning Rate = {lr}', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
    
    # Summary
    axes[5].axis('off')
    summary_text = """
    Learning Rate Effects:
    
    Too small (< 0.01):
    - Slow convergence
    - Many iterations needed
    
    Optimal (0.01 - 0.2):
    - Fast convergence
    - Stable progress
    
    Too large (> 0.5):
    - Oscillations
    - May diverge
    """
    axes[5].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('learning_rate_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main execution function."""
    print("="*70)
    print("Gradient Descent Variants Comparison")
    print("="*70)
    
    # Example 1: Compare optimizers
    print("\n1. Optimizer Comparison")
    print("-" * 70)
    results = compare_optimizers()
    
    for name, result in results.items():
        print(f"{name}:")
        print(f"  Final objective: {result['objective']:.6e}")
        print(f"  Iterations: {result['iterations']}")
    
    print("\nComparison plot saved to 'gradient_descent_comparison.png'")
    
    # Example 2: Learning rate sensitivity
    print("\n2. Learning Rate Sensitivity Analysis")
    print("-" * 70)
    learning_rate_sensitivity()
    print("Sensitivity analysis saved to 'learning_rate_sensitivity.png'")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
