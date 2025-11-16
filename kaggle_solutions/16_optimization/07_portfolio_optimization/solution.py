"""
Portfolio Optimization
======================

This example demonstrates portfolio optimization using Modern Portfolio Theory
and various optimization techniques to balance risk and return.

Problem: Allocate capital across assets to maximize return for given risk level,
or minimize risk for given return level.

Author: Kaggle Solutions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, linprog
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class PortfolioOptimizer:
    """
    Portfolio optimization using Modern Portfolio Theory.
    """

    def __init__(self, seed=42):
        """Initialize the optimizer."""
        self.seed = seed
        np.random.seed(seed)
        self.results = {}

    def generate_market_data(self, n_assets=8, n_periods=252) -> Dict:
        """
        Generate synthetic market data.

        Args:
            n_assets: Number of assets
            n_periods: Number of time periods (e.g., 252 trading days)

        Returns:
            Dictionary with market data
        """
        # Generate random returns
        mean_returns = np.random.uniform(0.05, 0.20, n_assets)  # 5-20% annual
        volatilities = np.random.uniform(0.10, 0.40, n_assets)  # 10-40% annual

        # Generate correlation matrix
        L = np.random.randn(n_assets, n_assets)
        corr_matrix = L @ L.T
        # Normalize to correlation matrix
        d = np.sqrt(np.diag(corr_matrix))
        corr_matrix = corr_matrix / np.outer(d, d)

        # Convert to covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * corr_matrix

        data = {
            'n_assets': n_assets,
            'mean_returns': mean_returns,
            'cov_matrix': cov_matrix,
            'volatilities': volatilities,
            'asset_names': [f'Asset {chr(65+i)}' for i in range(n_assets)]
        }

        return data

    def portfolio_metrics(self, weights: np.ndarray, data: Dict) -> Tuple[float, float]:
        """
        Calculate portfolio return and risk.

        Args:
            weights: Asset weights
            data: Market data dictionary

        Returns:
            Tuple of (expected_return, volatility)
        """
        expected_return = np.sum(weights * data['mean_returns'])
        variance = weights @ data['cov_matrix'] @ weights
        volatility = np.sqrt(variance)

        return expected_return, volatility

    def solve_equal_weight(self, data: Dict) -> Dict:
        """
        Naive equal-weight portfolio (1/n rule).

        Args:
            data: Market data dictionary

        Returns:
            Solution dictionary
        """
        print("\n" + "="*60)
        print("Method 1: Equal Weight (1/n)")
        print("="*60)

        n = data['n_assets']
        weights = np.ones(n) / n

        ret, vol = self.portfolio_metrics(weights, data)

        solution = {
            'method': 'Equal Weight',
            'weights': weights,
            'return': ret,
            'volatility': vol,
            'sharpe_ratio': ret / vol if vol > 0 else 0,
            'success': True
        }

        print(f"Expected Return: {ret*100:.2f}%")
        print(f"Volatility: {vol*100:.2f}%")
        print(f"Sharpe Ratio: {solution['sharpe_ratio']:.3f}")

        return solution

    def solve_min_variance(self, data: Dict) -> Dict:
        """
        Minimum variance portfolio (lowest risk).

        Args:
            data: Market data dictionary

        Returns:
            Solution dictionary
        """
        print("\n" + "="*60)
        print("Method 2: Minimum Variance Portfolio")
        print("="*60)

        n = data['n_assets']

        # Objective: minimize w^T Σ w
        def objective(w):
            return w @ data['cov_matrix'] @ w

        # Constraints: weights sum to 1
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        # Bounds: 0 <= w_i <= 1 (no short selling)
        bounds = [(0, 1) for _ in range(n)]

        # Initial guess
        w0 = np.ones(n) / n

        result = minimize(objective, w0, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        if result.success:
            weights = result.x
            ret, vol = self.portfolio_metrics(weights, data)

            solution = {
                'method': 'Minimum Variance',
                'weights': weights,
                'return': ret,
                'volatility': vol,
                'sharpe_ratio': ret / vol if vol > 0 else 0,
                'success': True
            }

            print(f"Expected Return: {ret*100:.2f}%")
            print(f"Volatility: {vol*100:.2f}%")
            print(f"Sharpe Ratio: {solution['sharpe_ratio']:.3f}")

        else:
            solution = {
                'method': 'Minimum Variance',
                'success': False,
                'message': result.message
            }
            print(f"Failed: {result.message}")

        return solution

    def solve_max_sharpe(self, data: Dict, risk_free_rate=0.02) -> Dict:
        """
        Maximum Sharpe ratio portfolio (best risk-adjusted return).

        Args:
            data: Market data dictionary
            risk_free_rate: Risk-free rate

        Returns:
            Solution dictionary
        """
        print("\n" + "="*60)
        print("Method 3: Maximum Sharpe Ratio Portfolio")
        print("="*60)

        n = data['n_assets']

        # Objective: maximize (return - rf) / volatility
        # Equivalent to minimizing negative Sharpe ratio
        def objective(w):
            ret, vol = self.portfolio_metrics(w, data)
            sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0
            return -sharpe  # Minimize negative

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        bounds = [(0, 1) for _ in range(n)]
        w0 = np.ones(n) / n

        result = minimize(objective, w0, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        if result.success:
            weights = result.x
            ret, vol = self.portfolio_metrics(weights, data)

            solution = {
                'method': 'Maximum Sharpe Ratio',
                'weights': weights,
                'return': ret,
                'volatility': vol,
                'sharpe_ratio': (ret - risk_free_rate) / vol if vol > 0 else 0,
                'success': True
            }

            print(f"Expected Return: {ret*100:.2f}%")
            print(f"Volatility: {vol*100:.2f}%")
            print(f"Sharpe Ratio: {solution['sharpe_ratio']:.3f}")

        else:
            solution = {
                'method': 'Maximum Sharpe Ratio',
                'success': False,
                'message': result.message
            }
            print(f"Failed: {result.message}")

        return solution

    def solve_target_return(self, data: Dict, target_return=0.12) -> Dict:
        """
        Minimum variance portfolio with target return constraint.

        Args:
            data: Market data dictionary
            target_return: Desired return level

        Returns:
            Solution dictionary
        """
        print("\n" + "="*60)
        print(f"Method 4: Target Return ({target_return*100:.0f}%) Portfolio")
        print("="*60)

        n = data['n_assets']

        def objective(w):
            return w @ data['cov_matrix'] @ w

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.sum(w * data['mean_returns']) - target_return}
        ]

        bounds = [(0, 1) for _ in range(n)]
        w0 = np.ones(n) / n

        result = minimize(objective, w0, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        if result.success:
            weights = result.x
            ret, vol = self.portfolio_metrics(weights, data)

            solution = {
                'method': f'Target Return ({target_return*100:.0f}%)',
                'weights': weights,
                'return': ret,
                'volatility': vol,
                'sharpe_ratio': ret / vol if vol > 0 else 0,
                'success': True
            }

            print(f"Expected Return: {ret*100:.2f}%")
            print(f"Volatility: {vol*100:.2f}%")
            print(f"Sharpe Ratio: {solution['sharpe_ratio']:.3f}")

        else:
            solution = {
                'method': f'Target Return ({target_return*100:.0f}%)',
                'success': False,
                'message': result.message
            }
            print(f"Failed: {result.message}")

        return solution

    def compute_efficient_frontier(self, data: Dict, n_points=50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the efficient frontier.

        Args:
            data: Market data dictionary
            n_points: Number of points to compute

        Returns:
            Tuple of (returns, volatilities)
        """
        print("\n" + "="*60)
        print("Computing Efficient Frontier")
        print("="*60)

        n = data['n_assets']
        min_ret = np.min(data['mean_returns'])
        max_ret = np.max(data['mean_returns'])

        target_returns = np.linspace(min_ret, max_ret, n_points)
        frontier_vols = []
        frontier_rets = []

        for target_ret in target_returns:
            def objective(w):
                return w @ data['cov_matrix'] @ w

            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: np.sum(w * data['mean_returns']) - target_ret}
            ]

            bounds = [(0, 1) for _ in range(n)]
            w0 = np.ones(n) / n

            result = minimize(objective, w0, method='SLSQP',
                            bounds=bounds, constraints=constraints,
                            options={'disp': False})

            if result.success:
                weights = result.x
                ret, vol = self.portfolio_metrics(weights, data)
                frontier_rets.append(ret)
                frontier_vols.append(vol)

        print(f"Computed {len(frontier_rets)} points on efficient frontier")

        return np.array(frontier_rets), np.array(frontier_vols)

    def solve_all_methods(self, data: Dict):
        """Solve using all methods."""
        print("\nPORTFOLIO OPTIMIZATION")
        print("="*60)
        print(f"Number of assets: {data['n_assets']}")
        print(f"Average return: {np.mean(data['mean_returns'])*100:.2f}%")
        print(f"Average volatility: {np.mean(data['volatilities'])*100:.2f}%")
        print("="*60)

        self.results['equal'] = self.solve_equal_weight(data)
        self.results['min_var'] = self.solve_min_variance(data)
        self.results['max_sharpe'] = self.solve_max_sharpe(data)
        self.results['target'] = self.solve_target_return(data, target_return=0.12)

        return self.results

    def visualize_portfolios(self, data: Dict):
        """Visualize portfolio allocations and performance."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Asset allocation comparison
        ax = axes[0, 0]

        methods = [v['method'] for v in self.results.values() if v['success']]
        n_methods = len(methods)
        x = np.arange(data['n_assets'])
        width = 0.8 / n_methods

        for idx, method_key in enumerate(['equal', 'min_var', 'max_sharpe', 'target']):
            result = self.results[method_key]
            if result['success']:
                offset = (idx - n_methods/2) * width + width/2
                ax.bar(x + offset, result['weights'], width,
                      label=result['method'], alpha=0.8)

        ax.set_xlabel('Assets', fontsize=12)
        ax.set_ylabel('Weight', fontsize=12)
        ax.set_title('Portfolio Allocations', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(data['asset_names'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 2: Risk-Return scatter with efficient frontier
        ax = axes[0, 1]

        # Compute efficient frontier
        frontier_rets, frontier_vols = self.compute_efficient_frontier(data, n_points=50)

        ax.plot(frontier_vols * 100, frontier_rets * 100, 'b-',
               linewidth=2, label='Efficient Frontier')

        # Plot portfolios
        colors = ['green', 'blue', 'red', 'orange']
        markers = ['o', 's', '^', 'D']

        for idx, method_key in enumerate(['equal', 'min_var', 'max_sharpe', 'target']):
            result = self.results[method_key]
            if result['success']:
                ax.scatter(result['volatility'] * 100, result['return'] * 100,
                          s=200, c=colors[idx], marker=markers[idx],
                          label=result['method'], edgecolors='black', linewidth=2,
                          zorder=3)

        # Plot individual assets
        ax.scatter(data['volatilities'] * 100, data['mean_returns'] * 100,
                  s=100, c='gray', marker='x', label='Individual Assets',
                  linewidths=2, zorder=2)

        ax.set_xlabel('Volatility (Risk) %', fontsize=12)
        ax.set_ylabel('Expected Return %', fontsize=12)
        ax.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Sharpe ratio comparison
        ax = axes[1, 0]

        sharpe_ratios = [v['sharpe_ratio'] for v in self.results.values() if v['success']]
        valid_methods = [v['method'] for v in self.results.values() if v['success']]

        bars = ax.bar(valid_methods, sharpe_ratios, color=colors[:len(valid_methods)],
                     alpha=0.7, edgecolor='black', linewidth=2)

        # Highlight best
        if sharpe_ratios:
            best_idx = np.argmax(sharpe_ratios)
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(4)

        for bar, sr in zip(bars, sharpe_ratios):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{sr:.3f}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_ylabel('Sharpe Ratio', fontsize=12)
        ax.set_title('Risk-Adjusted Performance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 4: Portfolio weights pie chart (best portfolio)
        ax = axes[1, 1]

        best_result = max([v for v in self.results.values() if v['success']],
                         key=lambda x: x['sharpe_ratio'])

        # Only show significant weights
        weights = best_result['weights']
        significant = weights > 0.01
        labels = [data['asset_names'][i] if significant[i] else ''
                 for i in range(len(weights))]

        wedges, texts, autotexts = ax.pie(weights, labels=labels,
                                          autopct=lambda pct: f'{pct:.1f}%' if pct > 1 else '',
                                          startangle=90,
                                          textprops={'fontsize': 10, 'fontweight': 'bold'})

        ax.set_title(f'{best_result["method"]} Allocation\nSharpe Ratio: {best_result["sharpe_ratio"]:.3f}',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/16_optimization/07_portfolio_optimization/portfolio_analysis.png',
                    dpi=300, bbox_inches='tight')
        print("\nPortfolio visualization saved to: portfolio_analysis.png")
        plt.show()


def main():
    """Main execution function."""
    print("="*60)
    print("PORTFOLIO OPTIMIZATION")
    print("="*60)

    # Create optimizer
    optimizer = PortfolioOptimizer(seed=42)

    # Generate market data
    data = optimizer.generate_market_data(n_assets=8)

    # Display asset characteristics
    print("\nAsset Characteristics:")
    for i, name in enumerate(data['asset_names']):
        print(f"  {name}: Return {data['mean_returns'][i]*100:.2f}%, "
              f"Volatility {data['volatilities'][i]*100:.2f}%")

    # Solve using all methods
    results = optimizer.solve_all_methods(data)

    # Compare results
    print("\n" + "="*60)
    print("Comparison of Portfolios")
    print("="*60)

    comparison_data = []
    for method_key, result in results.items():
        if result['success']:
            comparison_data.append({
                'Method': result['method'],
                'Return': f"{result['return']*100:.2f}%",
                'Volatility': f"{result['volatility']*100:.2f}%",
                'Sharpe Ratio': f"{result['sharpe_ratio']:.3f}"
            })

    df_comparison = pd.DataFrame(comparison_data)
    print("\n", df_comparison.to_string(index=False))

    # Find best portfolio
    successful = {k: v for k, v in results.items() if v['success']}
    best = max(successful.items(), key=lambda x: x[1]['sharpe_ratio'])
    print(f"\nBest portfolio: {best[1]['method']} with Sharpe Ratio {best[1]['sharpe_ratio']:.3f}")

    # Visualize
    optimizer.visualize_portfolios(data)

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
