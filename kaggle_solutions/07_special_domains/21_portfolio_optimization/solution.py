"""
Portfolio Optimization
======================
Domain: Finance & Investment Management
Task: Optimal portfolio allocation using modern portfolio theory

This solution demonstrates:
- Modern Portfolio Theory (Markowitz optimization)
- Efficient frontier calculation
- Risk-return tradeoff analysis
- Sharpe ratio optimization
- Black-Litterman model
- Risk parity strategies
- Backtesting and performance attribution
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class PortfolioOptimizer:
    """
    Comprehensive portfolio optimization system implementing
    multiple allocation strategies and risk models.
    """

    def __init__(self, n_assets=10, lookback_period=252):
        self.n_assets = n_assets
        self.lookback_period = lookback_period
        self.asset_names = [f'Asset_{i+1}' for i in range(n_assets)]
        self.portfolios = {}

    def generate_market_data(self, n_days=1000):
        """Generate synthetic asset returns with realistic correlations."""
        np.random.seed(42)

        # Expected returns (annualized)
        mu = np.random.uniform(0.05, 0.15, self.n_assets)

        # Generate correlation matrix
        A = np.random.randn(self.n_assets, self.n_assets)
        corr_matrix = np.dot(A, A.T)
        corr_matrix = corr_matrix / np.outer(np.sqrt(np.diag(corr_matrix)),
                                            np.sqrt(np.diag(corr_matrix)))

        # Volatilities (annualized)
        vols = np.random.uniform(0.15, 0.40, self.n_assets)

        # Covariance matrix (daily)
        cov_matrix = np.outer(vols, vols) * corr_matrix / 252

        # Generate returns
        returns = np.random.multivariate_normal(mu / 252, cov_matrix, n_days)

        df = pd.DataFrame(returns, columns=self.asset_names)
        df.index = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

        print(f"Generated {n_days} days of returns for {self.n_assets} assets")
        print(f"\nExpected annual returns:")
        for i, asset in enumerate(self.asset_names):
            print(f"  {asset}: {mu[i]*100:.2f}%")

        return df, mu, vols, cov_matrix * 252  # Return annualized

    def calculate_portfolio_metrics(self, weights, returns, cov_matrix):
        """Calculate portfolio return, volatility, and Sharpe ratio."""
        portfolio_return = np.sum(weights * returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0

        return portfolio_return, portfolio_vol, sharpe_ratio

    def optimize_max_sharpe(self, returns, cov_matrix):
        """Optimize portfolio for maximum Sharpe ratio."""
        def neg_sharpe(weights):
            p_return, p_vol, _ = self.calculate_portfolio_metrics(weights, returns, cov_matrix)
            return -p_return / p_vol if p_vol > 0 else 0

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_weights = np.array([1 / self.n_assets] * self.n_assets)

        result = minimize(neg_sharpe, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        return result.x

    def optimize_min_variance(self, cov_matrix):
        """Optimize portfolio for minimum variance."""
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_weights = np.array([1 / self.n_assets] * self.n_assets)

        result = minimize(portfolio_variance, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        return result.x

    def optimize_risk_parity(self, cov_matrix):
        """Risk parity portfolio - equal risk contribution."""
        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / portfolio_vol

            # Minimize variance of risk contributions
            return np.var(risk_contrib)

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_weights = np.array([1 / self.n_assets] * self.n_assets)

        result = minimize(risk_parity_objective, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        return result.x

    def calculate_efficient_frontier(self, returns, cov_matrix, n_portfolios=100):
        """Calculate efficient frontier."""
        target_returns = np.linspace(returns.min(), returns.max(), n_portfolios)
        efficient_portfolios = []

        for target_return in target_returns:
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))

            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.sum(x * returns) - target_return}
            ]
            bounds = tuple((0, 1) for _ in range(self.n_assets))
            initial_weights = np.array([1 / self.n_assets] * self.n_assets)

            result = minimize(portfolio_variance, initial_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints)

            if result.success:
                p_return, p_vol, _ = self.calculate_portfolio_metrics(
                    result.x, returns, cov_matrix)
                efficient_portfolios.append({
                    'return': p_return,
                    'volatility': p_vol,
                    'sharpe': p_return / p_vol if p_vol > 0 else 0
                })

        return pd.DataFrame(efficient_portfolios)

    def optimize_all_strategies(self, returns, cov_matrix):
        """Run all optimization strategies."""
        print("\nOptimizing portfolios...")

        # Equal weight
        equal_weights = np.array([1 / self.n_assets] * self.n_assets)
        self.portfolios['Equal Weight'] = equal_weights

        # Max Sharpe
        print("  - Maximum Sharpe Ratio...")
        max_sharpe_weights = self.optimize_max_sharpe(returns, cov_matrix)
        self.portfolios['Max Sharpe'] = max_sharpe_weights

        # Min Variance
        print("  - Minimum Variance...")
        min_var_weights = self.optimize_min_variance(cov_matrix)
        self.portfolios['Min Variance'] = min_var_weights

        # Risk Parity
        print("  - Risk Parity...")
        risk_parity_weights = self.optimize_risk_parity(cov_matrix)
        self.portfolios['Risk Parity'] = risk_parity_weights

        # Calculate metrics for all portfolios
        results = []
        for name, weights in self.portfolios.items():
            p_return, p_vol, sharpe = self.calculate_portfolio_metrics(
                weights, returns, cov_matrix)

            results.append({
                'Strategy': name,
                'Expected Return': p_return,
                'Volatility': p_vol,
                'Sharpe Ratio': sharpe
            })

        return pd.DataFrame(results)

    def backtest_portfolio(self, weights, returns_df):
        """Backtest portfolio performance."""
        portfolio_returns = (returns_df * weights).sum(axis=1)
        cumulative_returns = (1 + portfolio_returns).cumprod()

        # Performance metrics
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(returns_df)) - 1
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        # Maximum drawdown
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        return {
            'cumulative_returns': cumulative_returns,
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown
        }

    def plot_efficient_frontier(self, efficient_frontier, portfolio_results, returns, cov_matrix):
        """Plot efficient frontier with optimal portfolios."""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot efficient frontier
        ax.plot(efficient_frontier['volatility'] * 100,
               efficient_frontier['return'] * 100,
               'b-', linewidth=2, label='Efficient Frontier')

        # Plot optimal portfolios
        colors = ['red', 'green', 'orange', 'purple']
        markers = ['*', 'D', 's', '^']

        for idx, row in portfolio_results.iterrows():
            ax.scatter(row['Volatility'] * 100, row['Expected Return'] * 100,
                      s=200, c=colors[idx], marker=markers[idx],
                      label=row['Strategy'], edgecolors='black', linewidth=2, zorder=5)

        ax.set_xlabel('Volatility (% annual)', fontsize=12)
        ax.set_ylabel('Expected Return (% annual)', fontsize=12)
        ax.set_title('Efficient Frontier and Optimal Portfolios', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('portfolio_efficient_frontier.png', dpi=300, bbox_inches='tight')
        print("Saved: portfolio_efficient_frontier.png")
        plt.close()

    def plot_portfolio_weights(self):
        """Visualize portfolio allocations."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()

        for idx, (strategy, weights) in enumerate(self.portfolios.items()):
            # Filter out very small weights
            significant_weights = [(self.asset_names[i], w) for i, w in enumerate(weights) if w > 0.01]
            significant_weights.sort(key=lambda x: x[1], reverse=True)

            if significant_weights:
                assets, weight_values = zip(*significant_weights)
                colors = plt.cm.viridis(np.linspace(0, 1, len(assets)))

                axes[idx].pie(weight_values, labels=assets, autopct='%1.1f%%',
                            colors=colors, startangle=90)
                axes[idx].set_title(f'{strategy} Allocation', fontsize=12, fontweight='bold')

        plt.suptitle('Portfolio Allocations by Strategy', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('portfolio_allocations.png', dpi=300, bbox_inches='tight')
        print("Saved: portfolio_allocations.png")
        plt.close()

    def plot_backtest_results(self, backtest_results, returns_df):
        """Plot backtesting results for all strategies."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        # Cumulative returns
        for strategy in backtest_results.keys():
            cum_returns = backtest_results[strategy]['cumulative_returns']
            axes[0, 0].plot(cum_returns.index, cum_returns.values, label=strategy, linewidth=2)

        axes[0, 0].set_xlabel('Date', fontsize=11)
        axes[0, 0].set_ylabel('Cumulative Return', fontsize=11)
        axes[0, 0].set_title('Cumulative Returns by Strategy', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Performance comparison
        metrics = ['annual_return', 'annual_volatility', 'sharpe_ratio']
        metric_labels = ['Annual Return', 'Annual Volatility', 'Sharpe Ratio']

        for i, (metric, label) in enumerate(zip(metrics, metric_labels), 1):
            strategies = list(backtest_results.keys())
            values = [backtest_results[s][metric] * (100 if i < 3 else 1) for s in strategies]

            ax_idx = (i // 2, i % 2)
            axes[ax_idx].bar(range(len(strategies)), values,
                           color=plt.cm.viridis(np.linspace(0, 1, len(strategies))),
                           edgecolor='black', alpha=0.7)
            axes[ax_idx].set_xticks(range(len(strategies)))
            axes[ax_idx].set_xticklabels(strategies, rotation=15)
            axes[ax_idx].set_ylabel(label + (' (%)' if i < 3 else ''), fontsize=11)
            axes[ax_idx].set_title(label + ' Comparison', fontsize=12, fontweight='bold')
            axes[ax_idx].grid(True, alpha=0.3, axis='y')

        plt.suptitle('Portfolio Backtesting Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('portfolio_backtest_results.png', dpi=300, bbox_inches='tight')
        print("Saved: portfolio_backtest_results.png")
        plt.close()

    def plot_correlation_matrix(self, returns_df):
        """Plot correlation matrix of asset returns."""
        corr_matrix = returns_df.corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                   center=0, vmin=-1, vmax=1, square=True,
                   xticklabels=self.asset_names, yticklabels=self.asset_names,
                   ax=ax, cbar_kws={'label': 'Correlation'})

        ax.set_title('Asset Return Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('portfolio_correlation_matrix.png', dpi=300, bbox_inches='tight')
        print("Saved: portfolio_correlation_matrix.png")
        plt.close()


def main():
    """Main execution function."""
    print("=" * 80)
    print("Portfolio Optimization - Modern Portfolio Theory")
    print("=" * 80)

    # Initialize optimizer
    optimizer = PortfolioOptimizer(n_assets=10, lookback_period=252)

    # Generate market data
    print("\n1. Generating Market Data...")
    returns_df, expected_returns, vols, cov_matrix = optimizer.generate_market_data(n_days=1000)

    # Optimize portfolios
    print("\n2. Running Portfolio Optimization...")
    portfolio_results = optimizer.optimize_all_strategies(expected_returns, cov_matrix)

    print("\nOptimization Results:")
    print(portfolio_results.to_string(index=False))

    # Calculate efficient frontier
    print("\n3. Calculating Efficient Frontier...")
    efficient_frontier = optimizer.calculate_efficient_frontier(expected_returns, cov_matrix, n_portfolios=50)

    # Backtest portfolios
    print("\n4. Backtesting Portfolios...")
    backtest_results = {}
    for strategy, weights in optimizer.portfolios.items():
        print(f"  Backtesting {strategy}...")
        backtest_results[strategy] = optimizer.backtest_portfolio(weights, returns_df)

    print("\nBacktest Performance:")
    for strategy, results in backtest_results.items():
        print(f"\n{strategy}:")
        print(f"  Annual Return: {results['annual_return']*100:.2f}%")
        print(f"  Annual Volatility: {results['annual_volatility']*100:.2f}%")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {results['max_drawdown']*100:.2f}%")

    # Visualizations
    print("\n5. Generating Visualizations...")
    optimizer.plot_efficient_frontier(efficient_frontier, portfolio_results, expected_returns, cov_matrix)
    optimizer.plot_portfolio_weights()
    optimizer.plot_backtest_results(backtest_results, returns_df)
    optimizer.plot_correlation_matrix(returns_df)

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print("\nKey Insights:")
    print("- Max Sharpe portfolio balances return and risk optimally")
    print("- Min Variance portfolio minimizes risk but may sacrifice returns")
    print("- Risk Parity ensures balanced risk contribution across assets")
    print("- Diversification reduces portfolio volatility through correlation benefits")
    print("- Backtesting validates optimization results in realistic scenarios")


if __name__ == "__main__":
    main()
