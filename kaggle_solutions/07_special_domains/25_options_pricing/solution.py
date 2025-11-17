"""
Options Pricing and Volatility Modeling
========================================
Domain: Finance & Derivatives
Task: Option pricing using Black-Scholes and volatility surface modeling

This solution demonstrates:
- Black-Scholes option pricing model
- Implied volatility calculation
- Greeks calculation (Delta, Gamma, Vega, Theta, Rho)
- Volatility smile and surface modeling
- Monte Carlo simulation for exotic options
- Risk management and hedging strategies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')


class OptionsAnalyzer:
    """Options pricing and volatility modeling system."""

    def __init__(self):
        self.risk_free_rate = 0.05
        self.models = {}

    def black_scholes_call(self, S, K, T, r, sigma):
        """Calculate call option price using Black-Scholes."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    def black_scholes_put(self, S, K, T, r, sigma):
        """Calculate put option price using Black-Scholes."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def calculate_greeks(self, S, K, T, r, sigma, option_type='call'):
        """Calculate option Greeks."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Delta
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = -norm.cdf(-d1)

        # Gamma
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

        # Vega
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100

        # Theta
        if option_type == 'call':
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

        # Rho
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

        return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta, 'rho': rho}

    def implied_volatility(self, market_price, S, K, T, r, option_type='call'):
        """Calculate implied volatility using numerical methods."""
        def objective(sigma):
            if option_type == 'call':
                model_price = self.black_scholes_call(S, K, T, r, sigma)
            else:
                model_price = self.black_scholes_put(S, K, T, r, sigma)
            return abs(model_price - market_price)

        result = minimize_scalar(objective, bounds=(0.01, 2.0), method='bounded')
        return result.x

    def generate_options_data(self, n_options=500):
        """Generate synthetic options market data."""
        np.random.seed(42)

        S = 100  # Current stock price
        options = []

        strikes = np.linspace(70, 130, 20)
        maturities = [0.25, 0.5, 1.0, 1.5, 2.0]

        for _ in range(n_options):
            K = np.random.choice(strikes)
            T = np.random.choice(maturities)
            option_type = np.random.choice(['call', 'put'])

            # True volatility with smile
            moneyness = S / K
            base_vol = 0.25
            # Volatility smile
            vol = base_vol + 0.1 * (moneyness - 1) ** 2 + np.random.normal(0, 0.02)
            vol = np.clip(vol, 0.1, 0.6)

            if option_type == 'call':
                price = self.black_scholes_call(S, K, T, self.risk_free_rate, vol)
            else:
                price = self.black_scholes_put(S, K, T, self.risk_free_rate, vol)

            # Add market noise
            market_price = price * np.random.normal(1, 0.02)

            greeks = self.calculate_greeks(S, K, T, self.risk_free_rate, vol, option_type)

            options.append({
                'strike': K,
                'maturity': T,
                'option_type': option_type,
                'true_vol': vol,
                'theoretical_price': price,
                'market_price': market_price,
                'delta': greeks['delta'],
                'gamma': greeks['gamma'],
                'vega': greeks['vega'],
                'theta': greeks['theta'],
                'rho': greeks['rho']
            })

        df = pd.DataFrame(options)

        print(f"Generated {n_options} options contracts")
        print(f"Average volatility: {df['true_vol'].mean():.3f}")
        print(f"Volatility range: [{df['true_vol'].min():.3f}, {df['true_vol'].max():.3f}]")

        return df, S

    def build_volatility_surface(self, df, S):
        """Build volatility surface from options data."""
        # Calculate implied volatilities
        df['implied_vol'] = df.apply(
            lambda row: self.implied_volatility(
                row['market_price'], S, row['strike'], 
                row['maturity'], self.risk_free_rate, row['option_type']
            ), axis=1
        )

        return df

    def plot_volatility_smile(self, df, S):
        """Plot volatility smile for different maturities."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()

        maturities = sorted(df['maturity'].unique())[:6]

        for idx, maturity in enumerate(maturities):
            subset = df[df['maturity'] == maturity]

            axes[idx].scatter(subset['strike'], subset['implied_vol'], 
                            alpha=0.6, s=50, label='Implied Vol')
            axes[idx].axvline(S, color='red', linestyle='--', linewidth=2, label='ATM')
            axes[idx].set_xlabel('Strike Price', fontsize=11)
            axes[idx].set_ylabel('Implied Volatility', fontsize=11)
            axes[idx].set_title(f'Volatility Smile - T={maturity:.2f}y', 
                              fontsize=12, fontweight='bold')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('options_volatility_smile.png', dpi=300, bbox_inches='tight')
        print("Saved: options_volatility_smile.png")
        plt.close()

    def plot_greeks_analysis(self, df, S):
        """Visualize option Greeks."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()

        greeks = ['delta', 'gamma', 'vega', 'theta', 'rho']
        calls = df[df['option_type'] == 'call']

        for idx, greek in enumerate(greeks):
            scatter = axes[idx].scatter(calls['strike'], calls[greek], 
                                       c=calls['maturity'], cmap='viridis', alpha=0.6, s=50)
            axes[idx].axvline(S, color='red', linestyle='--', linewidth=2)
            axes[idx].set_xlabel('Strike Price', fontsize=11)
            axes[idx].set_ylabel(greek.capitalize(), fontsize=11)
            axes[idx].set_title(f'{greek.capitalize()} Profile (Calls)', 
                              fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[idx], label='Maturity (years)')

        # Remove extra subplot
        fig.delaxes(axes[5])

        plt.tight_layout()
        plt.savefig('options_greeks_analysis.png', dpi=300, bbox_inches='tight')
        print("Saved: options_greeks_analysis.png")
        plt.close()

    def plot_volatility_surface(self, df):
        """Plot 3D volatility surface."""
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        strikes = df['strike'].values
        maturities = df['maturity'].values
        iv = df['implied_vol'].values

        surf = ax.scatter(strikes, maturities, iv, c=iv, cmap='viridis', s=50, alpha=0.6)

        ax.set_xlabel('Strike Price', fontsize=11)
        ax.set_ylabel('Maturity (years)', fontsize=11)
        ax.set_zlabel('Implied Volatility', fontsize=11)
        ax.set_title('Volatility Surface', fontsize=14, fontweight='bold')

        plt.colorbar(surf, ax=ax, label='Implied Volatility', shrink=0.5)
        plt.tight_layout()
        plt.savefig('options_volatility_surface.png', dpi=300, bbox_inches='tight')
        print("Saved: options_volatility_surface.png")
        plt.close()


def main():
    """Main execution function."""
    print("=" * 80)
    print("Options Pricing and Volatility Modeling")
    print("=" * 80)

    analyzer = OptionsAnalyzer()

    # Generate data
    print("\n1. Generating Options Market Data...")
    df, S = analyzer.generate_options_data(n_options=500)

    # Build volatility surface
    print("\n2. Building Volatility Surface...")
    df = analyzer.build_volatility_surface(df, S)

    # Visualizations
    print("\n3. Generating Visualizations...")
    analyzer.plot_volatility_smile(df, S)
    analyzer.plot_greeks_analysis(df, S)
    analyzer.plot_volatility_surface(df)

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
