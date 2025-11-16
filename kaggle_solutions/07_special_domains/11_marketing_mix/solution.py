"""
Marketing Mix Modeling System
==============================

Problem: Quantify the impact of different marketing channels on sales/revenue
using regression analysis and optimize marketing budget allocation

Kaggle-style competition: Marketing Analytics
Difficulty: ⭐⭐⭐⭐

This solution demonstrates:
- Multi-channel attribution modeling
- Ad stock effects (carryover)
- Diminishing returns analysis
- Budget optimization
- ROI calculation by channel
- Seasonality and trend decomposition
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class MarketingMixModel:
    """Marketing mix modeling and budget optimization"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.channels = ['TV', 'Radio', 'Digital', 'Print', 'Social', 'Email']

    def create_sample_data(self, n_weeks=104):
        """Generate realistic marketing and sales data (2 years)"""
        np.random.seed(42)

        dates = pd.date_range(start='2022-01-03', periods=n_weeks, freq='W')

        # Base sales with trend and seasonality
        trend = np.linspace(100000, 150000, n_weeks)
        seasonality = 20000 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)
        base_sales = trend + seasonality

        # Marketing spend by channel
        data = {'week': dates}

        for channel in self.channels:
            # Different spend patterns by channel
            if channel == 'TV':
                spend = np.random.uniform(10000, 30000, n_weeks)
            elif channel == 'Digital':
                spend = np.random.uniform(15000, 40000, n_weeks)
            elif channel == 'Social':
                spend = np.random.uniform(5000, 15000, n_weeks)
            else:
                spend = np.random.uniform(5000, 20000, n_weeks)

            # Add some variation
            spend = spend * (1 + 0.3 * np.sin(2 * np.pi * np.arange(n_weeks) / 52))
            data[f'{channel}_spend'] = spend.clip(0)

        df = pd.DataFrame(data)

        # Generate sales as function of marketing with diminishing returns
        sales = base_sales.copy()

        # Channel effectiveness (different ROI)
        effectiveness = {
            'TV': 2.5,
            'Radio': 1.8,
            'Digital': 3.2,
            'Print': 1.2,
            'Social': 2.8,
            'Email': 3.5
        }

        # Ad stock decay rates (carryover effect)
        decay_rates = {
            'TV': 0.7,
            'Radio': 0.5,
            'Digital': 0.3,
            'Print': 0.4,
            'Social': 0.4,
            'Email': 0.2
        }

        for channel in self.channels:
            spend = df[f'{channel}_spend'].values

            # Apply ad stock (carryover effect)
            adstock = self.calculate_adstock(spend, decay_rates[channel])

            # Diminishing returns (log transformation)
            effect = effectiveness[channel] * np.log1p(adstock / 1000)

            sales += effect

        # Add noise
        sales += np.random.normal(0, 5000, n_weeks)

        df['sales'] = sales.clip(0)

        return df

    def calculate_adstock(self, x, decay_rate, max_lag=8):
        """Calculate ad stock effect with exponential decay"""
        adstocked = np.zeros_like(x)

        for t in range(len(x)):
            adstocked[t] = x[t]
            for lag in range(1, min(t + 1, max_lag)):
                adstocked[t] += x[t - lag] * (decay_rate ** lag)

        return adstocked

    def transform_features(self, df):
        """Transform marketing spend with adstock and diminishing returns"""
        df_transformed = df.copy()

        # Assume decay rates (in practice, these would be optimized)
        decay_rates = {
            'TV': 0.7,
            'Radio': 0.5,
            'Digital': 0.3,
            'Print': 0.4,
            'Social': 0.4,
            'Email': 0.2
        }

        for channel in self.channels:
            spend = df[f'{channel}_spend'].values

            # Apply adstock
            adstock = self.calculate_adstock(spend, decay_rates[channel])

            # Apply diminishing returns (log transformation)
            df_transformed[f'{channel}_transformed'] = np.log1p(adstock / 1000)

        # Add time features
        df_transformed['trend'] = np.arange(len(df))
        df_transformed['sin_seasonality'] = np.sin(
            2 * np.pi * np.arange(len(df)) / 52
        )
        df_transformed['cos_seasonality'] = np.cos(
            2 * np.pi * np.arange(len(df)) / 52
        )

        return df_transformed

    def train_model(self, df):
        """Train marketing mix model"""
        df_transformed = self.transform_features(df)

        # Features
        feature_cols = [f'{ch}_transformed' for ch in self.channels]
        feature_cols += ['trend', 'sin_seasonality', 'cos_seasonality']

        X = df_transformed[feature_cols]
        y = df['sales']

        # Use Ridge regression for stability
        self.model = Ridge(alpha=100)
        self.model.fit(X, y)

        # Predictions
        y_pred = self.model.predict(X)

        # Calculate metrics
        r2 = r2_score(y, y_pred)
        mape = mean_absolute_percentage_error(y, y_pred) * 100

        # Extract channel contributions
        contributions = {}
        for i, channel in enumerate(self.channels):
            coef = self.model.coef_[i]
            feature_values = X[f'{channel}_transformed'].values
            contributions[channel] = coef * feature_values

        return {
            'r2': r2,
            'mape': mape,
            'predictions': y_pred,
            'contributions': contributions,
            'coefficients': dict(zip(feature_cols, self.model.coef_))
        }

    def calculate_roi(self, df, contributions):
        """Calculate ROI for each channel"""
        roi_data = []

        for channel in self.channels:
            total_spend = df[f'{channel}_spend'].sum()
            total_contribution = contributions[channel].sum()
            roi = (total_contribution / total_spend) if total_spend > 0 else 0

            roi_data.append({
                'channel': channel,
                'spend': total_spend,
                'contribution': total_contribution,
                'roi': roi
            })

        return pd.DataFrame(roi_data)

    def optimize_budget(self, total_budget, constraints=None):
        """Optimize budget allocation across channels"""
        n_channels = len(self.channels)

        # Objective function (negative because we minimize)
        def objective(allocation):
            total_sales = 0
            for i, channel in enumerate(self.channels):
                spend = allocation[i]
                # Diminishing returns model
                contrib = self.model.coef_[i] * np.log1p(spend / 1000)
                total_sales += contrib
            return -total_sales  # Negative for minimization

        # Constraints
        cons = [{'type': 'eq', 'fun': lambda x: x.sum() - total_budget}]  # Budget constraint

        # Bounds (min/max spend per channel)
        bounds = [(total_budget * 0.05, total_budget * 0.5) for _ in range(n_channels)]

        # Initial guess (equal allocation)
        x0 = np.ones(n_channels) * (total_budget / n_channels)

        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)

        if result.success:
            optimal_allocation = dict(zip(self.channels, result.x))
            expected_sales = -result.fun

            return {
                'optimal_allocation': optimal_allocation,
                'expected_sales': expected_sales
            }
        else:
            return None

    def plot_results(self, df, model_results, roi_df):
        """Visualize marketing mix results"""
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Actual vs Predicted Sales
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(df['week'], df['sales'], label='Actual Sales',
                linewidth=2, color='#3498db', alpha=0.7)
        ax1.plot(df['week'], model_results['predictions'], label='Predicted Sales',
                linewidth=2, color='#e74c3c', linestyle='--')
        ax1.set_xlabel('Week', fontsize=11)
        ax1.set_ylabel('Sales ($)', fontsize=11)
        ax1.set_title(f"Actual vs Predicted Sales (R² = {model_results['r2']:.3f})",
                     fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Channel Contributions Over Time
        ax2 = fig.add_subplot(gs[1, :])
        bottom = np.zeros(len(df))
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.channels)))

        for i, channel in enumerate(self.channels):
            ax2.bar(df['week'], model_results['contributions'][channel],
                   bottom=bottom, label=channel, color=colors[i], alpha=0.8)
            bottom += model_results['contributions'][channel]

        ax2.set_xlabel('Week', fontsize=11)
        ax2.set_ylabel('Sales Contribution ($)', fontsize=11)
        ax2.set_title('Channel Contributions to Sales Over Time',
                     fontsize=13, fontweight='bold')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3, axis='y')

        # Total Contribution by Channel
        ax3 = fig.add_subplot(gs[2, 0])
        total_contrib = roi_df.sort_values('contribution', ascending=False)
        bars = ax3.bar(total_contrib['channel'], total_contrib['contribution'] / 1000,
                      color='#2ecc71', edgecolor='black', linewidth=1.5, alpha=0.7)
        ax3.set_ylabel('Total Contribution ($1000s)', fontsize=11)
        ax3.set_title('Total Sales Contribution by Channel', fontsize=12, fontweight='bold')
        ax3.set_xticklabels(total_contrib['channel'], rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')

        # ROI by Channel
        ax4 = fig.add_subplot(gs[2, 1])
        roi_sorted = roi_df.sort_values('roi', ascending=False)
        colors_roi = ['#2ecc71' if r > 0 else '#e74c3c' for r in roi_sorted['roi']]
        bars = ax4.bar(roi_sorted['channel'], roi_sorted['roi'],
                      color=colors_roi, edgecolor='black', linewidth=1.5, alpha=0.7)
        ax4.set_ylabel('ROI (Revenue / Spend)', fontsize=11)
        ax4.set_title('Return on Investment by Channel', fontsize=12, fontweight='bold')
        ax4.set_xticklabels(roi_sorted['channel'], rotation=45, ha='right')
        ax4.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Break-even')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.legend()

        # Spend vs Contribution
        ax5 = fig.add_subplot(gs[2, 2])
        for channel in self.channels:
            ax5.scatter(roi_df[roi_df['channel'] == channel]['spend'] / 1000,
                       roi_df[roi_df['channel'] == channel]['contribution'] / 1000,
                       s=200, label=channel, alpha=0.7, edgecolor='black', linewidth=1.5)

        ax5.plot([0, roi_df['spend'].max() / 1000],
                [0, roi_df['spend'].max() / 1000],
                'r--', linewidth=2, label='Break-even line')
        ax5.set_xlabel('Total Spend ($1000s)', fontsize=11)
        ax5.set_ylabel('Total Contribution ($1000s)', fontsize=11)
        ax5.set_title('Spend vs Contribution by Channel', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        plt.savefig('marketing_mix_analysis.png', dpi=300, bbox_inches='tight')
        print("\n📊 Visualization saved as 'marketing_mix_analysis.png'")
        plt.show()


def main():
    """Main execution function"""
    print("📢 Marketing Mix Modeling System")
    print("=" * 80)

    mmm = MarketingMixModel()

    # Generate data
    print("\n📊 Generating marketing and sales data...")
    df = mmm.create_sample_data(n_weeks=104)
    print(f"Dataset shape: {df.shape}")
    print(f"Time period: {len(df)} weeks")

    # Train model
    print("\n🤖 Training marketing mix model...")
    model_results = mmm.train_model(df)
    print(f"Model R²: {model_results['r2']:.3f}")
    print(f"Model MAPE: {model_results['mape']:.2f}%")

    # Calculate ROI
    print("\n💰 Calculating ROI by channel...")
    roi_df = mmm.calculate_roi(df, model_results['contributions'])
    print("\nROI Summary:")
    print(roi_df.to_string(index=False))

    # Optimize budget
    print("\n🎯 Optimizing budget allocation...")
    total_budget = df[[f'{ch}_spend' for ch in mmm.channels]].sum().sum()
    optimization = mmm.optimize_budget(total_budget)

    if optimization:
        print("\nOptimal Budget Allocation:")
        for channel, allocation in optimization['optimal_allocation'].items():
            print(f"  {channel:10s}: ${allocation:>12,.0f}")
        print(f"\nExpected Sales: ${optimization['expected_sales']:,.0f}")

    # Plot results
    print("\n📈 Generating visualizations...")
    mmm.plot_results(df, model_results, roi_df)

    print("\n✅ Marketing mix modeling complete!")


if __name__ == "__main__":
    main()
