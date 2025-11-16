"""
Dynamic Pricing Optimization System
====================================

Problem: Optimize product pricing dynamically based on demand elasticity,
competition, inventory, and market conditions to maximize revenue/profit

Kaggle-style competition: Pricing Strategy
Difficulty: ⭐⭐⭐⭐

This solution demonstrates:
- Price elasticity estimation
- Demand forecasting
- Revenue optimization
- Competitive pricing analysis
- Dynamic pricing strategies
- Inventory-aware pricing
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class PricingOptimizer:
    """Dynamic pricing optimization system"""

    def __init__(self):
        self.demand_model = None
        self.elasticity = {}

    def create_sample_data(self, n_products=10, n_days=365):
        """Generate realistic pricing and sales data"""
        np.random.seed(42)

        products = [f'Product_{i}' for i in range(n_products)]
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')

        data = []
        for product in products:
            # Product characteristics
            base_price = np.random.uniform(20, 200)
            base_demand = np.random.uniform(100, 1000)
            price_elasticity = -np.random.uniform(1.2, 2.5)  # Elastic demand
            cost = base_price * np.random.uniform(0.4, 0.7)

            for date in dates:
                # Day of week effect
                day_of_week = date.dayofweek
                weekend_factor = 1.2 if day_of_week >= 5 else 1.0

                # Seasonality
                season_factor = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)

                # Price variation
                price_change = np.random.uniform(-0.2, 0.2)
                price = base_price * (1 + price_change)

                # Competitor price
                competitor_price = price * (1 + np.random.uniform(-0.15, 0.15))

                # Inventory level (affects pricing pressure)
                inventory = np.random.uniform(50, 500)

                # Demand as function of price (price elasticity)
                price_factor = (price / base_price) ** price_elasticity
                demand = (base_demand * price_factor * weekend_factor *
                         season_factor * (1 + np.random.normal(0, 0.1)))

                # Competitor effect
                if competitor_price < price:
                    demand *= 0.8  # Lose customers to competitor
                else:
                    demand *= 1.1  # Gain customers

                # Constrain by inventory
                sales = min(demand, inventory)

                data.append({
                    'date': date,
                    'product': product,
                    'price': price,
                    'cost': cost,
                    'competitor_price': competitor_price,
                    'inventory': inventory,
                    'day_of_week': day_of_week,
                    'is_weekend': 1 if day_of_week >= 5 else 0,
                    'month': date.month,
                    'sales': max(0, sales),
                    'revenue': price * max(0, sales),
                    'profit': (price - cost) * max(0, sales)
                })

        return pd.DataFrame(data)

    def engineer_features(self, df):
        """Create pricing features"""
        df = df.copy()

        # Price features
        df['price_vs_competitor'] = df['price'] / (df['competitor_price'] + 1)
        df['price_discount'] = (df['competitor_price'] - df['price']) / (df['price'] + 1)

        # Cyclical time features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Lagged features by product
        df = df.sort_values(['product', 'date'])
        for lag in [1, 7]:
            df[f'sales_lag_{lag}'] = df.groupby('product')['sales'].shift(lag)
            df[f'price_lag_{lag}'] = df.groupby('product')['price'].shift(lag)

        # Rolling features
        df['sales_rolling_7'] = df.groupby('product')['sales'].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )

        df = df.fillna(method='bfill')

        return df

    def train_demand_model(self, df):
        """Train demand forecasting model"""
        feature_cols = ['price', 'competitor_price', 'price_vs_competitor',
                       'inventory', 'is_weekend', 'month_sin', 'month_cos',
                       'dow_sin', 'dow_cos', 'sales_lag_1', 'sales_lag_7',
                       'sales_rolling_7']

        X = df[feature_cols]
        y = df['sales']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        self.demand_model = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
        )
        self.demand_model.fit(X_train, y_train)

        # Predictions
        y_pred = self.demand_model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {
            'model': self.demand_model,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred,
            'actuals': y_test
        }

    def estimate_price_elasticity(self, df):
        """Estimate price elasticity for each product"""
        elasticities = {}

        for product in df['product'].unique():
            product_data = df[df['product'] == product].copy()

            # Log-log regression: log(sales) = a + b*log(price)
            # Elasticity = b
            product_data['log_sales'] = np.log(product_data['sales'] + 1)
            product_data['log_price'] = np.log(product_data['price'])

            model = LinearRegression()
            X = product_data[['log_price']].values
            y = product_data['log_sales'].values

            model.fit(X, y)
            elasticity = model.coef_[0]

            elasticities[product] = {
                'elasticity': elasticity,
                'avg_price': product_data['price'].mean(),
                'avg_sales': product_data['sales'].mean(),
                'avg_cost': product_data['cost'].mean()
            }

        return elasticities

    def optimize_price(self, product_info, competitor_price, inventory):
        """Find optimal price for a product"""
        elasticity = product_info['elasticity']
        avg_sales = product_info['avg_sales']
        avg_price = product_info['avg_price']
        cost = product_info['avg_cost']

        def profit_function(price):
            # Demand based on price elasticity
            price_ratio = price / avg_price
            demand = avg_sales * (price_ratio ** elasticity)

            # Competitor effect
            if competitor_price < price:
                demand *= 0.8
            else:
                demand *= 1.1

            # Constrained by inventory
            sales = min(demand, inventory)

            # Profit
            profit = (price - cost) * sales

            return -profit  # Negative because we minimize

        # Optimize
        result = minimize_scalar(
            profit_function,
            bounds=(cost * 1.1, avg_price * 2),  # Min 10% markup, max 2x avg price
            method='bounded'
        )

        if result.success:
            optimal_price = result.x
            expected_profit = -result.fun
            # Calculate expected sales
            price_ratio = optimal_price / avg_price
            expected_demand = avg_sales * (price_ratio ** elasticity)
            if competitor_price < optimal_price:
                expected_demand *= 0.8
            else:
                expected_demand *= 1.1
            expected_sales = min(expected_demand, inventory)

            return {
                'optimal_price': optimal_price,
                'expected_profit': expected_profit,
                'expected_sales': expected_sales
            }
        return None

    def plot_results(self, df, model_results, elasticities):
        """Visualize pricing optimization results"""
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Price vs Sales Scatter
        ax1 = fig.add_subplot(gs[0, 0])
        for product in df['product'].unique()[:5]:  # Top 5 products
            product_data = df[df['product'] == product]
            ax1.scatter(product_data['price'], product_data['sales'],
                       label=product, alpha=0.6, s=30)
        ax1.set_xlabel('Price ($)', fontsize=11)
        ax1.set_ylabel('Sales (units)', fontsize=11)
        ax1.set_title('Price vs Sales Relationship', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Demand Model Performance
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(model_results['actuals'], model_results['predictions'],
                   alpha=0.5, s=30)
        max_val = max(model_results['actuals'].max(), model_results['predictions'].max())
        ax2.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax2.set_xlabel('Actual Sales', fontsize=11)
        ax2.set_ylabel('Predicted Sales', fontsize=11)
        ax2.set_title(f"Demand Model (R² = {model_results['r2']:.3f})",
                     fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Price Elasticity by Product
        ax3 = fig.add_subplot(gs[0, 2])
        products = list(elasticities.keys())
        elasticity_values = [elasticities[p]['elasticity'] for p in products]

        colors = ['#e74c3c' if e < -2 else '#f39c12' if e < -1 else '#2ecc71'
                 for e in elasticity_values]
        ax3.barh(products, elasticity_values, color=colors,
                edgecolor='black', linewidth=1.5, alpha=0.7)
        ax3.set_xlabel('Price Elasticity', fontsize=11)
        ax3.set_title('Price Elasticity by Product', fontsize=12, fontweight='bold')
        ax3.axvline(x=-1, color='red', linestyle='--', linewidth=2,
                   label='Unit Elastic')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='x')

        # Revenue Over Time
        ax4 = fig.add_subplot(gs[1, :])
        daily_revenue = df.groupby('date')['revenue'].sum()
        ax4.plot(daily_revenue.index, daily_revenue.values / 1000,
                linewidth=2, color='#2ecc71', alpha=0.8)
        ax4.set_xlabel('Date', fontsize=11)
        ax4.set_ylabel('Revenue ($1000s)', fontsize=11)
        ax4.set_title('Daily Revenue Over Time', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # Price Distribution
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.hist(df['price'], bins=30, color='#3498db', edgecolor='black', alpha=0.7)
        ax5.axvline(x=df['price'].mean(), color='red', linestyle='--',
                   linewidth=2, label=f"Mean: ${df['price'].mean():.2f}")
        ax5.set_xlabel('Price ($)', fontsize=11)
        ax5.set_ylabel('Frequency', fontsize=11)
        ax5.set_title('Price Distribution', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')

        # Profit Margin Distribution
        ax6 = fig.add_subplot(gs[2, 1])
        df['margin'] = (df['price'] - df['cost']) / df['price'] * 100
        ax6.hist(df['margin'], bins=30, color='#9b59b6', edgecolor='black', alpha=0.7)
        ax6.axvline(x=df['margin'].mean(), color='red', linestyle='--',
                   linewidth=2, label=f"Mean: {df['margin'].mean():.1f}%")
        ax6.set_xlabel('Profit Margin (%)', fontsize=11)
        ax6.set_ylabel('Frequency', fontsize=11)
        ax6.set_title('Profit Margin Distribution', fontsize=12, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')

        # Optimal Price Example
        ax7 = fig.add_subplot(gs[2, 2])
        sample_product = list(elasticities.keys())[0]
        product_info = elasticities[sample_product]

        prices = np.linspace(product_info['avg_cost'] * 1.1,
                            product_info['avg_price'] * 2, 100)
        profits = []

        for price in prices:
            price_ratio = price / product_info['avg_price']
            demand = product_info['avg_sales'] * (price_ratio ** product_info['elasticity'])
            profit = (price - product_info['avg_cost']) * demand
            profits.append(profit)

        ax7.plot(prices, profits, linewidth=2, color='#2ecc71')
        optimal_idx = np.argmax(profits)
        ax7.scatter([prices[optimal_idx]], [profits[optimal_idx]],
                   s=200, color='red', zorder=5, edgecolor='black', linewidth=2)
        ax7.set_xlabel('Price ($)', fontsize=11)
        ax7.set_ylabel('Expected Profit ($)', fontsize=11)
        ax7.set_title(f'Profit Curve - {sample_product}', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3)

        plt.savefig('pricing_optimization_analysis.png', dpi=300, bbox_inches='tight')
        print("\n📊 Visualization saved as 'pricing_optimization_analysis.png'")
        plt.show()


def main():
    """Main execution function"""
    print("💵 Dynamic Pricing Optimization System")
    print("=" * 80)

    optimizer = PricingOptimizer()

    # Generate data
    print("\n📊 Generating pricing and sales data...")
    df = optimizer.create_sample_data(n_products=10, n_days=365)
    print(f"Dataset shape: {df.shape}")
    print(f"Products: {df['product'].nunique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Engineer features
    print("\n🔧 Engineering pricing features...")
    df = optimizer.engineer_features(df)

    # Train demand model
    print("\n🤖 Training demand forecasting model...")
    model_results = optimizer.train_demand_model(df)
    print(f"Demand model R²: {model_results['r2']:.3f}")
    print(f"Demand model MAE: {model_results['mae']:.2f}")

    # Estimate price elasticity
    print("\n📈 Estimating price elasticity...")
    elasticities = optimizer.estimate_price_elasticity(df)
    print("\nPrice Elasticities:")
    for product, info in list(elasticities.items())[:5]:
        print(f"  {product}: {info['elasticity']:.2f}")

    # Optimize pricing
    print("\n🎯 Optimizing prices...")
    sample_product = list(elasticities.keys())[0]
    optimization = optimizer.optimize_price(
        elasticities[sample_product],
        competitor_price=100,
        inventory=300
    )

    if optimization:
        print(f"\nOptimization for {sample_product}:")
        print(f"  Optimal Price: ${optimization['optimal_price']:.2f}")
        print(f"  Expected Sales: {optimization['expected_sales']:.0f} units")
        print(f"  Expected Profit: ${optimization['expected_profit']:.2f}")

    # Plot results
    print("\n📈 Generating visualizations...")
    optimizer.plot_results(df, model_results, elasticities)

    print("\n✅ Pricing optimization complete!")


if __name__ == "__main__":
    main()
