"""
Inventory Management Optimization
==================================

Problem: Optimize inventory levels to minimize costs while maintaining
service levels using EOQ models and demand forecasting

Kaggle-style competition: Inventory Optimization
Difficulty: ⭐⭐⭐

This solution demonstrates:
- Economic Order Quantity (EOQ)
- Safety stock calculation
- Reorder point optimization
- ABC analysis for inventory
- Carrying cost minimization
- Stockout prevention
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class InventoryOptimizer:
    """Inventory management and optimization system"""

    def __init__(self):
        self.demand_model = None

    def create_sample_data(self, n_products=50, n_weeks=52):
        """Generate realistic inventory and demand data"""
        np.random.seed(42)

        products = [f'SKU_{i:04d}' for i in range(n_products)]
        weeks = range(1, n_weeks + 1)

        data = []
        for product_idx, product in enumerate(products):
            # Product characteristics
            base_demand = np.random.uniform(50, 500)
            unit_cost = np.random.uniform(10, 200)
            lead_time_days = np.random.choice([7, 14, 21, 28])

            # Category (for ABC analysis)
            if product_idx < n_products * 0.2:  # A items (20%)
                category = 'A'
                base_demand *= 2  # High volume
            elif product_idx < n_products * 0.5:  # B items (30%)
                category = 'B'
            else:  # C items (50%)
                category = 'C'
                base_demand *= 0.5  # Low volume

            for week in weeks:
                # Seasonal demand
                season_factor = 1 + 0.4 * np.sin(2 * np.pi * week / 52)

                # Trend
                trend_factor = 1 + (week / n_weeks) * 0.2

                # Random variation
                demand = base_demand * season_factor * trend_factor
                demand = max(0, demand + np.random.normal(0, demand * 0.15))

                # Current inventory level
                inventory = np.random.uniform(demand * 0.5, demand * 3)

                # Orders placed
                order_quantity = 0
                if inventory < demand * 1.5:  # Reorder trigger
                    order_quantity = demand * 2

                # Stockouts
                stockout = 1 if inventory < demand else 0

                data.append({
                    'product': product,
                    'week': week,
                    'demand': int(demand),
                    'inventory': int(inventory),
                    'order_quantity': int(order_quantity),
                    'unit_cost': unit_cost,
                    'lead_time_days': lead_time_days,
                    'category': category,
                    'stockout': stockout
                })

        return pd.DataFrame(data)

    def calculate_eoq(self, annual_demand, ordering_cost, holding_cost_rate, unit_cost):
        """Calculate Economic Order Quantity"""
        holding_cost = unit_cost * holding_cost_rate

        eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)

        return eoq

    def calculate_safety_stock(self, demand_std, lead_time_days, service_level=0.95):
        """Calculate safety stock for given service level"""
        from scipy.stats import norm

        # Z-score for service level
        z = norm.ppf(service_level)

        # Safety stock
        safety_stock = z * demand_std * np.sqrt(lead_time_days / 7)

        return safety_stock

    def calculate_reorder_point(self, avg_demand_weekly, lead_time_days, safety_stock):
        """Calculate reorder point"""
        lead_time_demand = avg_demand_weekly * (lead_time_days / 7)
        reorder_point = lead_time_demand + safety_stock

        return reorder_point

    def abc_analysis(self, df):
        """Perform ABC analysis on inventory"""
        # Calculate annual value for each product
        product_value = df.groupby('product').agg({
            'demand': 'sum',
            'unit_cost': 'first'
        })
        product_value['annual_value'] = product_value['demand'] * product_value['unit_cost']
        product_value = product_value.sort_values('annual_value', ascending=False)

        # Calculate cumulative percentage
        total_value = product_value['annual_value'].sum()
        product_value['cumulative_pct'] = (
            product_value['annual_value'].cumsum() / total_value * 100
        )

        # Assign ABC classification
        def classify_abc(cum_pct):
            if cum_pct <= 80:
                return 'A'
            elif cum_pct <= 95:
                return 'B'
            else:
                return 'C'

        product_value['abc_class'] = product_value['cumulative_pct'].apply(classify_abc)

        return product_value

    def optimize_inventory(self, df):
        """Calculate optimal inventory parameters for all products"""
        ORDERING_COST = 100  # Fixed cost per order
        HOLDING_COST_RATE = 0.25  # 25% of unit cost per year
        SERVICE_LEVEL = 0.95  # 95% service level

        inventory_params = []

        for product in df['product'].unique():
            product_data = df[df['product'] == product]

            # Calculate parameters
            annual_demand = product_data['demand'].sum() * (52 / len(product_data))
            avg_weekly_demand = product_data['demand'].mean()
            demand_std = product_data['demand'].std()
            unit_cost = product_data['unit_cost'].iloc[0]
            lead_time = product_data['lead_time_days'].iloc[0]

            # EOQ
            eoq = self.calculate_eoq(annual_demand, ORDERING_COST,
                                     HOLDING_COST_RATE, unit_cost)

            # Safety stock
            safety_stock = self.calculate_safety_stock(demand_std, lead_time,
                                                       SERVICE_LEVEL)

            # Reorder point
            reorder_point = self.calculate_reorder_point(avg_weekly_demand,
                                                         lead_time, safety_stock)

            # Current metrics
            avg_inventory = product_data['inventory'].mean()
            stockout_rate = product_data['stockout'].mean()

            inventory_params.append({
                'product': product,
                'eoq': eoq,
                'safety_stock': safety_stock,
                'reorder_point': reorder_point,
                'avg_weekly_demand': avg_weekly_demand,
                'annual_demand': annual_demand,
                'current_avg_inventory': avg_inventory,
                'stockout_rate': stockout_rate,
                'unit_cost': unit_cost,
                'lead_time_days': lead_time
            })

        return pd.DataFrame(inventory_params)

    def calculate_costs(self, df, inventory_params):
        """Calculate inventory costs"""
        HOLDING_COST_RATE = 0.25 / 52  # Weekly rate
        ORDERING_COST = 100
        STOCKOUT_COST = 1000  # Cost per stockout

        total_holding_cost = 0
        total_ordering_cost = 0
        total_stockout_cost = 0

        for product in df['product'].unique():
            product_data = df[df['product'] == product]
            params = inventory_params[inventory_params['product'] == product].iloc[0]

            # Holding cost
            avg_inventory = product_data['inventory'].mean()
            unit_cost = params['unit_cost']
            holding_cost = avg_inventory * unit_cost * HOLDING_COST_RATE * len(product_data)

            # Ordering cost
            n_orders = (product_data['order_quantity'] > 0).sum()
            ordering_cost = n_orders * ORDERING_COST

            # Stockout cost
            stockouts = product_data['stockout'].sum()
            stockout_cost = stockouts * STOCKOUT_COST

            total_holding_cost += holding_cost
            total_ordering_cost += ordering_cost
            total_stockout_cost += stockout_cost

        return {
            'holding_cost': total_holding_cost,
            'ordering_cost': total_ordering_cost,
            'stockout_cost': total_stockout_cost,
            'total_cost': total_holding_cost + total_ordering_cost + total_stockout_cost
        }

    def plot_results(self, df, inventory_params, abc_results, costs):
        """Visualize inventory optimization results"""
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # ABC Analysis
        ax1 = fig.add_subplot(gs[0, :2])
        abc_data = abc_results.reset_index()
        ax1.bar(range(len(abc_data)), abc_data['annual_value'] / 1000,
               color=['#2ecc71' if c == 'A' else '#f39c12' if c == 'B' else '#e74c3c'
                     for c in abc_data['abc_class']],
               edgecolor='black', linewidth=0.5, alpha=0.7)
        ax1.plot(range(len(abc_data)), abc_data['cumulative_pct'],
                color='red', linewidth=3, marker='o', markersize=3,
                label='Cumulative %')
        ax1.axhline(y=80, color='green', linestyle='--', linewidth=2, alpha=0.7)
        ax1.axhline(y=95, color='orange', linestyle='--', linewidth=2, alpha=0.7)
        ax1.set_xlabel('Product (sorted by value)', fontsize=11)
        ax1.set_ylabel('Annual Value ($1000s)', fontsize=11)
        ax1_twin = ax1.twinx()
        ax1_twin.set_ylabel('Cumulative %', fontsize=11)
        ax1_twin.set_ylim(0, 110)
        ax1.set_title('ABC Analysis - Pareto Chart', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Cost Breakdown
        ax2 = fig.add_subplot(gs[0, 2])
        cost_categories = ['Holding', 'Ordering', 'Stockout']
        cost_values = [costs['holding_cost'], costs['ordering_cost'], costs['stockout_cost']]
        colors_cost = ['#3498db', '#f39c12', '#e74c3c']

        bars = ax2.bar(cost_categories, np.array(cost_values) / 1000,
                      color=colors_cost, edgecolor='black', linewidth=1.5, alpha=0.7)
        ax2.set_ylabel('Cost ($1000s)', fontsize=11)
        ax2.set_title('Inventory Cost Breakdown', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        for bar, value in zip(bars, np.array(cost_values) / 1000):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'${value:.0f}K', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

        # EOQ Distribution
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(inventory_params['eoq'], bins=20, color='#2ecc71',
                edgecolor='black', alpha=0.7)
        ax3.axvline(x=inventory_params['eoq'].median(), color='red',
                   linestyle='--', linewidth=2,
                   label=f"Median: {inventory_params['eoq'].median():.0f}")
        ax3.set_xlabel('Economic Order Quantity', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('EOQ Distribution', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        # Current vs Optimal Inventory
        ax4 = fig.add_subplot(gs[1, 1])
        top_products = inventory_params.nlargest(10, 'annual_demand')
        x = np.arange(len(top_products))
        width = 0.35

        ax4.bar(x - width/2, top_products['current_avg_inventory'], width,
               label='Current Avg', color='#e74c3c', edgecolor='black')
        ax4.bar(x + width/2, top_products['reorder_point'], width,
               label='Optimal Reorder Point', color='#2ecc71', edgecolor='black')
        ax4.set_ylabel('Units', fontsize=11)
        ax4.set_title('Top 10 Products: Current vs Optimal', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels([p[:10] for p in top_products['product']],
                           rotation=45, ha='right', fontsize=8)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        # Stockout Rate by Product
        ax5 = fig.add_subplot(gs[1, 2])
        high_stockout = inventory_params.nlargest(10, 'stockout_rate')
        ax5.barh(range(len(high_stockout)), high_stockout['stockout_rate'] * 100,
                color='#e74c3c', edgecolor='black', linewidth=1.5, alpha=0.7)
        ax5.set_yticks(range(len(high_stockout)))
        ax5.set_yticklabels([p[:12] for p in high_stockout['product']], fontsize=8)
        ax5.set_xlabel('Stockout Rate (%)', fontsize=11)
        ax5.set_title('Top 10 Products with Highest Stockout Rate',
                     fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')

        # Demand Pattern for Sample Product
        ax6 = fig.add_subplot(gs[2, :])
        sample_product = inventory_params.iloc[0]['product']
        sample_data = df[df['product'] == sample_product]

        ax6.plot(sample_data['week'], sample_data['demand'],
                label='Demand', linewidth=2, color='#3498db', marker='o')
        ax6.plot(sample_data['week'], sample_data['inventory'],
                label='Inventory', linewidth=2, color='#2ecc71', alpha=0.7)
        ax6.axhline(y=inventory_params[inventory_params['product'] == sample_product]['reorder_point'].iloc[0],
                   color='red', linestyle='--', linewidth=2, label='Reorder Point')
        ax6.set_xlabel('Week', fontsize=11)
        ax6.set_ylabel('Units', fontsize=11)
        ax6.set_title(f'Demand and Inventory Pattern - {sample_product}',
                     fontsize=12, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.savefig('inventory_optimization_analysis.png', dpi=300, bbox_inches='tight')
        print("\n📊 Visualization saved as 'inventory_optimization_analysis.png'")
        plt.show()


def main():
    """Main execution function"""
    print("📦 Inventory Management Optimization System")
    print("=" * 80)

    optimizer = InventoryOptimizer()

    # Generate data
    print("\n📊 Generating inventory data...")
    df = optimizer.create_sample_data(n_products=50, n_weeks=52)
    print(f"Dataset shape: {df.shape}")
    print(f"Products: {df['product'].nunique()}")
    print(f"Time period: {df['week'].nunique()} weeks")

    # ABC Analysis
    print("\n🔍 Performing ABC analysis...")
    abc_results = optimizer.abc_analysis(df)
    print(f"\nABC Classification:")
    print(f"  A items (80% value): {(abc_results['abc_class'] == 'A').sum()}")
    print(f"  B items (15% value): {(abc_results['abc_class'] == 'B').sum()}")
    print(f"  C items (5% value): {(abc_results['abc_class'] == 'C').sum()}")

    # Optimize inventory
    print("\n🎯 Calculating optimal inventory parameters...")
    inventory_params = optimizer.optimize_inventory(df)

    # Calculate costs
    print("\n💰 Calculating inventory costs...")
    costs = optimizer.calculate_costs(df, inventory_params)
    print(f"\nCost Breakdown:")
    print(f"  Holding Cost: ${costs['holding_cost']:,.2f}")
    print(f"  Ordering Cost: ${costs['ordering_cost']:,.2f}")
    print(f"  Stockout Cost: ${costs['stockout_cost']:,.2f}")
    print(f"  TOTAL COST: ${costs['total_cost']:,.2f}")

    # Plot results
    print("\n📈 Generating visualizations...")
    optimizer.plot_results(df, inventory_params, abc_results, costs)

    print("\n✅ Inventory optimization complete!")


if __name__ == "__main__":
    main()
