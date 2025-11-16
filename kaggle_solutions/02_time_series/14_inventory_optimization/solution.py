#!/usr/bin/env python3
"""
Inventory Level Prediction and Optimization
============================================
Predicts optimal inventory levels using demand forecasting and optimization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


def generate_inventory_data(n_days=365):
    """
    Generate synthetic inventory and demand data.

    Includes:
    - Daily demand with seasonality
    - Order patterns (periodic restocking)
    - Lead times
    - Stockout events
    - Safety stock requirements
    """
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')

    # Demand generation with multiple patterns
    base_demand = 100

    # Weekly seasonality (higher on weekdays)
    day_of_week = dates.dayofweek
    weekly_pattern = np.where(day_of_week < 5, 20, -30)  # Weekday vs weekend

    # Monthly seasonality
    day_of_month = dates.day
    monthly_pattern = 15 * np.sin(2 * np.pi * day_of_month / 30)

    # Quarterly trend
    quarter_effect = np.where(
        dates.quarter == 1, -10,
        np.where(dates.quarter == 2, 0,
                 np.where(dates.quarter == 3, 5, 20))
    )

    # Random daily variation
    random_demand = np.random.gamma(shape=2, scale=10, size=n_days)

    # Total demand
    demand = base_demand + weekly_pattern + monthly_pattern + quarter_effect + random_demand
    demand = np.maximum(demand, 0)  # Ensure non-negative

    # Simulate inventory levels
    initial_inventory = 1000
    reorder_point = 400
    order_quantity = 600
    lead_time = 3  # days

    inventory = []
    orders = []
    stockouts = []
    current_inventory = initial_inventory
    pending_orders = []

    for i in range(n_days):
        # Process incoming orders
        incoming = 0
        for order in pending_orders[:]:
            if order['arrival_day'] == i:
                incoming += order['quantity']
                pending_orders.remove(order)

        # Update inventory
        current_inventory += incoming

        # Satisfy demand (or stockout)
        daily_demand = demand[i]
        if current_inventory >= daily_demand:
            current_inventory -= daily_demand
            stockout = 0
        else:
            stockout = daily_demand - current_inventory
            current_inventory = 0

        # Check reorder point
        ordered = 0
        if current_inventory <= reorder_point and not any(o['placed_day'] > i - lead_time for o in pending_orders):
            ordered = order_quantity
            pending_orders.append({
                'quantity': order_quantity,
                'placed_day': i,
                'arrival_day': i + lead_time
            })

        inventory.append(current_inventory)
        orders.append(ordered)
        stockouts.append(stockout)

    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'demand': demand,
        'inventory': inventory,
        'orders': orders,
        'stockouts': stockouts,
        'day_of_week': day_of_week,
        'day_of_month': day_of_month,
        'month': dates.month,
        'quarter': dates.quarter,
        'is_weekend': (day_of_week >= 5).astype(int)
    })

    return df


def calculate_inventory_metrics(df):
    """Calculate key inventory management metrics."""
    metrics = {}

    # Service level (% of demand met)
    total_demand = df['demand'].sum()
    total_stockouts = df['stockouts'].sum()
    metrics['service_level'] = (1 - total_stockouts / total_demand) * 100

    # Average inventory
    metrics['avg_inventory'] = df['inventory'].mean()

    # Inventory turnover
    metrics['inventory_turnover'] = total_demand / metrics['avg_inventory']

    # Stockout frequency
    metrics['stockout_days'] = (df['stockouts'] > 0).sum()
    metrics['stockout_rate'] = metrics['stockout_days'] / len(df) * 100

    # Order frequency
    metrics['orders_placed'] = (df['orders'] > 0).sum()
    metrics['total_ordered'] = df['orders'].sum()

    # Holding cost (assuming $1 per unit per day)
    metrics['holding_cost'] = df['inventory'].sum()

    # Stockout cost (assuming $10 per unit shortage)
    metrics['stockout_cost'] = df['stockouts'].sum() * 10

    # Total cost
    metrics['total_cost'] = metrics['holding_cost'] + metrics['stockout_cost']

    return metrics


def create_demand_features(df):
    """Create features for demand prediction."""
    df_features = df.copy()

    # Lag features
    for lag in [1, 2, 3, 7, 14, 30]:
        df_features[f'demand_lag_{lag}'] = df_features['demand'].shift(lag)

    # Rolling statistics
    df_features['demand_rolling_mean_7d'] = df_features['demand'].shift(1).rolling(window=7).mean()
    df_features['demand_rolling_std_7d'] = df_features['demand'].shift(1).rolling(window=7).std()
    df_features['demand_rolling_mean_30d'] = df_features['demand'].shift(1).rolling(window=30).mean()

    # Cyclical features
    df_features['day_of_week_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['day_of_week_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['day_of_month_sin'] = np.sin(2 * np.pi * df_features['day_of_month'] / 30)
    df_features['day_of_month_cos'] = np.cos(2 * np.pi * df_features['day_of_month'] / 30)

    return df_features


def optimize_reorder_point(df, demand_forecast, lead_time=3, service_level_target=0.95):
    """Calculate optimal reorder point based on forecasted demand."""
    # Average demand during lead time
    avg_demand_lt = demand_forecast * lead_time

    # Standard deviation during lead time
    std_demand = df['demand'].std()
    std_demand_lt = std_demand * np.sqrt(lead_time)

    # Z-score for desired service level
    z_score = stats.norm.ppf(service_level_target)

    # Safety stock
    safety_stock = z_score * std_demand_lt

    # Reorder point
    reorder_point = avg_demand_lt + safety_stock

    return {
        'reorder_point': reorder_point,
        'safety_stock': safety_stock,
        'avg_demand_lt': avg_demand_lt
    }


def main():
    """Main execution function."""
    print("=" * 80)
    print("INVENTORY LEVEL PREDICTION AND OPTIMIZATION")
    print("=" * 80)

    # Generate data
    print("\n1. Generating synthetic inventory data...")
    df = generate_inventory_data(n_days=365)
    print(f"   Generated {len(df)} days of data")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Demand range: {df['demand'].min():.2f} - {df['demand'].max():.2f} units/day")
    print(f"   Average demand: {df['demand'].mean():.2f} units/day")

    # Calculate current metrics
    print("\n2. Current Inventory Performance Metrics:")
    current_metrics = calculate_inventory_metrics(df)
    print(f"   Service Level: {current_metrics['service_level']:.2f}%")
    print(f"   Average Inventory: {current_metrics['avg_inventory']:.2f} units")
    print(f"   Inventory Turnover: {current_metrics['inventory_turnover']:.2f}x/year")
    print(f"   Stockout Days: {current_metrics['stockout_days']} ({current_metrics['stockout_rate']:.2f}%)")
    print(f"   Orders Placed: {current_metrics['orders_placed']}")
    print(f"   Holding Cost: ${current_metrics['holding_cost']:,.2f}")
    print(f"   Stockout Cost: ${current_metrics['stockout_cost']:,.2f}")
    print(f"   Total Cost: ${current_metrics['total_cost']:,.2f}")

    # Create features for demand forecasting
    print("\n3. Creating demand prediction features...")
    df_features = create_demand_features(df)
    df_features.dropna(inplace=True)

    # Split data
    split_idx = int(len(df_features) * 0.8)
    train_df = df_features[:split_idx]
    test_df = df_features[split_idx:]

    print(f"   Training set: {len(train_df)} days")
    print(f"   Test set: {len(test_df)} days")

    # Define features
    feature_cols = [col for col in df_features.columns
                    if col not in ['date', 'demand', 'inventory', 'orders', 'stockouts']]
    X_train = train_df[feature_cols]
    y_train = train_df['demand']
    X_test = test_df[feature_cols]
    y_test = test_df['demand']

    # Train Gradient Boosting model
    print("\n4. Training Gradient Boosting model for demand prediction...")
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    gb_model.fit(X_train, y_train)

    # Make predictions
    y_pred = gb_model.predict(X_test)

    # Evaluate demand prediction
    print("\n5. Demand Prediction Performance:")
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print(f"   MAE: {mae:.2f} units")
    print(f"   RMSE: {rmse:.2f} units")
    print(f"   R²: {r2:.4f}")
    print(f"   MAPE: {mape:.2f}%")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': gb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n6. Top 10 Most Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")

    # Optimize reorder point
    print("\n7. Optimizing Reorder Point...")
    avg_forecast = y_pred.mean()

    for target_sl in [0.90, 0.95, 0.99]:
        optimization = optimize_reorder_point(
            df,
            avg_forecast,
            lead_time=3,
            service_level_target=target_sl
        )
        print(f"\n   Target Service Level: {target_sl*100:.0f}%")
        print(f"      Optimal Reorder Point: {optimization['reorder_point']:.2f} units")
        print(f"      Safety Stock: {optimization['safety_stock']:.2f} units")
        print(f"      Average Lead Time Demand: {optimization['avg_demand_lt']:.2f} units")

    # Visualization
    print("\n8. Creating visualizations...")
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    # Plot 1: Demand over time
    axes[0, 0].plot(df['date'], df['demand'], linewidth=1, alpha=0.7)
    axes[0, 0].set_title('Daily Demand', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Demand (units)')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Inventory levels
    axes[0, 1].plot(df['date'], df['inventory'], linewidth=1, color='green')
    axes[0, 1].axhline(y=400, color='red', linestyle='--', label='Reorder Point', alpha=0.5)
    axes[0, 1].set_title('Inventory Levels', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Inventory (units)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Stockouts
    axes[0, 2].plot(df['date'], df['stockouts'], linewidth=1, color='red', alpha=0.7)
    axes[0, 2].set_title('Stockouts', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Date')
    axes[0, 2].set_ylabel('Stockout (units)')
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: Demand by day of week
    dow_demand = df.groupby('day_of_week')['demand'].mean()
    dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[1, 0].bar(range(7), dow_demand.values, color='steelblue')
    axes[1, 0].set_title('Average Demand by Day of Week', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Day of Week')
    axes[1, 0].set_ylabel('Average Demand')
    axes[1, 0].set_xticks(range(7))
    axes[1, 0].set_xticklabels(dow_labels)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Plot 5: Demand by quarter
    quarter_demand = df.groupby('quarter')['demand'].mean()
    axes[1, 1].bar(range(1, 5), quarter_demand.values, color='coral')
    axes[1, 1].set_title('Average Demand by Quarter', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Quarter')
    axes[1, 1].set_ylabel('Average Demand')
    axes[1, 1].set_xticks(range(1, 5))
    axes[1, 1].set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # Plot 6: Demand distribution
    axes[1, 2].hist(df['demand'], bins=40, color='purple', edgecolor='black', alpha=0.7)
    axes[1, 2].set_title('Demand Distribution', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Demand (units)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].grid(True, alpha=0.3, axis='y')

    # Plot 7: Actual vs Predicted demand
    axes[2, 0].scatter(y_test, y_pred, alpha=0.5, s=20)
    axes[2, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                    'r--', linewidth=2, label='Perfect Prediction')
    axes[2, 0].set_title('Actual vs Predicted Demand', fontsize=12, fontweight='bold')
    axes[2, 0].set_xlabel('Actual Demand')
    axes[2, 0].set_ylabel('Predicted Demand')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # Plot 8: Feature importance
    top_features = feature_importance.head(10)
    axes[2, 1].barh(range(len(top_features)), top_features['importance'].values)
    axes[2, 1].set_yticks(range(len(top_features)))
    axes[2, 1].set_yticklabels(top_features['feature'].values, fontsize=9)
    axes[2, 1].set_title('Top 10 Feature Importance', fontsize=12, fontweight='bold')
    axes[2, 1].set_xlabel('Importance')
    axes[2, 1].invert_yaxis()
    axes[2, 1].grid(True, alpha=0.3, axis='x')

    # Plot 9: Cost breakdown
    cost_components = ['Holding\nCost', 'Stockout\nCost']
    cost_values = [current_metrics['holding_cost'], current_metrics['stockout_cost']]
    colors = ['green', 'red']
    axes[2, 2].bar(cost_components, cost_values, color=colors, alpha=0.7)
    axes[2, 2].set_title('Cost Breakdown', fontsize=12, fontweight='bold')
    axes[2, 2].set_ylabel('Cost ($)')
    axes[2, 2].grid(True, alpha=0.3, axis='y')

    # Add total cost annotation
    total_text = f'Total: ${current_metrics["total_cost"]:,.0f}'
    axes[2, 2].text(0.5, max(cost_values) * 0.9, total_text,
                    ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('inventory_optimization.png', dpi=300, bbox_inches='tight')
    print("   Saved: inventory_optimization.png")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
