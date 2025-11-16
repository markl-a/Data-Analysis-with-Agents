"""
Customer Lifetime Value Prediction
===================================

Problem: Predict customer lifetime value (LTV/CLV) to optimize customer
acquisition costs, retention strategies, and marketing spend

Kaggle-style competition: Customer Value Prediction
Difficulty: ⭐⭐⭐

This solution demonstrates:
- LTV calculation methods
- Customer segmentation by value
- Churn probability integration
- ML-based LTV prediction
- CAC to LTV ratio optimization
- Retention strategy recommendations
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class LTVPredictor:
    """Customer lifetime value prediction system"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def create_sample_data(self, n_customers=5000):
        """Generate realistic customer data"""
        np.random.seed(42)

        # Customer acquisition info
        data = {
            'customer_id': [f'CUST_{i:05d}' for i in range(n_customers)],
            'acquisition_channel': np.random.choice(
                ['Organic', 'Paid Search', 'Social', 'Referral', 'Email'],
                n_customers, p=[0.3, 0.25, 0.2, 0.15, 0.1]
            ),
            'months_active': np.random.exponential(12, n_customers).clip(1, 60),
            'purchase_frequency': np.random.poisson(3, n_customers),
            'avg_order_value': np.random.lognormal(4, 0.6, n_customers).clip(20, 500),
            'total_orders': np.random.poisson(5, n_customers),
            'days_since_last_purchase': np.random.exponential(30, n_customers).clip(0, 365),
            'returns_count': np.random.poisson(0.3, n_customers),
            'support_tickets': np.random.poisson(0.5, n_customers),
            'email_engagement_rate': np.random.beta(3, 2, n_customers),
            'website_visits': np.random.poisson(10, n_customers),
            'app_usage_days': np.random.poisson(8, n_customers),
            'referrals_made': np.random.poisson(0.8, n_customers),
            'discount_usage_rate': np.random.beta(2, 3, n_customers),
            'age': np.random.normal(38, 15, n_customers).clip(18, 80),
            'gender': np.random.choice(['M', 'F'], n_customers),
            'location_tier': np.random.choice([1, 2, 3], n_customers, p=[0.3, 0.5, 0.2])
        }

        df = pd.DataFrame(data)

        # Calculate actual LTV (historical)
        # LTV = Average Order Value × Purchase Frequency × Customer Lifespan
        df['historical_ltv'] = (
            df['avg_order_value'] *
            df['total_orders'] *
            (1 - 0.01 * df['returns_count']) *  # Returns reduce value
            (1 + 0.05 * df['referrals_made']) *  # Referrals add value
            (1 - 0.02 * df['support_tickets'])   # Support tickets slightly reduce
        )

        # Predicted future value (what we want to predict)
        churn_risk = 1 / (1 + np.exp(-(df['days_since_last_purchase'] / 30 - 3)))
        retention_months = df['months_active'] * (1 - churn_risk * 0.7)

        df['predicted_ltv'] = (
            df['avg_order_value'] *
            (df['purchase_frequency'] + 1) *
            (retention_months / 12) *
            (1 + 0.1 * df['email_engagement_rate']) *
            (1 - 0.05 * churn_risk)
        )

        return df

    def engineer_features(self, df):
        """Create LTV prediction features"""
        df = df.copy()

        # One-hot encode categoricals
        df = pd.get_dummies(df, columns=['acquisition_channel', 'gender'],
                           prefix=['channel', 'gender'])

        # Behavioral metrics
        df['avg_days_between_purchases'] = (
            df['months_active'] * 30 / (df['total_orders'] + 1)
        )
        df['return_rate'] = df['returns_count'] / (df['total_orders'] + 1)
        df['support_rate'] = df['support_tickets'] / (df['months_active'] + 1)
        df['engagement_score'] = (
            df['email_engagement_rate'] * 0.3 +
            (df['website_visits'] / df['months_active']) * 0.3 +
            (df['app_usage_days'] / df['months_active']) * 0.4
        )

        # Customer value indicators
        df['high_value'] = (df['avg_order_value'] > df['avg_order_value'].median()).astype(int)
        df['frequent_buyer'] = (df['purchase_frequency'] > 2).astype(int)
        df['at_risk'] = (df['days_since_last_purchase'] > 90).astype(int)
        df['brand_advocate'] = (df['referrals_made'] > 0).astype(int)

        # Lifetime metrics
        df['customer_age_years'] = df['months_active'] / 12
        df['purchases_per_year'] = df['total_orders'] / (df['customer_age_years'] + 0.1)
        df['revenue_per_month'] = df['historical_ltv'] / (df['months_active'] + 1)

        # Interaction features
        df['value_frequency_score'] = df['avg_order_value'] * df['purchase_frequency']
        df['engagement_value_score'] = df['engagement_score'] * df['avg_order_value']

        return df

    def calculate_segments(self, df):
        """Segment customers by LTV"""
        # Using quartiles
        df = df.copy()

        ltv_quartiles = df['predicted_ltv'].quantile([0.25, 0.5, 0.75])

        def assign_segment(ltv):
            if ltv < ltv_quartiles[0.25]:
                return 'Low Value'
            elif ltv < ltv_quartiles[0.5]:
                return 'Medium Value'
            elif ltv < ltv_quartiles[0.75]:
                return 'High Value'
            else:
                return 'VIP'

        df['segment'] = df['predicted_ltv'].apply(assign_segment)

        return df

    def train_model(self, df):
        """Train LTV prediction model"""
        # Features
        feature_cols = [col for col in df.columns if col not in
                       ['customer_id', 'historical_ltv', 'predicted_ltv', 'segment']]

        X = df[feature_cols]
        y = df['predicted_ltv']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100,
                                                           learning_rate=0.1,
                                                           max_depth=5, random_state=42)
        }

        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            results[name] = {
                'model': model,
                'predictions': y_pred,
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred),
                'actuals': y_test
            }

        return results, X_test, y_test, X_train.columns.tolist()

    def calculate_segment_metrics(self, df):
        """Calculate metrics by customer segment"""
        segment_metrics = df.groupby('segment').agg({
            'customer_id': 'count',
            'predicted_ltv': ['mean', 'median', 'sum'],
            'avg_order_value': 'mean',
            'purchase_frequency': 'mean',
            'months_active': 'mean',
            'email_engagement_rate': 'mean'
        }).round(2)

        segment_metrics.columns = ['_'.join(col).strip() for col in segment_metrics.columns.values]

        return segment_metrics

    def plot_results(self, df, model_results, segment_metrics):
        """Visualize LTV prediction results"""
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['r2'])
        best_result = model_results[best_model_name]

        # Actual vs Predicted
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(best_result['actuals'], best_result['predictions'],
                   alpha=0.5, s=30)
        max_val = max(best_result['actuals'].max(), best_result['predictions'].max())
        ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
        ax1.set_xlabel('Actual LTV ($)', fontsize=11)
        ax1.set_ylabel('Predicted LTV ($)', fontsize=11)
        ax1.set_title(f"LTV Prediction - {best_model_name} (R² = {best_result['r2']:.3f})",
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Model Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        model_names = list(model_results.keys())
        mae_scores = [model_results[m]['mae'] for m in model_names]
        r2_scores = [model_results[m]['r2'] for m in model_names]

        x = np.arange(len(model_names))
        width = 0.35
        ax2_twin = ax2.twinx()

        ax2.bar(x - width/2, mae_scores, width, label='MAE', color='#e74c3c')
        ax2_twin.bar(x + width/2, r2_scores, width, label='R²', color='#2ecc71')

        ax2.set_ylabel('MAE ($)', fontsize=11, color='#e74c3c')
        ax2_twin.set_ylabel('R² Score', fontsize=11, color='#2ecc71')
        ax2.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')

        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        # LTV Distribution by Segment
        ax3 = fig.add_subplot(gs[0, 2])
        segment_order = ['Low Value', 'Medium Value', 'High Value', 'VIP']
        colors_seg = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']

        for segment, color in zip(segment_order, colors_seg):
            segment_data = df[df['segment'] == segment]['predicted_ltv']
            ax3.hist(segment_data, bins=20, alpha=0.6, label=segment,
                    color=color, edgecolor='black')

        ax3.set_xlabel('Predicted LTV ($)', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('LTV Distribution by Segment', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        # Segment Sizes and Total Value
        ax4 = fig.add_subplot(gs[1, 0])
        segment_counts = df['segment'].value_counts()[segment_order]
        colors_pie = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']

        ax4.pie(segment_counts.values, labels=segment_counts.index,
               autopct='%1.1f%%', colors=colors_pie, startangle=90)
        ax4.set_title('Customer Distribution by Segment', fontsize=12, fontweight='bold')

        # Average LTV by Segment
        ax5 = fig.add_subplot(gs[1, 1])
        avg_ltv_by_segment = df.groupby('segment')['predicted_ltv'].mean()[segment_order]
        bars = ax5.bar(segment_order, avg_ltv_by_segment, color=colors_seg,
                      edgecolor='black', linewidth=1.5, alpha=0.7)
        ax5.set_ylabel('Average LTV ($)', fontsize=11)
        ax5.set_title('Average LTV by Segment', fontsize=12, fontweight='bold')
        ax5.set_xticklabels(segment_order, rotation=45, ha='right')
        ax5.grid(True, alpha=0.3, axis='y')

        for bar, value in zip(bars, avg_ltv_by_segment.values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'${value:.0f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

        # LTV by Acquisition Channel
        ax6 = fig.add_subplot(gs[1, 2])
        channel_cols = [col for col in df.columns if col.startswith('channel_')]
        if channel_cols:
            channel_ltv = {}
            for col in channel_cols:
                channel_name = col.replace('channel_', '')
                channel_ltv[channel_name] = df[df[col] == 1]['predicted_ltv'].mean()

            channels = list(channel_ltv.keys())
            ltv_values = list(channel_ltv.values())

            ax6.barh(channels, ltv_values, color='#9b59b6',
                    edgecolor='black', linewidth=1.5, alpha=0.7)
            ax6.set_xlabel('Average LTV ($)', fontsize=11)
            ax6.set_title('LTV by Acquisition Channel', fontsize=12, fontweight='bold')
            ax6.grid(True, alpha=0.3, axis='x')

        # Purchase Frequency vs LTV
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.scatter(df['purchase_frequency'], df['predicted_ltv'],
                   c=df['segment'].map({'Low Value': 0, 'Medium Value': 1,
                                       'High Value': 2, 'VIP': 3}),
                   cmap='RdYlGn', alpha=0.5, s=30, edgecolor='black', linewidth=0.5)
        ax7.set_xlabel('Purchase Frequency', fontsize=11)
        ax7.set_ylabel('Predicted LTV ($)', fontsize=11)
        ax7.set_title('Purchase Frequency vs LTV', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3)

        # Average Order Value vs LTV
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.scatter(df['avg_order_value'], df['predicted_ltv'],
                   c=df['segment'].map({'Low Value': 0, 'Medium Value': 1,
                                       'High Value': 2, 'VIP': 3}),
                   cmap='RdYlGn', alpha=0.5, s=30, edgecolor='black', linewidth=0.5)
        ax8.set_xlabel('Average Order Value ($)', fontsize=11)
        ax8.set_ylabel('Predicted LTV ($)', fontsize=11)
        ax8.set_title('AOV vs LTV', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3)

        # Summary Statistics
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')

        total_customers = len(df)
        total_ltv = df['predicted_ltv'].sum()
        avg_ltv = df['predicted_ltv'].mean()
        median_ltv = df['predicted_ltv'].median()

        summary_text = f"""
        ╔════════════════════════════════════════╗
        ║    CUSTOMER LTV PREDICTION SUMMARY      ║
        ╚════════════════════════════════════════╝

        Best Model: {best_model_name}
        R² Score: {best_result['r2']:.3f}
        MAE: ${best_result['mae']:.2f}
        RMSE: ${best_result['rmse']:.2f}

        ┌──────────────────────────────────────┐
        │ CUSTOMER METRICS                      │
        ├──────────────────────────────────────┤
        │ Total Customers:    {total_customers:>10,d}    │
        │ Average LTV:        ${avg_ltv:>10,.2f}    │
        │ Median LTV:         ${median_ltv:>10,.2f}    │
        │ Total LTV:          ${total_ltv:>10,.0f}    │
        └──────────────────────────────────────┘

        ┌──────────────────────────────────────┐
        │ SEGMENT BREAKDOWN                     │
        ├──────────────────────────────────────┤
        │ VIP:           {(df['segment']=='VIP').sum():>6d} ({(df['segment']=='VIP').mean()*100:>5.1f}%)  │
        │ High Value:    {(df['segment']=='High Value').sum():>6d} ({(df['segment']=='High Value').mean()*100:>5.1f}%)  │
        │ Medium Value:  {(df['segment']=='Medium Value').sum():>6d} ({(df['segment']=='Medium Value').mean()*100:>5.1f}%)  │
        │ Low Value:     {(df['segment']=='Low Value').sum():>6d} ({(df['segment']=='Low Value').mean()*100:>5.1f}%)  │
        └──────────────────────────────────────┘
        """
        ax9.text(0.05, 0.5, summary_text, fontsize=9, family='monospace',
                verticalalignment='center')

        plt.savefig('customer_ltv_analysis.png', dpi=300, bbox_inches='tight')
        print("\n📊 Visualization saved as 'customer_ltv_analysis.png'")
        plt.show()


def main():
    """Main execution function"""
    print("💎 Customer Lifetime Value Prediction System")
    print("=" * 80)

    predictor = LTVPredictor()

    # Generate data
    print("\n📊 Generating customer data...")
    df = predictor.create_sample_data(n_customers=5000)
    print(f"Dataset shape: {df.shape}")
    print(f"Average LTV: ${df['predicted_ltv'].mean():.2f}")

    # Engineer features
    print("\n🔧 Engineering LTV features...")
    df = predictor.engineer_features(df)

    # Calculate segments
    df = predictor.calculate_segments(df)
    print(f"\nCustomer segments:")
    print(df['segment'].value_counts())

    # Train model
    print("\n🤖 Training LTV prediction models...")
    model_results, X_test, y_test, feature_names = predictor.train_model(df)

    # Calculate segment metrics
    segment_metrics = predictor.calculate_segment_metrics(df)

    # Plot results
    print("\n📈 Generating visualizations...")
    predictor.plot_results(df, model_results, segment_metrics)

    print("\n✅ Customer LTV prediction complete!")


if __name__ == "__main__":
    main()
