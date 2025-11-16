"""
Customer Cohort Analysis System
================================

Problem: Analyze customer behavior over time by grouping users into cohorts
based on acquisition date and tracking retention, engagement, and revenue

Kaggle-style competition: Customer Analytics
Difficulty: ⭐⭐⭐

This solution demonstrates:
- Cohort creation and tracking
- Retention analysis
- Customer lifetime value by cohort
- Engagement metrics over time
- Revenue cohort analysis
- Churn prediction by cohort
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class CohortAnalyzer:
    """Customer cohort analysis and tracking"""

    def __init__(self):
        self.cohorts = None

    def create_sample_data(self, n_users=5000):
        """Generate realistic customer transaction data"""
        np.random.seed(42)

        # Generate user signup dates over 12 months
        start_date = datetime(2023, 1, 1)
        signup_dates = []
        user_ids = []

        for i in range(n_users):
            # More signups in early months
            days_offset = int(np.random.beta(2, 5) * 365)
            signup_dates.append(start_date + timedelta(days=days_offset))
            user_ids.append(f'user_{i:05d}')

        df_users = pd.DataFrame({
            'user_id': user_ids,
            'signup_date': signup_dates,
            'cohort': pd.to_datetime(signup_dates).to_period('M')
        })

        # Generate transactions
        transactions = []
        for _, user in df_users.iterrows():
            signup = user['signup_date']

            # Retention probability decreases over time
            n_months_active = np.random.geometric(p=0.25)  # Average 4 months

            for month_offset in range(min(n_months_active, 12)):
                # Probability of activity decreases
                if np.random.random() < 0.7 ** month_offset:
                    trans_date = signup + timedelta(days=30*month_offset +
                                                   np.random.randint(0, 30))

                    # Number of transactions in month
                    n_trans = np.random.poisson(2) + 1

                    for _ in range(n_trans):
                        transactions.append({
                            'user_id': user['user_id'],
                            'transaction_date': trans_date,
                            'revenue': np.random.lognormal(3, 0.8),
                            'cohort': user['cohort']
                        })

        df_trans = pd.DataFrame(transactions)
        df_trans['transaction_month'] = pd.to_datetime(
            df_trans['transaction_date']
        ).dt.to_period('M')

        return df_users, df_trans

    def calculate_retention(self, df_users, df_trans):
        """Calculate retention rates by cohort"""
        # Create cohort-month matrix
        df_trans = df_trans.copy()
        df_users = df_users.copy()

        # Calculate months since signup
        df_trans = df_trans.merge(df_users[['user_id', 'signup_date']], on='user_id')
        df_trans['months_since_signup'] = (
            (df_trans['transaction_month'] - df_trans['cohort']).apply(lambda x: x.n)
        )

        # Count active users by cohort and month
        retention_data = df_trans.groupby(
            ['cohort', 'months_since_signup']
        )['user_id'].nunique().reset_index()
        retention_data.columns = ['cohort', 'months_since_signup', 'active_users']

        # Get cohort sizes
        cohort_sizes = df_users.groupby('cohort').size().reset_index()
        cohort_sizes.columns = ['cohort', 'cohort_size']

        # Calculate retention rate
        retention = retention_data.merge(cohort_sizes, on='cohort')
        retention['retention_rate'] = retention['active_users'] / retention['cohort_size']

        # Pivot for heatmap
        retention_matrix = retention.pivot(
            index='cohort',
            columns='months_since_signup',
            values='retention_rate'
        )

        return retention_matrix

    def calculate_revenue_cohorts(self, df_users, df_trans):
        """Calculate revenue by cohort and time"""
        df_trans = df_trans.merge(df_users[['user_id', 'signup_date']], on='user_id')
        df_trans['months_since_signup'] = (
            (df_trans['transaction_month'] - df_trans['cohort']).apply(lambda x: x.n)
        )

        # Revenue by cohort and month
        revenue_data = df_trans.groupby(
            ['cohort', 'months_since_signup']
        )['revenue'].sum().reset_index()

        # Get cohort sizes for per-user metrics
        cohort_sizes = df_users.groupby('cohort').size().reset_index()
        cohort_sizes.columns = ['cohort', 'cohort_size']

        revenue_data = revenue_data.merge(cohort_sizes, on='cohort')
        revenue_data['revenue_per_user'] = (
            revenue_data['revenue'] / revenue_data['cohort_size']
        )

        # Pivot for heatmap
        revenue_matrix = revenue_data.pivot(
            index='cohort',
            columns='months_since_signup',
            values='revenue_per_user'
        )

        return revenue_matrix

    def calculate_ltv_by_cohort(self, df_users, df_trans):
        """Calculate customer lifetime value by cohort"""
        # Total revenue per user
        user_revenue = df_trans.groupby('user_id')['revenue'].sum().reset_index()
        user_revenue.columns = ['user_id', 'total_revenue']

        # Merge with cohort info
        ltv_data = df_users.merge(user_revenue, on='user_id', how='left')
        ltv_data['total_revenue'] = ltv_data['total_revenue'].fillna(0)

        # LTV by cohort
        cohort_ltv = ltv_data.groupby('cohort').agg({
            'total_revenue': ['mean', 'median', 'std', 'sum'],
            'user_id': 'count'
        }).reset_index()

        cohort_ltv.columns = ['cohort', 'mean_ltv', 'median_ltv',
                             'std_ltv', 'total_revenue', 'user_count']

        return cohort_ltv

    def calculate_engagement_metrics(self, df_users, df_trans):
        """Calculate engagement metrics by cohort"""
        # Transactions per user by cohort
        user_trans_count = df_trans.groupby('user_id').size().reset_index()
        user_trans_count.columns = ['user_id', 'transaction_count']

        engagement = df_users.merge(user_trans_count, on='user_id', how='left')
        engagement['transaction_count'] = engagement['transaction_count'].fillna(0)

        # Calculate months active
        user_activity = df_trans.groupby('user_id').agg({
            'transaction_month': lambda x: x.nunique()
        }).reset_index()
        user_activity.columns = ['user_id', 'months_active']

        engagement = engagement.merge(user_activity, on='user_id', how='left')
        engagement['months_active'] = engagement['months_active'].fillna(0)

        # Aggregate by cohort
        cohort_engagement = engagement.groupby('cohort').agg({
            'transaction_count': 'mean',
            'months_active': 'mean',
            'user_id': 'count'
        }).reset_index()

        cohort_engagement.columns = ['cohort', 'avg_transactions',
                                     'avg_months_active', 'user_count']

        return cohort_engagement

    def plot_results(self, retention_matrix, revenue_matrix, cohort_ltv, engagement):
        """Visualize cohort analysis results"""
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Retention Heatmap
        ax1 = fig.add_subplot(gs[0, :])
        sns.heatmap(retention_matrix * 100, annot=True, fmt='.0f', cmap='RdYlGn',
                   ax=ax1, cbar_kws={'label': 'Retention Rate (%)'}, vmin=0, vmax=100)
        ax1.set_xlabel('Months Since Signup', fontsize=11)
        ax1.set_ylabel('Cohort (Month-Year)', fontsize=11)
        ax1.set_title('Cohort Retention Analysis (%)', fontsize=13, fontweight='bold')

        # Revenue per User Heatmap
        ax2 = fig.add_subplot(gs[1, :])
        sns.heatmap(revenue_matrix, annot=True, fmt='.0f', cmap='YlGnBu',
                   ax=ax2, cbar_kws={'label': 'Revenue per User ($)'})
        ax2.set_xlabel('Months Since Signup', fontsize=11)
        ax2.set_ylabel('Cohort (Month-Year)', fontsize=11)
        ax2.set_title('Revenue per User by Cohort ($)', fontsize=13, fontweight='bold')

        # LTV by Cohort
        ax3 = fig.add_subplot(gs[2, 0])
        cohort_ltv_sorted = cohort_ltv.sort_values('cohort')
        ax3.bar(range(len(cohort_ltv_sorted)), cohort_ltv_sorted['mean_ltv'],
               color='#3498db', edgecolor='black', linewidth=1.5, alpha=0.7)
        ax3.set_xlabel('Cohort', fontsize=11)
        ax3.set_ylabel('Average LTV ($)', fontsize=11)
        ax3.set_title('Customer Lifetime Value by Cohort', fontsize=12, fontweight='bold')
        ax3.set_xticks(range(len(cohort_ltv_sorted)))
        ax3.set_xticklabels([str(c) for c in cohort_ltv_sorted['cohort']],
                           rotation=45, ha='right', fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')

        # Cohort Sizes
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.bar(range(len(cohort_ltv_sorted)), cohort_ltv_sorted['user_count'],
               color='#2ecc71', edgecolor='black', linewidth=1.5, alpha=0.7)
        ax4.set_xlabel('Cohort', fontsize=11)
        ax4.set_ylabel('Number of Users', fontsize=11)
        ax4.set_title('Cohort Sizes', fontsize=12, fontweight='bold')
        ax4.set_xticks(range(len(cohort_ltv_sorted)))
        ax4.set_xticklabels([str(c) for c in cohort_ltv_sorted['cohort']],
                           rotation=45, ha='right', fontsize=8)
        ax4.grid(True, alpha=0.3, axis='y')

        # Engagement Metrics
        ax5 = fig.add_subplot(gs[2, 2])
        engagement_sorted = engagement.sort_values('cohort')
        x = np.arange(len(engagement_sorted))
        width = 0.35

        ax5.bar(x - width/2, engagement_sorted['avg_transactions'], width,
               label='Avg Transactions', color='#9b59b6', edgecolor='black')
        ax5_twin = ax5.twinx()
        ax5_twin.bar(x + width/2, engagement_sorted['avg_months_active'], width,
                    label='Avg Months Active', color='#e74c3c', edgecolor='black')

        ax5.set_xlabel('Cohort', fontsize=11)
        ax5.set_ylabel('Avg Transactions', fontsize=11, color='#9b59b6')
        ax5_twin.set_ylabel('Avg Months Active', fontsize=11, color='#e74c3c')
        ax5.set_title('Engagement by Cohort', fontsize=12, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels([str(c) for c in engagement_sorted['cohort']],
                           rotation=45, ha='right', fontsize=8)
        ax5.grid(True, alpha=0.3, axis='y')

        lines1, labels1 = ax5.get_legend_handles_labels()
        lines2, labels2 = ax5_twin.get_legend_handles_labels()
        ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

        plt.savefig('cohort_analysis.png', dpi=300, bbox_inches='tight')
        print("\n📊 Visualization saved as 'cohort_analysis.png'")
        plt.show()


def main():
    """Main execution function"""
    print("👥 Customer Cohort Analysis System")
    print("=" * 80)

    analyzer = CohortAnalyzer()

    # Generate data
    print("\n📊 Generating customer transaction data...")
    df_users, df_trans = analyzer.create_sample_data(n_users=5000)
    print(f"Users: {len(df_users)}")
    print(f"Transactions: {len(df_trans)}")
    print(f"Cohorts: {df_users['cohort'].nunique()}")

    # Calculate retention
    print("\n📈 Calculating retention rates...")
    retention_matrix = analyzer.calculate_retention(df_users, df_trans)

    # Calculate revenue cohorts
    print("\n💰 Calculating revenue by cohort...")
    revenue_matrix = analyzer.calculate_revenue_cohorts(df_users, df_trans)

    # Calculate LTV
    print("\n📊 Calculating customer lifetime value...")
    cohort_ltv = analyzer.calculate_ltv_by_cohort(df_users, df_trans)

    # Calculate engagement
    print("\n🎯 Calculating engagement metrics...")
    engagement = analyzer.calculate_engagement_metrics(df_users, df_trans)

    # Print summary
    print("\n" + "="*80)
    print("COHORT ANALYSIS SUMMARY")
    print("="*80)
    print(f"\nRetention Rate (Month 1): {retention_matrix.iloc[:, 0].mean():.1%}")
    print(f"Retention Rate (Month 3): {retention_matrix.iloc[:, 2].mean():.1%}")
    print(f"Retention Rate (Month 6): {retention_matrix.iloc[:, 5].mean():.1%}")
    print(f"\nAverage LTV: ${cohort_ltv['mean_ltv'].mean():.2f}")
    print(f"Total Revenue: ${cohort_ltv['total_revenue'].sum():,.0f}")

    # Plot results
    print("\n📈 Generating visualizations...")
    analyzer.plot_results(retention_matrix, revenue_matrix, cohort_ltv, engagement)

    print("\n✅ Cohort analysis complete!")


if __name__ == "__main__":
    main()
