"""
RFM Customer Segmentation - Clustering Analysis

This module implements RFM-based customer segmentation using various
clustering algorithms to identify different customer value groups.

Dataset: https://www.kaggle.com/datasets/yasserh/customer-segmentation-dataset
Difficulty: ⭐⭐ Intermediate Level
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from datetime import datetime, timedelta

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class RFMCustomerSegmentation:
    """RFM-based Customer Segmentation using Clustering."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.models: Dict[str, Any] = {}
        self.best_model = None
        self.n_clusters = 5
        self.rfm_data = None

    def create_sample_data(self) -> pd.DataFrame:
        """Create realistic e-commerce transaction data."""
        np.random.seed(42)
        n_customers = 2000
        n_transactions = 15000

        # Generate customer IDs
        customer_ids = np.random.randint(1, n_customers + 1, n_transactions)

        # Generate dates over 2 years
        base_date = datetime(2023, 12, 31)
        days_ago = np.random.exponential(180, n_transactions).clip(1, 730).astype(int)
        invoice_dates = [base_date - timedelta(days=int(d)) for d in days_ago]

        # Generate quantities and prices
        quantities = np.random.poisson(3, n_transactions).clip(1, 20)
        unit_prices = np.random.lognormal(2, 0.8, n_transactions).clip(1, 500).round(2)

        # Invoice numbers
        invoice_nos = [f'INV{i:06d}' for i in range(1, n_transactions + 1)]

        # Stock codes and descriptions
        products = [
            ('85123A', 'WHITE HANGING HEART T-LIGHT HOLDER'),
            ('71053', 'WHITE METAL LANTERN'),
            ('84406B', 'CREAM CUPID HEARTS COAT HANGER'),
            ('84029G', 'KNITTED UNION FLAG HOT WATER BOTTLE'),
            ('84029E', 'RED WOOLLY HOTTIE WHITE HEART'),
            ('22752', 'SET 7 BABUSHKA NESTING BOXES'),
            ('21730', 'GLASS STAR FROSTED T-LIGHT HOLDER'),
            ('22633', 'HAND WARMER UNION JACK'),
            ('22632', 'HAND WARMER RED POLKA DOT'),
            ('84879', 'ASSORTED COLOUR BIRD ORNAMENT')
        ]
        product_indices = np.random.choice(len(products), n_transactions)
        stock_codes = [products[i][0] for i in product_indices]
        descriptions = [products[i][1] for i in product_indices]

        countries = np.random.choice(
            ['United Kingdom', 'Germany', 'France', 'Spain', 'Netherlands'],
            n_transactions, p=[0.85, 0.05, 0.04, 0.03, 0.03]
        )

        return pd.DataFrame({
            'InvoiceNo': invoice_nos,
            'StockCode': stock_codes,
            'Description': descriptions,
            'Quantity': quantities,
            'InvoiceDate': invoice_dates,
            'UnitPrice': unit_prices,
            'CustomerID': customer_ids,
            'Country': countries
        })

    def calculate_rfm(self, df: pd.DataFrame, analysis_date: datetime = None) -> pd.DataFrame:
        """Calculate RFM metrics for each customer."""
        if analysis_date is None:
            analysis_date = df['InvoiceDate'].max() + timedelta(days=1)

        # Calculate total amount per transaction
        df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

        # Group by customer
        rfm = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (analysis_date - x.max()).days,  # Recency
            'InvoiceNo': 'nunique',  # Frequency
            'TotalAmount': 'sum'  # Monetary
        }).reset_index()

        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

        # Remove outliers (top 1%)
        for col in ['Recency', 'Frequency', 'Monetary']:
            rfm = rfm[rfm[col] <= rfm[col].quantile(0.99)]

        self.rfm_data = rfm
        return rfm

    def assign_rfm_scores(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """Assign RFM scores (1-5) to each customer."""
        rfm = rfm.copy()

        # Recency: lower is better, so reverse the scoring
        rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')

        # Frequency: higher is better
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])

        # Monetary: higher is better
        rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])

        # Convert to numeric
        for col in ['R_Score', 'F_Score', 'M_Score']:
            rfm[col] = rfm[col].astype(int)

        # Combined RFM score
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        rfm['RFM_Total'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']

        return rfm

    def assign_customer_segments(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """Assign customer segments based on RFM scores."""
        rfm = rfm.copy()

        def segment_customer(row):
            r, f, m = row['R_Score'], row['F_Score'], row['M_Score']

            if r >= 4 and f >= 4 and m >= 4:
                return 'Champions'
            elif r >= 3 and f >= 3 and m >= 3:
                return 'Loyal Customers'
            elif r >= 4 and f <= 2:
                return 'Potential Loyalists'
            elif r <= 2 and f >= 3:
                return 'At Risk'
            elif r <= 2 and f <= 2 and m <= 2:
                return 'Lost'
            elif r <= 3 and f <= 2:
                return 'Hibernating'
            else:
                return 'Others'

        rfm['Segment'] = rfm.apply(segment_customer, axis=1)
        return rfm

    def find_optimal_clusters(self, X_scaled: np.ndarray, max_k: int = 10) -> int:
        """Find optimal number of clusters using elbow method and silhouette score."""
        silhouette_scores = []
        inertias = []

        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            silhouette_scores.append(silhouette_score(X_scaled, labels))
            inertias.append(kmeans.inertia_)

        # Find optimal k based on silhouette score
        optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
        return optimal_k

    def train_clustering_models(self, rfm: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Train multiple clustering algorithms."""
        # Prepare features
        X = rfm[['Recency', 'Frequency', 'Monetary']].values
        X_scaled = self.scaler.fit_transform(X)

        # Find optimal clusters
        self.n_clusters = self.find_optimal_clusters(X_scaled)
        print(f"Optimal number of clusters: {self.n_clusters}")

        results = {}

        # K-Means
        print("\nTraining K-Means...")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        results['K-Means'] = kmeans.fit_predict(X_scaled)
        self.models['K-Means'] = kmeans

        # Hierarchical Clustering
        print("Training Hierarchical Clustering...")
        hierarchical = AgglomerativeClustering(n_clusters=self.n_clusters)
        results['Hierarchical'] = hierarchical.fit_predict(X_scaled)
        self.models['Hierarchical'] = hierarchical

        # Gaussian Mixture Model
        print("Training Gaussian Mixture Model...")
        gmm = GaussianMixture(n_components=self.n_clusters, random_state=42)
        results['GMM'] = gmm.fit_predict(X_scaled)
        self.models['GMM'] = gmm

        return results, X_scaled

    def evaluate_clustering(self, X_scaled: np.ndarray, labels_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Evaluate clustering quality."""
        results = []

        for name, labels in labels_dict.items():
            if len(set(labels)) > 1:  # Need at least 2 clusters
                results.append({
                    'Algorithm': name,
                    'Silhouette': silhouette_score(X_scaled, labels),
                    'Davies-Bouldin': davies_bouldin_score(X_scaled, labels),
                    'Calinski-Harabasz': calinski_harabasz_score(X_scaled, labels)
                })

        results_df = pd.DataFrame(results)
        # Best model has highest silhouette score
        best_model_name = results_df.loc[results_df['Silhouette'].idxmax(), 'Algorithm']
        self.best_model = self.models[best_model_name]

        return results_df

    def plot_exploratory_analysis(self, rfm: pd.DataFrame, output_dir: str = '.') -> None:
        """Generate EDA visualizations."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('RFM Analysis - Exploratory Data Analysis', fontsize=16)

        # Recency distribution
        rfm['Recency'].hist(bins=30, ax=axes[0, 0], color='steelblue', alpha=0.7)
        axes[0, 0].set_title('Recency Distribution')
        axes[0, 0].set_xlabel('Days Since Last Purchase')

        # Frequency distribution
        rfm['Frequency'].hist(bins=30, ax=axes[0, 1], color='coral', alpha=0.7)
        axes[0, 1].set_title('Frequency Distribution')
        axes[0, 1].set_xlabel('Number of Purchases')

        # Monetary distribution
        rfm['Monetary'].hist(bins=30, ax=axes[0, 2], color='green', alpha=0.7)
        axes[0, 2].set_title('Monetary Distribution')
        axes[0, 2].set_xlabel('Total Spending')

        # RFM Score distribution
        if 'RFM_Total' in rfm.columns:
            rfm['RFM_Total'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 0], color='purple')
            axes[1, 0].set_title('RFM Total Score Distribution')

        # Recency vs Monetary scatter
        axes[1, 1].scatter(rfm['Recency'], rfm['Monetary'], alpha=0.5, c='steelblue')
        axes[1, 1].set_title('Recency vs Monetary')
        axes[1, 1].set_xlabel('Recency')
        axes[1, 1].set_ylabel('Monetary')

        # Frequency vs Monetary scatter
        axes[1, 2].scatter(rfm['Frequency'], rfm['Monetary'], alpha=0.5, c='coral')
        axes[1, 2].set_title('Frequency vs Monetary')
        axes[1, 2].set_xlabel('Frequency')
        axes[1, 2].set_ylabel('Monetary')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/rfm_eda.png', dpi=300, bbox_inches='tight')
        print(f"EDA saved to {output_dir}/rfm_eda.png")
        plt.close()

    def plot_clustering_results(self, rfm: pd.DataFrame, labels: np.ndarray,
                               output_dir: str = '.') -> None:
        """Visualize clustering results."""
        rfm = rfm.copy()
        rfm['Cluster'] = labels

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Customer Segmentation Results', fontsize=16)

        # 3D-like scatter (R vs F, colored by cluster)
        scatter = axes[0, 0].scatter(rfm['Recency'], rfm['Frequency'],
                                     c=rfm['Cluster'], cmap='viridis', alpha=0.6)
        axes[0, 0].set_title('Recency vs Frequency')
        axes[0, 0].set_xlabel('Recency (days)')
        axes[0, 0].set_ylabel('Frequency')
        plt.colorbar(scatter, ax=axes[0, 0], label='Cluster')

        # F vs M scatter
        scatter = axes[0, 1].scatter(rfm['Frequency'], rfm['Monetary'],
                                     c=rfm['Cluster'], cmap='viridis', alpha=0.6)
        axes[0, 1].set_title('Frequency vs Monetary')
        axes[0, 1].set_xlabel('Frequency')
        axes[0, 1].set_ylabel('Monetary')
        plt.colorbar(scatter, ax=axes[0, 1], label='Cluster')

        # Cluster distribution
        cluster_counts = rfm['Cluster'].value_counts().sort_index()
        cluster_counts.plot(kind='bar', ax=axes[1, 0], color='steelblue')
        axes[1, 0].set_title('Customers per Cluster')
        axes[1, 0].set_xlabel('Cluster')
        axes[1, 0].set_ylabel('Count')

        # Cluster profiles (mean RFM values)
        cluster_profiles = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
        cluster_profiles_norm = (cluster_profiles - cluster_profiles.min()) / (cluster_profiles.max() - cluster_profiles.min())
        cluster_profiles_norm.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Cluster Profiles (Normalized)')
        axes[1, 1].legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/clustering_results.png', dpi=300, bbox_inches='tight')
        print(f"Clustering results saved to {output_dir}/clustering_results.png")
        plt.close()

    def generate_segment_report(self, rfm: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        """Generate detailed segment analysis report."""
        rfm = rfm.copy()
        rfm['Cluster'] = labels

        report = rfm.groupby('Cluster').agg({
            'CustomerID': 'count',
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': ['mean', 'sum']
        }).round(2)

        report.columns = ['Customer_Count', 'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Total_Revenue']
        report['Revenue_Share'] = (report['Total_Revenue'] / report['Total_Revenue'].sum() * 100).round(2)

        return report


def main():
    """Main execution."""
    print("=" * 70)
    print("RFM CUSTOMER SEGMENTATION ANALYSIS")
    print("=" * 70)

    segmenter = RFMCustomerSegmentation()

    # Create sample data
    df = segmenter.create_sample_data()
    print(f"\nTransaction data: {df.shape}")
    print(f"Unique customers: {df['CustomerID'].nunique()}")

    # Calculate RFM
    rfm = segmenter.calculate_rfm(df)
    print(f"RFM data: {rfm.shape}")

    # Assign RFM scores
    rfm = segmenter.assign_rfm_scores(rfm)
    rfm = segmenter.assign_customer_segments(rfm)

    # Plot EDA
    segmenter.plot_exploratory_analysis(rfm)

    # Train clustering models
    labels_dict, X_scaled = segmenter.train_clustering_models(rfm)

    # Evaluate models
    eval_results = segmenter.evaluate_clustering(X_scaled, labels_dict)
    print(f"\n=== Clustering Evaluation ===\n{eval_results.to_string(index=False)}")

    # Use best model labels
    best_labels = labels_dict[eval_results.loc[eval_results['Silhouette'].idxmax(), 'Algorithm']]

    # Plot results
    segmenter.plot_clustering_results(rfm, best_labels)

    # Generate segment report
    report = segmenter.generate_segment_report(rfm, best_labels)
    print(f"\n=== Segment Report ===\n{report}")

    print("\n" + "=" * 70)
    print("SEGMENTATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
