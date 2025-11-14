"""K-Means clustering for customer segmentation."""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import Optional, List, Dict, Any, Tuple
from loguru import logger


class KMeansClusterer:
    """Perform K-Means clustering for customer segmentation."""

    def __init__(
        self,
        n_clusters: int = 5,
        random_state: int = 42,
        max_iter: int = 300,
        n_init: int = 10
    ):
        """
        Initialize the K-Means Clusterer.

        Args:
            n_clusters: Number of clusters
            random_state: Random state for reproducibility
            max_iter: Maximum number of iterations
            n_init: Number of times K-means will be run with different centroid seeds
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init

        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.scaled_data = None

    def fit(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        scale_features: bool = True
    ) -> 'KMeansClusterer':
        """
        Fit the K-Means model.

        Args:
            df: DataFrame containing features
            feature_columns: List of columns to use for clustering
            scale_features: Whether to scale features before clustering

        Returns:
            self
        """
        logger.info(f"Fitting K-Means with {self.n_clusters} clusters...")

        # Validate feature columns
        missing_cols = set(feature_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

        self.feature_columns = feature_columns

        # Prepare data
        X = df[feature_columns].copy()

        # Handle missing values
        if X.isnull().any().any():
            logger.warning("Missing values detected. Filling with median values.")
            X = X.fillna(X.median())

        # Scale features
        if scale_features:
            self.scaled_data = self.scaler.fit_transform(X)
        else:
            self.scaled_data = X.values

        # Fit K-Means
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            max_iter=self.max_iter,
            n_init=self.n_init
        )

        self.model.fit(self.scaled_data)

        logger.success(f"K-Means clustering completed. Inertia: {self.model.inertia_:.2f}")

        return self

    def predict(self, df: pd.DataFrame, scale_features: bool = True) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Args:
            df: DataFrame containing features
            scale_features: Whether to scale features

        Returns:
            Cluster labels
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = df[self.feature_columns].copy()

        # Handle missing values
        if X.isnull().any().any():
            X = X.fillna(X.median())

        # Scale features
        if scale_features:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values

        return self.model.predict(X_scaled)

    def fit_predict(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        scale_features: bool = True
    ) -> np.ndarray:
        """
        Fit the model and predict cluster labels.

        Args:
            df: DataFrame containing features
            feature_columns: List of columns to use for clustering
            scale_features: Whether to scale features

        Returns:
            Cluster labels
        """
        self.fit(df, feature_columns, scale_features)
        return self.model.labels_

    def get_cluster_centers(self, inverse_transform: bool = True) -> pd.DataFrame:
        """
        Get cluster centers.

        Args:
            inverse_transform: Whether to inverse transform scaled centers

        Returns:
            DataFrame with cluster centers
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        centers = self.model.cluster_centers_

        if inverse_transform and hasattr(self.scaler, 'inverse_transform'):
            centers = self.scaler.inverse_transform(centers)

        return pd.DataFrame(centers, columns=self.feature_columns)

    def get_cluster_summary(
        self,
        df: pd.DataFrame,
        cluster_labels: np.ndarray,
        summary_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get summary statistics for each cluster.

        Args:
            df: Original DataFrame
            cluster_labels: Cluster labels
            summary_columns: Columns to summarize (default: feature columns)

        Returns:
            DataFrame with cluster statistics
        """
        if summary_columns is None:
            summary_columns = self.feature_columns

        df_with_clusters = df.copy()
        df_with_clusters['Cluster'] = cluster_labels

        logger.info("Generating cluster summary...")

        # Calculate summary statistics
        summary = df_with_clusters.groupby('Cluster')[summary_columns].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(2)

        # Add cluster sizes
        cluster_sizes = df_with_clusters['Cluster'].value_counts().sort_index()
        cluster_percentages = (cluster_sizes / len(df_with_clusters) * 100).round(2)

        logger.success("Cluster summary generated")

        return summary

    def evaluate_clustering(self) -> Dict[str, float]:
        """
        Evaluate clustering quality.

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None or self.scaled_data is None:
            raise ValueError("Model not fitted. Call fit() first.")

        logger.info("Evaluating clustering quality...")

        metrics = {
            'inertia': float(self.model.inertia_),
            'n_clusters': self.n_clusters,
            'n_samples': len(self.scaled_data)
        }

        # Calculate silhouette score (only if we have more than 1 cluster and less than n_samples)
        if 1 < self.n_clusters < len(self.scaled_data):
            try:
                silhouette = silhouette_score(self.scaled_data, self.model.labels_)
                metrics['silhouette_score'] = float(silhouette)
            except Exception as e:
                logger.warning(f"Could not calculate silhouette score: {e}")

        # Calculate Davies-Bouldin index (lower is better)
        if self.n_clusters > 1:
            try:
                db_score = davies_bouldin_score(self.scaled_data, self.model.labels_)
                metrics['davies_bouldin_score'] = float(db_score)
            except Exception as e:
                logger.warning(f"Could not calculate Davies-Bouldin score: {e}")

        logger.success("Clustering evaluation completed")

        return metrics

    def find_optimal_clusters(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        k_range: Optional[List[int]] = None,
        scale_features: bool = True
    ) -> Dict[int, Dict[str, float]]:
        """
        Find optimal number of clusters using elbow method and silhouette analysis.

        Args:
            df: DataFrame containing features
            feature_columns: List of columns to use for clustering
            k_range: Range of K values to try (default: 2 to 10)
            scale_features: Whether to scale features

        Returns:
            Dictionary with metrics for each K value
        """
        if k_range is None:
            k_range = list(range(2, 11))

        logger.info(f"Finding optimal number of clusters for K in {k_range}...")

        # Prepare data
        X = df[feature_columns].copy()
        if X.isnull().any().any():
            X = X.fillna(X.median())

        if scale_features:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.values

        results = {}

        for k in k_range:
            logger.info(f"Testing K = {k}...")

            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                max_iter=self.max_iter,
                n_init=self.n_init
            )

            labels = kmeans.fit_predict(X_scaled)

            metrics = {
                'inertia': float(kmeans.inertia_),
                'n_clusters': k
            }

            # Silhouette score
            if k > 1:
                try:
                    silhouette = silhouette_score(X_scaled, labels)
                    metrics['silhouette_score'] = float(silhouette)
                except Exception as e:
                    logger.warning(f"Could not calculate silhouette score for K={k}: {e}")

            # Davies-Bouldin score
            try:
                db_score = davies_bouldin_score(X_scaled, labels)
                metrics['davies_bouldin_score'] = float(db_score)
            except Exception as e:
                logger.warning(f"Could not calculate Davies-Bouldin score for K={k}: {e}")

            results[k] = metrics

        logger.success("Optimal cluster search completed")

        return results

    def get_cluster_distribution(self, cluster_labels: np.ndarray) -> pd.DataFrame:
        """
        Get distribution of samples across clusters.

        Args:
            cluster_labels: Cluster labels

        Returns:
            DataFrame with cluster distribution
        """
        distribution = pd.Series(cluster_labels).value_counts().sort_index()

        df = pd.DataFrame({
            'Cluster': distribution.index,
            'Count': distribution.values,
            'Percentage': (distribution.values / len(cluster_labels) * 100).round(2)
        })

        return df

    def assign_cluster_names(
        self,
        cluster_names: Dict[int, str]
    ) -> Dict[int, str]:
        """
        Assign custom names to clusters.

        Args:
            cluster_names: Dictionary mapping cluster IDs to names

        Returns:
            Cluster name mapping
        """
        self.cluster_names = cluster_names
        logger.info(f"Assigned names to {len(cluster_names)} clusters")
        return cluster_names
