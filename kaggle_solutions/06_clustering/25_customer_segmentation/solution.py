"""
客戶分群進階
使用多種聚類算法比較
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

class CustomerSegmenter:
    def __init__(self):
        self.scaler = StandardScaler()

    def create_data(self, n=500):
        np.random.seed(42)
        df = pd.DataFrame({
            'Age': np.random.randint(18, 70, n),
            'Income': np.random.normal(50000, 20000, n).clip(10000, 150000),
            'SpendingScore': np.random.randint(1, 101, n),
            'Tenure': np.random.randint(1, 121, n)
        })
        return df

    def compare_algorithms(self, df):
        X = self.scaler.fit_transform(df)

        # KMeans
        kmeans = KMeans(n_clusters=4, random_state=42)
        kmeans_labels = kmeans.fit_predict(X)
        kmeans_score = silhouette_score(X, kmeans_labels)

        # DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X)
        if len(set(dbscan_labels)) > 1:
            dbscan_score = silhouette_score(X, dbscan_labels)
        else:
            dbscan_score = -1

        # Hierarchical
        hierarchical = AgglomerativeClustering(n_clusters=4)
        hier_labels = hierarchical.fit_predict(X)
        hier_score = silhouette_score(X, hier_labels)

        print("=== 聚類算法比較 ===")
        print(f"KMeans - Silhouette Score: {kmeans_score:.4f}, Clusters: {len(set(kmeans_labels))}")
        print(f"DBSCAN - Silhouette Score: {dbscan_score:.4f}, Clusters: {len(set(dbscan_labels))}")
        print(f"Hierarchical - Silhouette Score: {hier_score:.4f}, Clusters: {len(set(hier_labels))}")

        return kmeans_labels

if __name__ == "__main__":
    print("客戶分群進階 - 多算法比較")
    segmenter = CustomerSegmenter()
    df = segmenter.create_data()
    print(f"數據形狀: {df.shape}\n")
    labels = segmenter.compare_algorithms(df)
