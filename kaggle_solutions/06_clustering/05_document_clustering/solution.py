"""
Document Clustering Analysis
Clustering text documents using TF-IDF and various clustering algorithms
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')


class DocumentClustering:
    """Cluster text documents based on content similarity"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        np.random.seed(random_state)

    def generate_documents(self, n_docs=300):
        """
        Generate synthetic documents from different topics
        Topics: Technology, Sports, Health, Finance, Science
        """
        documents = []
        labels_true = []

        # Technology documents
        tech_words = ['software', 'computer', 'programming', 'algorithm', 'data', 'code',
                     'developer', 'technology', 'application', 'system', 'network', 'cloud',
                     'database', 'security', 'artificial intelligence', 'machine learning']
        tech_templates = [
            "The new {0} {1} revolutionizes how we use {2} in modern {3}.",
            "Developers are using {0} to build better {1} with improved {2}.",
            "Latest {0} technology enables faster {1} processing and enhanced {2}.",
            "{0} integration with {1} provides seamless {2} experience.",
            "Advanced {0} algorithms improve {1} efficiency in {2} systems."
        ]

        for _ in range(n_docs // 5):
            template = np.random.choice(tech_templates)
            words = np.random.choice(tech_words, size=4, replace=False)
            documents.append(template.format(*words))
            labels_true.append(0)

        # Sports documents
        sports_words = ['football', 'basketball', 'soccer', 'player', 'team', 'game',
                       'championship', 'coach', 'training', 'victory', 'tournament', 'athlete',
                       'competition', 'score', 'match', 'performance']
        sports_templates = [
            "The {0} team won the {1} after intense {2} and excellent {3}.",
            "Top {0} players showcase amazing {1} during the {2} match.",
            "The coach developed new {0} strategies for the upcoming {1} tournament.",
            "{0} championship features world-class {1} and competitive {2}.",
            "Athletes improve their {0} through rigorous {1} and dedicated {2}."
        ]

        for _ in range(n_docs // 5):
            template = np.random.choice(sports_templates)
            words = np.random.choice(sports_words, size=4, replace=False)
            documents.append(template.format(*words))
            labels_true.append(1)

        # Health documents
        health_words = ['medicine', 'patient', 'treatment', 'doctor', 'hospital', 'therapy',
                       'diagnosis', 'symptoms', 'healthcare', 'wellness', 'disease', 'cure',
                       'clinical', 'research', 'pharmaceutical', 'nutrition']
        health_templates = [
            "New {0} research reveals effective {1} for treating {2} conditions.",
            "Doctors recommend {0} therapy combined with proper {1} for better {2}.",
            "The {0} study shows promising results in {1} treatment and {2} prevention.",
            "Healthcare providers use advanced {0} for accurate {1} and effective {2}.",
            "Latest {0} developments improve patient {1} and overall {2} outcomes."
        ]

        for _ in range(n_docs // 5):
            template = np.random.choice(health_templates)
            words = np.random.choice(health_words, size=4, replace=False)
            documents.append(template.format(*words))
            labels_true.append(2)

        # Finance documents
        finance_words = ['investment', 'stock', 'market', 'trading', 'portfolio', 'profit',
                        'asset', 'revenue', 'financial', 'banking', 'economy', 'capital',
                        'dividend', 'equity', 'interest', 'risk']
        finance_templates = [
            "Investors see strong {0} growth in {1} markets with increased {2}.",
            "The {0} portfolio shows excellent {1} performance and reduced {2}.",
            "Financial analysts predict {0} trends will impact {1} and affect {2}.",
            "Banking sector reports higher {0} from strategic {1} and improved {2}.",
            "Market {0} creates opportunities for {1} investments with manageable {2}."
        ]

        for _ in range(n_docs // 5):
            template = np.random.choice(finance_templates)
            words = np.random.choice(finance_words, size=4, replace=False)
            documents.append(template.format(*words))
            labels_true.append(3)

        # Science documents
        science_words = ['research', 'experiment', 'discovery', 'theory', 'scientist', 'study',
                        'laboratory', 'analysis', 'hypothesis', 'observation', 'physics',
                        'chemistry', 'biology', 'innovation', 'methodology', 'evidence']
        science_templates = [
            "Scientists conduct {0} to validate the new {1} through careful {2}.",
            "Breakthrough {0} in the laboratory leads to important {1} about {2}.",
            "The research team's {0} supports the {1} hypothesis with strong {2}.",
            "Novel {0} methodology enables precise {1} and detailed {2} of phenomena.",
            "Scientific {0} reveals fascinating {1} about natural {2} processes."
        ]

        for _ in range(n_docs // 5):
            template = np.random.choice(science_templates)
            words = np.random.choice(science_words, size=4, replace=False)
            documents.append(template.format(*words))
            labels_true.append(4)

        # Create DataFrame
        df = pd.DataFrame({
            'doc_id': [f'DOC_{i:04d}' for i in range(len(documents))],
            'text': documents,
            'true_label': labels_true
        })

        # Shuffle
        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        return df

    def create_tfidf_features(self, documents):
        """Create TF-IDF features from documents"""
        X = self.vectorizer.fit_transform(documents)
        feature_names = self.vectorizer.get_feature_names_out()
        return X, feature_names

    def optimal_clusters_analysis(self, X, max_k=10):
        """Analyze optimal number of clusters"""
        X_dense = X.toarray() if hasattr(X, 'toarray') else X

        silhouette_scores = []
        inertias = []
        K_range = range(2, max_k + 1)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X_dense)
            silhouette_scores.append(silhouette_score(X_dense, labels))
            inertias.append(kmeans.inertia_)

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters', fontsize=12)
        ax1.set_ylabel('Inertia', fontsize=12)
        ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        ax2.plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/tmp/document_optimal_k.png', dpi=300, bbox_inches='tight')
        plt.show()

        return K_range[np.argmax(silhouette_scores)]

    def compare_algorithms(self, X, n_clusters=5):
        """Compare different clustering algorithms"""
        X_dense = X.toarray() if hasattr(X, 'toarray') else X
        results = {}

        # K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_dense)
        results['KMeans'] = {
            'labels': kmeans_labels,
            'silhouette': silhouette_score(X_dense, kmeans_labels),
            'davies_bouldin': davies_bouldin_score(X_dense, kmeans_labels),
            'n_clusters': len(set(kmeans_labels))
        }

        # Agglomerative
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        agg_labels = agg.fit_predict(X_dense)
        results['Agglomerative'] = {
            'labels': agg_labels,
            'silhouette': silhouette_score(X_dense, agg_labels),
            'davies_bouldin': davies_bouldin_score(X_dense, agg_labels),
            'n_clusters': len(set(agg_labels))
        }

        return results

    def visualize_clusters(self, X, results):
        """Visualize clusters using t-SNE"""
        X_dense = X.toarray() if hasattr(X, 'toarray') else X

        # Use t-SNE for better visualization of high-dimensional text data
        tsne = TSNE(n_components=2, random_state=self.random_state, perplexity=30)
        X_tsne = tsne.fit_transform(X_dense)

        n_algorithms = len(results)
        fig, axes = plt.subplots(1, n_algorithms, figsize=(7*n_algorithms, 6))
        if n_algorithms == 1:
            axes = [axes]

        for idx, (algo_name, result) in enumerate(results.items()):
            labels = result['labels']
            scatter = axes[idx].scatter(X_tsne[:, 0], X_tsne[:, 1],
                                       c=labels, cmap='Set1',
                                       s=60, alpha=0.6, edgecolors='black', linewidth=0.5)
            axes[idx].set_title(f'{algo_name}\n{result["n_clusters"]} clusters\n'
                               f'Silhouette: {result["silhouette"]:.3f}',
                               fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('t-SNE Component 1', fontsize=10)
            axes[idx].set_ylabel('t-SNE Component 2', fontsize=10)
            axes[idx].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[idx], label='Cluster')

        plt.tight_layout()
        plt.savefig('/tmp/document_clusters_tsne.png', dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_cluster_topics(self, df, labels, feature_names, top_n=10):
        """Extract top terms for each cluster"""
        df_analysis = df.copy()
        df_analysis['cluster'] = labels

        # Get TF-IDF matrix
        X_tfidf = self.vectorizer.transform(df['text'])

        print("\n" + "="*80)
        print("CLUSTER TOPIC ANALYSIS")
        print("="*80)

        for cluster_id in sorted(set(labels)):
            cluster_docs = df_analysis[df_analysis['cluster'] == cluster_id]
            print(f"\n{'='*80}")
            print(f"CLUSTER {cluster_id} - {len(cluster_docs)} documents "
                  f"({len(cluster_docs)/len(df)*100:.1f}%)")
            print(f"{'='*80}")

            # Get average TF-IDF scores for this cluster
            cluster_indices = df_analysis[df_analysis['cluster'] == cluster_id].index
            cluster_tfidf = X_tfidf[cluster_indices].mean(axis=0).A1

            # Get top terms
            top_indices = cluster_tfidf.argsort()[-top_n:][::-1]
            top_terms = [(feature_names[i], cluster_tfidf[i]) for i in top_indices]

            print("\nTop Terms:")
            for term, score in top_terms:
                print(f"  {term:20s}: {score:.4f}")

            print(f"\nSample documents:")
            for doc in cluster_docs.head(3)['text'].values:
                print(f"  - {doc}")


def main():
    print("="*80)
    print("DOCUMENT CLUSTERING ANALYSIS")
    print("="*80)

    # Initialize
    clustering = DocumentClustering(random_state=42)

    # Generate documents
    print("\n[1/5] Generating synthetic documents...")
    df = clustering.generate_documents(n_docs=300)
    print(f"Generated {len(df)} documents")
    print(f"\nSample documents:")
    for i, row in df.head(5).iterrows():
        print(f"  {row['doc_id']}: {row['text']}")

    # Create TF-IDF features
    print("\n[2/5] Creating TF-IDF features...")
    X, feature_names = clustering.create_tfidf_features(df['text'])
    print(f"TF-IDF matrix shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")

    # Find optimal k
    print("\n[3/5] Finding optimal number of clusters...")
    optimal_k = clustering.optimal_clusters_analysis(X, max_k=10)
    print(f"Suggested optimal k: {optimal_k}")

    # Compare algorithms
    print(f"\n[4/5] Comparing clustering algorithms with k={optimal_k}...")
    results = clustering.compare_algorithms(X, n_clusters=optimal_k)

    print("\nClustering Performance:")
    print("-" * 70)
    print(f"{'Algorithm':<20} {'Silhouette':>12} {'Davies-Bouldin':>17} {'Clusters':>10}")
    print("-" * 70)
    for algo_name, metrics in results.items():
        print(f"{algo_name:<20} {metrics['silhouette']:>12.4f} "
              f"{metrics['davies_bouldin']:>17.4f} {metrics['n_clusters']:>10d}")

    # Visualize
    print("\n[5/5] Visualizing clusters...")
    clustering.visualize_clusters(X, results)

    # Analyze topics
    best_algo = max(results.items(), key=lambda x: x[1]['silhouette'])
    clustering.analyze_cluster_topics(df, best_algo[1]['labels'], feature_names, top_n=8)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
