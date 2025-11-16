"""
Topic Modeling with LDA - Kaggle NLP Solution
==============================================
This solution demonstrates topic modeling using Latent Dirichlet Allocation (LDA)
to discover hidden topics in a collection of documents.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import warnings
import re
from collections import Counter
from wordcloud import WordCloud

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

class TopicModeler:
    """Topic modeling using LDA and NMF"""

    def __init__(self, n_topics=5, method='lda'):
        """
        Args:
            n_topics: Number of topics to extract
            method: 'lda' or 'nmf'
        """
        self.n_topics = n_topics
        self.method = method
        self.vectorizer = CountVectorizer(
            max_features=1000,
            stop_words='english',
            max_df=0.8,
            min_df=2
        )

        if method == 'lda':
            self.model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=20,
                learning_method='online'
            )
        else:  # NMF
            self.model = NMF(
                n_components=n_topics,
                random_state=42,
                max_iter=200
            )

        self.feature_names = None

    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def fit(self, documents):
        """Fit topic model on documents"""
        # Preprocess
        processed_docs = [self.preprocess_text(doc) for doc in documents]

        # Vectorize
        self.doc_term_matrix = self.vectorizer.fit_transform(processed_docs)
        self.feature_names = self.vectorizer.get_feature_names_out()

        # Fit model
        self.model.fit(self.doc_term_matrix)

        # Get document-topic distribution
        self.doc_topic_dist = self.model.transform(self.doc_term_matrix)

        return self

    def get_top_words(self, topic_idx, n_words=10):
        """Get top words for a topic"""
        topic_weights = self.model.components_[topic_idx]
        top_indices = np.argsort(topic_weights)[-n_words:][::-1]
        top_words = [(self.feature_names[i], topic_weights[i])
                     for i in top_indices]
        return top_words

    def get_topic_names(self, n_words=3):
        """Generate descriptive names for topics"""
        topic_names = []
        for topic_idx in range(self.n_topics):
            top_words = self.get_top_words(topic_idx, n_words)
            name = ', '.join([word for word, _ in top_words])
            topic_names.append(f"Topic {topic_idx + 1}: {name}")
        return topic_names

    def get_document_topics(self):
        """Get dominant topic for each document"""
        dominant_topics = np.argmax(self.doc_topic_dist, axis=1)
        return dominant_topics

def generate_document_collection():
    """Generate synthetic document collection on various topics"""

    documents = [
        # Technology documents
        "Artificial intelligence and machine learning are revolutionizing the tech industry. "
        "Neural networks and deep learning algorithms enable computers to learn from data. "
        "Natural language processing helps machines understand human language.",

        "Cloud computing provides scalable infrastructure for businesses. Companies use AWS, "
        "Azure, and Google Cloud for hosting applications. Serverless architecture reduces "
        "operational costs and improves scalability.",

        "Cybersecurity is crucial in the digital age. Encryption protects sensitive data from "
        "hackers. Companies invest heavily in security measures to prevent data breaches and "
        "protect customer information.",

        # Health & Medicine documents
        "Medical research focuses on developing new treatments for diseases. Clinical trials "
        "test drug efficacy and safety. Researchers work on cancer therapies, vaccines, and "
        "personalized medicine approaches.",

        "Healthy lifestyle includes regular exercise and balanced nutrition. Physical activity "
        "reduces risk of heart disease and diabetes. A diet rich in fruits and vegetables "
        "promotes overall wellness.",

        "Mental health awareness is growing worldwide. Therapy and counseling help people cope "
        "with anxiety and depression. Meditation and mindfulness practices improve psychological "
        "well-being.",

        # Environment documents
        "Climate change affects global ecosystems and weather patterns. Rising temperatures "
        "cause ice caps to melt and sea levels to rise. Reducing carbon emissions is essential "
        "for environmental sustainability.",

        "Renewable energy sources like solar and wind power reduce fossil fuel dependence. "
        "Solar panels convert sunlight into electricity. Wind turbines generate clean energy "
        "without greenhouse gas emissions.",

        "Wildlife conservation protects endangered species from extinction. Habitat preservation "
        "maintains biodiversity. National parks and reserves provide safe environments for "
        "animals and plants.",

        # Finance & Business documents
        "Stock market trading involves buying and selling company shares. Investors analyze "
        "financial statements and market trends. Diversification reduces investment risk across "
        "different asset classes.",

        "Cryptocurrency and blockchain technology disrupt traditional finance. Bitcoin and "
        "Ethereum enable decentralized transactions. Smart contracts automate financial "
        "agreements without intermediaries.",

        "Entrepreneurship drives innovation and economic growth. Startups develop new products "
        "and services. Venture capital funding helps companies scale their operations.",

        # Education documents
        "Online learning platforms democratize education access. Students take courses from "
        "top universities worldwide. Video lectures and interactive exercises enhance learning "
        "experiences.",

        "STEM education prepares students for technology careers. Science, technology, "
        "engineering, and mathematics skills are increasingly important. Coding and programming "
        "are essential modern competencies.",

        "Early childhood education shapes cognitive development. Preschool programs build "
        "foundational skills in reading and mathematics. Play-based learning encourages "
        "creativity and social interaction.",

        # Sports documents
        "Professional athletes train rigorously to improve performance. Coaches develop "
        "strategic game plans and training regimens. Sports science optimizes nutrition "
        "and recovery protocols.",

        "Olympic games showcase world-class athletic competition. Athletes from different "
        "countries compete in various sports. Training for Olympics requires years of "
        "dedication and sacrifice.",

        "Team sports teach cooperation and leadership skills. Basketball, soccer, and "
        "football require coordinated teamwork. Players develop communication and strategic "
        "thinking abilities.",
    ]

    return documents

def create_visualizations(topic_modeler, documents):
    """Create visualizations for topic modeling results"""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

    # 1. Topic word clouds (3 topics)
    for idx in range(min(3, topic_modeler.n_topics)):
        ax = fig.add_subplot(gs[0, idx])
        top_words = topic_modeler.get_top_words(idx, n_words=20)
        word_freq = {word: weight for word, weight in top_words}

        if word_freq:
            wordcloud = WordCloud(
                width=400, height=300,
                background_color='white',
                colormap='viridis'
            ).generate_from_frequencies(word_freq)

            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            topic_name = ', '.join([w for w, _ in top_words[:3]])
            ax.set_title(f'Topic {idx+1}: {topic_name}', fontsize=10, fontweight='bold')

    # 2. Document-Topic Distribution Heatmap
    ax2 = fig.add_subplot(gs[1, :])
    doc_topic_dist = topic_modeler.doc_topic_dist

    sns.heatmap(doc_topic_dist.T, cmap='YlOrRd', cbar_kws={'label': 'Probability'},
                yticklabels=[f'T{i+1}' for i in range(topic_modeler.n_topics)],
                xticklabels=[f'D{i+1}' for i in range(len(documents))],
                ax=ax2)
    ax2.set_xlabel('Documents')
    ax2.set_ylabel('Topics')
    ax2.set_title('Document-Topic Distribution', fontsize=12, fontweight='bold')

    # 3. Topic sizes (number of documents per topic)
    ax3 = fig.add_subplot(gs[2, 0])
    dominant_topics = topic_modeler.get_document_topics()
    topic_counts = Counter(dominant_topics)

    topics = sorted(topic_counts.keys())
    counts = [topic_counts[t] for t in topics]
    colors = plt.cm.Set3(range(len(topics)))

    bars = ax3.bar(topics, counts, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Topic')
    ax3.set_ylabel('Number of Documents')
    ax3.set_title('Documents per Topic', fontsize=12, fontweight='bold')
    ax3.set_xticks(topics)
    ax3.set_xticklabels([f'T{t+1}' for t in topics])
    ax3.grid(axis='y', alpha=0.3)

    # 4. Topic coherence (average probability)
    ax4 = fig.add_subplot(gs[2, 1])
    avg_probs = doc_topic_dist.mean(axis=0)

    ax4.bar(range(topic_modeler.n_topics), avg_probs,
            color=colors[:topic_modeler.n_topics], alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Topic')
    ax4.set_ylabel('Average Probability')
    ax4.set_title('Topic Prevalence', fontsize=12, fontweight='bold')
    ax4.set_xticks(range(topic_modeler.n_topics))
    ax4.set_xticklabels([f'T{i+1}' for i in range(topic_modeler.n_topics)])
    ax4.grid(axis='y', alpha=0.3)

    # 5. Topic entropy (diversity)
    ax5 = fig.add_subplot(gs[2, 2])
    entropies = []
    for i in range(len(documents)):
        probs = doc_topic_dist[i]
        # Calculate entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropies.append(entropy)

    ax5.hist(entropies, bins=15, color='skyblue', alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Entropy')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Topic Distribution Entropy', fontsize=12, fontweight='bold')
    ax5.axvline(np.mean(entropies), color='red', linestyle='--',
                label=f'Mean: {np.mean(entropies):.2f}')
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)

    plt.savefig('topic_modeling_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'topic_modeling_analysis.png'")
    plt.close()

def main():
    """Main execution function"""
    print("=" * 60)
    print("Topic Modeling with LDA - Kaggle NLP Solution")
    print("=" * 60)

    # Generate documents
    print("\n1. Loading Document Collection...")
    documents = generate_document_collection()
    print(f"   - Number of documents: {len(documents)}")
    print(f"   - Average document length: {np.mean([len(d) for d in documents]):.0f} chars")

    # Fit LDA model
    print("\n2. Training LDA Topic Model...")
    n_topics = 5
    lda_modeler = TopicModeler(n_topics=n_topics, method='lda')
    lda_modeler.fit(documents)

    print(f"   - Number of topics: {n_topics}")
    print(f"   - Vocabulary size: {len(lda_modeler.feature_names)}")
    print(f"   - Model iterations: {lda_modeler.model.n_iter_}")

    # Display topics
    print("\n3. Discovered Topics:")
    print("-" * 60)

    for topic_idx in range(n_topics):
        top_words = lda_modeler.get_top_words(topic_idx, n_words=10)
        print(f"\nTopic {topic_idx + 1}:")
        word_str = ', '.join([f"{word}({weight:.3f})" for word, weight in top_words])
        print(f"   {word_str}")

    # Document classification
    print("\n4. Document-Topic Assignments:")
    print("-" * 60)

    dominant_topics = lda_modeler.get_document_topics()
    topic_names = lda_modeler.get_topic_names(n_words=3)

    for doc_idx in range(min(10, len(documents))):
        topic_dist = lda_modeler.doc_topic_dist[doc_idx]
        dominant_topic = dominant_topics[doc_idx]

        print(f"\nDocument {doc_idx + 1}:")
        print(f"   Text: {documents[doc_idx][:70]}...")
        print(f"   Main Topic: {dominant_topic + 1} ({topic_dist[dominant_topic]:.2%})")
        print(f"   Distribution: {', '.join([f'T{i+1}:{p:.2%}' for i, p in enumerate(topic_dist)])}")

    # Compare with NMF
    print("\n5. Comparing with NMF...")
    nmf_modeler = TopicModeler(n_topics=n_topics, method='nmf')
    nmf_modeler.fit(documents)

    print("\nNMF Topics:")
    for topic_idx in range(n_topics):
        top_words = nmf_modeler.get_top_words(topic_idx, n_words=8)
        print(f"   Topic {topic_idx + 1}: {', '.join([word for word, _ in top_words])}")

    # Topic statistics
    print("\n6. Topic Statistics:")
    print("-" * 60)

    topic_counts = Counter(dominant_topics)
    for topic_id, count in sorted(topic_counts.items()):
        percentage = count / len(documents) * 100
        print(f"   Topic {topic_id + 1}: {count} documents ({percentage:.1f}%)")

    # Create visualizations
    print("\n7. Creating Visualizations...")
    create_visualizations(lda_modeler, documents)

    # Topic coherence
    print("\n8. Model Quality Metrics:")
    print("-" * 60)

    # Perplexity (lower is better)
    perplexity = lda_modeler.model.perplexity(lda_modeler.doc_term_matrix)
    print(f"   - Perplexity: {perplexity:.2f}")

    # Average topic entropy
    avg_entropy = np.mean([
        -np.sum(lda_modeler.doc_topic_dist[i] * np.log(lda_modeler.doc_topic_dist[i] + 1e-10))
        for i in range(len(documents))
    ])
    print(f"   - Average document entropy: {avg_entropy:.3f}")

    # Topic separation (average distance between topics)
    from scipy.spatial.distance import cosine
    topic_vectors = lda_modeler.model.components_
    distances = []
    for i in range(n_topics):
        for j in range(i + 1, n_topics):
            dist = cosine(topic_vectors[i], topic_vectors[j])
            distances.append(dist)
    avg_distance = np.mean(distances)
    print(f"   - Average topic separation: {avg_distance:.3f}")

    print("\n9. Interactive Topic Explorer:")
    print("-" * 60)
    print("\nEnter a document to see its topic distribution:")
    sample_doc = "Machine learning and artificial intelligence are transforming healthcare"
    print(f"\nSample: '{sample_doc}'")

    # Process sample
    processed = [lda_modeler.preprocess_text(sample_doc)]
    doc_vector = lda_modeler.vectorizer.transform(processed)
    topic_dist = lda_modeler.model.transform(doc_vector)[0]

    print("\nTopic probabilities:")
    for i, prob in enumerate(topic_dist):
        if prob > 0.05:  # Show only significant topics
            top_words = lda_modeler.get_top_words(i, n_words=5)
            words = ', '.join([w for w, _ in top_words])
            print(f"   Topic {i+1} ({prob:.1%}): {words}")

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
