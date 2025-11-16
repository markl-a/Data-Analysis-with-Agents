"""
Extractive Text Summarization - Kaggle NLP Solution
===================================================
This solution demonstrates extractive text summarization using TF-IDF,
sentence scoring, and TextRank algorithm for automatic text summarization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import networkx as nx
import warnings
import re
from collections import Counter

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# Download required NLTK data
for resource in ['punkt', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        try:
            nltk.download(resource, quiet=True)
        except:
            pass

class TextSummarizer:
    """Extractive text summarization using multiple approaches"""

    def __init__(self, method='tfidf'):
        """
        Args:
            method: 'tfidf', 'textrank', or 'frequency'
        """
        self.method = method
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def sentence_similarity(self, sent1, sent2):
        """Calculate similarity between two sentences"""
        words1 = [w.lower() for w in word_tokenize(sent1) if w.isalnum()]
        words2 = [w.lower() for w in word_tokenize(sent2) if w.isalnum()]

        all_words = list(set(words1 + words2))

        # Create vectors
        vector1 = [1 if w in words1 else 0 for w in all_words]
        vector2 = [1 if w in words2 else 0 for w in all_words]

        # Calculate cosine similarity
        dot_product = sum(v1 * v2 for v1, v2 in zip(vector1, vector2))
        magnitude1 = sum(v**2 for v in vector1) ** 0.5
        magnitude2 = sum(v**2 for v in vector2) ** 0.5

        if magnitude1 * magnitude2 == 0:
            return 0
        return dot_product / (magnitude1 * magnitude2)

    def tfidf_summarize(self, text, num_sentences=3):
        """Summarize using TF-IDF scoring"""
        sentences = sent_tokenize(text)

        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)

        # Score sentences by sum of TF-IDF values
        sentence_scores = tfidf_matrix.sum(axis=1).A1

        # Get top sentences
        ranked_indices = np.argsort(sentence_scores)[::-1]
        top_indices = sorted(ranked_indices[:num_sentences])

        summary = ' '.join([sentences[i] for i in top_indices])
        return summary, sentence_scores

    def textrank_summarize(self, text, num_sentences=3):
        """Summarize using TextRank algorithm"""
        sentences = sent_tokenize(text)

        # Build similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))

        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    similarity_matrix[i][j] = self.sentence_similarity(
                        sentences[i], sentences[j]
                    )

        # Create graph and apply PageRank
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)

        # Get top sentences
        ranked_sentences = sorted(
            ((scores[i], i) for i in range(len(sentences))),
            reverse=True
        )
        top_indices = sorted([idx for _, idx in ranked_sentences[:num_sentences]])

        summary = ' '.join([sentences[i] for i in top_indices])
        sentence_scores = np.array([scores[i] for i in range(len(sentences))])

        return summary, sentence_scores

    def frequency_summarize(self, text, num_sentences=3):
        """Summarize using word frequency scoring"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())

        # Calculate word frequencies
        word_freq = Counter([
            w for w in words
            if w.isalnum() and w not in self.stop_words
        ])

        # Score sentences by sum of word frequencies
        sentence_scores = np.zeros(len(sentences))

        for i, sentence in enumerate(sentences):
            words_in_sentence = word_tokenize(sentence.lower())
            sentence_scores[i] = sum(
                word_freq.get(w, 0) for w in words_in_sentence
                if w.isalnum() and w not in self.stop_words
            )

        # Normalize by sentence length
        for i, sentence in enumerate(sentences):
            word_count = len([w for w in word_tokenize(sentence) if w.isalnum()])
            if word_count > 0:
                sentence_scores[i] /= word_count

        # Get top sentences
        ranked_indices = np.argsort(sentence_scores)[::-1]
        top_indices = sorted(ranked_indices[:num_sentences])

        summary = ' '.join([sentences[i] for i in top_indices])
        return summary, sentence_scores

    def summarize(self, text, num_sentences=3):
        """Summarize text using selected method"""
        text = self.preprocess_text(text)

        if self.method == 'tfidf':
            return self.tfidf_summarize(text, num_sentences)
        elif self.method == 'textrank':
            return self.textrank_summarize(text, num_sentences)
        elif self.method == 'frequency':
            return self.frequency_summarize(text, num_sentences)
        else:
            raise ValueError(f"Unknown method: {self.method}")

def generate_documents():
    """Generate sample documents for summarization"""
    documents = [
        {
            'title': 'Artificial Intelligence Revolution',
            'text': """Artificial intelligence is transforming the world at an unprecedented pace.
            Machine learning algorithms are now capable of performing tasks that were once thought
            to be exclusively human. Deep learning, a subset of machine learning, uses neural
            networks with multiple layers to analyze complex patterns. Companies across industries
            are investing billions in AI research and development. The healthcare sector uses AI
            for disease diagnosis and drug discovery. Autonomous vehicles rely on AI for navigation
            and decision-making. However, concerns about job displacement and ethical implications
            continue to grow. Researchers emphasize the importance of developing responsible AI
            systems. The future of AI promises both opportunities and challenges for society."""
        },
        {
            'title': 'Climate Change Impact',
            'text': """Climate change represents one of the most pressing challenges of our time.
            Global temperatures have risen by approximately 1.1 degrees Celsius since pre-industrial
            times. Extreme weather events are becoming more frequent and severe across the globe.
            Rising sea levels threaten coastal communities and island nations. The Arctic ice is
            melting at an alarming rate, affecting polar ecosystems. Scientists warn that we have
            limited time to prevent catastrophic consequences. Renewable energy sources like solar
            and wind are crucial for reducing carbon emissions. International cooperation is
            essential for addressing this global crisis. Many countries have committed to achieving
            net-zero emissions by 2050. Individual actions, combined with policy changes, can make
            a significant difference in combating climate change."""
        },
        {
            'title': 'Space Exploration Advances',
            'text': """Space exploration has entered a new era of innovation and discovery. Private
            companies are now competing with government agencies in space missions. SpaceX has
            successfully launched reusable rockets, dramatically reducing launch costs. NASA's
            Perseverance rover is exploring Mars, searching for signs of ancient life. The James
            Webb Space Telescope is revealing unprecedented details about distant galaxies. Plans
            for establishing permanent lunar bases are progressing rapidly. International
            collaboration in space has strengthened with the International Space Station. Mining
            asteroids for valuable resources is no longer science fiction. The commercialization
            of space travel promises to make it accessible to more people. These advances mark
            humanity's growing capability to explore beyond Earth."""
        }
    ]
    return documents

def evaluate_summary_quality(original, summary):
    """Calculate simple quality metrics for summary"""
    original_sentences = sent_tokenize(original)
    summary_sentences = sent_tokenize(summary)

    compression_ratio = len(summary) / len(original)
    sentence_ratio = len(summary_sentences) / len(original_sentences)

    # Calculate coverage (what percentage of original words appear in summary)
    original_words = set(word_tokenize(original.lower()))
    summary_words = set(word_tokenize(summary.lower()))
    coverage = len(original_words & summary_words) / len(original_words)

    return {
        'compression_ratio': compression_ratio,
        'sentence_ratio': sentence_ratio,
        'coverage': coverage,
        'original_length': len(original),
        'summary_length': len(summary)
    }

def create_visualizations(results_df, method_comparison):
    """Create visualizations for summarization results"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Compression ratios
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(range(len(results_df)), results_df['compression_ratio'],
            color='skyblue', alpha=0.7, edgecolor='black')
    ax1.set_xticks(range(len(results_df)))
    ax1.set_xticklabels([f"Doc {i+1}" for i in range(len(results_df))], rotation=0)
    ax1.set_ylabel('Compression Ratio')
    ax1.set_title('Summary Compression Ratios', fontsize=12, fontweight='bold')
    ax1.axhline(y=results_df['compression_ratio'].mean(), color='red',
                linestyle='--', label=f'Mean: {results_df["compression_ratio"].mean():.2f}')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 2. Coverage scores
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(range(len(results_df)), results_df['coverage'],
            color='lightgreen', alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(results_df)))
    ax2.set_xticklabels([f"Doc {i+1}" for i in range(len(results_df))], rotation=0)
    ax2.set_ylabel('Coverage Score')
    ax2.set_title('Word Coverage in Summaries', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3)

    # 3. Method comparison
    ax3 = fig.add_subplot(gs[1, :])
    methods = list(method_comparison.keys())
    metrics = ['compression_ratio', 'coverage']
    x = np.arange(len(methods))
    width = 0.35

    for i, metric in enumerate(metrics):
        values = [method_comparison[m][metric] for m in methods]
        ax3.bar(x + i*width, values, width, label=metric.replace('_', ' ').title(),
                alpha=0.7)

    ax3.set_xlabel('Method')
    ax3.set_ylabel('Score')
    ax3.set_title('Comparison of Summarization Methods', fontsize=12, fontweight='bold')
    ax3.set_xticks(x + width/2)
    ax3.set_xticklabels(methods)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # 4. Length comparison
    ax4 = fig.add_subplot(gs[2, :])
    x_pos = np.arange(len(results_df))
    width = 0.35

    ax4.bar(x_pos - width/2, results_df['original_length'],
            width, label='Original', alpha=0.7, color='orange')
    ax4.bar(x_pos + width/2, results_df['summary_length'],
            width, label='Summary', alpha=0.7, color='green')

    ax4.set_xlabel('Document')
    ax4.set_ylabel('Length (characters)')
    ax4.set_title('Original vs Summary Length', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f"Doc {i+1}" for i in range(len(results_df))])
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    plt.savefig('summarization_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'summarization_analysis.png'")
    plt.close()

def main():
    """Main execution function"""
    print("=" * 60)
    print("Extractive Text Summarization - Kaggle NLP Solution")
    print("=" * 60)

    # Generate documents
    print("\n1. Loading Documents...")
    documents = generate_documents()
    print(f"   - Number of documents: {len(documents)}")

    # Summarize using TF-IDF
    print("\n2. Generating Summaries (TF-IDF method)...")
    summarizer = TextSummarizer(method='tfidf')
    results = []

    for i, doc in enumerate(documents):
        summary, scores = summarizer.summarize(doc['text'], num_sentences=3)
        metrics = evaluate_summary_quality(doc['text'], summary)

        results.append({
            'document': doc['title'],
            'summary': summary,
            **metrics
        })

        print(f"\n   Document: {doc['title']}")
        print(f"   Summary: {summary[:100]}...")
        print(f"   Compression: {metrics['compression_ratio']:.2%}")

    results_df = pd.DataFrame(results)

    # Compare different methods
    print("\n3. Comparing Summarization Methods...")
    methods = ['tfidf', 'textrank', 'frequency']
    method_comparison = {}

    for method in methods:
        summarizer = TextSummarizer(method=method)
        method_metrics = []

        for doc in documents:
            summary, _ = summarizer.summarize(doc['text'], num_sentences=3)
            metrics = evaluate_summary_quality(doc['text'], summary)
            method_metrics.append(metrics)

        # Average metrics
        avg_compression = np.mean([m['compression_ratio'] for m in method_metrics])
        avg_coverage = np.mean([m['coverage'] for m in method_metrics])

        method_comparison[method] = {
            'compression_ratio': avg_compression,
            'coverage': avg_coverage
        }

        print(f"   {method.upper():12} - Compression: {avg_compression:.2%}, "
              f"Coverage: {avg_coverage:.2%}")

    # Create visualizations
    print("\n4. Creating Visualizations...")
    create_visualizations(results_df, method_comparison)

    # Show detailed example
    print("\n5. Detailed Example:")
    print("-" * 60)
    doc = documents[0]
    print(f"\nOriginal Text ({len(doc['text'])} chars):")
    print(doc['text'][:200] + "...\n")

    for method in methods:
        summarizer = TextSummarizer(method=method)
        summary, _ = summarizer.summarize(doc['text'], num_sentences=3)
        print(f"{method.upper()} Summary ({len(summary)} chars):")
        print(summary)
        print()

    # Summary statistics
    print("\n6. Overall Statistics:")
    print("-" * 60)
    print(f"   Average compression ratio: {results_df['compression_ratio'].mean():.2%}")
    print(f"   Average coverage: {results_df['coverage'].mean():.2%}")
    print(f"   Average original length: {results_df['original_length'].mean():.0f} chars")
    print(f"   Average summary length: {results_df['summary_length'].mean():.0f} chars")

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
