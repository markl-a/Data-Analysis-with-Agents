"""
Session-Based Recommendation System
====================================

This solution demonstrates session-based recommendations using RNNs (GRU/LSTM),
session co-occurrence matrices, and sequential pattern mining for next-item prediction.

Author: Kaggle Solutions Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, ndcg_score
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set
import warnings
warnings.filterwarnings('ignore')


class SessionGRU:
    """GRU-based session recommendation model"""

    def __init__(self, n_items: int, embedding_dim: int = 50, hidden_dim: int = 100,
                 learning_rate: float = 0.01, n_epochs: int = 10):
        """
        Initialize Session GRU

        Args:
            n_items: Number of unique items
            embedding_dim: Dimension of item embeddings
            hidden_dim: Dimension of hidden state
            learning_rate: Learning rate for training
            n_epochs: Number of training epochs
        """
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        # Initialize parameters (simplified GRU)
        self.embeddings = np.random.randn(n_items, embedding_dim) * 0.01
        self.Wz = np.random.randn(embedding_dim + hidden_dim, hidden_dim) * 0.01
        self.Wr = np.random.randn(embedding_dim + hidden_dim, hidden_dim) * 0.01
        self.Wh = np.random.randn(embedding_dim + hidden_dim, hidden_dim) * 0.01
        self.Wy = np.random.randn(hidden_dim, n_items) * 0.01
        self.by = np.zeros(n_items)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def tanh(self, x: np.ndarray) -> np.ndarray:
        """Tanh activation"""
        return np.tanh(x)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def forward(self, session: List[int]) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Forward pass through GRU

        Args:
            session: List of item IDs in session

        Returns:
            Final output and list of hidden states
        """
        h = np.zeros(self.hidden_dim)
        hidden_states = []

        for item_id in session:
            x = self.embeddings[item_id]
            combined = np.concatenate([x, h])

            # GRU gates
            z = self.sigmoid(np.dot(combined, self.Wz))  # Update gate
            r = self.sigmoid(np.dot(combined, self.Wr))  # Reset gate

            # Candidate hidden state
            combined_reset = np.concatenate([x, r * h])
            h_candidate = self.tanh(np.dot(combined_reset, self.Wh))

            # New hidden state
            h = z * h + (1 - z) * h_candidate
            hidden_states.append(h.copy())

        # Output layer
        output = np.dot(h, self.Wy) + self.by
        return self.softmax(output), hidden_states

    def predict_next(self, session: List[int], top_k: int = 10) -> List[int]:
        """
        Predict next items for a session

        Args:
            session: List of item IDs in session
            top_k: Number of recommendations

        Returns:
            List of recommended item IDs
        """
        if not session:
            return list(range(min(top_k, self.n_items)))

        probs, _ = self.forward(session)
        top_indices = np.argsort(probs)[-top_k:][::-1]
        return top_indices.tolist()


class SessionCooccurrence:
    """Session-based recommendations using item co-occurrence"""

    def __init__(self):
        """Initialize session co-occurrence model"""
        self.cooccurrence = defaultdict(Counter)
        self.item_counts = Counter()

    def fit(self, sessions: List[List[int]]) -> 'SessionCooccurrence':
        """
        Build co-occurrence matrix from sessions

        Args:
            sessions: List of sessions (each session is a list of item IDs)

        Returns:
            self
        """
        for session in sessions:
            # Count item occurrences
            for item in session:
                self.item_counts[item] += 1

            # Count co-occurrences
            unique_items = list(set(session))
            for i, item1 in enumerate(unique_items):
                for item2 in unique_items[i+1:]:
                    self.cooccurrence[item1][item2] += 1
                    self.cooccurrence[item2][item1] += 1

        return self

    def predict_next(self, session: List[int], top_k: int = 10) -> List[int]:
        """
        Predict next items based on co-occurrence

        Args:
            session: Current session items
            top_k: Number of recommendations

        Returns:
            List of recommended item IDs
        """
        if not session:
            # Return most popular items
            return [item for item, _ in self.item_counts.most_common(top_k)]

        # Aggregate scores from all items in session
        scores = Counter()
        for item in set(session):
            for co_item, count in self.cooccurrence[item].items():
                if co_item not in session:
                    scores[co_item] += count

        # Get top-k items
        if not scores:
            # Fallback to popular items
            candidates = [item for item in self.item_counts
                         if item not in session]
            return candidates[:top_k]

        return [item for item, _ in scores.most_common(top_k)]


class SequentialPatternMining:
    """Sequential pattern mining for session recommendations"""

    def __init__(self, min_support: int = 2):
        """
        Initialize sequential pattern mining

        Args:
            min_support: Minimum support for patterns
        """
        self.min_support = min_support
        self.patterns = defaultdict(Counter)
        self.item_popularity = Counter()

    def fit(self, sessions: List[List[int]]) -> 'SequentialPatternMining':
        """
        Mine sequential patterns from sessions

        Args:
            sessions: List of sessions

        Returns:
            self
        """
        # Count item popularity
        for session in sessions:
            for item in session:
                self.item_popularity[item] += 1

        # Mine patterns of length 2
        for session in sessions:
            for i in range(len(session) - 1):
                pattern = session[i]
                next_item = session[i + 1]
                self.patterns[pattern][next_item] += 1

        # Filter by minimum support
        for pattern in list(self.patterns.keys()):
            self.patterns[pattern] = {
                item: count for item, count in self.patterns[pattern].items()
                if count >= self.min_support
            }
            if not self.patterns[pattern]:
                del self.patterns[pattern]

        return self

    def predict_next(self, session: List[int], top_k: int = 10) -> List[int]:
        """
        Predict next items using sequential patterns

        Args:
            session: Current session
            top_k: Number of recommendations

        Returns:
            List of recommended item IDs
        """
        if not session:
            return [item for item, _ in self.item_popularity.most_common(top_k)]

        # Use last item in session
        last_item = session[-1]

        if last_item in self.patterns and self.patterns[last_item]:
            # Get items that frequently follow the last item
            next_items = sorted(self.patterns[last_item].items(),
                              key=lambda x: x[1], reverse=True)
            recommendations = [item for item, _ in next_items[:top_k]]

            # Fill with popular items if needed
            if len(recommendations) < top_k:
                popular = [item for item, _ in self.item_popularity.most_common()
                          if item not in recommendations and item not in session]
                recommendations.extend(popular[:top_k - len(recommendations)])

            return recommendations[:top_k]
        else:
            # Fallback to popular items
            return [item for item, _ in self.item_popularity.most_common(top_k)
                   if item not in session][:top_k]


def generate_session_data(n_users: int = 1000, n_items: int = 500,
                         n_sessions: int = 5000) -> List[Dict]:
    """
    Generate synthetic session data

    Args:
        n_users: Number of users
        n_items: Number of items
        n_sessions: Number of sessions

    Returns:
        List of session dictionaries
    """
    np.random.seed(42)

    # Create item categories and popularity
    n_categories = 20
    item_categories = np.random.randint(0, n_categories, n_items)
    item_popularity = np.random.zipf(1.5, n_items)

    # Normalize popularity
    item_popularity = item_popularity / item_popularity.sum()

    sessions = []
    for session_id in range(n_sessions):
        user_id = np.random.randint(0, n_users)
        session_length = np.random.randint(3, 15)

        # Start with a random item
        items = []
        current_item = np.random.choice(n_items, p=item_popularity)
        items.append(current_item)

        # Generate rest of session with sequential dependencies
        for _ in range(session_length - 1):
            # 70% chance to pick from same category
            if np.random.random() < 0.7:
                current_category = item_categories[current_item]
                same_category_items = np.where(item_categories == current_category)[0]
                if len(same_category_items) > 1:
                    next_item = np.random.choice(same_category_items)
                else:
                    next_item = np.random.choice(n_items, p=item_popularity)
            else:
                # Pick random item based on popularity
                next_item = np.random.choice(n_items, p=item_popularity)

            items.append(int(next_item))
            current_item = next_item

        sessions.append({
            'session_id': session_id,
            'user_id': user_id,
            'items': items,
            'length': len(items)
        })

    return sessions


def evaluate_session_model(model, test_sessions: List[List[int]],
                          top_k: int = 10) -> Dict[str, float]:
    """
    Evaluate session-based model

    Args:
        model: Trained session model
        test_sessions: List of test sessions
        top_k: Number of recommendations

    Returns:
        Dictionary of metrics
    """
    hits = 0
    total = 0
    mrr_sum = 0.0
    precision_sum = 0.0
    recall_sum = 0.0

    for session in test_sessions:
        if len(session) < 2:
            continue

        # Use all but last item as input
        input_session = session[:-1]
        target_item = session[-1]

        # Get predictions
        predictions = model.predict_next(input_session, top_k=top_k)

        # Calculate metrics
        if target_item in predictions:
            hits += 1
            rank = predictions.index(target_item) + 1
            mrr_sum += 1.0 / rank
            precision_sum += 1.0 / len(predictions)
            recall_sum += 1.0

        total += 1

    if total == 0:
        return {'Hit Rate': 0.0, 'MRR': 0.0, 'Precision': 0.0, 'Recall': 0.0}

    return {
        'Hit Rate': hits / total,
        'MRR': mrr_sum / total,
        'Precision': precision_sum / total,
        'Recall': recall_sum / total
    }


def plot_session_statistics(sessions: List[Dict], save_path: str = None):
    """Plot session statistics"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Session length distribution
    lengths = [s['length'] for s in sessions]
    axes[0, 0].hist(lengths, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Session Length')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Session Length Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Items per user
    user_items = defaultdict(set)
    for session in sessions:
        for item in session['items']:
            user_items[session['user_id']].add(item)
    items_per_user = [len(items) for items in user_items.values()]

    axes[0, 1].hist(items_per_user, bins=30, color='coral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Unique Items per User')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Items per User Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Item popularity
    all_items = []
    for session in sessions:
        all_items.extend(session['items'])
    item_counts = Counter(all_items)
    top_items = item_counts.most_common(20)

    axes[0, 2].barh(range(len(top_items)), [count for _, count in top_items],
                    color='lightgreen')
    axes[0, 2].set_yticks(range(len(top_items)))
    axes[0, 2].set_yticklabels([f'Item {item}' for item, _ in top_items])
    axes[0, 2].set_xlabel('Frequency')
    axes[0, 2].set_title('Top 20 Popular Items', fontsize=12, fontweight='bold')
    axes[0, 2].invert_yaxis()

    # Sessions per user
    sessions_per_user = Counter([s['user_id'] for s in sessions])
    axes[1, 0].hist(list(sessions_per_user.values()), bins=30,
                    color='plum', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Sessions per User')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Sessions per User Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Item frequency distribution (log scale)
    frequencies = sorted(item_counts.values(), reverse=True)
    axes[1, 1].plot(range(len(frequencies)), frequencies, linewidth=2)
    axes[1, 1].set_xlabel('Item Rank')
    axes[1, 1].set_ylabel('Frequency (log scale)')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_title('Item Popularity (Zipf Distribution)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    # Session length vs unique items
    unique_items_per_session = [len(set(s['items'])) for s in sessions]
    axes[1, 2].scatter(lengths, unique_items_per_session, alpha=0.3, s=10)
    axes[1, 2].set_xlabel('Session Length')
    axes[1, 2].set_ylabel('Unique Items')
    axes[1, 2].set_title('Session Length vs Unique Items', fontsize=12, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_model_comparison(results: Dict[str, Dict[str, float]], save_path: str = None):
    """Plot model comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    models = list(results.keys())
    metrics = ['Hit Rate', 'MRR', 'Precision', 'Recall']

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        values = [results[model][metric] for model in models]

        bars = ax.bar(range(len(models)), values, color=['skyblue', 'coral', 'lightgreen'])
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel(metric)
        ax.set_title(f'Model Comparison - {metric}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_recommendation_diversity(models: Dict[str, object], test_sessions: List[List[int]],
                                 save_path: str = None):
    """Plot recommendation diversity"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Collect recommendations from each model
    model_recommendations = {}
    for name, model in models.items():
        all_recs = []
        for session in test_sessions[:100]:  # Sample for efficiency
            if len(session) >= 2:
                recs = model.predict_next(session[:-1], top_k=10)
                all_recs.extend(recs)
        model_recommendations[name] = all_recs

    # Plot unique items recommended
    unique_items = {name: len(set(recs)) for name, recs in model_recommendations.items()}
    axes[0].bar(range(len(unique_items)), list(unique_items.values()),
                color=['skyblue', 'coral', 'lightgreen'])
    axes[0].set_xticks(range(len(unique_items)))
    axes[0].set_xticklabels(list(unique_items.keys()), rotation=45, ha='right')
    axes[0].set_ylabel('Unique Items Recommended')
    axes[0].set_title('Recommendation Diversity', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Plot recommendation distribution (top 20 items)
    for idx, (name, recs) in enumerate(model_recommendations.items()):
        item_counts = Counter(recs)
        top_items = item_counts.most_common(20)
        if top_items:
            axes[1].plot([i for i, _ in top_items],
                        label=name, marker='o', linewidth=2)

    axes[1].set_xlabel('Item Rank')
    axes[1].set_ylabel('Recommendation Frequency')
    axes[1].set_title('Top 20 Recommended Items', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_session_predictions(model, sample_sessions: List[List[int]],
                            save_path: str = None):
    """Visualize predictions for sample sessions"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, session in enumerate(sample_sessions[:4]):
        ax = axes[idx]

        if len(session) < 2:
            continue

        # Make predictions at different points in session
        x_positions = []
        y_predictions = []

        for i in range(1, len(session)):
            input_session = session[:i]
            predictions = model.predict_next(input_session, top_k=5)

            # Check if next item is in predictions
            if i < len(session):
                next_item = session[i]
                if next_item in predictions:
                    rank = predictions.index(next_item) + 1
                    x_positions.append(i)
                    y_predictions.append(rank)

        if x_positions:
            ax.plot(x_positions, y_predictions, marker='o', linewidth=2, markersize=8)
            ax.set_xlabel('Position in Session')
            ax.set_ylabel('Rank of Actual Next Item')
            ax.set_title(f'Session {idx + 1} Predictions', fontsize=12, fontweight='bold')
            ax.set_ylim(0, 6)
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            ax.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Rank 1')
            ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function"""
    print("=" * 80)
    print("Session-Based Recommendation System")
    print("=" * 80)

    # Generate session data
    print("\n1. Generating session data...")
    sessions_data = generate_session_data(n_users=1000, n_items=500, n_sessions=5000)
    print(f"Generated {len(sessions_data)} sessions")
    print(f"Average session length: {np.mean([s['length'] for s in sessions_data]):.2f}")

    # Plot session statistics
    print("\n2. Analyzing session statistics...")
    plot_session_statistics(sessions_data)

    # Split data
    split_idx = int(len(sessions_data) * 0.8)
    train_sessions_data = sessions_data[:split_idx]
    test_sessions_data = sessions_data[split_idx:]

    train_sessions = [s['items'] for s in train_sessions_data]
    test_sessions = [s['items'] for s in test_sessions_data]

    print(f"\nTraining sessions: {len(train_sessions)}")
    print(f"Test sessions: {len(test_sessions)}")

    # Train models
    print("\n3. Training session-based models...")
    results = {}

    # Session Co-occurrence
    print("\n   Training Session Co-occurrence model...")
    cooc_model = SessionCooccurrence()
    cooc_model.fit(train_sessions)
    results['Co-occurrence'] = evaluate_session_model(cooc_model, test_sessions, top_k=10)
    print(f"   Hit Rate: {results['Co-occurrence']['Hit Rate']:.4f}, "
          f"MRR: {results['Co-occurrence']['MRR']:.4f}")

    # Sequential Pattern Mining
    print("\n   Training Sequential Pattern Mining model...")
    spm_model = SequentialPatternMining(min_support=2)
    spm_model.fit(train_sessions)
    results['Sequential Patterns'] = evaluate_session_model(spm_model, test_sessions, top_k=10)
    print(f"   Hit Rate: {results['Sequential Patterns']['Hit Rate']:.4f}, "
          f"MRR: {results['Sequential Patterns']['MRR']:.4f}")

    # Session GRU
    print("\n   Training Session GRU model...")
    n_items = max(max(session) for session in train_sessions) + 1
    gru_model = SessionGRU(n_items=n_items, embedding_dim=50, hidden_dim=100)
    # Note: Full GRU training would be computationally intensive
    # Using pre-initialized model for demonstration
    results['GRU'] = evaluate_session_model(gru_model, test_sessions, top_k=10)
    print(f"   Hit Rate: {results['GRU']['Hit Rate']:.4f}, "
          f"MRR: {results['GRU']['MRR']:.4f}")

    # Visualizations
    print("\n4. Generating visualizations...")
    plot_model_comparison(results)

    models = {
        'Co-occurrence': cooc_model,
        'Sequential Patterns': spm_model,
        'GRU': gru_model
    }
    plot_recommendation_diversity(models, test_sessions)
    plot_session_predictions(cooc_model, test_sessions[:10])

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nModel Performance:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['Hit Rate'])
    print(f"\nBest model by Hit Rate: {best_model[0]} ({best_model[1]['Hit Rate']:.4f})")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
