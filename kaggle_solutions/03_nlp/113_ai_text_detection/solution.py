"""
AI Generated Text Detection - NLP Classification

Detect whether text was written by AI (LLM) or humans using
various text features and machine learning models.

Dataset: https://www.kaggle.com/competitions/llm-detect-ai-generated-text
Difficulty: ⭐⭐⭐ Advanced Level
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import warnings
import re
from collections import Counter
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class AITextDetector:
    """AI Generated Text Detection Model."""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
        self.scaler = StandardScaler()
        self.best_model = None

    def create_sample_data(self) -> pd.DataFrame:
        """Create synthetic AI vs Human text dataset."""
        np.random.seed(42)

        # Human-written text patterns (more varied, informal)
        human_templates = [
            "I think this is really interesting because {reason}. What do you think?",
            "So I was wondering about {topic} and honestly, it's quite complex...",
            "The thing is, {point} - at least that's my take on it.",
            "Hmm, I'm not sure about {topic}, but maybe {suggestion}?",
            "You know what? {observation} - pretty cool if you ask me!",
            "Let me tell you about {topic}. So basically, {explanation}",
            "Ok so here's the thing about {topic} - it's complicated...",
            "I've been thinking a lot about {topic} lately and {thought}",
        ]

        # AI-written text patterns (more formal, structured)
        ai_templates = [
            "The concept of {topic} is a multifaceted subject that encompasses various aspects. {elaboration}",
            "In examining {topic}, it is essential to consider the following key factors. {points}",
            "The significance of {topic} cannot be overstated. This phenomenon {description}",
            "When analyzing {topic}, several important considerations emerge. {analysis}",
            "The field of {topic} has undergone significant developments. {details}",
            "It is important to note that {topic} plays a crucial role in {context}.",
            "The implications of {topic} are far-reaching and multifaceted. {implications}",
            "A comprehensive understanding of {topic} requires examination of {aspects}.",
        ]

        topics = ['technology', 'climate change', 'education', 'healthcare', 'economics',
                 'artificial intelligence', 'social media', 'renewable energy']
        reasons = ['it affects everyone', 'we need to understand it better',
                  'the data shows clear patterns', 'experts have been discussing this']

        texts = []
        labels = []

        # Generate human texts
        for _ in range(500):
            template = np.random.choice(human_templates)
            topic = np.random.choice(topics)
            reason = np.random.choice(reasons)

            text = template.format(
                topic=topic, reason=reason, point=f"understanding {topic} matters",
                suggestion=f"we could explore {topic} more", observation=f"{topic} is evolving",
                explanation=f"it involves many aspects", thought=f"there's more to learn"
            )
            # Add human-like variations
            if np.random.random() > 0.7:
                text = text.replace('.', '...')
            if np.random.random() > 0.8:
                text = text + " lol" if np.random.random() > 0.5 else text + "!"

            texts.append(text)
            labels.append(0)  # Human

        # Generate AI texts
        for _ in range(500):
            template = np.random.choice(ai_templates)
            topic = np.random.choice(topics)

            text = template.format(
                topic=topic, elaboration=f"Furthermore, the implications extend to multiple domains.",
                points=f"First, the fundamental principles. Second, the practical applications.",
                description=f"demonstrates remarkable characteristics in modern contexts.",
                analysis=f"The data suggests a correlation between multiple variables.",
                details=f"Recent advancements have transformed our understanding.",
                context=f"contemporary discourse and practical applications.",
                implications=f"These findings suggest important considerations for future research.",
                aspects=f"both theoretical frameworks and empirical evidence."
            )

            texts.append(text)
            labels.append(1)  # AI

        return pd.DataFrame({'text': texts, 'label': labels})

    def extract_text_features(self, texts: pd.Series) -> pd.DataFrame:
        """Extract statistical features from text."""
        features = []

        for text in texts:
            # Basic stats
            words = text.split()
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]

            # Word-level features
            word_lengths = [len(w) for w in words]
            avg_word_length = np.mean(word_lengths) if word_lengths else 0
            word_length_std = np.std(word_lengths) if len(word_lengths) > 1 else 0

            # Sentence-level features
            sent_lengths = [len(s.split()) for s in sentences]
            avg_sent_length = np.mean(sent_lengths) if sent_lengths else 0
            sent_length_std = np.std(sent_lengths) if len(sent_lengths) > 1 else 0

            # Vocabulary richness
            unique_words = len(set(w.lower() for w in words))
            vocab_richness = unique_words / len(words) if words else 0

            # Punctuation features
            punct_count = sum(1 for c in text if c in '.,!?;:')
            punct_ratio = punct_count / len(text) if text else 0

            # Special patterns
            formal_words = ['furthermore', 'moreover', 'consequently', 'therefore',
                          'significant', 'essential', 'comprehensive', 'multifaceted']
            formal_count = sum(1 for w in words if w.lower() in formal_words)

            informal_markers = ['lol', 'hmm', 'ok', 'so', 'basically', 'honestly', '...']
            informal_count = sum(1 for w in words if w.lower() in informal_markers)
            informal_count += text.count('...')

            # Repetition
            word_freq = Counter(w.lower() for w in words)
            if word_freq:
                max_freq = max(word_freq.values())
                repetition_ratio = max_freq / len(words)
            else:
                repetition_ratio = 0

            features.append({
                'word_count': len(words),
                'sentence_count': len(sentences),
                'avg_word_length': avg_word_length,
                'word_length_std': word_length_std,
                'avg_sent_length': avg_sent_length,
                'sent_length_std': sent_length_std,
                'vocab_richness': vocab_richness,
                'punct_ratio': punct_ratio,
                'formal_word_count': formal_count,
                'informal_marker_count': informal_count,
                'repetition_ratio': repetition_ratio,
                'char_count': len(text),
                'question_count': text.count('?'),
                'exclamation_count': text.count('!')
            })

        return pd.DataFrame(features)

    def plot_analysis(self, df: pd.DataFrame, output_dir: str = '.') -> None:
        """Generate EDA visualizations."""
        features_df = self.extract_text_features(df['text'])
        features_df['label'] = df['label'].values

        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('AI vs Human Text Analysis', fontsize=16)

        # Label distribution
        df['label'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['green', 'red'])
        axes[0, 0].set_title('Text Distribution')
        axes[0, 0].set_xticklabels(['Human', 'AI'], rotation=0)

        # Word count
        features_df.boxplot(column='word_count', by='label', ax=axes[0, 1])
        axes[0, 1].set_title('Word Count by Label')
        axes[0, 1].set_xticklabels(['Human', 'AI'])
        plt.suptitle('')

        # Vocabulary richness
        features_df.boxplot(column='vocab_richness', by='label', ax=axes[0, 2])
        axes[0, 2].set_title('Vocabulary Richness by Label')
        plt.suptitle('')

        # Formal words
        features_df.boxplot(column='formal_word_count', by='label', ax=axes[1, 0])
        axes[1, 0].set_title('Formal Word Count by Label')
        plt.suptitle('')

        # Informal markers
        features_df.boxplot(column='informal_marker_count', by='label', ax=axes[1, 1])
        axes[1, 1].set_title('Informal Markers by Label')
        plt.suptitle('')

        # Average sentence length
        features_df.boxplot(column='avg_sent_length', by='label', ax=axes[1, 2])
        axes[1, 2].set_title('Avg Sentence Length by Label')
        plt.suptitle('')

        # Sentence length variability
        features_df.boxplot(column='sent_length_std', by='label', ax=axes[2, 0])
        axes[2, 0].set_title('Sentence Length Variability')
        plt.suptitle('')

        # Feature correlation
        corr_cols = ['vocab_richness', 'formal_word_count', 'informal_marker_count',
                    'avg_sent_length', 'sent_length_std', 'label']
        sns.heatmap(features_df[corr_cols].corr(), annot=True, fmt='.2f',
                   cmap='coolwarm', ax=axes[2, 1])
        axes[2, 1].set_title('Feature Correlations')

        # Word length distribution
        for label, color, name in [(0, 'green', 'Human'), (1, 'red', 'AI')]:
            subset = features_df[features_df['label'] == label]['avg_word_length']
            axes[2, 2].hist(subset, bins=20, alpha=0.5, color=color, label=name)
        axes[2, 2].set_title('Average Word Length Distribution')
        axes[2, 2].legend()

        plt.tight_layout()
        plt.savefig(f'{output_dir}/ai_text_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Analysis saved to {output_dir}/ai_text_analysis.png")
        plt.close()

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare combined features."""
        # TF-IDF features
        tfidf_features = self.tfidf.fit_transform(df['text']).toarray()

        # Statistical features
        stat_features = self.extract_text_features(df['text']).values
        stat_features = self.scaler.fit_transform(stat_features)

        # Combine
        return np.hstack([tfidf_features, stat_features])

    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train classification models."""
        print("\nTraining models...")

        self.models['Logistic Regression'] = LogisticRegression(max_iter=1000, random_state=42)
        self.models['Logistic Regression'].fit(X_train, y_train)

        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
        )
        self.models['Random Forest'].fit(X_train, y_train)

        self.models['Gradient Boosting'] = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
        )
        self.models['Gradient Boosting'].fit(X_train, y_train)

        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42, use_label_encoder=False, eval_metric='logloss'
            )
            self.models['XGBoost'].fit(X_train, y_train)

        print(f"Trained {len(self.models)} models!")

    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """Evaluate all models."""
        results = []

        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            results.append({
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred),
                'AUC-ROC': roc_auc_score(y_test, y_pred_proba)
            })

        results_df = pd.DataFrame(results).sort_values('AUC-ROC', ascending=False)
        self.best_model = self.models[results_df.iloc[0]['Model']]
        return results_df

    def plot_results(self, results_df: pd.DataFrame, X_test: np.ndarray,
                    y_test: np.ndarray, output_dir: str = '.') -> None:
        """Visualize results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Model comparison
        results_df.set_index('Model')[['Accuracy', 'F1-Score', 'AUC-ROC']].plot(
            kind='bar', ax=axes[0, 0]
        )
        axes[0, 0].set_title('Model Performance')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # ROC curves
        for name, model in self.models.items():
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            axes[0, 1].plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(y_test, y_proba):.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--')
        axes[0, 1].set_title('ROC Curves')
        axes[0, 1].legend(loc='lower right')

        # Confusion matrix
        y_pred = self.best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                   xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
        axes[1, 0].set_title('Confusion Matrix')

        # Score distribution
        y_proba = self.best_model.predict_proba(X_test)[:, 1]
        axes[1, 1].hist(y_proba[y_test == 0], bins=30, alpha=0.5, label='Human', color='green')
        axes[1, 1].hist(y_proba[y_test == 1], bins=30, alpha=0.5, label='AI', color='red')
        axes[1, 1].set_title('Prediction Score Distribution')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(f'{output_dir}/detection_results.png', dpi=300, bbox_inches='tight')
        print(f"Results saved to {output_dir}/detection_results.png")
        plt.close()


def main():
    """Main execution."""
    print("=" * 70)
    print("AI GENERATED TEXT DETECTION")
    print("=" * 70)

    detector = AITextDetector()

    # Create data
    df = detector.create_sample_data()
    print(f"\nDataset: {df.shape}")
    print(f"Human: {(df['label'] == 0).sum()}, AI: {(df['label'] == 1).sum()}")

    # Analysis
    detector.plot_analysis(df)

    # Prepare features
    X = detector.prepare_features(df)
    y = df['label'].values
    print(f"Feature matrix: {X.shape}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train and evaluate
    detector.train_models(X_train, y_train)
    results = detector.evaluate_models(X_test, y_test)

    print(f"\n{results.to_string(index=False)}")

    detector.plot_results(results, X_test, y_test)

    print("\n" + "=" * 70)
    print(f"Best Model: {results.iloc[0]['Model']}")
    print(f"Best AUC-ROC: {results.iloc[0]['AUC-ROC']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
