"""
Toxic Comment Classification - Kaggle NLP Solution
==================================================
This solution demonstrates multi-label classification for detecting toxic
comments across multiple categories: toxic, severe_toxic, obscene,
threat, insult, and identity_hate.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, hamming_loss, accuracy_score,
                            f1_score, roc_auc_score, roc_curve, auc)
import warnings
import re

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

class ToxicCommentClassifier:
    """Multi-label toxic comment classification"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            strip_accents='unicode',
            lowercase=True
        )
        # One classifier per label (multi-label)
        self.classifier = OneVsRestClassifier(
            LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        )
        self.label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def fit(self, texts, labels):
        """Train the classifier"""
        # Preprocess
        processed_texts = [self.preprocess_text(text) for text in texts]

        # Vectorize
        X = self.vectorizer.fit_transform(processed_texts)

        # Train
        self.classifier.fit(X, labels)

    def predict(self, texts):
        """Predict toxicity labels"""
        processed_texts = [self.preprocess_text(text) for text in texts]
        X = self.vectorizer.transform(processed_texts)
        return self.classifier.predict(X)

    def predict_proba(self, texts):
        """Get prediction probabilities"""
        processed_texts = [self.preprocess_text(text) for text in texts]
        X = self.vectorizer.transform(processed_texts)
        return self.classifier.predict_proba(X)

def generate_toxic_comment_dataset():
    """Generate synthetic toxic comment dataset"""

    # Non-toxic comments
    clean_comments = [
        "This is a great article! Very informative.",
        "I really enjoyed reading this. Thank you for sharing.",
        "Interesting perspective on the topic.",
        "Well written and thoroughly researched.",
        "I appreciate your insights on this matter.",
        "This helped me understand the concept better.",
        "Great explanation! Very clear and concise.",
        "Thank you for taking the time to write this.",
        "I learned something new today from this post.",
        "Excellent work! Keep it up.",
        "This is exactly what I was looking for.",
        "Very helpful information. Much appreciated.",
        "I agree with your analysis here.",
        "This brings up some good points to consider.",
        "Looking forward to more content like this.",
        "Nice work on presenting both sides of the argument.",
        "This is a thoughtful and balanced view.",
        "Thanks for the detailed explanation.",
        "I found this very useful for my research.",
        "Great job summarizing the key points.",
    ]

    # Toxic comments (mild toxicity)
    toxic_comments = [
        "This is stupid and makes no sense at all.",
        "What a waste of time reading this garbage.",
        "You clearly don't know what you're talking about.",
        "This is the dumbest thing I've ever read.",
        "Stop spreading misinformation, you're clueless.",
        "Anyone who believes this is an idiot.",
        "This is complete nonsense and totally wrong.",
        "You have no idea what you're saying, do you?",
        "This article is trash and poorly written.",
        "What a ridiculous and absurd argument.",
    ]

    # Severely toxic and obscene
    severe_comments = [
        "You're absolutely pathetic and worthless.",
        "This is offensive garbage from a terrible person.",
        "Shut up, nobody wants to hear your stupid opinions.",
        "You're a disgrace and an embarrassment.",
        "This is the worst thing ever posted online.",
    ]

    # Threatening comments
    threat_comments = [
        "You better watch your back if you keep this up.",
        "I know where you live, be careful what you say.",
        "Someone should teach you a lesson you won't forget.",
        "You're going to regret posting this.",
        "I'm coming for you if you don't delete this.",
    ]

    # Insulting comments
    insult_comments = [
        "You're such a pathetic loser with no life.",
        "What a moron, can't believe people like you exist.",
        "You're an embarrassment to humanity, seriously.",
        "Only a complete fool would think this way.",
        "You're a disgrace and everyone knows it.",
    ]

    # Identity-based hate
    identity_hate_comments = [
        "People like you should not be allowed to speak.",
        "Your kind always ruins everything for everyone.",
        "Go back to where you came from, not welcome here.",
        "Your group is the problem with society today.",
        "We don't need your type around here at all.",
    ]

    # Create dataset
    data = []

    # Clean comments (all labels 0)
    for comment in clean_comments:
        data.append({
            'text': comment,
            'toxic': 0, 'severe_toxic': 0, 'obscene': 0,
            'threat': 0, 'insult': 0, 'identity_hate': 0
        })

    # Toxic comments
    for comment in toxic_comments:
        data.append({
            'text': comment,
            'toxic': 1, 'severe_toxic': 0, 'obscene': 0,
            'threat': 0, 'insult': 0, 'identity_hate': 0
        })

    # Severe toxic (toxic + severe + obscene)
    for comment in severe_comments:
        data.append({
            'text': comment,
            'toxic': 1, 'severe_toxic': 1, 'obscene': 1,
            'threat': 0, 'insult': 0, 'identity_hate': 0
        })

    # Threats (toxic + threat)
    for comment in threat_comments:
        data.append({
            'text': comment,
            'toxic': 1, 'severe_toxic': 0, 'obscene': 0,
            'threat': 1, 'insult': 0, 'identity_hate': 0
        })

    # Insults (toxic + insult)
    for comment in insult_comments:
        data.append({
            'text': comment,
            'toxic': 1, 'severe_toxic': 0, 'obscene': 0,
            'threat': 0, 'insult': 1, 'identity_hate': 0
        })

    # Identity hate (toxic + identity_hate + insult)
    for comment in identity_hate_comments:
        data.append({
            'text': comment,
            'toxic': 1, 'severe_toxic': 0, 'obscene': 0,
            'threat': 0, 'insult': 1, 'identity_hate': 1
        })

    df = pd.DataFrame(data)
    return df

def create_visualizations(df, y_true, y_pred, y_proba, label_names):
    """Create visualizations for toxic comment classification"""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Label distribution
    ax1 = fig.add_subplot(gs[0, 0])
    label_counts = df[label_names].sum()
    colors = plt.cm.Set3(range(len(label_names)))

    ax1.barh(range(len(label_names)), label_counts, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_yticks(range(len(label_names)))
    ax1.set_yticklabels(label_names)
    ax1.set_xlabel('Count')
    ax1.set_title('Label Distribution in Dataset', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)

    # 2. F1 scores per label
    ax2 = fig.add_subplot(gs[0, 1])
    f1_scores = []
    for i, label in enumerate(label_names):
        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        f1_scores.append(f1)

    bars = ax2.bar(range(len(label_names)), f1_scores, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(label_names)))
    ax2.set_xticklabels(label_names, rotation=45, ha='right')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score by Label', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    # 3. Label co-occurrence matrix
    ax3 = fig.add_subplot(gs[0, 2])
    cooccurrence = np.zeros((len(label_names), len(label_names)))

    for i in range(len(label_names)):
        for j in range(len(label_names)):
            cooccurrence[i, j] = ((df[label_names[i]] == 1) &
                                 (df[label_names[j]] == 1)).sum()

    sns.heatmap(cooccurrence, annot=True, fmt='.0f', cmap='YlOrRd',
                xticklabels=label_names, yticklabels=label_names,
                cbar_kws={'label': 'Co-occurrence'}, ax=ax3)
    ax3.set_title('Label Co-occurrence Matrix', fontsize=12, fontweight='bold')

    # 4. ROC curves for each label
    ax4 = fig.add_subplot(gs[1, :])
    for i, label in enumerate(label_names):
        if len(np.unique(y_true[:, i])) > 1:  # Only if both classes present
            fpr, tpr, _ = roc_curve(y_true[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax4.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.3f})',
                    color=colors[i], linewidth=2)

    ax4.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title('ROC Curves for All Labels', fontsize=12, fontweight='bold')
    ax4.legend(loc='lower right')
    ax4.grid(alpha=0.3)

    # 5. Precision and Recall per label
    ax5 = fig.add_subplot(gs[2, 0])
    from sklearn.metrics import precision_score, recall_score

    precisions = []
    recalls = []

    for i, label in enumerate(label_names):
        prec = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        rec = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        precisions.append(prec)
        recalls.append(rec)

    x = np.arange(len(label_names))
    width = 0.35

    ax5.bar(x - width/2, precisions, width, label='Precision', alpha=0.7, color='skyblue')
    ax5.bar(x + width/2, recalls, width, label='Recall', alpha=0.7, color='lightcoral')

    ax5.set_xlabel('Label')
    ax5.set_ylabel('Score')
    ax5.set_title('Precision and Recall by Label', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(label_names, rotation=45, ha='right')
    ax5.legend()
    ax5.set_ylim([0, 1])
    ax5.grid(axis='y', alpha=0.3)

    # 6. Prediction confidence distribution
    ax6 = fig.add_subplot(gs[2, 1])
    max_probas = y_proba.max(axis=1)

    ax6.hist(max_probas, bins=20, color='green', alpha=0.7, edgecolor='black')
    ax6.set_xlabel('Maximum Prediction Probability')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Prediction Confidence Distribution', fontsize=12, fontweight='bold')
    ax6.axvline(max_probas.mean(), color='red', linestyle='--',
                label=f'Mean: {max_probas.mean():.3f}')
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3)

    # 7. Multi-label statistics
    ax7 = fig.add_subplot(gs[2, 2])
    labels_per_comment = y_true.sum(axis=1)
    label_count_dist = pd.Series(labels_per_comment).value_counts().sort_index()

    ax7.bar(label_count_dist.index, label_count_dist.values,
            color='purple', alpha=0.7, edgecolor='black')
    ax7.set_xlabel('Number of Labels per Comment')
    ax7.set_ylabel('Frequency')
    ax7.set_title('Multi-Label Distribution', fontsize=12, fontweight='bold')
    ax7.grid(axis='y', alpha=0.3)

    plt.savefig('toxic_comment_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'toxic_comment_analysis.png'")
    plt.close()

def main():
    """Main execution function"""
    print("=" * 60)
    print("Toxic Comment Classification - Kaggle NLP Solution")
    print("=" * 60)

    # Generate dataset
    print("\n1. Generating Toxic Comment Dataset...")
    df = generate_toxic_comment_dataset()

    label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    print(f"   - Total comments: {len(df)}")
    print(f"   - Label types: {len(label_names)}")
    print(f"   - Toxic comments: {df['toxic'].sum()}")
    print(f"   - Clean comments: {(df['toxic'] == 0).sum()}")

    # Display label distribution
    print("\n   Label distribution:")
    for label in label_names:
        count = df[label].sum()
        pct = count / len(df) * 100
        print(f"      {label}: {count} ({pct:.1f}%)")

    # Split dataset
    print("\n2. Splitting Dataset...")
    X = df['text'].values
    y = df[label_names].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    print(f"   - Training samples: {len(X_train)}")
    print(f"   - Test samples: {len(X_test)}")

    # Train classifier
    print("\n3. Training Toxic Comment Classifier...")
    classifier = ToxicCommentClassifier()
    classifier.fit(X_train, y_train)

    print(f"   - Feature dimension: {len(classifier.vectorizer.get_feature_names_out())}")
    print(f"   - N-gram range: 1-2")

    # Make predictions
    print("\n4. Evaluating Model...")
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)

    # Calculate metrics
    hamming = hamming_loss(y_test, y_pred)
    subset_acc = accuracy_score(y_test, y_pred)

    print(f"   - Hamming Loss: {hamming:.3f}")
    print(f"   - Subset Accuracy: {subset_acc:.3f}")

    # Per-label metrics
    print("\n5. Per-Label Performance:")
    print("-" * 60)

    for i, label in enumerate(label_names):
        f1 = f1_score(y_test[:, i], y_pred[:, i], zero_division=0)
        if len(np.unique(y_test[:, i])) > 1:
            auc_score = roc_auc_score(y_test[:, i], y_proba[:, i])
            print(f"   {label:15} - F1: {f1:.3f}, AUC: {auc_score:.3f}")
        else:
            print(f"   {label:15} - F1: {f1:.3f}, AUC: N/A")

    # Create visualizations
    print("\n6. Creating Visualizations...")
    create_visualizations(df, y_test, y_pred, y_proba, label_names)

    # Interactive demo
    print("\n7. Interactive Demo:")
    print("-" * 60)

    test_comments = [
        "This is a wonderful article, thank you for sharing!",
        "This is stupid and you're an idiot for posting it.",
        "I completely disagree with your viewpoint here.",
        "You should be ashamed of yourself, pathetic loser.",
        "I'm going to find you and make you pay for this.",
        "People like you are ruining everything for everyone.",
    ]

    for comment in test_comments:
        pred = classifier.predict([comment])[0]
        proba = classifier.predict_proba([comment])[0]

        print(f"\nComment: '{comment}'")
        print("Predictions:")

        detected_labels = []
        for i, label in enumerate(label_names):
            if pred[i] == 1:
                detected_labels.append(f"{label} ({proba[i]:.2%})")

        if detected_labels:
            print(f"   {', '.join(detected_labels)}")
        else:
            print("   Clean (non-toxic)")

    # Summary statistics
    print("\n8. Summary Statistics:")
    print("-" * 60)
    print(f"   - Overall Hamming Loss: {hamming:.3f}")
    print(f"   - Subset Accuracy: {subset_acc:.3f}")
    print(f"   - Average F1 Score: {np.mean([f1_score(y_test[:, i], y_pred[:, i], zero_division=0) for i in range(len(label_names))]):.3f}")
    print(f"   - Comments with multiple labels: {(y_test.sum(axis=1) > 1).sum()}/{len(y_test)}")

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
