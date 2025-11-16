"""
Chatbot Intent Classification - Kaggle NLP Solution
===================================================
This solution demonstrates intent classification for chatbots using
TF-IDF features and various classifiers to understand user intentions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
import re
from collections import Counter

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

class IntentClassifier:
    """Intent classification for chatbot conversations"""

    def __init__(self, classifier_type='logistic'):
        """
        Args:
            classifier_type: 'logistic', 'naive_bayes', 'svm', or 'random_forest'
        """
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            lowercase=True,
            stop_words='english'
        )

        # Select classifier
        if classifier_type == 'logistic':
            self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        elif classifier_type == 'naive_bayes':
            self.classifier = MultinomialNB(alpha=0.1)
        elif classifier_type == 'svm':
            self.classifier = SVC(kernel='linear', probability=True, random_state=42)
        else:  # random_forest
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)

        self.classifier_type = classifier_type
        self.intents = []

    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters except question marks and exclamation
        text = re.sub(r'[^\w\s\?\!]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def fit(self, texts, intents):
        """Train the intent classifier"""
        self.intents = sorted(list(set(intents)))

        # Preprocess
        processed_texts = [self.preprocess_text(text) for text in texts]

        # Vectorize
        X = self.vectorizer.fit_transform(processed_texts)

        # Train
        self.classifier.fit(X, intents)

    def predict(self, texts):
        """Predict intents for texts"""
        processed_texts = [self.preprocess_text(text) for text in texts]
        X = self.vectorizer.transform(processed_texts)
        return self.classifier.predict(X)

    def predict_proba(self, texts):
        """Get prediction probabilities"""
        processed_texts = [self.preprocess_text(text) for text in texts]
        X = self.vectorizer.transform(processed_texts)
        return self.classifier.predict_proba(X)

    def get_top_features(self, intent, n=10):
        """Get top features for an intent"""
        if self.classifier_type == 'logistic':
            feature_names = self.vectorizer.get_feature_names_out()
            intent_idx = self.intents.index(intent)
            coef = self.classifier.coef_[intent_idx]
            top_indices = np.argsort(coef)[-n:][::-1]
            return [(feature_names[i], coef[i]) for i in top_indices]
        return []

def generate_intent_dataset():
    """Generate synthetic chatbot intent dataset"""

    data = {
        'greeting': [
            "Hello there!",
            "Hi, how are you?",
            "Hey!",
            "Good morning!",
            "Good evening!",
            "Hi there, nice to meet you",
            "Hello, what's up?",
            "Hey, how's it going?",
            "Greetings!",
            "Hi, good to see you",
        ],
        'goodbye': [
            "Goodbye!",
            "See you later!",
            "Bye!",
            "Take care!",
            "Have a nice day!",
            "See you soon!",
            "Catch you later!",
            "Farewell!",
            "Until next time!",
            "Talk to you later!",
        ],
        'booking': [
            "I want to book a table for 4 people",
            "Can I reserve a room for tomorrow?",
            "I'd like to make a reservation",
            "Book a flight to New York",
            "Reserve a table for tonight",
            "I need to book an appointment",
            "Can you schedule a meeting for me?",
            "Make a reservation for Friday evening",
            "Book a hotel room please",
            "I want to reserve seats for the show",
        ],
        'cancel': [
            "I need to cancel my reservation",
            "Cancel my booking please",
            "I want to cancel my appointment",
            "Can you cancel my order?",
            "Please cancel my subscription",
            "I'd like to cancel the meeting",
            "Cancel my flight booking",
            "I need to cancel that",
            "Can you help me cancel my reservation?",
            "Please remove my booking",
        ],
        'price_inquiry': [
            "How much does it cost?",
            "What's the price?",
            "How much is it?",
            "Can you tell me the cost?",
            "What are your rates?",
            "How expensive is this?",
            "What's the pricing?",
            "Do you have a price list?",
            "How much do I need to pay?",
            "What does this cost?",
        ],
        'product_inquiry': [
            "Do you have this in stock?",
            "Tell me about this product",
            "What are the features?",
            "Is this available?",
            "Can you describe this item?",
            "What colors does this come in?",
            "Do you have other sizes?",
            "Tell me more about this",
            "What's included with this?",
            "Is this product good?",
        ],
        'complaint': [
            "I'm not happy with the service",
            "This is unacceptable",
            "I have a complaint",
            "This product is defective",
            "I want to speak to a manager",
            "This is terrible quality",
            "I'm very disappointed",
            "This doesn't work properly",
            "I demand a refund",
            "The service was awful",
        ],
        'thanks': [
            "Thank you so much!",
            "Thanks a lot!",
            "I appreciate your help",
            "Thank you for your assistance",
            "Thanks!",
            "That's very helpful, thank you",
            "I'm grateful for your help",
            "Thanks for everything",
            "Much appreciated!",
            "Thank you very much!",
        ],
        'help': [
            "I need help",
            "Can you help me?",
            "I'm having trouble",
            "I don't understand",
            "Can you assist me?",
            "I need assistance",
            "Help me please",
            "I'm confused",
            "Can you explain this?",
            "I need support",
        ],
        'hours': [
            "What are your opening hours?",
            "When are you open?",
            "What time do you close?",
            "Are you open on weekends?",
            "What's your schedule?",
            "When can I visit?",
            "What are your business hours?",
            "Are you open today?",
            "What time do you open?",
            "Do you work on holidays?",
        ],
    }

    # Create dataset
    texts = []
    intents = []

    for intent, sentences in data.items():
        texts.extend(sentences)
        intents.extend([intent] * len(sentences))

    return texts, intents

def create_visualizations(y_true, y_pred, y_proba, intents, conf_matrix, classifier_comparison):
    """Create visualizations for intent classification"""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

    # 1. Confusion Matrix
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=intents, yticklabels=intents,
                cbar_kws={'label': 'Count'}, ax=ax1)
    ax1.set_xlabel('Predicted Intent')
    ax1.set_ylabel('True Intent')
    ax1.set_title('Intent Classification Confusion Matrix', fontsize=12, fontweight='bold')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax1.get_yticklabels(), rotation=0)

    # 2. Intent distribution
    ax2 = fig.add_subplot(gs[0, 2])
    intent_counts = pd.Series(y_true).value_counts()
    colors = plt.cm.Set3(range(len(intent_counts)))

    ax2.pie(intent_counts.values, labels=intent_counts.index,
            autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Intent Distribution', fontsize=10, fontweight='bold')

    # 3. Accuracy by intent
    ax3 = fig.add_subplot(gs[1, 2])
    accuracies = []
    for intent in intents:
        mask = np.array(y_true) == intent
        if mask.sum() > 0:
            acc = accuracy_score(np.array(y_true)[mask], np.array(y_pred)[mask])
            accuracies.append(acc)
        else:
            accuracies.append(0)

    y_pos = np.arange(len(intents))
    bars = ax3.barh(y_pos, accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(intents, fontsize=8)
    ax3.set_xlabel('Accuracy')
    ax3.set_title('Per-Intent Accuracy', fontsize=10, fontweight='bold')
    ax3.set_xlim([0, 1])
    ax3.invert_yaxis()
    ax3.grid(axis='x', alpha=0.3)

    # 4. Classifier comparison
    ax4 = fig.add_subplot(gs[2, 0])
    classifiers = list(classifier_comparison.keys())
    scores = [classifier_comparison[c]['accuracy'] for c in classifiers]
    colors_comp = plt.cm.Set2(range(len(classifiers)))

    bars = ax4.bar(range(len(classifiers)), scores, color=colors_comp,
                   alpha=0.7, edgecolor='black')
    ax4.set_xticks(range(len(classifiers)))
    ax4.set_xticklabels(classifiers, rotation=45, ha='right')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Classifier Comparison', fontsize=12, fontweight='bold')
    ax4.set_ylim([0, 1])
    ax4.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # 5. Prediction confidence distribution
    ax5 = fig.add_subplot(gs[2, 1])
    max_probas = y_proba.max(axis=1)

    ax5.hist(max_probas, bins=20, color='green', alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Maximum Prediction Probability')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Prediction Confidence', fontsize=12, fontweight='bold')
    ax5.axvline(max_probas.mean(), color='red', linestyle='--',
                label=f'Mean: {max_probas.mean():.3f}')
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)

    # 6. F1 scores by intent
    ax6 = fig.add_subplot(gs[2, 2])
    from sklearn.metrics import f1_score

    f1_scores = []
    for intent in intents:
        y_true_binary = [1 if i == intent else 0 for i in y_true]
        y_pred_binary = [1 if i == intent else 0 for i in y_pred]
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        f1_scores.append(f1)

    y_pos = np.arange(len(intents))
    ax6.barh(y_pos, f1_scores, color=colors, alpha=0.7, edgecolor='black')
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels(intents, fontsize=8)
    ax6.set_xlabel('F1 Score')
    ax6.set_title('F1 Score by Intent', fontsize=10, fontweight='bold')
    ax6.set_xlim([0, 1])
    ax6.invert_yaxis()
    ax6.grid(axis='x', alpha=0.3)

    plt.savefig('chatbot_intent_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'chatbot_intent_analysis.png'")
    plt.close()

def main():
    """Main execution function"""
    print("=" * 60)
    print("Chatbot Intent Classification - Kaggle NLP Solution")
    print("=" * 60)

    # Generate dataset
    print("\n1. Generating Intent Dataset...")
    texts, intents_list = generate_intent_dataset()
    unique_intents = sorted(list(set(intents_list)))

    print(f"   - Total samples: {len(texts)}")
    print(f"   - Number of intents: {len(unique_intents)}")
    print(f"   - Intents: {', '.join(unique_intents)}")
    print(f"   - Samples per intent: {len(texts) // len(unique_intents)}")

    # Split dataset
    print("\n2. Splitting Dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, intents_list, test_size=0.25, random_state=42, stratify=intents_list
    )

    print(f"   - Training samples: {len(X_train)}")
    print(f"   - Test samples: {len(X_test)}")

    # Train classifier
    print("\n3. Training Intent Classifier (Logistic Regression)...")
    classifier = IntentClassifier(classifier_type='logistic')
    classifier.fit(X_train, y_train)

    print(f"   - Feature dimension: {len(classifier.vectorizer.get_feature_names_out())}")
    print(f"   - N-gram range: 1-2")

    # Make predictions
    print("\n4. Evaluating Model...")
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"   - Test Accuracy: {accuracy:.2%}")

    # Classification report
    print("\n5. Classification Report:")
    print("-" * 60)
    report = classification_report(y_test, y_pred, target_names=unique_intents)
    print(report)

    # Cross-validation
    print("\n6. Cross-Validation Scores:")
    print("-" * 60)
    cv_scores = cross_val_score(
        classifier.classifier,
        classifier.vectorizer.transform([classifier.preprocess_text(t) for t in X_train]),
        y_train,
        cv=5
    )
    print(f"   - CV Scores: {cv_scores}")
    print(f"   - Mean CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std() * 2:.2%})")

    # Top features per intent
    print("\n7. Top Features (Keywords) per Intent:")
    print("-" * 60)

    for intent in unique_intents[:5]:  # Show first 5
        top_features = classifier.get_top_features(intent, n=5)
        if top_features:
            keywords = ', '.join([f[0] for f in top_features])
            print(f"   {intent:15}: {keywords}")

    # Compare classifiers
    print("\n8. Comparing Different Classifiers...")
    classifier_types = ['logistic', 'naive_bayes', 'svm', 'random_forest']
    classifier_comparison = {}

    for clf_type in classifier_types:
        clf = IntentClassifier(classifier_type=clf_type)
        clf.fit(X_train, y_train)
        y_pred_clf = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred_clf)
        classifier_comparison[clf_type] = {'accuracy': acc}
        print(f"   {clf_type:15}: {acc:.2%}")

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=unique_intents)

    # Create visualizations
    print("\n9. Creating Visualizations...")
    create_visualizations(y_test, y_pred, y_proba, unique_intents,
                         conf_matrix, classifier_comparison)

    # Interactive demo
    print("\n10. Interactive Intent Detection Demo:")
    print("-" * 60)

    test_queries = [
        "Hello, how can I help you?",
        "I want to book a table for two",
        "How much does this cost?",
        "Can you cancel my reservation?",
        "Thank you for your help!",
        "I need assistance with my order",
        "What time are you open?",
        "This product is broken, I'm very unhappy",
        "Do you have this in blue?",
        "Goodbye, have a great day!",
    ]

    for query in test_queries:
        pred_intent = classifier.predict([query])[0]
        proba = classifier.predict_proba([query])[0]
        confidence = proba.max()

        print(f"\nQuery: '{query}'")
        print(f"Intent: {pred_intent} (confidence: {confidence:.2%})")

        # Show top 3 predictions
        top_3_idx = np.argsort(proba)[-3:][::-1]
        print("Top 3 predictions:")
        for idx in top_3_idx:
            print(f"   {classifier.intents[idx]}: {proba[idx]:.2%}")

    # Performance summary
    print("\n11. Performance Summary:")
    print("-" * 60)
    print(f"   - Best classifier: {max(classifier_comparison.items(), key=lambda x: x[1]['accuracy'])[0]}")
    print(f"   - Best accuracy: {max([v['accuracy'] for v in classifier_comparison.values()]):.2%}")
    print(f"   - Average accuracy: {np.mean([v['accuracy'] for v in classifier_comparison.values()]):.2%}")
    print(f"   - Cross-validation score: {cv_scores.mean():.2%}")

    # Misclassification analysis
    print("\n12. Misclassification Analysis:")
    print("-" * 60)

    misclassified = [(X_test[i], y_test[i], y_pred[i])
                     for i in range(len(X_test))
                     if y_test[i] != y_pred[i]]

    if misclassified:
        print(f"   - Total misclassifications: {len(misclassified)}")
        print("\n   Sample misclassifications:")
        for text, true_intent, pred_intent in misclassified[:3]:
            print(f"      Text: '{text}'")
            print(f"      True: {true_intent}, Predicted: {pred_intent}")
    else:
        print("   - No misclassifications! Perfect accuracy!")

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
