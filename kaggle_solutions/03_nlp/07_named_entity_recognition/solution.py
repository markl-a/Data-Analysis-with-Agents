"""
Named Entity Recognition (NER) - Kaggle NLP Solution
====================================================
This solution demonstrates NER for extracting entities (Person, Organization,
Location, Date) from text using both rule-based and ML approaches.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import nltk
from nltk import pos_tag, word_tokenize
from nltk.chunk import ne_chunk
import warnings
import re
from collections import Counter, defaultdict

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# Download required NLTK data
for resource in ['punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        try:
            nltk.download(resource, quiet=True)
        except:
            pass

class NERExtractor:
    """Named Entity Recognition using feature-based approach"""

    def __init__(self):
        self.vectorizer = DictVectorizer(sparse=True)
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.entity_types = ['B-PER', 'I-PER', 'B-ORG', 'I-ORG',
                            'B-LOC', 'I-LOC', 'B-DATE', 'I-DATE', 'O']

    def word_features(self, sentence, idx):
        """Extract features for a word at position idx"""
        word = sentence[idx]
        features = {
            'word': word.lower(),
            'is_first': idx == 0,
            'is_last': idx == len(sentence) - 1,
            'is_capitalized': word[0].upper() == word[0],
            'is_all_caps': word.upper() == word,
            'is_all_lower': word.lower() == word,
            'is_numeric': word.isdigit(),
            'prefix-1': word[0] if len(word) > 0 else '',
            'prefix-2': word[:2] if len(word) > 1 else '',
            'prefix-3': word[:3] if len(word) > 2 else '',
            'suffix-1': word[-1] if len(word) > 0 else '',
            'suffix-2': word[-2:] if len(word) > 1 else '',
            'suffix-3': word[-3:] if len(word) > 2 else '',
            'has_hyphen': '-' in word,
            'word_length': len(word),
        }

        # Previous word features
        if idx > 0:
            prev_word = sentence[idx - 1]
            features.update({
                'prev_word': prev_word.lower(),
                'prev_is_capitalized': prev_word[0].upper() == prev_word[0],
            })

        # Next word features
        if idx < len(sentence) - 1:
            next_word = sentence[idx + 1]
            features.update({
                'next_word': next_word.lower(),
                'next_is_capitalized': next_word[0].upper() == next_word[0],
            })

        return features

    def fit(self, sentences, labels):
        """Train the NER model"""
        features = []
        y = []

        for sentence, sentence_labels in zip(sentences, labels):
            for idx in range(len(sentence)):
                features.append(self.word_features(sentence, idx))
                y.append(sentence_labels[idx])

        X = self.vectorizer.fit_transform(features)
        self.classifier.fit(X, y)

    def predict(self, sentences):
        """Predict entities for sentences"""
        all_predictions = []

        for sentence in sentences:
            features = [self.word_features(sentence, idx)
                       for idx in range(len(sentence))]
            X = self.vectorizer.transform(features)
            predictions = self.classifier.predict(X)
            all_predictions.append(list(predictions))

        return all_predictions

    def extract_entities(self, sentence, predictions):
        """Extract named entities from predictions"""
        entities = []
        current_entity = []
        current_type = None

        for word, label in zip(sentence, predictions):
            if label.startswith('B-'):
                if current_entity:
                    entities.append((' '.join(current_entity), current_type))
                current_entity = [word]
                current_type = label[2:]
            elif label.startswith('I-') and current_entity:
                current_entity.append(word)
            else:
                if current_entity:
                    entities.append((' '.join(current_entity), current_type))
                current_entity = []
                current_type = None

        if current_entity:
            entities.append((' '.join(current_entity), current_type))

        return entities

def generate_ner_dataset():
    """Generate realistic NER dataset"""
    # Training sentences with entities
    sentences = [
        "Apple Inc was founded by Steve Jobs in California".split(),
        "Microsoft CEO Satya Nadella visited New York yesterday".split(),
        "Amazon opened new offices in Seattle on January 15".split(),
        "Barack Obama was born in Hawaii in August 1961".split(),
        "Google acquired YouTube in October 2006 for billions".split(),
        "Tesla factory in Austin produces electric vehicles".split(),
        "The Eiffel Tower in Paris attracts millions of visitors".split(),
        "Mark Zuckerberg created Facebook at Harvard University".split(),
        "NASA launched the mission from Cape Canaveral in March".split(),
        "Bill Gates founded Microsoft with Paul Allen in 1975".split(),
        "The Amazon River flows through South America".split(),
        "Tokyo Olympics were postponed to July 2021".split(),
        "Albert Einstein worked in Princeton until April 1955".split(),
        "The United Nations building stands in New York City".split(),
        "Leonardo da Vinci painted Mona Lisa in Florence".split(),
        "Mount Everest is located in Nepal and Tibet".split(),
        "The Beatles performed in Liverpool in the 1960s".split(),
        "Nelson Mandela became president of South Africa in May 1994".split(),
        "The Great Wall stretches across northern China".split(),
        "Marie Curie won Nobel Prize in December 1903".split(),
    ]

    # Labels (B=Beginning, I=Inside, O=Outside)
    labels = [
        ['B-ORG', 'I-ORG', 'O', 'O', 'O', 'B-PER', 'I-PER', 'O', 'B-LOC'],
        ['B-ORG', 'O', 'B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'B-DATE'],
        ['B-ORG', 'O', 'O', 'O', 'O', 'B-LOC', 'O', 'B-DATE', 'I-DATE'],
        ['B-PER', 'I-PER', 'O', 'O', 'O', 'B-LOC', 'O', 'B-DATE', 'I-DATE'],
        ['B-ORG', 'O', 'B-ORG', 'O', 'B-DATE', 'I-DATE', 'O', 'O'],
        ['B-ORG', 'O', 'O', 'B-LOC', 'O', 'O', 'O'],
        ['O', 'B-LOC', 'I-LOC', 'O', 'B-LOC', 'O', 'O', 'O', 'O'],
        ['B-PER', 'I-PER', 'O', 'B-ORG', 'O', 'B-ORG', 'I-ORG'],
        ['B-ORG', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC', 'O', 'B-DATE'],
        ['B-PER', 'I-PER', 'O', 'B-ORG', 'O', 'B-PER', 'I-PER', 'O', 'B-DATE'],
        ['O', 'B-LOC', 'I-LOC', 'O', 'O', 'B-LOC', 'I-LOC'],
        ['B-LOC', 'B-ORG', 'O', 'O', 'O', 'B-DATE', 'I-DATE'],
        ['B-PER', 'I-PER', 'O', 'O', 'B-LOC', 'O', 'B-DATE', 'I-DATE'],
        ['O', 'B-ORG', 'I-ORG', 'O', 'O', 'O', 'B-LOC', 'I-LOC', 'I-LOC'],
        ['B-PER', 'I-PER', 'I-PER', 'O', 'B-ORG', 'I-ORG', 'O', 'B-LOC'],
        ['B-LOC', 'I-LOC', 'O', 'O', 'O', 'B-LOC', 'O', 'B-LOC'],
        ['O', 'B-ORG', 'O', 'O', 'B-LOC', 'O', 'O', 'B-DATE'],
        ['B-PER', 'I-PER', 'O', 'O', 'O', 'B-LOC', 'I-LOC', 'O', 'B-DATE', 'I-DATE'],
        ['O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'B-LOC'],
        ['B-PER', 'I-PER', 'O', 'B-ORG', 'I-ORG', 'O', 'B-DATE', 'I-DATE'],
    ]

    # Test sentences
    test_sentences = [
        "Tim Cook leads Apple from Cupertino California".split(),
        "The Statue of Liberty stands in New York Harbor".split(),
        "Jeff Bezos started Amazon in Seattle in July 1994".split(),
        "The Louvre Museum in Paris opened in August 1793".split(),
    ]

    return sentences, labels, test_sentences

def evaluate_ner(y_true, y_pred):
    """Evaluate NER performance"""
    # Flatten lists
    y_true_flat = [label for sentence in y_true for label in sentence]
    y_pred_flat = [label for sentence in y_pred for label in sentence]

    return classification_report(y_true_flat, y_pred_flat, output_dict=True)

def create_visualizations(report_dict, entities_by_type):
    """Create visualizations for NER results"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. F1 scores by entity type
    ax1 = fig.add_subplot(gs[0, :2])
    entity_types = ['B-PER', 'B-ORG', 'B-LOC', 'B-DATE']
    f1_scores = [report_dict.get(et, {}).get('f1-score', 0) for et in entity_types]
    colors = plt.cm.Set3(range(len(entity_types)))

    bars = ax1.bar(range(len(entity_types)), f1_scores, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xticks(range(len(entity_types)))
    ax1.set_xticklabels(['Person', 'Organization', 'Location', 'Date'])
    ax1.set_ylabel('F1 Score')
    ax1.set_title('NER Performance by Entity Type', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')

    # 2. Entity count by type
    ax2 = fig.add_subplot(gs[0, 2])
    entity_counts = {k: len(v) for k, v in entities_by_type.items()}
    ax2.pie(entity_counts.values(), labels=entity_counts.keys(),
            autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Entity Distribution', fontsize=12, fontweight='bold')

    # 3. Precision, Recall, F1 comparison
    ax3 = fig.add_subplot(gs[1, :])
    metrics = ['precision', 'recall', 'f1-score']
    x = np.arange(len(entity_types))
    width = 0.25

    for i, metric in enumerate(metrics):
        values = [report_dict.get(et, {}).get(metric, 0) for et in entity_types]
        ax3.bar(x + i*width, values, width, label=metric.capitalize(), alpha=0.8)

    ax3.set_xlabel('Entity Type')
    ax3.set_ylabel('Score')
    ax3.set_title('Precision, Recall, and F1-Score Comparison', fontsize=12, fontweight='bold')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(['Person', 'Organization', 'Location', 'Date'])
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim([0, 1])

    # 4. Most common entities
    ax4 = fig.add_subplot(gs[2, :])
    all_entities = []
    for entities in entities_by_type.values():
        all_entities.extend(entities)

    if all_entities:
        entity_counts = Counter(all_entities)
        top_10 = entity_counts.most_common(10)
        entities, counts = zip(*top_10)

        y_pos = np.arange(len(entities))
        ax4.barh(y_pos, counts, color='coral', alpha=0.7, edgecolor='black')
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(entities)
        ax4.set_xlabel('Frequency')
        ax4.set_title('Top 10 Most Common Entities', fontsize=12, fontweight='bold')
        ax4.invert_yaxis()
        ax4.grid(axis='x', alpha=0.3)

    plt.savefig('ner_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'ner_analysis.png'")
    plt.close()

def main():
    """Main execution function"""
    print("=" * 60)
    print("Named Entity Recognition - Kaggle NLP Solution")
    print("=" * 60)

    # Generate dataset
    print("\n1. Generating NER Dataset...")
    sentences, labels, test_sentences = generate_ner_dataset()
    print(f"   - Training sentences: {len(sentences)}")
    print(f"   - Test sentences: {len(test_sentences)}")

    # Split data
    train_sents, val_sents, train_labels, val_labels = train_test_split(
        sentences, labels, test_size=0.2, random_state=42
    )

    # Train NER model
    print("\n2. Training NER Model...")
    ner = NERExtractor()
    ner.fit(train_sents, train_labels)
    print(f"   - Feature dimension: {len(ner.vectorizer.feature_names_)}")
    print(f"   - Entity types: {len(ner.entity_types)}")

    # Evaluate on validation set
    print("\n3. Evaluating on Validation Set...")
    val_predictions = ner.predict(val_sents)
    report_dict = evaluate_ner(val_labels, val_predictions)

    print(f"   - Overall F1 Score: {report_dict['weighted avg']['f1-score']:.3f}")
    print(f"   - Precision: {report_dict['weighted avg']['precision']:.3f}")
    print(f"   - Recall: {report_dict['weighted avg']['recall']:.3f}")

    # Extract entities from test sentences
    print("\n4. Extracting Entities from Test Sentences...")
    test_predictions = ner.predict(test_sentences)

    entities_by_type = defaultdict(list)

    print("\n" + "-" * 60)
    for sentence, predictions in zip(test_sentences, test_predictions):
        entities = ner.extract_entities(sentence, predictions)
        print(f"\nSentence: {' '.join(sentence)}")
        print("Entities:")
        for entity, entity_type in entities:
            print(f"  - {entity} ({entity_type})")
            entities_by_type[entity_type].append(entity)

    # Collect entities from all predictions
    for sentence, predictions in zip(val_sents + train_sents,
                                    val_predictions + ner.predict(train_sents)):
        entities = ner.extract_entities(sentence, predictions)
        for entity, entity_type in entities:
            entities_by_type[entity_type].append(entity)

    # Create visualizations
    print("\n5. Creating Visualizations...")
    create_visualizations(report_dict, entities_by_type)

    # Summary statistics
    print("\n6. Entity Statistics:")
    print("-" * 60)
    for entity_type, entities in sorted(entities_by_type.items()):
        print(f"   {entity_type}: {len(entities)} entities")
        # Show top 3 most common
        counter = Counter(entities)
        print(f"      Top: {', '.join([e for e, _ in counter.most_common(3)])}")

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
