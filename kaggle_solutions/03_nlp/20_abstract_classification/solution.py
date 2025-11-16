"""
Scientific Abstract Classification
Classify research abstracts by field (Physics, Biology, Computer Science, Chemistry, Mathematics)

Dataset: Synthetic research abstracts
Difficulty: ⭐⭐⭐⭐ Advanced
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')


class AbstractClassifier:
    """Scientific abstract classification system"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=4000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2,
            max_df=0.85,
            sublinear_tf=True
        )
        self.model = None
        self.fields = ['Physics', 'Biology', 'Computer Science', 'Chemistry', 'Mathematics']

    def create_sample_data(self, n_samples=1000):
        """Create synthetic scientific abstract data"""
        np.random.seed(42)

        # Field-specific scientific vocabularies
        field_vocab = {
            'Physics': {
                'concepts': ['quantum mechanics', 'particle physics', 'thermodynamics', 'relativity',
                           'electromagnetic field', 'wave function', 'energy conservation',
                           'gravitational force', 'superconductivity', 'nuclear physics'],
                'methods': ['theoretical framework', 'experimental setup', 'numerical simulation',
                          'spectroscopy', 'accelerator experiment', 'quantum field theory'],
                'terms': ['photon', 'electron', 'momentum', 'entropy', 'wavelength', 'velocity',
                         'mass', 'force', 'temperature', 'pressure', 'magnetic field'],
                'verbs': ['observe', 'measure', 'calculate', 'demonstrate', 'investigate',
                         'derive', 'predict', 'simulate']
            },
            'Biology': {
                'concepts': ['gene expression', 'cellular mechanism', 'protein synthesis',
                           'evolutionary biology', 'ecosystem dynamics', 'molecular biology',
                           'neuroscience', 'genomics', 'metabolism', 'immunology'],
                'methods': ['in vivo experiment', 'in vitro analysis', 'sequencing',
                          'microscopy', 'cell culture', 'PCR', 'genetic analysis'],
                'terms': ['DNA', 'RNA', 'protein', 'cell', 'organism', 'species', 'mutation',
                         'enzyme', 'chromosome', 'tissue', 'antibody', 'receptor'],
                'verbs': ['express', 'regulate', 'interact', 'evolve', 'mutate',
                         'proliferate', 'differentiate', 'activate']
            },
            'Computer Science': {
                'concepts': ['machine learning', 'algorithm optimization', 'neural networks',
                           'distributed systems', 'computer vision', 'natural language processing',
                           'cybersecurity', 'data structures', 'software engineering', 'AI'],
                'methods': ['deep learning', 'reinforcement learning', 'supervised learning',
                          'convolutional networks', 'computational analysis', 'simulation'],
                'terms': ['algorithm', 'dataset', 'model', 'accuracy', 'training', 'network',
                         'computation', 'performance', 'optimization', 'classifier', 'features'],
                'verbs': ['train', 'classify', 'optimize', 'implement', 'develop',
                         'evaluate', 'propose', 'improve', 'process']
            },
            'Chemistry': {
                'concepts': ['organic synthesis', 'catalysis', 'chemical reaction', 'molecular structure',
                           'thermochemistry', 'electrochemistry', 'polymer chemistry',
                           'spectroscopy', 'crystallography', 'analytical chemistry'],
                'methods': ['synthesis', 'chromatography', 'NMR spectroscopy', 'mass spectrometry',
                          'titration', 'crystallization', 'purification'],
                'terms': ['molecule', 'compound', 'reaction', 'catalyst', 'solvent', 'bond',
                         'atom', 'element', 'concentration', 'pH', 'solution', 'yield'],
                'verbs': ['synthesize', 'react', 'catalyze', 'dissolve', 'precipitate',
                         'oxidize', 'reduce', 'polymerize']
            },
            'Mathematics': {
                'concepts': ['differential equations', 'topology', 'number theory', 'algebra',
                           'probability theory', 'optimization', 'graph theory', 'calculus',
                           'linear algebra', 'mathematical modeling'],
                'methods': ['proof', 'theorem', 'lemma', 'mathematical analysis', 'construction',
                          'derivation', 'computation', 'numerical methods'],
                'terms': ['equation', 'function', 'theorem', 'proof', 'variable', 'matrix',
                         'vector', 'space', 'convergence', 'solution', 'dimension', 'set'],
                'verbs': ['prove', 'derive', 'show', 'construct', 'solve', 'define',
                         'establish', 'generalize', 'extend']
            }
        }

        abstracts = []
        labels = []

        for _ in range(n_samples):
            field = np.random.choice(self.fields)
            vocab = field_vocab[field]

            # Generate abstract structure
            abstract_parts = []

            # Introduction/Background
            concept = np.random.choice(vocab['concepts'])
            term1 = np.random.choice(vocab['terms'])
            abstract_parts.append(f"We {np.random.choice(vocab['verbs'])} {concept} in the context of {term1}.")

            # Problem statement
            term2 = np.random.choice(vocab['terms'])
            abstract_parts.append(
                f"Understanding the relationship between {term1} and {term2} is crucial for advancing {field.lower()}."
            )

            # Methodology
            method = np.random.choice(vocab['methods'])
            verb = np.random.choice(vocab['verbs'])
            abstract_parts.append(f"We {verb} using {method}.")

            # Results/Findings
            concept2 = np.random.choice(vocab['concepts'])
            term3 = np.random.choice(vocab['terms'])
            abstract_parts.append(
                f"Our results demonstrate that {concept2} significantly affects {term3}."
            )

            # Conclusion/Impact
            abstract_parts.append(
                f"These findings have important implications for understanding {concept}."
            )

            # Optional: Add quantitative element
            if np.random.random() > 0.5:
                improvement = np.random.randint(10, 90)
                abstract_parts.append(f"We observe a {improvement}% improvement over previous approaches.")

            abstract_text = ' '.join(abstract_parts)
            abstracts.append(abstract_text)
            labels.append(field)

        return pd.DataFrame({
            'abstract': abstracts,
            'field': labels
        })

    def extract_features(self, df):
        """Extract abstract features"""
        df = df.copy()

        # Text statistics
        df['abstract_length'] = df['abstract'].apply(len)
        df['word_count'] = df['abstract'].apply(lambda x: len(x.split()))
        df['sentence_count'] = df['abstract'].apply(lambda x: x.count('.'))
        df['avg_word_length'] = df['abstract'].apply(
            lambda x: np.mean([len(word) for word in x.split()])
        )

        # Scientific writing features
        df['has_percentage'] = df['abstract'].str.contains(r'\d+%').astype(int)
        df['has_equation_ref'] = df['abstract'].str.contains(r'equation|formula').astype(int)
        df['method_mentions'] = df['abstract'].str.lower().str.count(
            r'method|approach|technique|algorithm'
        )
        df['result_mentions'] = df['abstract'].str.lower().str.count(
            r'result|finding|demonstrate|show'
        )

        # Domain-specific term detection
        df['has_quantum'] = df['abstract'].str.lower().str.contains('quantum').astype(int)
        df['has_gene'] = df['abstract'].str.lower().str.contains('gene|DNA|protein').astype(int)
        df['has_algorithm'] = df['abstract'].str.lower().str.contains('algorithm|model').astype(int)
        df['has_molecule'] = df['abstract'].str.lower().str.contains('molecule|compound|reaction').astype(int)
        df['has_proof'] = df['abstract'].str.lower().str.contains('prove|theorem|proof').astype(int)

        return df

    def train(self, X_train, y_train):
        """Train ensemble classifier"""
        # Create ensemble of models
        clf1 = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf2 = LinearSVC(C=1.0, random_state=42, max_iter=2000)
        clf3 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                         max_depth=5, random_state=42)

        # Voting classifier
        self.model = VotingClassifier(
            estimators=[('lr', clf1), ('svc', clf2), ('gb', clf3)],
            voting='hard'
        )

        self.model.fit(X_train, y_train)

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.model.predict(X_test)

        print("\n=== Model Evaluation ===")
        print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))

        # Confusion matrix
        cm = confusion_matrix(y_test, predictions, labels=self.fields)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.fields,
                   yticklabels=self.fields,
                   cbar_kws={'label': 'Number of Abstracts'})
        plt.title('Scientific Abstract Classification - Confusion Matrix', fontsize=14, pad=20)
        plt.ylabel('Actual Field')
        plt.xlabel('Predicted Field')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('abstract_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\nConfusion matrix saved as 'abstract_confusion_matrix.png'")

        return predictions

    def visualize_data(self, df):
        """Visualize abstract data distributions"""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # Field distribution
        df['field'].value_counts()[self.fields].plot(
            kind='bar', ax=axes[0, 0], color='steelblue'
        )
        axes[0, 0].set_title('Research Field Distribution', fontsize=12, pad=10)
        axes[0, 0].set_xlabel('Field')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Word count by field
        df.groupby('field')['word_count'].mean()[self.fields].plot(
            kind='barh', ax=axes[0, 1], color='coral'
        )
        axes[0, 1].set_title('Average Word Count by Field', fontsize=12, pad=10)
        axes[0, 1].set_xlabel('Average Words')

        # Average word length by field
        df.groupby('field')['avg_word_length'].mean()[self.fields].plot(
            kind='barh', ax=axes[0, 2], color='seagreen'
        )
        axes[0, 2].set_title('Average Word Length by Field', fontsize=12, pad=10)
        axes[0, 2].set_xlabel('Average Character Count')

        # Method mentions by field
        df.groupby('field')['method_mentions'].mean()[self.fields].plot(
            kind='bar', ax=axes[1, 0], color='orchid'
        )
        axes[1, 0].set_title('Avg Method Mentions by Field', fontsize=12, pad=10)
        axes[1, 0].set_xlabel('Field')
        axes[1, 0].set_ylabel('Average Count')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Percentage usage by field
        df.groupby('field')['has_percentage'].mean()[self.fields].plot(
            kind='bar', ax=axes[1, 1], color='crimson'
        )
        axes[1, 1].set_title('Proportion with Percentage Values', fontsize=12, pad=10)
        axes[1, 1].set_xlabel('Field')
        axes[1, 1].set_ylabel('Proportion')
        axes[1, 1].tick_params(axis='x', rotation=45)

        # Sentence count distribution
        axes[1, 2].hist(df['sentence_count'], bins=20, color='mediumseagreen',
                       edgecolor='black', alpha=0.7)
        axes[1, 2].set_title('Sentence Count Distribution', fontsize=12, pad=10)
        axes[1, 2].set_xlabel('Number of Sentences')
        axes[1, 2].set_ylabel('Frequency')

        plt.tight_layout()
        plt.savefig('abstract_analysis.png', dpi=300, bbox_inches='tight')
        print("Visualization saved as 'abstract_analysis.png'")


def main():
    """Main execution function"""
    print("=" * 70)
    print("Scientific Abstract Classification")
    print("=" * 70)

    # Initialize classifier
    classifier = AbstractClassifier()

    # Create sample data
    print("\nCreating synthetic scientific abstract data...")
    df = classifier.create_sample_data(n_samples=1000)
    print(f"Dataset size: {df.shape}")
    print(f"\nSample abstract:\n{df['abstract'].iloc[0]}")
    print(f"Field: {df['field'].iloc[0]}")

    # Data exploration
    print(f"\n=== Field Distribution ===")
    print(df['field'].value_counts())

    # Extract features
    print("\nExtracting features...")
    df = classifier.extract_features(df)

    # Visualize data
    print("\nGenerating visualizations...")
    classifier.visualize_data(df)

    # Prepare data for modeling
    print("\nPreparing data for modeling...")
    X = classifier.vectorizer.fit_transform(df['abstract'])
    y = df['field']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")

    # Train ensemble model
    print("\nTraining ensemble classifier...")
    classifier.train(X_train, y_train)

    # Evaluate model
    print("\nEvaluating model...")
    predictions = classifier.evaluate(X_test, y_test)

    # Sample predictions
    print("\n=== Sample Predictions ===")
    for i in range(3):
        sample_text = df['abstract'].iloc[i]
        sample_vectorized = classifier.vectorizer.transform([sample_text])
        prediction = classifier.model.predict(sample_vectorized)[0]
        actual = df['field'].iloc[i]

        print(f"\nAbstract: {sample_text[:200]}...")
        print(f"Predicted Field: {prediction}")
        print(f"Actual Field: {actual}")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
