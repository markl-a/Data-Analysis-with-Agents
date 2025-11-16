"""
Legal Document Classification
Classify legal documents into categories (Contract, Patent, Court Filing, Agreement, Regulation)

Dataset: Synthetic legal documents
Difficulty: ⭐⭐⭐⭐ Advanced
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import re
import warnings
warnings.filterwarnings('ignore')


class LegalDocumentClassifier:
    """Legal document classification system"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2,
            max_df=0.9,
            sublinear_tf=True
        )
        self.label_encoder = LabelEncoder()
        self.model = None
        self.categories = ['Contract', 'Patent', 'Court Filing', 'Agreement', 'Regulation']

    def create_sample_data(self, n_samples=1000):
        """Create synthetic legal document data"""
        np.random.seed(42)

        # Legal domain-specific vocabularies
        legal_vocab = {
            'Contract': {
                'terms': ['hereby agrees', 'pursuant to', 'in consideration of', 'parties agree',
                         'terms and conditions', 'breach of contract', 'indemnify', 'liability',
                         'termination clause', 'force majeure', 'confidentiality'],
                'phrases': ['This Agreement is entered into', 'The parties hereto agree',
                           'shall be binding upon', 'in witness whereof', 'executed as of',
                           'obligations hereunder', 'representations and warranties'],
                'sections': ['Article I', 'Section 1.1', 'Clause', 'Paragraph', 'Schedule A'],
                'parties': ['Party A', 'Party B', 'Contractor', 'Client', 'Vendor', 'Service Provider']
            },
            'Patent': {
                'terms': ['invention', 'claim', 'embodiment', 'apparatus', 'method comprising',
                         'prior art', 'specification', 'drawings', 'abstract', 'field of invention',
                         'technical solution', 'implementation'],
                'phrases': ['What is claimed is', 'in accordance with the invention',
                           'preferred embodiment', 'as shown in Figure', 'the present invention relates to',
                           'novel and non-obvious', 'technical effect'],
                'sections': ['Claim 1', 'Claim 2', 'Figure 1', 'Background', 'Summary', 'Detailed Description'],
                'parties': ['Inventor', 'Applicant', 'Assignee', 'Patent Holder']
            },
            'Court Filing': {
                'terms': ['plaintiff', 'defendant', 'motion', 'jurisdiction', 'complaint',
                         'discovery', 'deposition', 'testimony', 'exhibit', 'stipulation',
                         'ruling', 'precedent', 'allegation'],
                'phrases': ['comes now the plaintiff', 'respectfully requests', 'pursuant to Rule',
                           'in the matter of', 'court finds', 'hereby orders', 'based on the evidence',
                           'motion to dismiss', 'summary judgment'],
                'sections': ['Count I', 'Count II', 'Prayer for Relief', 'Wherefore', 'Respectfully submitted'],
                'parties': ['Plaintiff', 'Defendant', 'Petitioner', 'Respondent', 'The Court']
            },
            'Agreement': {
                'terms': ['mutual agreement', 'good faith', 'cooperation', 'amendment',
                         'waiver', 'assignment', 'severability', 'entire agreement',
                         'governing law', 'dispute resolution', 'arbitration'],
                'phrases': ['parties mutually agree', 'effective date', 'for and in consideration',
                           'jointly and severally', 'written consent', 'time is of the essence',
                           'supersedes all prior agreements'],
                'sections': ['Recitals', 'Definitions', 'Obligations', 'Term', 'Miscellaneous'],
                'parties': ['First Party', 'Second Party', 'Collaborator', 'Partner', 'Associate']
            },
            'Regulation': {
                'terms': ['regulation', 'compliance', 'shall comply', 'requirement', 'standard',
                         'prohibition', 'enforcement', 'violation', 'penalty', 'inspection',
                         'certification', 'authority', 'mandate'],
                'phrases': ['pursuant to authority', 'federal regulations require', 'must comply with',
                           'violation of this regulation', 'subject to penalties', 'enforcement action',
                           'regulatory framework', 'standards set forth'],
                'sections': ['Section 101', 'Subsection (a)', 'Part A', 'Chapter 1', 'Rule'],
                'parties': ['Regulatory Authority', 'Commission', 'Agency', 'Department', 'Board']
            }
        }

        documents = []
        labels = []

        for _ in range(n_samples):
            category = np.random.choice(self.categories)
            vocab = legal_vocab[category]

            # Generate document text
            num_paragraphs = np.random.randint(3, 6)
            paragraphs = []

            for _ in range(num_paragraphs):
                # Select components
                num_terms = np.random.randint(2, 4)
                terms = np.random.choice(vocab['terms'], num_terms, replace=False)
                phrase = np.random.choice(vocab['phrases'])
                section = np.random.choice(vocab['sections'])
                party = np.random.choice(vocab['parties'])

                # Generate paragraph
                paragraph_parts = [
                    f"{section}: {phrase}",
                    f"{party} {terms[0]}",
                ]

                if len(terms) > 1:
                    paragraph_parts.append(f"including {terms[1]}")
                if len(terms) > 2:
                    paragraph_parts.append(f"and {terms[2]}")

                paragraph = ' '.join(paragraph_parts) + '.'
                paragraphs.append(paragraph)

            document_text = ' '.join(paragraphs)
            documents.append(document_text)
            labels.append(category)

        return pd.DataFrame({
            'document': documents,
            'category': labels
        })

    def extract_features(self, df):
        """Extract legal document features"""
        df = df.copy()

        # Document statistics
        df['doc_length'] = df['document'].apply(len)
        df['word_count'] = df['document'].apply(lambda x: len(x.split()))
        df['avg_word_length'] = df['document'].apply(
            lambda x: np.mean([len(word) for word in x.split()])
        )

        # Legal-specific features
        df['section_count'] = df['document'].apply(
            lambda x: len(re.findall(r'Section|Article|Clause|Claim|Count', x))
        )
        df['party_mentions'] = df['document'].apply(
            lambda x: len(re.findall(r'Party|Plaintiff|Defendant|Inventor|Agency', x))
        )
        df['legal_terms'] = df['document'].apply(
            lambda x: len(re.findall(r'pursuant|hereby|aforementioned|thereof|wherein', x, re.IGNORECASE))
        )
        df['has_citations'] = df['document'].str.contains(r'\d+\.\d+').astype(int)

        return df

    def train(self, X_train, y_train):
        """Train ensemble model for legal classification"""
        # Create ensemble of classifiers
        clf1 = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf2 = LinearSVC(C=1.0, random_state=42, max_iter=2000)
        clf3 = MultinomialNB(alpha=0.1)

        # Voting classifier
        self.model = VotingClassifier(
            estimators=[('lr', clf1), ('svc', clf2), ('nb', clf3)],
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
        cm = confusion_matrix(y_test, predictions, labels=self.categories)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                   xticklabels=self.categories,
                   yticklabels=self.categories,
                   cbar_kws={'label': 'Number of Documents'})
        plt.title('Legal Document Classification - Confusion Matrix', fontsize=14, pad=20)
        plt.ylabel('Actual Category')
        plt.xlabel('Predicted Category')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('legal_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\nConfusion matrix saved as 'legal_confusion_matrix.png'")

        return predictions

    def visualize_data(self, df):
        """Visualize legal document distributions"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Category distribution
        df['category'].value_counts().plot(kind='bar', ax=axes[0, 0], color='darkblue')
        axes[0, 0].set_title('Legal Document Category Distribution', fontsize=12, pad=10)
        axes[0, 0].set_xlabel('Category')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Word count by category
        category_word_counts = df.groupby('category')['word_count'].mean().sort_values()
        category_word_counts.plot(kind='barh', ax=axes[0, 1], color='darkgreen')
        axes[0, 1].set_title('Average Word Count by Category', fontsize=12, pad=10)
        axes[0, 1].set_xlabel('Average Word Count')

        # Legal terms distribution
        df.groupby('category')['legal_terms'].mean().plot(
            kind='bar', ax=axes[1, 0], color='darkred'
        )
        axes[1, 0].set_title('Average Legal Terms by Category', fontsize=12, pad=10)
        axes[1, 0].set_xlabel('Category')
        axes[1, 0].set_ylabel('Average Count')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Section count distribution
        df.groupby('category')['section_count'].mean().plot(
            kind='bar', ax=axes[1, 1], color='darkorange'
        )
        axes[1, 1].set_title('Average Section Count by Category', fontsize=12, pad=10)
        axes[1, 1].set_xlabel('Category')
        axes[1, 1].set_ylabel('Average Count')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('legal_analysis.png', dpi=300, bbox_inches='tight')
        print("Visualization saved as 'legal_analysis.png'")


def main():
    """Main execution function"""
    print("=" * 70)
    print("Legal Document Classification")
    print("=" * 70)

    # Initialize classifier
    classifier = LegalDocumentClassifier()

    # Create sample data
    print("\nCreating synthetic legal document data...")
    df = classifier.create_sample_data(n_samples=1000)
    print(f"Dataset size: {df.shape}")
    print(f"\nSample document:\n{df['document'].iloc[0][:250]}...")
    print(f"Category: {df['category'].iloc[0]}")

    # Data exploration
    print(f"\n=== Category Distribution ===")
    print(df['category'].value_counts())

    # Extract features
    print("\nExtracting features...")
    df = classifier.extract_features(df)

    # Visualize data
    print("\nGenerating visualizations...")
    classifier.visualize_data(df)

    # Prepare data for modeling
    print("\nPreparing data for modeling...")
    X = classifier.vectorizer.fit_transform(df['document'])
    y = df['category']

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
        sample_text = df['document'].iloc[i]
        sample_vectorized = classifier.vectorizer.transform([sample_text])
        prediction = classifier.model.predict(sample_vectorized)[0]
        actual = df['category'].iloc[i]

        print(f"\nDocument {i+1}: {sample_text[:120]}...")
        print(f"Predicted: {prediction}")
        print(f"Actual: {actual}")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
