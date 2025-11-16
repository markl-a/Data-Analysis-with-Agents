"""
Resume Parsing - Extract Skills and Experience
Extract and classify technical skills, years of experience from resumes

Dataset: Synthetic resume data
Difficulty: ⭐⭐⭐ Intermediate
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import re
import warnings
warnings.filterwarnings('ignore')


class ResumeParser:
    """Resume parsing and skill extraction system"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2
        )
        self.label_encoder = LabelEncoder()
        self.model = None

    def create_sample_data(self, n_samples=1000):
        """Create synthetic resume data"""
        np.random.seed(42)

        # Domain-specific vocabularies
        categories = ['Data Science', 'Software Engineering', 'DevOps', 'Web Development', 'Mobile Development']

        skills_vocab = {
            'Data Science': [
                'python', 'machine learning', 'deep learning', 'tensorflow', 'pytorch',
                'scikit-learn', 'pandas', 'numpy', 'data analysis', 'statistics',
                'sql', 'tableau', 'power bi', 'r programming', 'jupyter',
                'neural networks', 'nlp', 'computer vision', 'feature engineering'
            ],
            'Software Engineering': [
                'java', 'c++', 'algorithms', 'data structures', 'object oriented',
                'design patterns', 'microservices', 'rest api', 'spring boot',
                'unit testing', 'agile', 'scrum', 'git', 'code review',
                'software architecture', 'multithreading', 'maven', 'gradle'
            ],
            'DevOps': [
                'docker', 'kubernetes', 'jenkins', 'ci/cd', 'aws', 'azure',
                'terraform', 'ansible', 'monitoring', 'prometheus', 'grafana',
                'linux', 'bash scripting', 'networking', 'cloud infrastructure',
                'container orchestration', 'gitlab', 'deployment automation'
            ],
            'Web Development': [
                'javascript', 'react', 'angular', 'vue.js', 'node.js', 'html',
                'css', 'responsive design', 'bootstrap', 'tailwind', 'webpack',
                'typescript', 'express.js', 'mongodb', 'postgresql', 'rest api',
                'graphql', 'redux', 'sass', 'web performance'
            ],
            'Mobile Development': [
                'android', 'ios', 'swift', 'kotlin', 'react native', 'flutter',
                'mobile ui/ux', 'xcode', 'android studio', 'firebase',
                'push notifications', 'app store', 'play store', 'mobile security',
                'offline storage', 'api integration', 'mobile testing'
            ]
        }

        experience_templates = {
            'junior': [
                'developed {} applications', 'worked on {} projects',
                'assisted in {} implementation', 'learned {}',
                'contributed to {} development'
            ],
            'mid': [
                'designed and implemented {}', 'led {} initiatives',
                'optimized {} performance', 'architected {} solutions',
                'mentored team on {}'
            ],
            'senior': [
                'architected enterprise {}', 'led team of developers on {}',
                'established best practices for {}', 'drove {} strategy',
                'scaled {} infrastructure'
            ]
        }

        resumes = []
        labels = []

        for _ in range(n_samples):
            category = np.random.choice(categories)
            experience_level = np.random.choice(['junior', 'mid', 'senior'], p=[0.3, 0.5, 0.2])

            # Generate resume text
            num_skills = np.random.randint(5, 12)
            selected_skills = np.random.choice(skills_vocab[category], num_skills, replace=False)

            # Build resume sections
            resume_parts = []

            # Summary section
            resume_parts.append(f"Professional {category} professional with expertise in:")
            resume_parts.extend(selected_skills[:3])

            # Experience section
            num_experiences = np.random.randint(2, 5)
            for _ in range(num_experiences):
                template = np.random.choice(experience_templates[experience_level])
                skill = np.random.choice(selected_skills)
                resume_parts.append(template.format(skill))

            # Skills section
            resume_parts.append("Technical Skills:")
            resume_parts.extend(selected_skills)

            # Add some noise words
            noise_words = ['strong', 'excellent', 'proven', 'track record', 'collaborative',
                          'team player', 'fast learner', 'detail oriented', 'passionate']
            resume_parts.extend(np.random.choice(noise_words, 3, replace=False))

            resume_text = ' '.join(resume_parts)
            resumes.append(resume_text)
            labels.append(category)

        return pd.DataFrame({
            'resume_text': resumes,
            'category': labels
        })

    def extract_features(self, df):
        """Extract features from resume text"""
        df = df.copy()

        # Text length features
        df['text_length'] = df['resume_text'].apply(len)
        df['word_count'] = df['resume_text'].apply(lambda x: len(x.split()))
        df['avg_word_length'] = df['resume_text'].apply(
            lambda x: np.mean([len(word) for word in x.split()])
        )

        # Count specific keywords
        df['num_technical_terms'] = df['resume_text'].str.lower().apply(
            lambda x: sum([1 for term in ['api', 'framework', 'library', 'database'] if term in x])
        )

        # Years of experience (simulated extraction)
        df['experience_indicators'] = df['resume_text'].str.lower().apply(
            lambda x: sum([1 for term in ['led', 'architected', 'designed', 'implemented'] if term in x])
        )

        return df

    def train(self, X_train, y_train):
        """Train the classification model"""
        # Use Gradient Boosting for better performance
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
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
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Resume Category Classification - Confusion Matrix', fontsize=14, pad=20)
        plt.ylabel('Actual Category')
        plt.xlabel('Predicted Category')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('resume_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\nConfusion matrix saved as 'resume_confusion_matrix.png'")

    def visualize_data(self, df):
        """Visualize resume data distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Category distribution
        df['category'].value_counts().plot(kind='bar', ax=axes[0, 0], color='steelblue')
        axes[0, 0].set_title('Resume Category Distribution', fontsize=12, pad=10)
        axes[0, 0].set_xlabel('Category')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Word count distribution
        axes[0, 1].hist(df['word_count'], bins=30, color='coral', edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Word Count Distribution', fontsize=12, pad=10)
        axes[0, 1].set_xlabel('Number of Words')
        axes[0, 1].set_ylabel('Frequency')

        # Average word length by category
        df.groupby('category')['avg_word_length'].mean().plot(
            kind='barh', ax=axes[1, 0], color='seagreen'
        )
        axes[1, 0].set_title('Average Word Length by Category', fontsize=12, pad=10)
        axes[1, 0].set_xlabel('Average Word Length')

        # Experience indicators by category
        df.groupby('category')['experience_indicators'].mean().plot(
            kind='barh', ax=axes[1, 1], color='mediumpurple'
        )
        axes[1, 1].set_title('Experience Indicators by Category', fontsize=12, pad=10)
        axes[1, 1].set_xlabel('Average Count')

        plt.tight_layout()
        plt.savefig('resume_analysis.png', dpi=300, bbox_inches='tight')
        print("Visualization saved as 'resume_analysis.png'")


def main():
    """Main execution function"""
    print("=" * 70)
    print("Resume Parsing - Skill Extraction and Classification")
    print("=" * 70)

    # Initialize parser
    parser = ResumeParser()

    # Create sample data
    print("\nCreating synthetic resume data...")
    df = parser.create_sample_data(n_samples=1000)
    print(f"Dataset size: {df.shape}")
    print(f"\nSample resume:\n{df['resume_text'].iloc[0][:200]}...")
    print(f"Category: {df['category'].iloc[0]}")

    # Data exploration
    print(f"\n=== Data Distribution ===")
    print(df['category'].value_counts())

    # Extract features
    print("\nExtracting features...")
    df = parser.extract_features(df)

    # Visualize data
    print("\nGenerating visualizations...")
    parser.visualize_data(df)

    # Prepare data for modeling
    print("\nPreparing data for modeling...")
    X_text = parser.vectorizer.fit_transform(df['resume_text'])
    y = parser.label_encoder.fit_transform(df['category'])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")

    # Train model
    print("\nTraining Gradient Boosting classifier...")
    parser.train(X_train, y_train)

    # Evaluate model
    print("\nEvaluating model...")
    parser.evaluate(X_test, y_test)

    # Sample prediction
    print("\n=== Sample Prediction ===")
    sample_idx = 0
    sample_text = df['resume_text'].iloc[sample_idx]
    sample_vectorized = parser.vectorizer.transform([sample_text])
    prediction = parser.model.predict(sample_vectorized)[0]
    predicted_category = parser.label_encoder.inverse_transform([prediction])[0]
    actual_category = df['category'].iloc[sample_idx]

    print(f"Resume snippet: {sample_text[:150]}...")
    print(f"Predicted category: {predicted_category}")
    print(f"Actual category: {actual_category}")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
