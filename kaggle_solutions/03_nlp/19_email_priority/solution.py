"""
Email Priority Classification
Classify emails by priority (High, Medium, Low) based on content

Dataset: Synthetic email data
Difficulty: ⭐⭐⭐ Intermediate
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import warnings
warnings.filterwarnings('ignore')


class EmailPriorityClassifier:
    """Email priority classification system"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=2500,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.9
        )
        self.model = None
        self.priorities = ['High', 'Medium', 'Low']

    def create_sample_data(self, n_samples=1000):
        """Create synthetic email data"""
        np.random.seed(42)

        # Priority-specific vocabularies
        priority_vocab = {
            'High': {
                'subjects': ['URGENT', 'CRITICAL', 'IMMEDIATE ACTION REQUIRED', 'Emergency',
                           'Security Alert', 'System Down', 'Client Issue', 'Deadline Today'],
                'keywords': ['urgent', 'critical', 'immediately', 'asap', 'emergency',
                           'deadline', 'important', 'action required', 'time-sensitive',
                           'escalation', 'priority', 'crucial', 'vital'],
                'senders': ['CEO', 'VP', 'Director', 'Client', 'Security Team', 'Manager'],
                'phrases': ['need this done today', 'top priority', 'cannot wait',
                          'requires immediate attention', 'please respond urgently',
                          'critical situation', 'time is running out']
            },
            'Medium': {
                'subjects': ['Follow up', 'Meeting Request', 'Project Update', 'Question',
                           'Feedback Needed', 'Review Required', 'Information Request'],
                'keywords': ['follow up', 'update', 'meeting', 'review', 'feedback',
                           'question', 'discuss', 'coordinate', 'plan', 'schedule',
                           'check in', 'status', 'progress'],
                'senders': ['Colleague', 'Team Member', 'Project Manager', 'Coordinator'],
                'phrases': ['when you get a chance', 'at your convenience',
                          'please review', 'wanted to follow up', 'quick question',
                          'need your input', 'would appreciate']
            },
            'Low': {
                'subjects': ['FYI', 'Newsletter', 'Info', 'Update', 'Announcement',
                           'Weekly Summary', 'Reminder', 'General Information'],
                'keywords': ['fyi', 'information', 'newsletter', 'update', 'reminder',
                           'announcement', 'notice', 'heads up', 'just so you know',
                           'for your reference', 'sharing', 'thought you might like'],
                'senders': ['HR', 'Marketing', 'IT Department', 'Admin', 'Newsletter'],
                'phrases': ['just wanted to share', 'for your information',
                          'no response needed', 'keeping you in the loop',
                          'thought this might interest you', 'weekly roundup']
            }
        }

        emails = []
        priorities = []

        for _ in range(n_samples):
            priority = np.random.choice(['High', 'Medium', 'Low'], p=[0.20, 0.45, 0.35])
            vocab = priority_vocab[priority]

            # Generate email components
            subject = np.random.choice(vocab['subjects'])
            sender = np.random.choice(vocab['senders'])
            keywords = np.random.choice(vocab['keywords'], np.random.randint(2, 4), replace=False)
            phrase = np.random.choice(vocab['phrases'])

            # Build email
            email_parts = []
            email_parts.append(f"From: {sender}")
            email_parts.append(f"Subject: {subject}")
            email_parts.append("")  # Empty line
            email_parts.append(f"{phrase.capitalize()}.")

            # Add body content
            if priority == 'High':
                email_parts.append(f"This is {keywords[0]} and requires {keywords[1]}.")
                email_parts.append(f"Please address this {keywords[2]} matter.")
            elif priority == 'Medium':
                email_parts.append(f"I wanted to {keywords[0]} regarding the {keywords[1]}.")
                email_parts.append(f"We should {keywords[2]} this when possible.")
            else:
                email_parts.append(f"This {keywords[0]} is for your {keywords[1]}.")
                email_parts.append(f"Feel free to {keywords[2]} at your leisure.")

            # Add signature
            email_parts.append("")
            email_parts.append(f"Best regards,\n{sender}")

            email_text = '\n'.join(email_parts)
            emails.append(email_text)
            priorities.append(priority)

        return pd.DataFrame({
            'email': emails,
            'priority': priorities
        })

    def extract_features(self, df):
        """Extract email features"""
        df = df.copy()

        # Extract subject line
        df['subject'] = df['email'].apply(
            lambda x: re.search(r'Subject: (.+)', x).group(1) if re.search(r'Subject: (.+)', x) else ''
        )

        # Text statistics
        df['email_length'] = df['email'].apply(len)
        df['word_count'] = df['email'].apply(lambda x: len(x.split()))
        df['line_count'] = df['email'].apply(lambda x: x.count('\n'))

        # Priority indicators
        urgent_words = ['urgent', 'critical', 'asap', 'emergency', 'immediately', 'deadline']
        df['urgent_word_count'] = df['email'].apply(
            lambda x: sum([1 for word in urgent_words if word in x.lower()])
        )

        # Subject line features
        df['subject_has_urgent'] = df['subject'].str.upper().str.contains(
            'URGENT|CRITICAL|EMERGENCY|ASAP', na=False
        ).astype(int)
        df['subject_all_caps'] = df['subject'].apply(
            lambda x: 1 if x.isupper() and len(x) > 0 else 0
        )
        df['subject_length'] = df['subject'].apply(len)

        # Sender analysis
        df['sender'] = df['email'].apply(
            lambda x: re.search(r'From: (.+)', x).group(1) if re.search(r'From: (.+)', x) else ''
        )
        executive_titles = ['CEO', 'VP', 'Director', 'President', 'Client']
        df['from_executive'] = df['sender'].apply(
            lambda x: 1 if any(title in x for title in executive_titles) else 0
        )

        # Punctuation
        df['exclamation_count'] = df['email'].str.count('!')
        df['question_count'] = df['email'].str.count(r'\?')

        return df

    def train(self, X_train, y_train):
        """Train priority classification model"""
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
        print(classification_report(y_test, predictions, target_names=self.priorities))

        # Confusion matrix
        cm = confusion_matrix(y_test, predictions, labels=self.priorities)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                   xticklabels=self.priorities,
                   yticklabels=self.priorities,
                   cbar_kws={'label': 'Number of Emails'})
        plt.title('Email Priority Classification - Confusion Matrix', fontsize=14, pad=20)
        plt.ylabel('Actual Priority')
        plt.xlabel('Predicted Priority')
        plt.tight_layout()
        plt.savefig('email_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\nConfusion matrix saved as 'email_confusion_matrix.png'")

        return predictions

    def visualize_data(self, df):
        """Visualize email data distributions"""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # Priority distribution
        df['priority'].value_counts()[self.priorities].plot(
            kind='bar', ax=axes[0, 0], color='steelblue'
        )
        axes[0, 0].set_title('Email Priority Distribution', fontsize=12, pad=10)
        axes[0, 0].set_xlabel('Priority')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=0)

        # Word count by priority
        df.boxplot(column='word_count', by='priority', ax=axes[0, 1])
        axes[0, 1].set_title('Word Count by Priority', fontsize=12, pad=10)
        axes[0, 1].set_xlabel('Priority')
        axes[0, 1].set_ylabel('Word Count')

        # Urgent word count by priority
        df.groupby('priority')['urgent_word_count'].mean()[self.priorities].plot(
            kind='bar', ax=axes[0, 2], color='coral'
        )
        axes[0, 2].set_title('Avg Urgent Words by Priority', fontsize=12, pad=10)
        axes[0, 2].set_xlabel('Priority')
        axes[0, 2].set_ylabel('Urgent Word Count')
        axes[0, 2].tick_params(axis='x', rotation=0)

        # Subject length by priority
        df.groupby('priority')['subject_length'].mean()[self.priorities].plot(
            kind='bar', ax=axes[1, 0], color='seagreen'
        )
        axes[1, 0].set_title('Avg Subject Length by Priority', fontsize=12, pad=10)
        axes[1, 0].set_xlabel('Priority')
        axes[1, 0].set_ylabel('Character Count')
        axes[1, 0].tick_params(axis='x', rotation=0)

        # Executive sender proportion
        exec_prop = df.groupby('priority')['from_executive'].mean()[self.priorities]
        exec_prop.plot(kind='bar', ax=axes[1, 1], color='orchid')
        axes[1, 1].set_title('Proportion from Executives', fontsize=12, pad=10)
        axes[1, 1].set_xlabel('Priority')
        axes[1, 1].set_ylabel('Proportion')
        axes[1, 1].tick_params(axis='x', rotation=0)

        # Subject all caps by priority
        caps_prop = df.groupby('priority')['subject_all_caps'].mean()[self.priorities]
        caps_prop.plot(kind='bar', ax=axes[1, 2], color='crimson')
        axes[1, 2].set_title('Proportion with ALL CAPS Subject', fontsize=12, pad=10)
        axes[1, 2].set_xlabel('Priority')
        axes[1, 2].set_ylabel('Proportion')
        axes[1, 2].tick_params(axis='x', rotation=0)

        plt.tight_layout()
        plt.savefig('email_analysis.png', dpi=300, bbox_inches='tight')
        print("Visualization saved as 'email_analysis.png'")


def main():
    """Main execution function"""
    print("=" * 70)
    print("Email Priority Classification")
    print("=" * 70)

    # Initialize classifier
    classifier = EmailPriorityClassifier()

    # Create sample data
    print("\nCreating synthetic email data...")
    df = classifier.create_sample_data(n_samples=1000)
    print(f"Dataset size: {df.shape}")
    print(f"\nSample email:\n{df['email'].iloc[0]}")
    print(f"Priority: {df['priority'].iloc[0]}")

    # Data exploration
    print(f"\n=== Priority Distribution ===")
    print(df['priority'].value_counts())

    # Extract features
    print("\nExtracting features...")
    df = classifier.extract_features(df)

    # Visualize data
    print("\nGenerating visualizations...")
    classifier.visualize_data(df)

    # Prepare data for modeling
    print("\nPreparing data for modeling...")
    X = classifier.vectorizer.fit_transform(df['email'])
    y = df['priority']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")

    # Train model
    print("\nTraining Gradient Boosting classifier...")
    classifier.train(X_train, y_train)

    # Evaluate model
    print("\nEvaluating model...")
    predictions = classifier.evaluate(X_test, y_test)

    # Sample predictions
    print("\n=== Sample Predictions ===")
    for i in range(3):
        sample_text = df['email'].iloc[i]
        sample_vectorized = classifier.vectorizer.transform([sample_text])
        prediction = classifier.model.predict(sample_vectorized)[0]
        actual = df['priority'].iloc[i]

        print(f"\nEmail Preview: {sample_text[:150]}...")
        print(f"Predicted Priority: {prediction}")
        print(f"Actual Priority: {actual}")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
