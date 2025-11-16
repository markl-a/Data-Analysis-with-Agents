"""
Medical Text Classification
Classify medical notes and reports into categories (Cardiology, Neurology, Oncology, Pediatrics, Radiology)

Dataset: Synthetic medical records
Difficulty: ⭐⭐⭐⭐ Advanced
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')


class MedicalTextClassifier:
    """Medical text classification system"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=4000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2,
            max_df=0.85
        )
        self.model = None
        self.categories = ['Cardiology', 'Neurology', 'Oncology', 'Pediatrics', 'Radiology']

    def create_sample_data(self, n_samples=1000):
        """Create synthetic medical text data"""
        np.random.seed(42)

        # Medical domain-specific vocabularies
        medical_vocab = {
            'Cardiology': {
                'conditions': ['hypertension', 'arrhythmia', 'coronary artery disease', 'heart failure',
                             'myocardial infarction', 'atrial fibrillation', 'cardiomyopathy',
                             'valve disease', 'angina', 'pericarditis'],
                'symptoms': ['chest pain', 'shortness of breath', 'palpitations', 'syncope',
                           'edema', 'fatigue', 'dizziness', 'irregular heartbeat'],
                'tests': ['ECG', 'echocardiogram', 'stress test', 'cardiac catheterization',
                         'holter monitor', 'angiography', 'troponin levels', 'BNP'],
                'treatments': ['beta blockers', 'ACE inhibitors', 'stent placement', 'bypass surgery',
                             'anticoagulation', 'cardioversion', 'pacemaker', 'diuretics'],
                'findings': ['ejection fraction', 'ST elevation', 'murmur', 'enlarged heart',
                           'valve regurgitation', 'coronary stenosis']
            },
            'Neurology': {
                'conditions': ['stroke', 'epilepsy', 'multiple sclerosis', 'Parkinsons disease',
                             'Alzheimers disease', 'migraine', 'neuropathy', 'brain tumor',
                             'seizure disorder', 'dementia'],
                'symptoms': ['headache', 'weakness', 'numbness', 'confusion', 'tremor',
                           'memory loss', 'vision changes', 'balance problems', 'speech difficulty'],
                'tests': ['MRI brain', 'CT scan', 'EEG', 'lumbar puncture', 'nerve conduction study',
                         'EMG', 'neurological exam', 'cognitive assessment'],
                'treatments': ['anticonvulsants', 'dopamine agonists', 'physical therapy',
                             'corticosteroids', 'disease-modifying therapy', 'pain management'],
                'findings': ['lesions', 'atrophy', 'hemorrhage', 'focal deficits', 'abnormal reflexes',
                           'white matter changes', 'seizure activity']
            },
            'Oncology': {
                'conditions': ['breast cancer', 'lung cancer', 'colon cancer', 'lymphoma',
                             'leukemia', 'melanoma', 'prostate cancer', 'metastatic disease',
                             'brain tumor', 'ovarian cancer'],
                'symptoms': ['weight loss', 'fatigue', 'pain', 'anemia', 'night sweats',
                           'mass', 'bleeding', 'loss of appetite', 'cough'],
                'tests': ['biopsy', 'CT scan', 'PET scan', 'tumor markers', 'bone scan',
                         'CBC', 'pathology', 'staging workup', 'genetic testing'],
                'treatments': ['chemotherapy', 'radiation therapy', 'immunotherapy', 'surgery',
                             'targeted therapy', 'hormone therapy', 'stem cell transplant'],
                'findings': ['malignancy', 'metastases', 'tumor size', 'lymph node involvement',
                           'stage IV', 'recurrence', 'progression', 'remission']
            },
            'Pediatrics': {
                'conditions': ['asthma', 'ear infection', 'RSV', 'croup', 'pneumonia',
                             'developmental delay', 'ADHD', 'growth disorder', 'allergies',
                             'viral illness'],
                'symptoms': ['fever', 'cough', 'rash', 'vomiting', 'diarrhea', 'irritability',
                           'poor feeding', 'wheezing', 'runny nose', 'behavioral issues'],
                'tests': ['growth chart', 'developmental screening', 'throat culture',
                         'chest x-ray', 'urinalysis', 'vaccine records', 'lead screening'],
                'treatments': ['antibiotics', 'nebulizer', 'acetaminophen', 'supportive care',
                             'behavioral therapy', 'growth hormone', 'immunizations'],
                'findings': ['normal development', 'weight percentile', 'height percentile',
                           'delayed milestones', 'fever of unknown origin', 'viral exanthem']
            },
            'Radiology': {
                'conditions': ['fracture', 'pneumonia', 'mass', 'effusion', 'obstruction',
                             'herniation', 'abscess', 'stenosis', 'aneurysm', 'embolism'],
                'symptoms': ['pain', 'swelling', 'deformity', 'limitation of movement'],
                'tests': ['x-ray', 'CT scan', 'MRI', 'ultrasound', 'fluoroscopy',
                         'mammography', 'contrast study', 'nuclear medicine scan'],
                'treatments': ['follow-up imaging', 'intervention radiology', 'guided biopsy',
                             'drainage procedure'],
                'findings': ['lucency', 'opacity', 'consolidation', 'fracture line',
                           'displacement', 'soft tissue swelling', 'no acute findings',
                           'contrast enhancement', 'architectural distortion', 'calcification']
            }
        }

        texts = []
        labels = []

        for _ in range(n_samples):
            category = np.random.choice(self.categories)
            vocab = medical_vocab[category]

            # Generate medical note
            note_parts = []

            # Chief complaint / History
            condition = np.random.choice(vocab['conditions'])
            symptoms = np.random.choice(vocab['symptoms'], np.random.randint(2, 4), replace=False)
            note_parts.append(f"Patient presents with {condition}.")
            note_parts.append(f"Reports {symptoms[0]} and {symptoms[1]}.")

            # Physical exam / Tests
            test = np.random.choice(vocab['tests'])
            finding = np.random.choice(vocab['findings'])
            note_parts.append(f"{test} reveals {finding}.")

            # Assessment and Plan
            treatment = np.random.choice(vocab['treatments'])
            note_parts.append(f"Diagnosis: {condition}.")
            note_parts.append(f"Plan: Initiate {treatment}.")

            # Follow-up
            if np.random.random() > 0.5:
                followup = np.random.choice(['Follow up in 2 weeks', 'Return if symptoms worsen',
                                            'Schedule repeat imaging', 'Monitor labs'])
                note_parts.append(followup)

            medical_text = ' '.join(note_parts)
            texts.append(medical_text)
            labels.append(category)

        return pd.DataFrame({
            'medical_text': texts,
            'specialty': labels
        })

    def extract_features(self, df):
        """Extract medical text features"""
        df = df.copy()

        # Text statistics
        df['text_length'] = df['medical_text'].apply(len)
        df['word_count'] = df['medical_text'].apply(lambda x: len(x.split()))
        df['sentence_count'] = df['medical_text'].apply(lambda x: x.count('.'))

        # Medical-specific features
        df['medication_mentions'] = df['medical_text'].str.count(
            r'therapy|medication|treatment|drug'
        )
        df['test_mentions'] = df['medical_text'].str.count(
            r'test|scan|imaging|study|exam'
        )
        df['symptom_mentions'] = df['medical_text'].str.count(
            r'pain|fever|fatigue|weakness|nausea'
        )
        df['diagnosis_present'] = df['medical_text'].str.contains('Diagnosis:').astype(int)

        return df

    def train(self, X_train, y_train):
        """Train gradient boosting classifier"""
        self.model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            subsample=0.8
        )

        self.model.fit(X_train, y_train)

        # Stratified cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=skf)
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
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
                   xticklabels=self.categories,
                   yticklabels=self.categories,
                   cbar_kws={'label': 'Number of Cases'})
        plt.title('Medical Text Classification - Confusion Matrix', fontsize=14, pad=20)
        plt.ylabel('Actual Specialty')
        plt.xlabel('Predicted Specialty')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('medical_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\nConfusion matrix saved as 'medical_confusion_matrix.png'")

        return predictions

    def visualize_data(self, df):
        """Visualize medical text distributions"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Specialty distribution
        df['specialty'].value_counts().plot(kind='bar', ax=axes[0, 0], color='mediumseagreen')
        axes[0, 0].set_title('Medical Specialty Distribution', fontsize=12, pad=10)
        axes[0, 0].set_xlabel('Specialty')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Word count by specialty
        df.groupby('specialty')['word_count'].mean().sort_values().plot(
            kind='barh', ax=axes[0, 1], color='steelblue'
        )
        axes[0, 1].set_title('Average Word Count by Specialty', fontsize=12, pad=10)
        axes[0, 1].set_xlabel('Average Words')

        # Test mentions by specialty
        df.groupby('specialty')['test_mentions'].mean().plot(
            kind='bar', ax=axes[1, 0], color='coral'
        )
        axes[1, 0].set_title('Average Test Mentions by Specialty', fontsize=12, pad=10)
        axes[1, 0].set_xlabel('Specialty')
        axes[1, 0].set_ylabel('Average Count')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Medication mentions by specialty
        df.groupby('specialty')['medication_mentions'].mean().plot(
            kind='bar', ax=axes[1, 1], color='orchid'
        )
        axes[1, 1].set_title('Average Medication Mentions by Specialty', fontsize=12, pad=10)
        axes[1, 1].set_xlabel('Specialty')
        axes[1, 1].set_ylabel('Average Count')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('medical_analysis.png', dpi=300, bbox_inches='tight')
        print("Visualization saved as 'medical_analysis.png'")


def main():
    """Main execution function"""
    print("=" * 70)
    print("Medical Text Classification")
    print("=" * 70)

    # Initialize classifier
    classifier = MedicalTextClassifier()

    # Create sample data
    print("\nCreating synthetic medical text data...")
    df = classifier.create_sample_data(n_samples=1000)
    print(f"Dataset size: {df.shape}")
    print(f"\nSample medical note:\n{df['medical_text'].iloc[0]}")
    print(f"Specialty: {df['specialty'].iloc[0]}")

    # Data exploration
    print(f"\n=== Specialty Distribution ===")
    print(df['specialty'].value_counts())

    # Extract features
    print("\nExtracting features...")
    df = classifier.extract_features(df)

    # Visualize data
    print("\nGenerating visualizations...")
    classifier.visualize_data(df)

    # Prepare data for modeling
    print("\nPreparing data for modeling...")
    X = classifier.vectorizer.fit_transform(df['medical_text'])
    y = df['specialty']

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
        sample_text = df['medical_text'].iloc[i]
        sample_vectorized = classifier.vectorizer.transform([sample_text])
        prediction = classifier.model.predict(sample_vectorized)[0]
        actual = df['specialty'].iloc[i]

        print(f"\nMedical Note {i+1}: {sample_text}")
        print(f"Predicted Specialty: {prediction}")
        print(f"Actual Specialty: {actual}")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
