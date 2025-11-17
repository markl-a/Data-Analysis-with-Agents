"""
Disease Prediction from Symptoms
=================================
Domain: Healthcare & Clinical Decision Support
Task: Multi-class disease classification from patient symptoms

This solution demonstrates:
- Symptom-based diagnostic modeling
- Handling sparse binary feature matrices
- Multiple ML algorithms comparison
- Probabilistic disease prediction
- Clinical decision trees
- Feature importance for medical interpretation
- Differential diagnosis ranking
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             f1_score, roc_auc_score, top_k_accuracy_score)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')


class DiseaseSymptomPredictor:
    """
    Clinical decision support system for disease prediction from symptoms.
    Implements multiple ML approaches with medical interpretability.
    """

    def __init__(self):
        self.models = {}
        self.symptom_list = []
        self.disease_list = []
        self.feature_importance = {}
        self.predictions = {}

    def generate_clinical_data(self, n_samples=3000):
        """
        Generate synthetic patient data with symptoms and diseases.
        Simulates realistic symptom-disease relationships.
        """
        np.random.seed(42)

        # Define diseases and their characteristic symptoms
        disease_symptoms = {
            'Common Cold': {
                'primary': ['runny_nose', 'sneezing', 'sore_throat', 'mild_fever'],
                'secondary': ['cough', 'fatigue', 'headache'],
                'probability': 0.20
            },
            'Influenza': {
                'primary': ['high_fever', 'body_aches', 'fatigue', 'headache'],
                'secondary': ['cough', 'sore_throat', 'chills'],
                'probability': 0.15
            },
            'Pneumonia': {
                'primary': ['high_fever', 'chest_pain', 'cough', 'shortness_breath'],
                'secondary': ['fatigue', 'chills', 'rapid_breathing'],
                'probability': 0.10
            },
            'COVID-19': {
                'primary': ['fever', 'dry_cough', 'loss_taste_smell', 'fatigue'],
                'secondary': ['shortness_breath', 'body_aches', 'sore_throat'],
                'probability': 0.12
            },
            'Migraine': {
                'primary': ['severe_headache', 'nausea', 'light_sensitivity'],
                'secondary': ['vomiting', 'visual_disturbances', 'dizziness'],
                'probability': 0.08
            },
            'Gastroenteritis': {
                'primary': ['diarrhea', 'vomiting', 'abdominal_pain', 'nausea'],
                'secondary': ['fever', 'fatigue', 'loss_appetite'],
                'probability': 0.10
            },
            'Bronchitis': {
                'primary': ['persistent_cough', 'mucus_production', 'chest_discomfort'],
                'secondary': ['fatigue', 'mild_fever', 'shortness_breath'],
                'probability': 0.08
            },
            'Urinary_Tract_Infection': {
                'primary': ['painful_urination', 'frequent_urination', 'pelvic_pain'],
                'secondary': ['fever', 'cloudy_urine', 'fatigue'],
                'probability': 0.07
            },
            'Allergic_Rhinitis': {
                'primary': ['sneezing', 'runny_nose', 'itchy_eyes', 'nasal_congestion'],
                'secondary': ['fatigue', 'headache'],
                'probability': 0.06
            },
            'Diabetes_Type2': {
                'primary': ['increased_thirst', 'frequent_urination', 'fatigue', 'blurred_vision'],
                'secondary': ['slow_healing', 'weight_loss', 'numbness_hands'],
                'probability': 0.04
            }
        }

        self.disease_list = list(disease_symptoms.keys())

        # Collect all unique symptoms
        all_symptoms = set()
        for disease_info in disease_symptoms.values():
            all_symptoms.update(disease_info['primary'])
            all_symptoms.update(disease_info['secondary'])
        self.symptom_list = sorted(list(all_symptoms))

        # Generate patient records
        patients = []
        disease_probs = [info['probability'] for info in disease_symptoms.values()]
        disease_probs = np.array(disease_probs) / sum(disease_probs)

        for i in range(n_samples):
            # Select disease
            disease = np.random.choice(self.disease_list, p=disease_probs)
            disease_info = disease_symptoms[disease]

            # Generate symptoms
            symptoms = []

            # Primary symptoms (high probability)
            for symptom in disease_info['primary']:
                if np.random.random() < 0.85:  # 85% chance
                    symptoms.append(symptom)

            # Secondary symptoms (moderate probability)
            for symptom in disease_info['secondary']:
                if np.random.random() < 0.50:  # 50% chance
                    symptoms.append(symptom)

            # Add noise (random symptoms)
            n_noise = np.random.choice([0, 0, 1, 1, 2])
            noise_symptoms = np.random.choice(
                [s for s in self.symptom_list if s not in symptoms],
                size=min(n_noise, len(self.symptom_list) - len(symptoms)),
                replace=False
            )
            symptoms.extend(noise_symptoms)

            # Patient metadata
            age = int(np.random.normal(45, 18))
            age = np.clip(age, 5, 90)

            patients.append({
                'patient_id': f'P_{i:05d}',
                'age': age,
                'gender': np.random.choice(['M', 'F']),
                'symptoms': symptoms,
                'num_symptoms': len(symptoms),
                'disease': disease,
                'severity': np.random.choice(['Mild', 'Moderate', 'Severe'], p=[0.5, 0.35, 0.15])
            })

        df = pd.DataFrame(patients)

        # Create binary symptom matrix
        symptom_matrix = np.zeros((n_samples, len(self.symptom_list)))
        for i, symptoms in enumerate(df['symptoms']):
            for symptom in symptoms:
                if symptom in self.symptom_list:
                    symptom_idx = self.symptom_list.index(symptom)
                    symptom_matrix[i, symptom_idx] = 1

        print(f"Generated {n_samples} patient records")
        print(f"Number of unique symptoms: {len(self.symptom_list)}")
        print(f"Number of diseases: {len(self.disease_list)}")
        print(f"\nDisease distribution:")
        print(df['disease'].value_counts())
        print(f"\nAverage symptoms per patient: {df['num_symptoms'].mean():.2f}")

        return symptom_matrix, df

    def train_models(self, X_train, y_train):
        """Train multiple classifiers for disease prediction."""
        print("\nTraining multiple models...")

        # 1. Random Forest
        print("  - Random Forest...")
        rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        self.models['Random Forest'] = rf
        self.feature_importance['Random Forest'] = rf.feature_importances_

        # 2. Gradient Boosting
        print("  - Gradient Boosting...")
        gb = GradientBoostingClassifier(n_estimators=150, max_depth=10, random_state=42)
        gb.fit(X_train, y_train)
        self.models['Gradient Boosting'] = gb
        self.feature_importance['Gradient Boosting'] = gb.feature_importances_

        # 3. Extra Trees
        print("  - Extra Trees...")
        et = ExtraTreesClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
        et.fit(X_train, y_train)
        self.models['Extra Trees'] = et
        self.feature_importance['Extra Trees'] = et.feature_importances_

        # 4. Logistic Regression
        print("  - Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
        lr.fit(X_train, y_train)
        self.models['Logistic Regression'] = lr

        # 5. Neural Network
        print("  - Neural Network...")
        nn = MLPClassifier(hidden_layers=(128, 64, 32), max_iter=500, random_state=42)
        nn.fit(X_train, y_train)
        self.models['Neural Network'] = nn

        # 6. Decision Tree (for interpretability)
        print("  - Decision Tree...")
        dt = DecisionTreeClassifier(max_depth=8, random_state=42)
        dt.fit(X_train, y_train)
        self.models['Decision Tree'] = dt
        self.feature_importance['Decision Tree'] = dt.feature_importances_

        print(f"\nTrained {len(self.models)} models successfully")

    def evaluate_models(self, X_test, y_test):
        """Evaluate all models on test set."""
        results = []

        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            # Top-3 accuracy (important in medical diagnosis)
            top3_acc = top_k_accuracy_score(y_test, y_pred_proba, k=3,
                                           labels=range(len(self.disease_list)))

            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'F1-Score': f1,
                'Top-3 Accuracy': top3_acc
            })

            self.predictions[name] = {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }

        results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
        return results_df

    def get_differential_diagnosis(self, symptoms, top_k=5):
        """
        Provide differential diagnosis with probabilities.
        Returns top K most likely diseases with confidence scores.
        """
        # Create symptom vector
        symptom_vector = np.zeros((1, len(self.symptom_list)))
        for symptom in symptoms:
            if symptom in self.symptom_list:
                idx = self.symptom_list.index(symptom)
                symptom_vector[0, idx] = 1

        # Get predictions from best model (Random Forest)
        model = self.models['Random Forest']
        probas = model.predict_proba(symptom_vector)[0]

        # Get top K diseases
        top_indices = np.argsort(probas)[::-1][:top_k]

        differential = []
        for idx in top_indices:
            differential.append({
                'disease': self.disease_list[idx],
                'probability': probas[idx],
                'confidence': 'High' if probas[idx] > 0.5 else 'Medium' if probas[idx] > 0.2 else 'Low'
            })

        return differential

    def plot_model_comparison(self, results_df):
        """Compare model performances."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        metrics = ['Accuracy', 'F1-Score', 'Top-3 Accuracy']
        colors = plt.cm.viridis(np.linspace(0, 1, len(results_df)))

        for idx, metric in enumerate(metrics):
            axes[idx].barh(results_df['Model'], results_df[metric], color=colors)
            axes[idx].set_xlabel(metric, fontsize=12)
            axes[idx].set_title(f'{metric} by Model', fontsize=14, fontweight='bold')
            axes[idx].grid(True, alpha=0.3, axis='x')

            # Add value labels
            for i, v in enumerate(results_df[metric]):
                axes[idx].text(v + 0.01, i, f'{v:.3f}', va='center')

        plt.tight_layout()
        plt.savefig('disease_model_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved: disease_model_comparison.png")
        plt.close()

    def plot_symptom_importance(self, top_n=20):
        """Plot most important symptoms for diagnosis."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.ravel()

        for idx, (model_name, importances) in enumerate(list(self.feature_importance.items())[:4]):
            # Get top N symptoms
            top_indices = np.argsort(importances)[::-1][:top_n]
            top_symptoms = [self.symptom_list[i].replace('_', ' ').title() for i in top_indices]
            top_values = importances[top_indices]

            axes[idx].barh(range(top_n), top_values, color=plt.cm.coolwarm(top_values / max(top_values)))
            axes[idx].set_yticks(range(top_n))
            axes[idx].set_yticklabels(top_symptoms, fontsize=9)
            axes[idx].set_xlabel('Importance Score', fontsize=11)
            axes[idx].set_title(f'Top {top_n} Symptoms - {model_name}', fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3, axis='x')
            axes[idx].invert_yaxis()

        plt.tight_layout()
        plt.savefig('disease_symptom_importance.png', dpi=300, bbox_inches='tight')
        print("Saved: disease_symptom_importance.png")
        plt.close()

    def plot_confusion_matrix(self, y_test, model_name='Random Forest'):
        """Plot confusion matrix for best model."""
        if model_name not in self.predictions:
            return

        y_pred = self.predictions[model_name]['y_pred']
        cm = confusion_matrix(y_test, y_pred)

        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd',
                   xticklabels=self.disease_list, yticklabels=self.disease_list,
                   ax=ax, cbar_kws={'label': 'Proportion'})

        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
        ax.set_ylabel('True Disease', fontsize=12)
        ax.set_xlabel('Predicted Disease', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig('disease_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("Saved: disease_confusion_matrix.png")
        plt.close()

    def plot_decision_tree(self, max_depth=4):
        """Visualize decision tree for interpretability."""
        if 'Decision Tree' not in self.models:
            return

        # Train a shallow tree for visualization
        dt_simple = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        # Use a subset of data for clarity
        dt_simple.fit(X_train[:500], y_train[:500])

        fig, ax = plt.subplots(figsize=(24, 12))
        plot_tree(dt_simple,
                 feature_names=[s.replace('_', ' ') for s in self.symptom_list],
                 class_names=self.disease_list,
                 filled=True,
                 rounded=True,
                 fontsize=8,
                 ax=ax)

        plt.title('Clinical Decision Tree for Disease Diagnosis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('disease_decision_tree.png', dpi=300, bbox_inches='tight')
        print("Saved: disease_decision_tree.png")
        plt.close()

    def plot_disease_symptom_heatmap(self, df):
        """Create heatmap showing symptom patterns for each disease."""
        # Calculate symptom frequency per disease
        disease_symptom_matrix = []

        for disease in self.disease_list:
            disease_patients = df[df['disease'] == disease]
            symptom_counts = np.zeros(len(self.symptom_list))

            for symptoms in disease_patients['symptoms']:
                for symptom in symptoms:
                    if symptom in self.symptom_list:
                        idx = self.symptom_list.index(symptom)
                        symptom_counts[idx] += 1

            # Normalize by number of patients
            symptom_freq = symptom_counts / len(disease_patients)
            disease_symptom_matrix.append(symptom_freq)

        # Select top symptoms (most variable across diseases)
        disease_symptom_matrix = np.array(disease_symptom_matrix)
        symptom_variance = np.var(disease_symptom_matrix, axis=0)
        top_symptom_indices = np.argsort(symptom_variance)[::-1][:25]

        # Create heatmap with top symptoms
        matrix_subset = disease_symptom_matrix[:, top_symptom_indices]
        symptom_labels = [self.symptom_list[i].replace('_', ' ').title() for i in top_symptom_indices]

        fig, ax = plt.subplots(figsize=(16, 10))
        sns.heatmap(matrix_subset, annot=True, fmt='.2f', cmap='RdYlBu_r',
                   xticklabels=symptom_labels, yticklabels=self.disease_list,
                   ax=ax, cbar_kws={'label': 'Symptom Frequency'})

        ax.set_title('Disease-Symptom Association Patterns', fontsize=16, fontweight='bold')
        ax.set_xlabel('Symptoms', fontsize=12)
        ax.set_ylabel('Diseases', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig('disease_symptom_heatmap.png', dpi=300, bbox_inches='tight')
        print("Saved: disease_symptom_heatmap.png")
        plt.close()


def main():
    """Main execution function."""
    print("=" * 80)
    print("Disease Prediction from Symptoms - Clinical Decision Support System")
    print("=" * 80)

    # Initialize predictor
    predictor = DiseaseSymptomPredictor()

    # Generate data
    print("\n1. Generating Clinical Data...")
    X, df = predictor.generate_clinical_data(n_samples=3000)
    y = df['disease'].values

    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"\nData split: {len(X_train)} train, {len(X_test)} test")

    # Train models
    print("\n2. Training Classification Models...")
    predictor.train_models(X_train, y_train)

    # Evaluate
    print("\n3. Evaluating Models...")
    results = predictor.evaluate_models(X_test, y_test)
    print("\nModel Performance:")
    print(results.to_string(index=False))

    # Example differential diagnosis
    print("\n4. Example Differential Diagnosis...")
    example_symptoms = ['high_fever', 'body_aches', 'fatigue', 'cough']
    print(f"\nPatient symptoms: {example_symptoms}")
    differential = predictor.get_differential_diagnosis(example_symptoms, top_k=5)
    print("\nDifferential Diagnosis (Top 5):")
    for i, diag in enumerate(differential, 1):
        print(f"  {i}. {diag['disease']}: {diag['probability']:.3f} ({diag['confidence']} confidence)")

    # Visualizations
    print("\n5. Generating Visualizations...")
    predictor.plot_model_comparison(results)
    predictor.plot_symptom_importance(top_n=20)
    predictor.plot_confusion_matrix(y_test, 'Random Forest')
    predictor.plot_decision_tree(max_depth=4)
    predictor.plot_disease_symptom_heatmap(df)

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print("\nKey Insights:")
    print("- Symptom-based diagnosis achieves high accuracy with ensemble methods")
    print("- Top-3 accuracy important for differential diagnosis in clinical practice")
    print("- Feature importance reveals critical diagnostic symptoms per disease")
    print("- Decision trees provide interpretable clinical pathways")
    print("- System can support medical decision-making with probabilistic outputs")


if __name__ == "__main__":
    main()
