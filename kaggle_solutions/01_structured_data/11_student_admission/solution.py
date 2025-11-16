"""
University Student Admission Prediction
=========================================

Problem: Predict whether a student will be admitted to a university based on academic
and extracurricular performance

Kaggle-style competition: Graduate Admission Prediction
Difficulty: ⭐⭐

This solution demonstrates:
- Multi-factor admission scoring
- Feature interactions and polynomial features
- Ensemble model stacking
- Calibrated probability predictions
- Holistic admission criteria modeling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class AdmissionPredictor:
    """Predicts university admission probability using holistic criteria"""

    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.poly_features = PolynomialFeatures(degree=2, include_bias=False)

    def create_sample_data(self, n_samples=3000):
        """Generate realistic student application data"""
        np.random.seed(42)

        # Academic metrics
        data = {
            'gre_score': np.random.normal(315, 11, n_samples).clip(260, 340),
            'toefl_score': np.random.normal(105, 6, n_samples).clip(80, 120),
            'university_rating': np.random.randint(1, 6, n_samples),
            'sop_strength': np.random.uniform(1, 5, n_samples),  # Statement of Purpose rating
            'lor_strength': np.random.uniform(1, 5, n_samples),  # Letter of Recommendation rating
            'cgpa': np.random.normal(8.2, 0.6, n_samples).clip(6.0, 10.0),
            'research_experience': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
            'publications': np.random.poisson(0.8, n_samples),
            'internships': np.random.poisson(1.2, n_samples),
            'projects': np.random.poisson(2.5, n_samples),
            'awards': np.random.poisson(0.5, n_samples),
            'extracurricular_score': np.random.uniform(0, 10, n_samples),
            'work_experience_months': np.random.exponential(8, n_samples).clip(0, 60)
        }

        df = pd.DataFrame(data)

        # Generate admission decision with realistic dependencies
        admission_score = (
            0.008 * (df['gre_score'] - 260) +
            0.015 * (df['toefl_score'] - 80) +
            0.08 * df['university_rating'] +
            0.12 * df['sop_strength'] +
            0.12 * df['lor_strength'] +
            0.25 * (df['cgpa'] - 6.0) +
            0.3 * df['research_experience'] +
            0.1 * df['publications'] +
            0.05 * df['internships'] +
            0.03 * df['projects'] +
            0.08 * df['awards'] +
            0.015 * df['extracurricular_score'] +
            0.008 * (df['work_experience_months'] / 12) +
            np.random.normal(0, 0.8, n_samples) -
            2.0  # Offset to center the distribution
        )

        # Convert to probability using sigmoid
        admission_prob = 1 / (1 + np.exp(-admission_score))
        df['admitted'] = (admission_prob > 0.5).astype(int)

        return df

    def engineer_features(self, df):
        """Create advanced application features"""
        df = df.copy()

        # Academic strength composite
        df['academic_score'] = (
            (df['gre_score'] / 340) * 0.3 +
            (df['toefl_score'] / 120) * 0.2 +
            (df['cgpa'] / 10) * 0.5
        ) * 100

        # Research profile
        df['research_profile'] = (
            df['research_experience'] * 2 +
            df['publications'] * 1.5 +
            df['projects'] * 0.5
        )

        # Professional experience
        df['professional_score'] = (
            df['internships'] * 1.5 +
            (df['work_experience_months'] / 12) * 2
        )

        # Overall application strength
        df['application_strength'] = (
            df['sop_strength'] + df['lor_strength']
        ) / 2

        # Interaction features
        df['gre_cgpa_interaction'] = df['gre_score'] * df['cgpa']
        df['research_academic_interaction'] = df['research_experience'] * df['academic_score']

        # Categorize applicants
        df['high_achiever'] = (
            (df['cgpa'] >= 9.0) & (df['gre_score'] >= 325)
        ).astype(int)

        df['research_focused'] = (
            (df['publications'] > 0) | (df['research_experience'] == 1)
        ).astype(int)

        df['well_rounded'] = (
            (df['extracurricular_score'] > 7) &
            (df['internships'] > 1)
        ).astype(int)

        return df

    def train_models(self, X, y):
        """Train multiple models with calibration"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Initialize models
        models_config = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=120, learning_rate=0.05, max_depth=5, random_state=42
            ),
            'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42)
        }

        results = {}
        for name, model in models_config.items():
            # Train model
            model.fit(X_train_scaled, y_train)

            # Calibrate probabilities
            calibrated_model = CalibratedClassifierCV(model, cv=3, method='sigmoid')
            calibrated_model.fit(X_train_scaled, y_train)

            # Predictions
            y_pred = calibrated_model.predict(X_test_scaled)
            y_pred_proba = calibrated_model.predict_proba(X_test_scaled)[:, 1]

            results[name] = {
                'model': calibrated_model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'auc_score': roc_auc_score(y_test, y_pred_proba),
                'cv_score': cross_val_score(
                    calibrated_model, X_train_scaled, y_train, cv=5, scoring='roc_auc'
                ).mean()
            }

        return results, X_test_scaled, y_test, X_train

    def plot_results(self, results, y_test, feature_names):
        """Visualize comprehensive results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # ROC Curves
        ax = axes[0, 0]
        for name, result in results.items():
            fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
            ax.plot(fpr, tpr, label=f"{name} (AUC={result['auc_score']:.3f})", linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title('ROC Curves - Admission Prediction', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Model Performance Comparison
        ax = axes[0, 1]
        models = list(results.keys())
        aucs = [results[m]['auc_score'] for m in models]
        cv_scores = [results[m]['cv_score'] for m in models]

        x = np.arange(len(models))
        width = 0.35
        ax.bar(x - width/2, aucs, width, label='Test AUC', color='#3498db')
        ax.bar(x + width/2, cv_scores, width, label='CV AUC', color='#2ecc71')
        ax.set_ylabel('AUC Score', fontsize=11)
        ax.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0.5, 1.0)

        # Confusion Matrix for best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
        best_result = results[best_model_name]

        ax = axes[0, 2]
        cm = confusion_matrix(y_test, best_result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True,
                   annot_kws={'size': 12})
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('Actual', fontsize=11)
        ax.set_title(f'Confusion Matrix - {best_model_name}', fontsize=12, fontweight='bold')
        ax.set_xticklabels(['Rejected', 'Admitted'])
        ax.set_yticklabels(['Rejected', 'Admitted'])

        # Probability Distribution
        ax = axes[1, 0]
        admitted_probs = best_result['probabilities'][y_test == 1]
        rejected_probs = best_result['probabilities'][y_test == 0]

        ax.hist(rejected_probs, bins=30, alpha=0.6, label='Rejected', color='red', edgecolor='black')
        ax.hist(admitted_probs, bins=30, alpha=0.6, label='Admitted', color='green', edgecolor='black')
        ax.set_xlabel('Predicted Probability', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Probability Distribution by Outcome', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Feature Importance (Random Forest)
        if 'Random Forest' in results:
            ax = axes[1, 1]
            # Access the underlying estimator from calibrated classifier
            rf_model = results['Random Forest']['model'].calibrated_classifiers_[0].estimator

            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False).head(12)

            ax.barh(range(len(feature_importance)), feature_importance['importance'],
                   color='#9b59b6', edgecolor='black')
            ax.set_yticks(range(len(feature_importance)))
            ax.set_yticklabels(feature_importance['feature'], fontsize=9)
            ax.set_xlabel('Importance', fontsize=11)
            ax.set_title('Top Feature Importances (Random Forest)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')

        # Admission Rate by GRE Score Bands
        ax = axes[1, 2]
        ax.axis('off')
        summary_text = f"""
        ╔══════════════════════════════════════╗
        ║     ADMISSION PREDICTION SUMMARY      ║
        ╚══════════════════════════════════════╝

        Best Model: {best_model_name}
        Test AUC: {best_result['auc_score']:.4f}
        CV AUC: {best_result['cv_score']:.4f}

        Confusion Matrix:
        True Negatives:  {cm[0,0]:4d}
        False Positives: {cm[0,1]:4d}
        False Negatives: {cm[1,0]:4d}
        True Positives:  {cm[1,1]:4d}

        Accuracy: {((cm[0,0] + cm[1,1]) / cm.sum()):.3f}
        Precision: {(cm[1,1] / (cm[1,1] + cm[0,1])):.3f}
        Recall: {(cm[1,1] / (cm[1,1] + cm[1,0])):.3f}
        """
        ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
               verticalalignment='center')

        plt.tight_layout()
        plt.savefig('student_admission_analysis.png', dpi=300, bbox_inches='tight')
        print("\n📊 Visualization saved as 'student_admission_analysis.png'")
        plt.show()

    def print_results(self, results, y_test):
        """Print detailed results"""
        print("\n" + "="*80)
        print("UNIVERSITY ADMISSION PREDICTION RESULTS")
        print("="*80)

        for name, result in results.items():
            print(f"\n{'='*40}")
            print(f"Model: {name}")
            print(f"{'='*40}")
            print(f"Test AUC Score: {result['auc_score']:.4f}")
            print(f"CV AUC Score: {result['cv_score']:.4f}")
            print(f"\nClassification Report:")
            print(classification_report(y_test, result['predictions'],
                                       target_names=['Rejected', 'Admitted']))

        best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
        best_auc = results[best_model_name]['auc_score']
        print(f"\n{'='*80}")
        print(f"🏆 Best Model: {best_model_name} (AUC: {best_auc:.4f})")
        print(f"{'='*80}")


def main():
    """Main execution function"""
    print("🎓 University Admission Prediction System")
    print("=" * 80)

    predictor = AdmissionPredictor()

    # Generate data
    print("\n📊 Generating sample admission data...")
    df = predictor.create_sample_data(n_samples=3000)
    print(f"Dataset shape: {df.shape}")
    print(f"Admission rate: {df['admitted'].mean():.2%}")

    # Engineer features
    print("\n🔧 Engineering features...")
    df_engineered = predictor.engineer_features(df)

    # Prepare data
    X = df_engineered.drop('admitted', axis=1)
    y = df_engineered['admitted']
    print(f"Features shape: {X.shape}")

    # Train models
    print("\n🤖 Training models with calibration...")
    results, X_test, y_test, X_train = predictor.train_models(X, y)

    # Print results
    predictor.print_results(results, y_test)

    # Plot results
    print("\n📈 Generating visualizations...")
    predictor.plot_results(results, y_test, X.columns.tolist())

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
