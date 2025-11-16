"""
Hospital Readmission Prediction
================================

Problem: Predict 30-day hospital readmission risk for patients to enable
proactive intervention and reduce healthcare costs

Kaggle-style competition: Healthcare Readmission Prediction
Difficulty: ⭐⭐⭐

This solution demonstrates:
- Healthcare risk stratification
- Comorbidity scoring (Charlson Index)
- Time-to-event features
- Cost-benefit analysis
- Clinical decision support
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class HealthcareReadmissionPredictor:
    """Predicts 30-day hospital readmission risk"""

    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.readmission_cost = 15000  # Average cost of readmission

    def create_sample_data(self, n_samples=6000):
        """Generate realistic hospital patient data"""
        np.random.seed(42)

        # Patient demographics
        data = {
            'patient_id': range(1, n_samples + 1),
            'age': np.random.normal(65, 15, n_samples).clip(18, 100),
            'gender': np.random.choice(['M', 'F'], n_samples, p=[0.48, 0.52]),
            'race': np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'],
                                    n_samples, p=[0.6, 0.2, 0.12, 0.05, 0.03]),

            # Admission details
            'admission_type': np.random.choice(['Emergency', 'Urgent', 'Elective'],
                                              n_samples, p=[0.5, 0.3, 0.2]),
            'discharge_disposition': np.random.choice(['Home', 'SNF', 'Home_Health', 'AMA'],
                                                     n_samples, p=[0.6, 0.25, 0.12, 0.03]),
            'length_of_stay': np.random.lognormal(1.5, 0.8, n_samples).clip(1, 30),
            'num_procedures': np.random.poisson(2, n_samples),
            'num_medications': np.random.poisson(8, n_samples) + 1,
            'num_lab_procedures': np.random.poisson(15, n_samples),

            # Comorbidities (Charlson Comorbidity Index components)
            'diabetes': np.random.choice([0, 1], n_samples, p=[0.65, 0.35]),
            'hypertension': np.random.choice([0, 1], n_samples, p=[0.45, 0.55]),
            'heart_failure': np.random.choice([0, 1], n_samples, p=[0.80, 0.20]),
            'copd': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
            'kidney_disease': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'liver_disease': np.random.choice([0, 1], n_samples, p=[0.92, 0.08]),
            'cancer': np.random.choice([0, 1], n_samples, p=[0.88, 0.12]),
            'stroke': np.random.choice([0, 1], n_samples, p=[0.90, 0.10]),

            # Hospital utilization history
            'num_prior_admissions': np.random.poisson(1.5, n_samples),
            'num_emergency_visits': np.random.poisson(1, n_samples),
            'num_outpatient_visits': np.random.poisson(3, n_samples),
            'days_since_last_admission': np.random.exponential(180, n_samples).clip(0, 730),

            # Clinical measurements
            'num_diagnoses': np.random.poisson(5, n_samples) + 1,
            'primary_diagnosis_severity': np.random.choice(['Minor', 'Moderate', 'Major', 'Extreme'],
                                                          n_samples, p=[0.2, 0.4, 0.3, 0.1]),
            'medication_changes': np.random.poisson(3, n_samples),
            'insulin_prescribed': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),

            # Social determinants
            'insurance_type': np.random.choice(['Medicare', 'Medicaid', 'Private', 'Self_Pay'],
                                              n_samples, p=[0.5, 0.2, 0.25, 0.05]),
            'has_caregiver': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
            'distance_to_hospital_miles': np.random.exponential(15, n_samples).clip(1, 100),
            'socioeconomic_score': np.random.normal(50, 20, n_samples).clip(0, 100)
        }

        df = pd.DataFrame(data)

        # Generate readmission with realistic clinical dependencies
        readmission_score = (
            0.02 * df['age'] +
            0.4 * (df['admission_type'] == 'Emergency').astype(int) +
            0.5 * (df['discharge_disposition'] == 'AMA').astype(int) +
            0.3 * (df['discharge_disposition'] == 'SNF').astype(int) +
            0.08 * df['length_of_stay'] +
            0.15 * df['num_procedures'] +
            0.05 * df['num_medications'] +
            0.6 * df['diabetes'] +
            0.5 * df['heart_failure'] +
            0.4 * df['kidney_disease'] +
            0.3 * df['copd'] +
            0.3 * df['stroke'] +
            0.2 * df['cancer'] +
            0.25 * df['num_prior_admissions'] +
            0.3 * df['num_emergency_visits'] +
            -0.01 * df['days_since_last_admission'] +
            0.1 * df['num_diagnoses'] +
            0.4 * (df['primary_diagnosis_severity'] == 'Extreme').astype(int) +
            0.2 * (df['primary_diagnosis_severity'] == 'Major').astype(int) +
            0.3 * df['medication_changes'] +
            0.2 * df['insulin_prescribed'] +
            0.3 * (df['insurance_type'] == 'Medicaid').astype(int) +
            -0.4 * df['has_caregiver'] +
            0.01 * df['distance_to_hospital_miles'] +
            -0.01 * df['socioeconomic_score'] +
            np.random.normal(0, 1, n_samples)
        )

        # Convert to probability
        readmission_prob = 1 / (1 + np.exp(-readmission_score))
        df['readmitted_30day'] = (readmission_prob > 0.65).astype(int)

        return df

    def calculate_charlson_index(self, df):
        """Calculate Charlson Comorbidity Index"""
        df = df.copy()

        # Charlson weights (simplified)
        charlson = (
            df['diabetes'] * 1 +
            df['heart_failure'] * 1 +
            df['copd'] * 1 +
            df['kidney_disease'] * 2 +
            df['liver_disease'] * 1 +
            df['cancer'] * 2 +
            df['stroke'] * 1
        )

        # Age adjustment
        age_score = ((df['age'] - 40) / 10).clip(0, 4).astype(int)
        df['charlson_index'] = charlson + age_score

        return df

    def engineer_features(self, df):
        """Create advanced healthcare features"""
        df = df.copy()

        # Calculate Charlson Index
        df = self.calculate_charlson_index(df)

        # Comorbidity burden
        df['num_comorbidities'] = (
            df['diabetes'] + df['hypertension'] + df['heart_failure'] +
            df['copd'] + df['kidney_disease'] + df['liver_disease'] +
            df['cancer'] + df['stroke']
        )

        df['high_comorbidity_burden'] = (df['num_comorbidities'] >= 3).astype(int)

        # Hospital utilization intensity
        df['total_prior_encounters'] = (
            df['num_prior_admissions'] +
            df['num_emergency_visits'] +
            df['num_outpatient_visits']
        )

        df['utilization_score'] = (
            df['num_prior_admissions'] * 3 +
            df['num_emergency_visits'] * 2 +
            df['num_outpatient_visits']
        )

        df['frequent_flyer'] = (df['num_prior_admissions'] >= 2).astype(int)

        # Clinical complexity
        df['clinical_complexity'] = (
            df['num_diagnoses'] * 0.3 +
            df['num_procedures'] * 0.4 +
            df['num_medications'] * 0.1 +
            df['num_lab_procedures'] * 0.05
        )

        df['high_medication_count'] = (df['num_medications'] > 10).astype(int)
        df['polypharmacy'] = (df['num_medications'] > 5).astype(int)

        # Length of stay categories
        df['extended_stay'] = (df['length_of_stay'] > 7).astype(int)
        df['los_category'] = pd.cut(df['length_of_stay'],
                                     bins=[0, 3, 7, 14, 100],
                                     labels=['short', 'medium', 'long', 'very_long'])

        # Risk factors
        df['high_risk_discharge'] = (
            (df['discharge_disposition'] == 'AMA') |
            (df['discharge_disposition'] == 'SNF')
        ).astype(int)

        df['emergency_admission'] = (df['admission_type'] == 'Emergency').astype(int)

        # Continuity of care
        df['recent_admission'] = (df['days_since_last_admission'] < 90).astype(int)
        df['recency_score'] = 1 / (df['days_since_last_admission'] + 1)

        # Medication management
        df['diabetes_medication_flag'] = (
            df['diabetes'] & df['insulin_prescribed']
        ).astype(int)

        df['medication_burden'] = (
            df['num_medications'] + df['medication_changes'] * 0.5
        )

        # Social risk factors
        df['social_risk_score'] = (
            (1 - df['has_caregiver']) * 2 +
            (df['insurance_type'] == 'Medicaid').astype(int) * 1.5 +
            (df['insurance_type'] == 'Self_Pay').astype(int) * 2 +
            (df['distance_to_hospital_miles'] > 30).astype(int) +
            (100 - df['socioeconomic_score']) / 50
        )

        # Age risk categories
        df['elderly'] = (df['age'] >= 65).astype(int)
        df['very_elderly'] = (df['age'] >= 80).astype(int)

        # Composite risk score
        df['composite_risk_score'] = (
            df['charlson_index'] * 0.3 +
            df['utilization_score'] * 0.2 +
            df['clinical_complexity'] * 0.2 +
            df['social_risk_score'] * 0.15 +
            df['num_comorbidities'] * 0.15
        )

        return df

    def train_models(self, X, y):
        """Train multiple clinical prediction models"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Initialize models
        models_config = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, class_weight='balanced', C=0.3, random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=18, min_samples_split=15,
                min_samples_leaf=5, class_weight='balanced', random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=180, learning_rate=0.08, max_depth=7,
                subsample=0.8, random_state=42
            ),
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=200, max_depth=16, min_samples_split=10,
                class_weight='balanced', random_state=42
            )
        }

        results = {}
        for name, model in models_config.items():
            # Train model
            model.fit(X_train_scaled, y_train)

            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

            # Cross-validation
            cv = StratifiedKFold(5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train_scaled, y_train,
                                       cv=cv, scoring='roc_auc')

            # Calculate cost savings
            cost_savings = self.calculate_cost_benefit(y_test, y_pred, y_pred_proba)

            results[name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'auc_score': roc_auc_score(y_test, y_pred_proba),
                'avg_precision': average_precision_score(y_test, y_pred_proba),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cost_savings': cost_savings
            }

        return results, X_test_scaled, y_test, X_train

    def calculate_cost_benefit(self, y_true, y_pred, y_pred_proba):
        """Calculate potential cost savings from intervention"""
        # Assume intervention costs $500 per patient
        intervention_cost = 500

        # Assume intervention prevents 40% of readmissions
        intervention_effectiveness = 0.4

        # Identify high-risk patients (predicted probability > 0.5)
        high_risk = y_pred_proba > 0.5

        # True positives: Correctly identified high-risk (can intervene)
        tp = np.sum((y_true == 1) & (high_risk == 1))
        prevented_readmissions = tp * intervention_effectiveness

        # Cost savings from prevented readmissions
        savings = prevented_readmissions * self.readmission_cost

        # Cost of interventions
        intervention_total_cost = np.sum(high_risk) * intervention_cost

        # Net benefit
        net_benefit = savings - intervention_total_cost

        return {
            'high_risk_identified': np.sum(high_risk),
            'true_positives': tp,
            'prevented_readmissions': prevented_readmissions,
            'gross_savings': savings,
            'intervention_cost': intervention_total_cost,
            'net_benefit': net_benefit,
            'roi': (savings / intervention_total_cost - 1) * 100 if intervention_total_cost > 0 else 0
        }

    def plot_results(self, results, y_test, feature_names):
        """Visualize comprehensive healthcare prediction results"""
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        # ROC Curves
        ax1 = fig.add_subplot(gs[0, 0])
        for name, result in results.items():
            fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
            ax1.plot(fpr, tpr, label=f"{name}\n(AUC={result['auc_score']:.3f})",
                    linewidth=2.5)
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1.5)
        ax1.set_xlabel('False Positive Rate', fontsize=12)
        ax1.set_ylabel('True Positive Rate', fontsize=12)
        ax1.set_title('ROC Curves - 30-Day Readmission', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9, loc='lower right')
        ax1.grid(True, alpha=0.3)

        # Precision-Recall Curves
        ax2 = fig.add_subplot(gs[0, 1])
        for name, result in results.items():
            precision, recall, _ = precision_recall_curve(y_test, result['probabilities'])
            ax2.plot(recall, precision,
                    label=f"{name}\n(AP={result['avg_precision']:.3f})",
                    linewidth=2.5)
        ax2.set_xlabel('Recall (Sensitivity)', fontsize=12)
        ax2.set_ylabel('Precision (PPV)', fontsize=12)
        ax2.set_title('Precision-Recall Curves', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=9, loc='upper right')
        ax2.grid(True, alpha=0.3)

        # Cost-Benefit Analysis
        ax3 = fig.add_subplot(gs[0, 2])
        models = list(results.keys())
        net_benefits = [results[m]['cost_savings']['net_benefit'] for m in models]
        colors_cb = ['green' if nb > 0 else 'red' for nb in net_benefits]

        bars = ax3.barh(models, net_benefits, color=colors_cb, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Net Benefit ($)', fontsize=12)
        ax3.set_title('Cost-Benefit Analysis', fontsize=13, fontweight='bold')
        ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)

        for bar, nb in zip(bars, net_benefits):
            ax3.text(nb, bar.get_y() + bar.get_height()/2,
                    f'${nb:,.0f}', ha='left' if nb > 0 else 'right',
                    va='center', fontweight='bold', fontsize=9)
        ax3.grid(True, alpha=0.3, axis='x')

        # Confusion Matrix
        best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
        best_result = results[best_model_name]

        ax4 = fig.add_subplot(gs[1, 0])
        cm = confusion_matrix(y_test, best_result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r', ax=ax4,
                   annot_kws={'size': 14}, cbar_kws={'label': 'Count'})
        ax4.set_xlabel('Predicted', fontsize=12)
        ax4.set_ylabel('Actual', fontsize=12)
        ax4.set_title(f'Confusion Matrix - {best_model_name}',
                     fontsize=13, fontweight='bold')
        ax4.set_xticklabels(['No Readmit', 'Readmit'])
        ax4.set_yticklabels(['No Readmit', 'Readmit'])

        # Risk Distribution
        ax5 = fig.add_subplot(gs[1, 1])
        readmit_probs = best_result['probabilities'][y_test == 1]
        no_readmit_probs = best_result['probabilities'][y_test == 0]

        ax5.hist(no_readmit_probs, bins=40, alpha=0.65, label='No Readmission',
                color='green', edgecolor='black')
        ax5.hist(readmit_probs, bins=40, alpha=0.65, label='Readmitted',
                color='red', edgecolor='black')
        ax5.axvline(x=0.5, color='black', linestyle='--', linewidth=2,
                   label='Intervention Threshold')
        ax5.set_xlabel('Predicted Readmission Risk', fontsize=12)
        ax5.set_ylabel('Frequency', fontsize=12)
        ax5.set_title('Risk Score Distribution', fontsize=13, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3, axis='y')

        # Feature Importance
        if 'Random Forest' in results:
            ax6 = fig.add_subplot(gs[1, 2])
            rf_model = results['Random Forest']['model']

            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)

            colors_feat = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(feature_importance)))
            ax6.barh(range(len(feature_importance)), feature_importance['importance'],
                    color=colors_feat, edgecolor='black')
            ax6.set_yticks(range(len(feature_importance)))
            ax6.set_yticklabels(feature_importance['feature'], fontsize=9)
            ax6.set_xlabel('Importance', fontsize=12)
            ax6.set_title('Top 15 Clinical Risk Factors', fontsize=13, fontweight='bold')
            ax6.grid(True, alpha=0.3, axis='x')

        # ROI Comparison
        ax7 = fig.add_subplot(gs[2, 0])
        rois = [results[m]['cost_savings']['roi'] for m in models]
        ax7.bar(range(len(models)), rois, color='#3498db', alpha=0.7, edgecolor='black')
        ax7.set_xticks(range(len(models)))
        ax7.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        ax7.set_ylabel('ROI (%)', fontsize=12)
        ax7.set_title('Return on Investment', fontsize=13, fontweight='bold')
        ax7.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax7.grid(True, alpha=0.3, axis='y')

        for i, roi in enumerate(rois):
            ax7.text(i, roi, f'{roi:.1f}%', ha='center',
                    va='bottom' if roi > 0 else 'top', fontweight='bold')

        # Risk Stratification
        ax8 = fig.add_subplot(gs[2, 1])
        risk_bins = [0, 0.3, 0.5, 0.7, 1.0]
        risk_labels = ['Low', 'Medium', 'High', 'Very High']

        risk_categories = pd.cut(best_result['probabilities'], bins=risk_bins, labels=risk_labels)
        risk_counts = risk_categories.value_counts().sort_index()

        colors_risk = ['green', 'yellow', 'orange', 'red']
        ax8.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
               colors=colors_risk, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax8.set_title('Patient Risk Stratification', fontsize=13, fontweight='bold')

        # Summary Statistics
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')

        cb = best_result['cost_savings']
        accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
        precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
        recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0

        summary_text = f"""
╔════════════════════════════════════════╗
║   READMISSION PREDICTION SUMMARY       ║
╚════════════════════════════════════════╝

Best Model: {best_model_name}

Performance:
  AUC:           {best_result['auc_score']:.4f}
  Avg Precision: {best_result['avg_precision']:.4f}
  Accuracy:      {accuracy:.4f}
  Sensitivity:   {recall:.4f}
  Specificity:   {cm[0,0]/(cm[0,0]+cm[0,1]):.4f}

Economic Impact (per 1000 patients):
  High Risk:     {cb['high_risk_identified']:.0f}
  Prevented:     {cb['prevented_readmissions']:.0f}
  Savings:       ${cb['gross_savings']:,.0f}
  Cost:          ${cb['intervention_cost']:,.0f}
  Net Benefit:   ${cb['net_benefit']:,.0f}
  ROI:           {cb['roi']:.1f}%

Baseline Readmission Rate: {y_test.mean():.1%}
        """

        ax9.text(0.05, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round',
                facecolor='lightblue', alpha=0.3))

        plt.savefig('healthcare_readmission_analysis.png', dpi=300, bbox_inches='tight')
        print("\n📊 Visualization saved as 'healthcare_readmission_analysis.png'")
        plt.show()

    def print_results(self, results, y_test):
        """Print detailed clinical results"""
        print("\n" + "="*80)
        print("HOSPITAL 30-DAY READMISSION PREDICTION RESULTS")
        print("="*80)

        for name, result in results.items():
            print(f"\n{'='*40}")
            print(f"Model: {name}")
            print(f"{'='*40}")
            print(f"ROC-AUC: {result['auc_score']:.4f}")
            print(f"Average Precision: {result['avg_precision']:.4f}")
            print(f"CV Score: {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})")

            cb = result['cost_savings']
            print(f"\nCost-Benefit Analysis:")
            print(f"  High-risk identified: {cb['high_risk_identified']}")
            print(f"  Prevented readmissions: {cb['prevented_readmissions']:.1f}")
            print(f"  Gross savings: ${cb['gross_savings']:,.2f}")
            print(f"  Intervention cost: ${cb['intervention_cost']:,.2f}")
            print(f"  Net benefit: ${cb['net_benefit']:,.2f}")
            print(f"  ROI: {cb['roi']:.1f}%")

            print(f"\nClassification Report:")
            print(classification_report(y_test, result['predictions'],
                                       target_names=['No Readmit', 'Readmit']))

        best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
        best_auc = results[best_model_name]['auc_score']
        print(f"\n{'='*80}")
        print(f"🏆 Best Model: {best_model_name} (AUC: {best_auc:.4f})")
        print(f"{'='*80}")


def main():
    """Main execution function"""
    print("🏥 Hospital 30-Day Readmission Prediction")
    print("=" * 80)

    predictor = HealthcareReadmissionPredictor()

    # Generate data
    print("\n📊 Generating patient readmission data...")
    df = predictor.create_sample_data(n_samples=6000)
    print(f"Dataset shape: {df.shape}")
    print(f"30-day readmission rate: {df['readmitted_30day'].mean():.2%}")

    # Engineer features
    print("\n🔧 Engineering clinical features...")
    df_engineered = predictor.engineer_features(df)
    print(f"Average Charlson Index: {df_engineered['charlson_index'].mean():.2f}")

    # Prepare data
    exclude_cols = ['patient_id', 'readmitted_30day', 'los_category']
    X = df_engineered.drop(exclude_cols, axis=1)

    # Encode categorical variables
    categorical_cols = ['gender', 'race', 'admission_type', 'discharge_disposition',
                       'primary_diagnosis_severity', 'insurance_type']
    for col in categorical_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            predictor.label_encoders[col] = le

    y = df_engineered['readmitted_30day']
    print(f"Features shape: {X.shape}")

    # Train models
    print("\n🤖 Training readmission prediction models...")
    results, X_test, y_test, X_train = predictor.train_models(X, y)

    # Print results
    predictor.print_results(results, y_test)

    # Plot results
    print("\n📈 Generating clinical visualizations...")
    predictor.plot_results(results, y_test, X.columns.tolist())

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
