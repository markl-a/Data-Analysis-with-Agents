"""
Employee Job Change Prediction
===============================

This solution predicts whether employees will change jobs using Random Forest
with comprehensive career trajectory and satisfaction features.

Business Context:
- Employee turnover costs companies 50-200% of annual salary
- Early identification enables retention interventions
- Data science talent particularly prone to job changes
- Predictive models help optimize retention strategies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def generate_employee_data(n_samples=3500):
    """
    Generate realistic employee job change data

    Features include:
    - Demographics and education
    - Work experience and career progression
    - Training and development
    - Company engagement
    - Job search activity
    """
    print("Generating employee job change data...")

    # Demographics
    ages = np.random.normal(32, 8, n_samples).clip(22, 60).astype(int)

    # City development index (normalized city tier)
    city_development_index = np.random.choice(
        [0.920, 0.880, 0.820, 0.740, 0.650, 0.520],
        n_samples,
        p=[0.15, 0.20, 0.25, 0.20, 0.15, 0.05]
    )

    # Gender
    gender = np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.70, 0.28, 0.02])

    # Education level
    education_level = np.random.choice(
        ['Graduate', 'Masters', 'High School', 'Phd', 'Primary'],
        n_samples,
        p=[0.50, 0.35, 0.10, 0.04, 0.01]
    )

    # Major discipline
    major_discipline = np.random.choice(
        ['STEM', 'Business', 'Humanities', 'Arts', 'Other', 'No Major'],
        n_samples,
        p=[0.50, 0.20, 0.10, 0.05, 0.10, 0.05]
    )

    # Experience in years
    experience = np.random.choice(
        ['<1', '1-5', '5-10', '10-15', '15-20', '>20'],
        n_samples,
        p=[0.08, 0.35, 0.30, 0.15, 0.08, 0.04]
    )

    # Company size
    company_size = np.random.choice(
        ['<10', '10-50', '50-100', '100-500', '500-1000', '1000-5000', '5000-10000', '>10000'],
        n_samples,
        p=[0.05, 0.10, 0.12, 0.20, 0.15, 0.18, 0.12, 0.08]
    )

    # Company type
    company_type = np.random.choice(
        ['Pvt Ltd', 'Public Sector', 'Funded Startup', 'Early Stage Startup', 'NGO', 'Other'],
        n_samples,
        p=[0.45, 0.15, 0.15, 0.10, 0.05, 0.10]
    )

    # Last new job (years since last job change)
    last_new_job = np.random.choice(
        ['never', '1', '2', '3', '4', '>4'],
        n_samples,
        p=[0.15, 0.25, 0.20, 0.15, 0.12, 0.13]
    )

    # Training hours
    training_hours_base = np.random.gamma(2, 20, n_samples)
    training_hours = training_hours_base.clip(0, 300).astype(int)

    # Years at current company
    years_at_company = np.minimum(
        ages - 22,
        np.random.exponential(3, n_samples)
    ).clip(0, 35)

    # Number of previous employers
    num_previous_employers = np.random.poisson(2, n_samples).clip(0, 10)

    # Current salary (in thousands)
    base_salary = np.random.lognormal(10.5, 0.5, n_samples)
    education_multiplier = np.where(education_level == 'Phd', 1.5,
                           np.where(education_level == 'Masters', 1.3,
                           np.where(education_level == 'Graduate', 1.1, 0.8)))
    experience_multiplier = 1 + (ages - 22) * 0.02

    salary = (base_salary * education_multiplier * experience_multiplier).clip(20, 300)

    # Salary growth rate (annual %)
    salary_growth = np.random.normal(8, 5, n_samples).clip(0, 30)

    # Job satisfaction (1-10)
    job_satisfaction = np.random.normal(6.5, 2, n_samples).clip(1, 10)

    # Work-life balance (1-10)
    work_life_balance = np.random.normal(6, 2.5, n_samples).clip(1, 10)

    # Career growth opportunities (1-10)
    career_growth = np.random.normal(6, 2, n_samples).clip(1, 10)

    # Management quality (1-10)
    management_quality = np.random.normal(6.5, 2, n_samples).clip(1, 10)

    # Enrolled in training
    enrolled_training = (training_hours > 20).astype(int)

    # Relevant experience (years in relevant field)
    relevant_experience = np.minimum(
        ages - 22,
        np.abs(np.random.normal(5, 4, n_samples))
    ).clip(0, 40)

    # Job search activity
    linkedin_activity = np.random.beta(2, 8, n_samples)  # Most people low activity
    resume_updates = np.random.poisson(0.5, n_samples).clip(0, 5)

    # Calculate job change probability
    job_change_probability = (
        0.12 * (1 - job_satisfaction / 10) +
        0.10 * (1 - work_life_balance / 10) +
        0.10 * (1 - career_growth / 10) +
        0.08 * (1 - management_quality / 10) +
        0.08 * (salary_growth < 5) +
        0.08 * (training_hours < 10) +
        0.08 * linkedin_activity +
        0.06 * (resume_updates > 0) +
        0.06 * (years_at_company < 1) +
        0.05 * (last_new_job == '1') +
        0.05 * (company_type == 'Early Stage Startup') +
        0.04 * (company_size == '<10') +
        0.05 * (num_previous_employers > 4) +
        0.05 * np.random.random(n_samples)
    )

    # Generate target (looking for job change)
    looking_for_job_change = (job_change_probability > 0.35).astype(int)

    # Create DataFrame
    data = pd.DataFrame({
        'enrollee_id': range(1, n_samples + 1),
        'city': ['City_' + str(int(cdi * 100)) for cdi in city_development_index],
        'city_development_index': city_development_index,
        'gender': gender,
        'relevant_experience': ['Yes' if re > 1 else 'No' for re in relevant_experience],
        'enrolled_university': np.random.choice(['no_enrollment', 'Part time course', 'Full time course'],
                                                n_samples, p=[0.70, 0.25, 0.05]),
        'education_level': education_level,
        'major_discipline': major_discipline,
        'experience': experience,
        'company_size': company_size,
        'company_type': company_type,
        'last_new_job': last_new_job,
        'training_hours': training_hours,
        'age': ages,
        'years_at_company': years_at_company,
        'num_previous_employers': num_previous_employers,
        'salary': salary,
        'salary_growth': salary_growth,
        'job_satisfaction': job_satisfaction,
        'work_life_balance': work_life_balance,
        'career_growth': career_growth,
        'management_quality': management_quality,
        'linkedin_activity': linkedin_activity,
        'resume_updates': resume_updates,
        'target': looking_for_job_change
    })

    return data

def engineer_career_features(df):
    """Create advanced career trajectory features"""
    print("Engineering career trajectory features...")

    df_eng = df.copy()

    # Career stability score
    df_eng['career_stability'] = (
        (df_eng['years_at_company'] / 10) * 0.4 +
        (1 - df_eng['num_previous_employers'] / 10) * 0.3 +
        (df_eng['job_satisfaction'] / 10) * 0.3
    ).clip(0, 1)

    # Overall satisfaction
    df_eng['overall_satisfaction'] = (
        df_eng['job_satisfaction'] * 0.3 +
        df_eng['work_life_balance'] * 0.25 +
        df_eng['career_growth'] * 0.25 +
        df_eng['management_quality'] * 0.2
    ) / 10

    # Flight risk score
    df_eng['flight_risk'] = (
        (1 - df_eng['job_satisfaction'] / 10) * 0.25 +
        df_eng['linkedin_activity'] * 0.20 +
        (df_eng['resume_updates'] / 5) * 0.20 +
        (df_eng['salary_growth'] < 5) * 0.15 +
        (df_eng['years_at_company'] < 1) * 0.10 +
        (1 - df_eng['career_growth'] / 10) * 0.10
    ).clip(0, 1)

    # Development investment
    df_eng['development_investment'] = (df_eng['training_hours'] / 100).clip(0, 3)

    # Career momentum
    df_eng['career_momentum'] = (
        (df_eng['salary_growth'] / 30) * 0.4 +
        (df_eng['training_hours'] / 300) * 0.3 +
        (df_eng['career_growth'] / 10) * 0.3
    )

    # Job hopping tendency
    df_eng['job_hopping_rate'] = df_eng['num_previous_employers'] / (df_eng['age'] - 21).clip(1, 100)

    # Salary satisfaction (comparing to expected based on experience)
    expected_salary = 50 + (df_eng['age'] - 22) * 3
    df_eng['salary_gap'] = (df_eng['salary'] - expected_salary) / expected_salary

    # Company attractiveness
    df_eng['company_attractiveness'] = (
        df_eng['management_quality'] / 10 * 0.4 +
        (df_eng['salary_growth'] / 30) * 0.3 +
        df_eng['overall_satisfaction'] * 0.3
    )

    # Active job seeker indicator
    df_eng['active_job_seeker'] = (
        (df_eng['linkedin_activity'] > 0.3) |
        (df_eng['resume_updates'] > 0)
    ).astype(int)

    # Years since last move
    df_eng['last_new_job_numeric'] = df_eng['last_new_job'].map({
        'never': 99,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '>4': 6
    })

    # Tenure-to-age ratio
    df_eng['tenure_ratio'] = df_eng['years_at_company'] / (df_eng['age'] - 21).clip(1, 100)

    return df_eng

def create_visualizations(df, y_test, y_pred, y_pred_proba, feature_importance, feature_names):
    """Create comprehensive employee turnover visualizations"""
    print("Creating visualizations...")

    fig = plt.figure(figsize=(20, 12))

    # 1. Job change rate by education
    ax1 = plt.subplot(3, 4, 1)
    job_change_by_edu = df.groupby('education_level')['target'].mean().sort_values(ascending=False)
    ax1.barh(job_change_by_edu.index, job_change_by_edu.values, color='steelblue', alpha=0.7)
    ax1.set_title('Job Change Rate by Education', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Job Change Rate')

    # 2. Job change by experience
    ax2 = plt.subplot(3, 4, 2)
    job_change_by_exp = df.groupby('experience')['target'].mean()
    exp_order = ['<1', '1-5', '5-10', '10-15', '15-20', '>20']
    job_change_by_exp = job_change_by_exp.reindex(exp_order)
    ax2.plot(range(len(job_change_by_exp)), job_change_by_exp.values, marker='o',
            linewidth=2, color='darkred')
    ax2.set_title('Job Change Rate by Experience', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Experience Level')
    ax2.set_ylabel('Job Change Rate')
    ax2.set_xticks(range(len(exp_order)))
    ax2.set_xticklabels(exp_order, rotation=45)
    ax2.grid(True, alpha=0.3)

    # 3. Satisfaction distribution by job change
    ax3 = plt.subplot(3, 4, 3)
    df[df['target'] == 0]['job_satisfaction'].hist(bins=20, alpha=0.6, label='Staying',
                                                    color='green', ax=ax3)
    df[df['target'] == 1]['job_satisfaction'].hist(bins=20, alpha=0.6, label='Leaving',
                                                    color='red', ax=ax3)
    ax3.set_title('Job Satisfaction Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Job Satisfaction (1-10)')
    ax3.set_ylabel('Frequency')
    ax3.legend()

    # 4. Work-life balance impact
    ax4 = plt.subplot(3, 4, 4)
    df.boxplot(column='work_life_balance', by='target', ax=ax4)
    ax4.set_title('Work-Life Balance by Job Change', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Target (0=Stay, 1=Leave)')
    ax4.set_ylabel('Work-Life Balance Score')
    plt.suptitle('')

    # 5. Training hours distribution
    ax5 = plt.subplot(3, 4, 5)
    df[df['target'] == 0]['training_hours'].hist(bins=30, alpha=0.6, label='Staying',
                                                  color='green', ax=ax5)
    df[df['target'] == 1]['training_hours'].hist(bins=30, alpha=0.6, label='Leaving',
                                                  color='red', ax=ax5)
    ax5.set_title('Training Hours Distribution', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Training Hours')
    ax5.set_ylabel('Frequency')
    ax5.legend()
    ax5.set_xlim(0, 200)

    # 6. Job change by company type
    ax6 = plt.subplot(3, 4, 6)
    job_change_by_company = df.groupby('company_type')['target'].mean().sort_values(ascending=False)
    ax6.bar(range(len(job_change_by_company)), job_change_by_company.values,
           color='purple', alpha=0.7)
    ax6.set_title('Job Change Rate by Company Type', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Company Type')
    ax6.set_ylabel('Job Change Rate')
    ax6.set_xticks(range(len(job_change_by_company)))
    ax6.set_xticklabels(job_change_by_company.index, rotation=45, ha='right')
    ax6.grid(axis='y', alpha=0.3)

    # 7. Salary growth impact
    ax7 = plt.subplot(3, 4, 7)
    df.boxplot(column='salary_growth', by='target', ax=ax7)
    ax7.set_title('Salary Growth by Job Change', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Target (0=Stay, 1=Leave)')
    ax7.set_ylabel('Salary Growth (%)')
    plt.suptitle('')

    # 8. Career growth vs job satisfaction
    ax8 = plt.subplot(3, 4, 8)
    for target in [0, 1]:
        subset = df[df['target'] == target].sample(min(300, len(df[df['target'] == target])))
        ax8.scatter(subset['career_growth'], subset['job_satisfaction'],
                   alpha=0.4, label=f"Target={target}")
    ax8.set_title('Career Growth vs Job Satisfaction', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Career Growth Score')
    ax8.set_ylabel('Job Satisfaction Score')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # 9. Job change by company size
    ax9 = plt.subplot(3, 4, 9)
    job_change_by_size = df.groupby('company_size')['target'].mean()
    size_order = ['<10', '10-50', '50-100', '100-500', '500-1000', '1000-5000', '5000-10000', '>10000']
    job_change_by_size = job_change_by_size.reindex([s for s in size_order if s in job_change_by_size.index])
    ax9.plot(range(len(job_change_by_size)), job_change_by_size.values, marker='s',
            linewidth=2, color='teal')
    ax9.set_title('Job Change Rate by Company Size', fontsize=12, fontweight='bold')
    ax9.set_xlabel('Company Size')
    ax9.set_ylabel('Job Change Rate')
    ax9.set_xticks(range(len(job_change_by_size)))
    ax9.set_xticklabels(job_change_by_size.index, rotation=45, ha='right')
    ax9.grid(True, alpha=0.3)

    # 10. Feature importance
    ax10 = plt.subplot(3, 4, 10)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=True).tail(15)

    ax10.barh(importance_df['feature'], importance_df['importance'], color='darkgreen')
    ax10.set_title('Top 15 Feature Importance', fontsize=12, fontweight='bold')
    ax10.set_xlabel('Importance')

    # 11. Confusion matrix
    ax11 = plt.subplot(3, 4, 11)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax11, cbar=False)
    ax11.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    ax11.set_xlabel('Predicted')
    ax11.set_ylabel('Actual')

    # 12. ROC Curve
    ax12 = plt.subplot(3, 4, 12)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    ax12.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc_score:.3f}', color='darkorange')
    ax12.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax12.set_title('ROC Curve', fontsize=12, fontweight='bold')
    ax12.set_xlabel('False Positive Rate')
    ax12.set_ylabel('True Positive Rate')
    ax12.legend()
    ax12.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('job_change_prediction_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'job_change_prediction_analysis.png'")
    plt.close()

def main():
    print("="*60)
    print("Employee Job Change Prediction")
    print("="*60)

    # Generate data
    df = generate_employee_data(n_samples=3500)
    print(f"\nDataset shape: {df.shape}")
    print(f"Job change rate: {df['target'].mean():.2%}")

    # Engineer features
    df_eng = engineer_career_features(df)

    # Encode categorical variables
    categorical_cols = ['city', 'gender', 'relevant_experience', 'enrolled_university',
                       'education_level', 'major_discipline', 'experience', 'company_size',
                       'company_type', 'last_new_job']

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_eng[col] = le.fit_transform(df_eng[col].astype(str))
        label_encoders[col] = le

    # Drop enrollee_id
    df_eng = df_eng.drop('enrollee_id', axis=1)

    # Prepare features
    X = df_eng.drop('target', axis=1)
    y = df_eng['target']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Train Random Forest model
    print("\nTraining Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)

    # Predictions
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

    # Evaluation
    print("\n" + "="*60)
    print("Model Performance (Random Forest)")
    print("="*60)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

    # Cross-validation
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
    print(f"Cross-validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Create visualizations
    feature_importance = rf_model.feature_importances_
    feature_names = X.columns.tolist()

    create_visualizations(df, y_test, y_pred, y_pred_proba, feature_importance, feature_names)

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
