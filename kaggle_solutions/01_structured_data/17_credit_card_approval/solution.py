"""
Credit Card Approval Prediction
================================

This solution predicts credit card application approval using Support Vector Machines
with comprehensive risk assessment features.

Business Context:
- Financial institutions need automated credit assessment
- Reduce manual review time and costs
- Ensure consistent, fair lending decisions
- Minimize default risk while maximizing approvals
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def generate_credit_data(n_samples=3000):
    """
    Generate realistic credit card application data

    Features include:
    - Applicant demographics
    - Financial metrics
    - Employment information
    - Credit history
    - Application characteristics
    """
    print("Generating credit card application data...")

    # Demographics
    ages = np.random.normal(40, 12, n_samples).clip(18, 75).astype(int)
    genders = np.random.choice(['Male', 'Female'], n_samples, p=[0.52, 0.48])

    # Marital status
    marital_status = np.random.choice(
        ['Single', 'Married', 'Divorced', 'Widowed'],
        n_samples,
        p=[0.30, 0.50, 0.15, 0.05]
    )

    # Dependents
    dependents = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.25, 0.25, 0.25, 0.15, 0.10])

    # Education level
    education = np.random.choice(
        ['High School', 'Bachelor', 'Master', 'PhD'],
        n_samples,
        p=[0.30, 0.45, 0.20, 0.05]
    )

    # Employment status
    employment_status = np.random.choice(
        ['Employed', 'Self-Employed', 'Unemployed', 'Student', 'Retired'],
        n_samples,
        p=[0.60, 0.15, 0.10, 0.08, 0.07]
    )

    # Years employed (influenced by age and employment status)
    years_employed = np.where(
        employment_status == 'Employed',
        np.random.uniform(0, (ages - 22).clip(0, 40)),
        np.where(employment_status == 'Self-Employed',
                np.random.uniform(0, (ages - 25).clip(0, 35)),
                0)
    ).clip(0, 40)

    # Income (influenced by education and employment)
    base_income = np.random.lognormal(10.5, 0.7, n_samples)
    education_multiplier = np.where(education == 'PhD', 1.5,
                           np.where(education == 'Master', 1.3,
                           np.where(education == 'Bachelor', 1.1, 0.8)))
    employment_multiplier = np.where(employment_status == 'Employed', 1.0,
                             np.where(employment_status == 'Self-Employed', 1.2,
                             np.where(employment_status == 'Retired', 0.6, 0.3)))

    annual_income = (base_income * education_multiplier * employment_multiplier).clip(15000, 500000)

    # Property ownership
    owns_property = np.random.choice([0, 1], n_samples, p=[0.40, 0.60])
    property_value = np.where(owns_property == 1,
                             np.random.lognormal(12.5, 0.6, n_samples).clip(50000, 2000000),
                             0)

    # Owns car
    owns_car = np.random.choice([0, 1], n_samples, p=[0.35, 0.65])

    # Credit history
    credit_history_years = np.minimum(ages - 18, np.random.uniform(0, 30, n_samples)).clip(0, 50)

    # Number of existing credit cards
    existing_cards = np.random.choice([0, 1, 2, 3, 4, 5], n_samples,
                                     p=[0.15, 0.25, 0.30, 0.20, 0.08, 0.02])

    # Credit utilization (for those with cards)
    credit_utilization = np.where(
        existing_cards > 0,
        np.random.beta(2, 5, n_samples),  # Most people use 20-40%
        0
    )

    # Payment history (percentage of on-time payments)
    payment_history = np.random.beta(8, 2, n_samples)  # Most people have good history

    # Number of recent inquiries
    recent_inquiries = np.random.choice([0, 1, 2, 3, 4, 5, 6], n_samples,
                                       p=[0.30, 0.25, 0.20, 0.15, 0.07, 0.02, 0.01])

    # Debt-to-income ratio
    total_debt = np.random.uniform(0, annual_income * 0.5, n_samples)
    debt_to_income = total_debt / annual_income

    # Requested credit limit
    requested_limit = np.random.choice(
        [2000, 5000, 10000, 15000, 20000, 30000, 50000],
        n_samples,
        p=[0.20, 0.25, 0.25, 0.15, 0.08, 0.05, 0.02]
    )

    # Calculate approval probability based on features
    approval_score = (
        0.15 * (annual_income / 100000) +
        0.10 * (credit_history_years / 30) +
        0.15 * payment_history +
        0.10 * (1 - credit_utilization) +
        0.10 * owns_property +
        0.08 * (employment_status == 'Employed') +
        0.07 * (years_employed / 20) +
        0.08 * (1 - debt_to_income) +
        0.07 * (existing_cards > 0) * (existing_cards < 4) +
        0.05 * (recent_inquiries == 0) +
        0.03 * (education == 'Bachelor') + 0.05 * (education == 'Master') + 0.07 * (education == 'PhD') +
        0.02 * owns_car -
        0.10 * (recent_inquiries / 6) -
        0.05 * (requested_limit / 50000)
    )

    # Add some randomness
    approval_score += np.random.normal(0, 0.1, n_samples)

    # Convert to binary approval (threshold around 0.5)
    approved = (approval_score > 0.50).astype(int)

    # Create DataFrame
    data = pd.DataFrame({
        'Age': ages,
        'Gender': genders,
        'MaritalStatus': marital_status,
        'Dependents': dependents,
        'Education': education,
        'EmploymentStatus': employment_status,
        'YearsEmployed': years_employed,
        'AnnualIncome': annual_income,
        'OwnsProperty': owns_property,
        'PropertyValue': property_value,
        'OwnsCar': owns_car,
        'CreditHistoryYears': credit_history_years,
        'ExistingCards': existing_cards,
        'CreditUtilization': credit_utilization,
        'PaymentHistory': payment_history,
        'RecentInquiries': recent_inquiries,
        'DebtToIncome': debt_to_income,
        'RequestedLimit': requested_limit,
        'Approved': approved
    })

    return data

def engineer_credit_features(df):
    """Create advanced credit risk features"""
    print("Engineering credit risk features...")

    df_eng = df.copy()

    # Income to limit ratio
    df_eng['IncomeToLimitRatio'] = df_eng['AnnualIncome'] / (df_eng['RequestedLimit'] * 12)

    # Total assets
    df_eng['TotalAssets'] = df_eng['PropertyValue'] + (df_eng['OwnsCar'] * 15000)

    # Asset to income ratio
    df_eng['AssetToIncomeRatio'] = df_eng['TotalAssets'] / (df_eng['AnnualIncome'] + 1)

    # Credit maturity score (age of credit history relative to age)
    df_eng['CreditMaturity'] = df_eng['CreditHistoryYears'] / (df_eng['Age'] - 17)

    # Financial stability score
    df_eng['FinancialStability'] = (
        (df_eng['YearsEmployed'] / 10) * 0.3 +
        (1 - df_eng['DebtToIncome']) * 0.4 +
        df_eng['PaymentHistory'] * 0.3
    )

    # Risk score (inverse of approval indicators)
    df_eng['RiskScore'] = (
        df_eng['RecentInquiries'] / 10 +
        df_eng['CreditUtilization'] * 0.5 +
        df_eng['DebtToIncome'] * 0.5
    ) / 3

    # Credit portfolio diversity
    df_eng['CreditDiversity'] = (df_eng['ExistingCards'] > 0) & (df_eng['ExistingCards'] <= 3)
    df_eng['CreditDiversity'] = df_eng['CreditDiversity'].astype(int)

    # Age group
    df_eng['AgeGroup'] = pd.cut(df_eng['Age'], bins=[0, 25, 35, 50, 100],
                                 labels=['Young', 'MiddleAge', 'Mature', 'Senior'])

    # Income bracket
    df_eng['IncomeBracket'] = pd.cut(df_eng['AnnualIncome'],
                                     bins=[0, 30000, 60000, 100000, 1000000],
                                     labels=['Low', 'Medium', 'High', 'VeryHigh'])

    return df_eng

def create_visualizations(df, y_test, y_pred, y_pred_proba, feature_importance, feature_names):
    """Create comprehensive credit analysis visualizations"""
    print("Creating visualizations...")

    fig = plt.figure(figsize=(20, 12))

    # 1. Approval rate by income bracket
    ax1 = plt.subplot(3, 4, 1)
    approval_by_income = df.groupby('IncomeBracket')['Approved'].agg(['mean', 'count'])
    ax1.bar(approval_by_income.index, approval_by_income['mean'], color='steelblue', alpha=0.7)
    ax1.set_title('Approval Rate by Income Bracket', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Income Bracket')
    ax1.set_ylabel('Approval Rate')
    ax1.grid(axis='y', alpha=0.3)

    # 2. Approval rate by education
    ax2 = plt.subplot(3, 4, 2)
    approval_by_edu = df.groupby('Education')['Approved'].mean().sort_values(ascending=False)
    ax2.barh(approval_by_edu.index, approval_by_edu.values, color='coral')
    ax2.set_title('Approval Rate by Education', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Approval Rate')

    # 3. Income distribution by approval
    ax3 = plt.subplot(3, 4, 3)
    df[df['Approved'] == 0]['AnnualIncome'].hist(bins=40, alpha=0.6, label='Rejected',
                                                  color='red', ax=ax3)
    df[df['Approved'] == 1]['AnnualIncome'].hist(bins=40, alpha=0.6, label='Approved',
                                                  color='green', ax=ax3)
    ax3.set_title('Income Distribution by Approval', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Annual Income')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.set_xlim(0, 200000)

    # 4. Debt-to-income ratio by approval
    ax4 = plt.subplot(3, 4, 4)
    df.boxplot(column='DebtToIncome', by='Approved', ax=ax4)
    ax4.set_title('Debt-to-Income Ratio by Approval', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Approved (0=No, 1=Yes)')
    ax4.set_ylabel('Debt-to-Income Ratio')
    plt.suptitle('')

    # 5. Credit utilization vs payment history
    ax5 = plt.subplot(3, 4, 5)
    for approved in [0, 1]:
        data_subset = df[df['Approved'] == approved]
        ax5.scatter(data_subset['CreditUtilization'], data_subset['PaymentHistory'],
                   alpha=0.5, label=f"Approved={approved}")
    ax5.set_title('Credit Utilization vs Payment History', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Credit Utilization')
    ax5.set_ylabel('Payment History')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Approval rate by employment status
    ax6 = plt.subplot(3, 4, 6)
    approval_by_emp = df.groupby('EmploymentStatus')['Approved'].mean().sort_values(ascending=False)
    ax6.bar(approval_by_emp.index, approval_by_emp.values, color='purple', alpha=0.7)
    ax6.set_title('Approval Rate by Employment', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Employment Status')
    ax6.set_ylabel('Approval Rate')
    plt.xticks(rotation=45, ha='right')
    ax6.grid(axis='y', alpha=0.3)

    # 7. Property ownership impact
    ax7 = plt.subplot(3, 4, 7)
    property_approval = df.groupby(['OwnsProperty', 'Approved']).size().unstack(fill_value=0)
    property_approval.plot(kind='bar', ax=ax7, color=['#d32f2f', '#388e3c'])
    ax7.set_title('Approval by Property Ownership', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Owns Property (0=No, 1=Yes)')
    ax7.set_ylabel('Count')
    ax7.legend(['Rejected', 'Approved'])
    plt.xticks(rotation=0)

    # 8. Age distribution by approval
    ax8 = plt.subplot(3, 4, 8)
    df.boxplot(column='Age', by='Approved', ax=ax8)
    ax8.set_title('Age Distribution by Approval', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Approved (0=No, 1=Yes)')
    ax8.set_ylabel('Age')
    plt.suptitle('')

    # 9. Feature importance
    ax9 = plt.subplot(3, 4, 9)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=True).tail(15)

    ax9.barh(importance_df['feature'], importance_df['importance'], color='darkblue')
    ax9.set_title('Top 15 Feature Importance', fontsize=12, fontweight='bold')
    ax9.set_xlabel('Importance')

    # 10. Confusion matrix
    ax10 = plt.subplot(3, 4, 10)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax10, cbar=False)
    ax10.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    ax10.set_xlabel('Predicted')
    ax10.set_ylabel('Actual')

    # 11. ROC Curve
    ax11 = plt.subplot(3, 4, 11)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    ax11.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc_score:.3f}', color='darkorange')
    ax11.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax11.set_title('ROC Curve', fontsize=12, fontweight='bold')
    ax11.set_xlabel('False Positive Rate')
    ax11.set_ylabel('True Positive Rate')
    ax11.legend()
    ax11.grid(True, alpha=0.3)

    # 12. Recent inquiries impact
    ax12 = plt.subplot(3, 4, 12)
    inquiry_approval = df.groupby('RecentInquiries')['Approved'].mean()
    ax12.plot(inquiry_approval.index, inquiry_approval.values, marker='o',
             linewidth=2, color='darkred')
    ax12.set_title('Approval Rate by Recent Inquiries', fontsize=12, fontweight='bold')
    ax12.set_xlabel('Number of Recent Inquiries')
    ax12.set_ylabel('Approval Rate')
    ax12.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('credit_card_approval_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'credit_card_approval_analysis.png'")
    plt.close()

def main():
    print("="*60)
    print("Credit Card Approval Prediction with SVM")
    print("="*60)

    # Generate data
    df = generate_credit_data(n_samples=3000)
    print(f"\nDataset shape: {df.shape}")
    print(f"Approval rate: {df['Approved'].mean():.2%}")

    # Engineer features
    df_eng = engineer_credit_features(df)

    # Encode categorical variables
    categorical_cols = ['Gender', 'MaritalStatus', 'Education', 'EmploymentStatus',
                       'AgeGroup', 'IncomeBracket']
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df_eng[col] = le.fit_transform(df_eng[col])
        label_encoders[col] = le

    # Prepare features
    X = df_eng.drop('Approved', axis=1)
    y = df_eng['Approved']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Feature scaling (important for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM model
    print("\nTraining SVM model with RBF kernel...")
    svm_model = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        probability=True,
        random_state=42
    )

    svm_model.fit(X_train_scaled, y_train)

    # Train Random Forest for feature importance
    print("Training Random Forest for feature importance...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    # Predictions
    y_pred = svm_model.predict(X_test_scaled)
    y_pred_proba = svm_model.predict_proba(X_test_scaled)[:, 1]

    # Evaluation
    print("\n" + "="*60)
    print("Model Performance (SVM)")
    print("="*60)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

    # Cross-validation
    cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
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
