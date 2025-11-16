"""
Credit Card Payment Default Prediction
=======================================

This solution predicts credit card payment defaults using Neural Networks
with temporal payment behavior analysis.

Business Context:
- Credit card issuers lose billions annually to defaults
- Early intervention can reduce default rates by 30-50%
- Proactive risk management preserves customer relationships
- Predictive models enable targeted collection strategies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def generate_payment_default_data(n_samples=4000):
    """
    Generate realistic credit card payment default data

    Features include:
    - Cardholder demographics
    - Account characteristics
    - Payment history (6 months)
    - Utilization and balance trends
    - Behavioral indicators
    """
    print("Generating credit card payment default data...")

    # Demographics
    ages = np.random.normal(35, 10, n_samples).clip(21, 70).astype(int)
    genders = np.random.choice([1, 2], n_samples, p=[0.46, 0.54])  # 1=Male, 2=Female

    # Education level (1=graduate, 2=university, 3=high school, 4=others)
    education = np.random.choice([1, 2, 3, 4], n_samples, p=[0.35, 0.40, 0.20, 0.05])

    # Marital status (1=married, 2=single, 3=others)
    marital_status = np.random.choice([1, 2, 3], n_samples, p=[0.55, 0.40, 0.05])

    # Credit limit
    credit_limits = np.random.lognormal(10.2, 1.0, n_samples).clip(10000, 1000000)

    # Account age (months)
    account_age = np.random.uniform(6, 120, n_samples).astype(int)

    # Current balance
    utilization_rate = np.random.beta(2, 3, n_samples)  # Most use 20-40%
    balance = (credit_limits * utilization_rate).clip(0, credit_limits)

    # Payment status for last 6 months
    # -1=pay duly, 0=revolving, 1=delay 1 month, 2=delay 2 months, etc.
    # Create patterns that lead to default

    # Define default probability base
    default_base_prob = np.random.uniform(0, 1, n_samples)

    # Create payment patterns
    pay_status_1 = np.random.choice([-1, 0, 1, 2, 3], n_samples, p=[0.50, 0.25, 0.15, 0.07, 0.03])
    pay_status_2 = np.random.choice([-1, 0, 1, 2, 3], n_samples, p=[0.48, 0.27, 0.15, 0.07, 0.03])
    pay_status_3 = np.random.choice([-1, 0, 1, 2, 3], n_samples, p=[0.47, 0.28, 0.15, 0.07, 0.03])
    pay_status_4 = np.random.choice([-1, 0, 1, 2, 3], n_samples, p=[0.46, 0.29, 0.15, 0.07, 0.03])
    pay_status_5 = np.random.choice([-1, 0, 1, 2, 3], n_samples, p=[0.45, 0.30, 0.15, 0.07, 0.03])
    pay_status_6 = np.random.choice([-1, 0, 1, 2, 3], n_samples, p=[0.44, 0.31, 0.15, 0.07, 0.03])

    # Bill amounts for last 6 months
    bill_amt_1 = balance * np.random.uniform(0.8, 1.2, n_samples)
    bill_amt_2 = bill_amt_1 * np.random.uniform(0.7, 1.3, n_samples)
    bill_amt_3 = bill_amt_2 * np.random.uniform(0.7, 1.3, n_samples)
    bill_amt_4 = bill_amt_3 * np.random.uniform(0.7, 1.3, n_samples)
    bill_amt_5 = bill_amt_4 * np.random.uniform(0.7, 1.3, n_samples)
    bill_amt_6 = bill_amt_5 * np.random.uniform(0.7, 1.3, n_samples)

    # Payment amounts for last 6 months
    # Good payers pay 20-100% of bill, poor payers pay 0-30%
    payment_rate_1 = np.where(pay_status_1 > 0,
                              np.random.uniform(0, 0.3, n_samples),
                              np.random.uniform(0.2, 1.0, n_samples))
    pay_amt_1 = bill_amt_1 * payment_rate_1

    payment_rate_2 = np.where(pay_status_2 > 0,
                              np.random.uniform(0, 0.3, n_samples),
                              np.random.uniform(0.2, 1.0, n_samples))
    pay_amt_2 = bill_amt_2 * payment_rate_2

    payment_rate_3 = np.where(pay_status_3 > 0,
                              np.random.uniform(0, 0.3, n_samples),
                              np.random.uniform(0.2, 1.0, n_samples))
    pay_amt_3 = bill_amt_3 * payment_rate_3

    payment_rate_4 = np.where(pay_status_4 > 0,
                              np.random.uniform(0, 0.3, n_samples),
                              np.random.uniform(0.2, 1.0, n_samples))
    pay_amt_4 = bill_amt_4 * payment_rate_4

    payment_rate_5 = np.where(pay_status_5 > 0,
                              np.random.uniform(0, 0.3, n_samples),
                              np.random.uniform(0.2, 1.0, n_samples))
    pay_amt_5 = bill_amt_5 * payment_rate_5

    payment_rate_6 = np.where(pay_status_6 > 0,
                              np.random.uniform(0, 0.3, n_samples),
                              np.random.uniform(0.2, 1.0, n_samples))
    pay_amt_6 = bill_amt_6 * payment_rate_6

    # Calculate default probability based on features
    avg_pay_status = (pay_status_1 + pay_status_2 + pay_status_3 +
                     pay_status_4 + pay_status_5 + pay_status_6) / 6

    recent_pay_trend = (pay_status_1 + pay_status_2 - pay_status_5 - pay_status_6) / 4

    avg_payment_rate = (payment_rate_1 + payment_rate_2 + payment_rate_3 +
                       payment_rate_4 + payment_rate_5 + payment_rate_6) / 6

    default_probability = (
        0.25 * (avg_pay_status + 1) / 4 +  # Payment history
        0.20 * (1 - avg_payment_rate) +     # Payment amount
        0.15 * (utilization_rate > 0.8) +   # High utilization
        0.10 * (recent_pay_trend > 0) +     # Worsening trend
        0.10 * (education == 4) +            # Education level
        0.08 * (ages < 25) +                 # Young age
        0.07 * (balance / credit_limits > 0.9) +  # Near limit
        0.05 * np.random.random(n_samples)
    )

    default = (default_probability > 0.35).astype(int)

    # Create DataFrame
    data = pd.DataFrame({
        'LIMIT_BAL': credit_limits,
        'SEX': genders,
        'EDUCATION': education,
        'MARRIAGE': marital_status,
        'AGE': ages,
        'PAY_0': pay_status_1,
        'PAY_2': pay_status_2,
        'PAY_3': pay_status_3,
        'PAY_4': pay_status_4,
        'PAY_5': pay_status_5,
        'PAY_6': pay_status_6,
        'BILL_AMT1': bill_amt_1,
        'BILL_AMT2': bill_amt_2,
        'BILL_AMT3': bill_amt_3,
        'BILL_AMT4': bill_amt_4,
        'BILL_AMT5': bill_amt_5,
        'BILL_AMT6': bill_amt_6,
        'PAY_AMT1': pay_amt_1,
        'PAY_AMT2': pay_amt_2,
        'PAY_AMT3': pay_amt_3,
        'PAY_AMT4': pay_amt_4,
        'PAY_AMT5': pay_amt_5,
        'PAY_AMT6': pay_amt_6,
        'default': default
    })

    return data

def engineer_payment_features(df):
    """Create advanced temporal payment behavior features"""
    print("Engineering payment behavior features...")

    df_eng = df.copy()

    # Utilization metrics
    df_eng['AvgUtilization'] = (df_eng['BILL_AMT1'] + df_eng['BILL_AMT2'] +
                                  df_eng['BILL_AMT3']) / (3 * df_eng['LIMIT_BAL'])
    df_eng['CurrentUtilization'] = df_eng['BILL_AMT1'] / df_eng['LIMIT_BAL']
    df_eng['MaxUtilization'] = df_eng[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
                                        'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].max(axis=1) / df_eng['LIMIT_BAL']

    # Payment status aggregations
    df_eng['AvgPayStatus'] = df_eng[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].mean(axis=1)
    df_eng['MaxPayStatus'] = df_eng[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].max(axis=1)
    df_eng['NumDelays'] = (df_eng[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']] > 0).sum(axis=1)

    # Payment trends
    df_eng['PayStatusTrend'] = (df_eng['PAY_0'] + df_eng['PAY_2'] - df_eng['PAY_5'] - df_eng['PAY_6']) / 4
    df_eng['RecentDelays'] = (df_eng[['PAY_0', 'PAY_2', 'PAY_3']] > 0).sum(axis=1)

    # Bill amount trends
    df_eng['BillTrend'] = (df_eng['BILL_AMT1'] - df_eng['BILL_AMT6']) / (df_eng['BILL_AMT6'] + 1)
    df_eng['BillVolatility'] = df_eng[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
                                        'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].std(axis=1)

    # Payment amount analysis
    df_eng['AvgPaymentRate'] = ((df_eng['PAY_AMT1'] / (df_eng['BILL_AMT1'] + 1)) +
                                 (df_eng['PAY_AMT2'] / (df_eng['BILL_AMT2'] + 1)) +
                                 (df_eng['PAY_AMT3'] / (df_eng['BILL_AMT3'] + 1))) / 3

    df_eng['PaymentTrend'] = ((df_eng['PAY_AMT1'] / (df_eng['BILL_AMT1'] + 1)) -
                               (df_eng['PAY_AMT6'] / (df_eng['BILL_AMT6'] + 1)))

    df_eng['TotalPayments6M'] = (df_eng['PAY_AMT1'] + df_eng['PAY_AMT2'] + df_eng['PAY_AMT3'] +
                                  df_eng['PAY_AMT4'] + df_eng['PAY_AMT5'] + df_eng['PAY_AMT6'])

    # Risk indicators
    df_eng['HighUtilization'] = (df_eng['CurrentUtilization'] > 0.8).astype(int)
    df_eng['ConsistentDelays'] = (df_eng['NumDelays'] >= 3).astype(int)
    df_eng['LowPaymentRate'] = (df_eng['AvgPaymentRate'] < 0.3).astype(int)

    # Composite risk score
    df_eng['RiskScore'] = (
        df_eng['AvgPayStatus'] * 0.3 +
        df_eng['CurrentUtilization'] * 0.25 +
        (1 - df_eng['AvgPaymentRate']) * 0.25 +
        df_eng['PayStatusTrend'] * 0.1 +
        df_eng['NumDelays'] / 6 * 0.1
    )

    return df_eng

def create_visualizations(df, y_test, y_pred, y_pred_proba):
    """Create comprehensive payment default visualizations"""
    print("Creating visualizations...")

    fig = plt.figure(figsize=(20, 12))

    # 1. Default rate by education
    ax1 = plt.subplot(3, 4, 1)
    default_by_edu = df.groupby('EDUCATION')['default'].mean()
    ax1.bar(default_by_edu.index, default_by_edu.values, color='steelblue', alpha=0.7)
    ax1.set_title('Default Rate by Education Level', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Education (1=Grad, 2=Uni, 3=HS, 4=Other)')
    ax1.set_ylabel('Default Rate')
    ax1.grid(axis='y', alpha=0.3)

    # 2. Default rate by age group
    ax2 = plt.subplot(3, 4, 2)
    df['AgeGroup'] = pd.cut(df['AGE'], bins=[0, 30, 40, 50, 100], labels=['<30', '30-40', '40-50', '50+'])
    default_by_age = df.groupby('AgeGroup')['default'].mean()
    ax2.bar(default_by_age.index, default_by_age.values, color='coral', alpha=0.7)
    ax2.set_title('Default Rate by Age Group', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Age Group')
    ax2.set_ylabel('Default Rate')
    ax2.grid(axis='y', alpha=0.3)

    # 3. Utilization distribution by default
    ax3 = plt.subplot(3, 4, 3)
    df_with_util = df.copy()
    df_with_util['Utilization'] = df_with_util['BILL_AMT1'] / df_with_util['LIMIT_BAL']
    df_with_util[df_with_util['default'] == 0]['Utilization'].hist(bins=30, alpha=0.6,
                                                                     label='No Default', color='green', ax=ax3)
    df_with_util[df_with_util['default'] == 1]['Utilization'].hist(bins=30, alpha=0.6,
                                                                     label='Default', color='red', ax=ax3)
    ax3.set_title('Utilization Distribution by Default', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Credit Utilization')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.set_xlim(0, 1.5)

    # 4. Payment status distribution
    ax4 = plt.subplot(3, 4, 4)
    pay_status_counts = df['PAY_0'].value_counts().sort_index()
    ax4.bar(pay_status_counts.index, pay_status_counts.values, color='purple', alpha=0.7)
    ax4.set_title('Payment Status Distribution (Most Recent)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Payment Status (-1=Duly, 0=Revolve, 1+=Delay)')
    ax4.set_ylabel('Count')
    ax4.grid(axis='y', alpha=0.3)

    # 5. Default rate by payment status
    ax5 = plt.subplot(3, 4, 5)
    default_by_pay = df.groupby('PAY_0')['default'].mean()
    ax5.plot(default_by_pay.index, default_by_pay.values, marker='o', linewidth=2, color='darkred')
    ax5.set_title('Default Rate by Payment Status', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Payment Status')
    ax5.set_ylabel('Default Rate')
    ax5.grid(True, alpha=0.3)

    # 6. Credit limit distribution by default
    ax6 = plt.subplot(3, 4, 6)
    df.boxplot(column='LIMIT_BAL', by='default', ax=ax6)
    ax6.set_title('Credit Limit by Default Status', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Default (0=No, 1=Yes)')
    ax6.set_ylabel('Credit Limit')
    plt.suptitle('')

    # 7. Payment amount vs bill amount
    ax7 = plt.subplot(3, 4, 7)
    for default in [0, 1]:
        subset = df[df['default'] == default].sample(min(200, len(df[df['default'] == default])))
        ax7.scatter(subset['BILL_AMT1'], subset['PAY_AMT1'], alpha=0.5, label=f"Default={default}")
    ax7.set_title('Payment vs Bill Amount', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Bill Amount (Month 1)')
    ax7.set_ylabel('Payment Amount (Month 1)')
    ax7.legend()
    ax7.set_xlim(0, 200000)
    ax7.set_ylim(0, 200000)
    ax7.grid(True, alpha=0.3)

    # 8. Gender vs default
    ax8 = plt.subplot(3, 4, 8)
    gender_default = df.groupby(['SEX', 'default']).size().unstack(fill_value=0)
    gender_default.plot(kind='bar', ax=ax8, color=['#388e3c', '#d32f2f'])
    ax8.set_title('Default by Gender', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Gender (1=Male, 2=Female)')
    ax8.set_ylabel('Count')
    ax8.legend(['No Default', 'Default'])
    plt.xticks(rotation=0)

    # 9. Marital status impact
    ax9 = plt.subplot(3, 4, 9)
    marriage_default = df.groupby('MARRIAGE')['default'].mean()
    ax9.bar(marriage_default.index, marriage_default.values, color='teal', alpha=0.7)
    ax9.set_title('Default Rate by Marital Status', fontsize=12, fontweight='bold')
    ax9.set_xlabel('Marital Status (1=Married, 2=Single, 3=Other)')
    ax9.set_ylabel('Default Rate')
    ax9.grid(axis='y', alpha=0.3)

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

    # 12. Precision-Recall Curve
    ax12 = plt.subplot(3, 4, 12)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    ax12.plot(recall, precision, linewidth=2, label=f'AP = {avg_precision:.3f}', color='darkblue')
    ax12.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
    ax12.set_xlabel('Recall')
    ax12.set_ylabel('Precision')
    ax12.legend()
    ax12.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('payment_default_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'payment_default_analysis.png'")
    plt.close()

def main():
    print("="*60)
    print("Credit Card Payment Default Prediction")
    print("="*60)

    # Generate data
    df = generate_payment_default_data(n_samples=4000)
    print(f"\nDataset shape: {df.shape}")
    print(f"Default rate: {df['default'].mean():.2%}")

    # Engineer features
    df_eng = engineer_payment_features(df)

    # Prepare features
    X = df_eng.drop('default', axis=1)
    y = df_eng['default']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Feature scaling (critical for neural networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Neural Network
    print("\nTraining Neural Network (MLP)...")
    nn_model = MLPClassifier(
        hidden_layer_sizes=(100, 50, 25),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=32,
        learning_rate='adaptive',
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )

    nn_model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = nn_model.predict(X_test_scaled)
    y_pred_proba = nn_model.predict_proba(X_test_scaled)[:, 1]

    # Evaluation
    print("\n" + "="*60)
    print("Model Performance (Neural Network)")
    print("="*60)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"Average Precision Score: {average_precision_score(y_test, y_pred_proba):.4f}")

    # Create visualizations
    create_visualizations(df, y_test, y_pred, y_pred_proba)

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
