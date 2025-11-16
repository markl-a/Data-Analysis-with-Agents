"""
Online Shopper Purchase Intention Prediction
=============================================

This solution predicts whether an online shopping session will end in a purchase
using XGBoost and comprehensive session-based feature engineering.

Business Context:
- E-commerce platforms want to predict purchase intent in real-time
- Helps optimize user experience and targeted marketing
- Identifies high-value browsing sessions for intervention
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def generate_shopping_data(n_samples=5000):
    """
    Generate realistic online shopping session data

    Features include:
    - Session metrics (pages viewed, duration, bounce rates)
    - User behavior (product vs info pages, exit rates)
    - Traffic source and visitor type
    - Temporal features (weekend, month, special day)
    """
    print("Generating online shopping session data...")

    # Traffic source distribution
    traffic_sources = np.random.choice(
        ['Direct', 'Organic', 'Paid', 'Social', 'Referral'],
        n_samples,
        p=[0.25, 0.30, 0.15, 0.20, 0.10]
    )

    # Visitor types
    visitor_types = np.random.choice(
        ['New_Visitor', 'Returning_Visitor', 'Other'],
        n_samples,
        p=[0.15, 0.80, 0.05]
    )

    # Browser types
    browsers = np.random.choice(
        ['Chrome', 'Safari', 'Firefox', 'Edge', 'Other'],
        n_samples,
        p=[0.45, 0.25, 0.15, 0.10, 0.05]
    )

    # Operating systems
    operating_systems = np.random.choice(
        ['Windows', 'MacOS', 'Linux', 'iOS', 'Android'],
        n_samples,
        p=[0.40, 0.25, 0.05, 0.15, 0.15]
    )

    # Device types
    devices = np.random.choice(
        ['Desktop', 'Mobile', 'Tablet'],
        n_samples,
        p=[0.50, 0.40, 0.10]
    )

    # Temporal features
    months = np.random.choice(range(1, 13), n_samples)
    weekends = np.random.choice([0, 1], n_samples, p=[0.70, 0.30])
    special_days = np.random.choice([0, 1], n_samples, p=[0.90, 0.10])

    # Session metrics (influenced by various factors)
    base_product_pages = np.random.poisson(5, n_samples)
    product_related_duration = np.abs(np.random.normal(300, 200, n_samples))

    info_pages = np.random.poisson(2, n_samples)
    info_duration = np.abs(np.random.normal(100, 80, n_samples))

    # Bounce and exit rates (lower for returning visitors)
    bounce_rates = np.where(
        visitor_types == 'Returning_Visitor',
        np.random.uniform(0.01, 0.05, n_samples),
        np.random.uniform(0.10, 0.40, n_samples)
    )

    exit_rates = np.where(
        visitor_types == 'Returning_Visitor',
        np.random.uniform(0.01, 0.08, n_samples),
        np.random.uniform(0.10, 0.35, n_samples)
    )

    # Page values (higher for product pages)
    page_values = product_related_duration * 0.1 + np.random.uniform(0, 50, n_samples)

    # Generate purchase outcome based on features
    purchase_probability = (
        0.20 * (visitor_types == 'Returning_Visitor') +
        0.15 * (traffic_sources == 'Direct') +
        0.10 * (devices == 'Desktop') +
        0.15 * (1 - bounce_rates) +
        0.15 * (1 - exit_rates) +
        0.10 * (page_values / 100) +
        0.10 * (special_days) +
        0.05 * np.random.random(n_samples)
    )

    revenue = (purchase_probability > 0.35).astype(int)

    # Create DataFrame
    data = pd.DataFrame({
        'Administrative': info_pages,
        'Administrative_Duration': info_duration,
        'Informational': np.random.poisson(1, n_samples),
        'Informational_Duration': np.abs(np.random.normal(50, 40, n_samples)),
        'ProductRelated': base_product_pages,
        'ProductRelated_Duration': product_related_duration,
        'BounceRates': bounce_rates,
        'ExitRates': exit_rates,
        'PageValues': page_values,
        'SpecialDay': special_days,
        'Month': months,
        'Weekend': weekends,
        'TrafficSource': traffic_sources,
        'VisitorType': visitor_types,
        'Browser': browsers,
        'OperatingSystem': operating_systems,
        'Device': devices,
        'Revenue': revenue
    })

    return data

def engineer_features(df):
    """Create advanced session-based features"""
    print("Engineering session-based features...")

    df_eng = df.copy()

    # Total pages viewed
    df_eng['TotalPages'] = (df_eng['Administrative'] +
                            df_eng['Informational'] +
                            df_eng['ProductRelated'])

    # Total session duration
    df_eng['TotalDuration'] = (df_eng['Administrative_Duration'] +
                               df_eng['Informational_Duration'] +
                               df_eng['ProductRelated_Duration'])

    # Average time per page
    df_eng['AvgTimePerPage'] = df_eng['TotalDuration'] / (df_eng['TotalPages'] + 1)

    # Product page ratio
    df_eng['ProductPageRatio'] = df_eng['ProductRelated'] / (df_eng['TotalPages'] + 1)

    # Engagement score
    df_eng['EngagementScore'] = (df_eng['PageValues'] * df_eng['TotalPages'] *
                                  (1 - df_eng['BounceRates']))

    # Exit to bounce ratio
    df_eng['ExitBounceRatio'] = df_eng['ExitRates'] / (df_eng['BounceRates'] + 0.01)

    # Session quality score
    df_eng['SessionQuality'] = (df_eng['ProductPageRatio'] *
                                 (1 - df_eng['ExitRates']) *
                                 df_eng['AvgTimePerPage'] / 100)

    return df_eng

def create_visualizations(df, y_test, y_pred, y_pred_proba, feature_importance, feature_names):
    """Create comprehensive visualization dashboard"""
    print("Creating visualizations...")

    fig = plt.figure(figsize=(20, 12))

    # 1. Revenue distribution by visitor type
    ax1 = plt.subplot(3, 4, 1)
    revenue_by_visitor = df.groupby(['VisitorType', 'Revenue']).size().unstack(fill_value=0)
    revenue_by_visitor.plot(kind='bar', ax=ax1, color=['#d32f2f', '#388e3c'])
    ax1.set_title('Revenue by Visitor Type', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Visitor Type')
    ax1.set_ylabel('Count')
    ax1.legend(['No Purchase', 'Purchase'])
    plt.xticks(rotation=45)

    # 2. Revenue distribution by traffic source
    ax2 = plt.subplot(3, 4, 2)
    revenue_by_traffic = df.groupby(['TrafficSource', 'Revenue']).size().unstack(fill_value=0)
    revenue_by_traffic.plot(kind='bar', ax=ax2, color=['#d32f2f', '#388e3c'])
    ax2.set_title('Revenue by Traffic Source', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Traffic Source')
    ax2.set_ylabel('Count')
    ax2.legend(['No Purchase', 'Purchase'])
    plt.xticks(rotation=45)

    # 3. Bounce rate vs Exit rate by revenue
    ax3 = plt.subplot(3, 4, 3)
    for revenue in [0, 1]:
        data_subset = df[df['Revenue'] == revenue]
        ax3.scatter(data_subset['BounceRates'], data_subset['ExitRates'],
                   alpha=0.5, label=f"Revenue={revenue}")
    ax3.set_title('Bounce vs Exit Rates by Revenue', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Bounce Rate')
    ax3.set_ylabel('Exit Rate')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Page values distribution
    ax4 = plt.subplot(3, 4, 4)
    df[df['Revenue'] == 0]['PageValues'].hist(bins=30, alpha=0.6, label='No Purchase', color='red', ax=ax4)
    df[df['Revenue'] == 1]['PageValues'].hist(bins=30, alpha=0.6, label='Purchase', color='green', ax=ax4)
    ax4.set_title('Page Values Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Page Values')
    ax4.set_ylabel('Frequency')
    ax4.legend()

    # 5. Session duration by device
    ax5 = plt.subplot(3, 4, 5)
    df.boxplot(column='ProductRelated_Duration', by='Device', ax=ax5)
    ax5.set_title('Product Duration by Device', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Device')
    ax5.set_ylabel('Duration (seconds)')
    plt.suptitle('')

    # 6. Weekend vs weekday revenue
    ax6 = plt.subplot(3, 4, 6)
    weekend_revenue = df.groupby(['Weekend', 'Revenue']).size().unstack(fill_value=0)
    weekend_revenue.plot(kind='bar', ax=ax6, color=['#d32f2f', '#388e3c'])
    ax6.set_title('Revenue: Weekday vs Weekend', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Weekend (0=No, 1=Yes)')
    ax6.set_ylabel('Count')
    ax6.legend(['No Purchase', 'Purchase'])
    plt.xticks(rotation=0)

    # 7. Feature importance
    ax7 = plt.subplot(3, 4, 7)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=True).tail(15)

    ax7.barh(importance_df['feature'], importance_df['importance'], color='steelblue')
    ax7.set_title('Top 15 Feature Importance', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Importance')

    # 8. Confusion matrix
    ax8 = plt.subplot(3, 4, 8)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax8, cbar=False)
    ax8.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Predicted')
    ax8.set_ylabel('Actual')

    # 9. ROC Curve
    ax9 = plt.subplot(3, 4, 9)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    ax9.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc_score:.3f}')
    ax9.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax9.set_title('ROC Curve', fontsize=12, fontweight='bold')
    ax9.set_xlabel('False Positive Rate')
    ax9.set_ylabel('True Positive Rate')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    # 10. Monthly revenue trend
    ax10 = plt.subplot(3, 4, 10)
    monthly_revenue = df.groupby('Month')['Revenue'].mean()
    ax10.plot(monthly_revenue.index, monthly_revenue.values, marker='o', linewidth=2, color='purple')
    ax10.set_title('Purchase Rate by Month', fontsize=12, fontweight='bold')
    ax10.set_xlabel('Month')
    ax10.set_ylabel('Purchase Rate')
    ax10.grid(True, alpha=0.3)

    # 11. Special day impact
    ax11 = plt.subplot(3, 4, 11)
    special_day_revenue = df.groupby(['SpecialDay', 'Revenue']).size().unstack(fill_value=0)
    special_day_revenue.plot(kind='bar', ax=ax11, color=['#d32f2f', '#388e3c'])
    ax11.set_title('Revenue: Special Days', fontsize=12, fontweight='bold')
    ax11.set_xlabel('Special Day (0=No, 1=Yes)')
    ax11.set_ylabel('Count')
    ax11.legend(['No Purchase', 'Purchase'])
    plt.xticks(rotation=0)

    # 12. Browser performance
    ax12 = plt.subplot(3, 4, 12)
    browser_revenue = df.groupby('Browser')['Revenue'].mean().sort_values(ascending=False)
    ax12.bar(browser_revenue.index, browser_revenue.values, color='coral')
    ax12.set_title('Purchase Rate by Browser', fontsize=12, fontweight='bold')
    ax12.set_xlabel('Browser')
    ax12.set_ylabel('Purchase Rate')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('online_shopper_intention_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'online_shopper_intention_analysis.png'")
    plt.close()

def main():
    print("="*60)
    print("Online Shopper Purchase Intention Prediction")
    print("="*60)

    # Generate data
    df = generate_shopping_data(n_samples=5000)
    print(f"\nDataset shape: {df.shape}")
    print(f"Purchase rate: {df['Revenue'].mean():.2%}")

    # Engineer features
    df_eng = engineer_features(df)

    # Prepare features for modeling
    categorical_cols = ['TrafficSource', 'VisitorType', 'Browser', 'OperatingSystem', 'Device']
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df_eng[col] = le.fit_transform(df_eng[col])
        label_encoders[col] = le

    # Split features and target
    X = df_eng.drop('Revenue', axis=1)
    y = df_eng['Revenue']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Train XGBoost model
    print("\nTraining XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Evaluation
    print("\n" + "="*60)
    print("Model Performance")
    print("="*60)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"Cross-validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Create visualizations
    feature_importance = model.feature_importances_
    feature_names = X.columns.tolist()

    create_visualizations(df, y_test, y_pred, y_pred_proba, feature_importance, feature_names)

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
