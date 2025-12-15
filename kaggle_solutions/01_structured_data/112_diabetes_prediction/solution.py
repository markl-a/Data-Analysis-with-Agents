"""
Diabetes Prediction - Comprehensive Machine Learning Solution

This module provides a complete solution for predicting diabetes risk based on
patient health indicators. It implements multiple machine learning algorithms,
feature engineering, model evaluation, and interpretability analysis.

Dataset: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset
Difficulty: ⭐⭐ Intermediate Level

Key Features:
- Multiple algorithms: Logistic Regression, Random Forest, XGBoost, LightGBM
- Advanced feature engineering for health data
- Handling class imbalance with SMOTE
- Comprehensive visualizations and model interpretability
- Focus on recall for medical diagnosis applications
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    GridSearchCV, learning_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, roc_auc_score,
    f1_score, recall_score, precision_score
)
from sklearn.feature_selection import SelectKBest, f_classif

# Advanced ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available")

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Warning: imbalanced-learn not available, SMOTE disabled")

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class DiabetesPredictionModel:
    """
    Comprehensive Diabetes Prediction Model with multiple ML algorithms.

    This class implements a complete pipeline for predicting diabetes risk
    including data preprocessing, feature engineering, model training,
    evaluation, and interpretability analysis.

    Attributes:
        models (Dict[str, Any]): Dictionary storing trained models
        scaler (StandardScaler): Feature scaler for normalization
        label_encoders (Dict[str, LabelEncoder]): Encoders for categorical variables
        best_model (Any): Best performing model after evaluation
        feature_names (List[str]): List of feature column names
    """

    def __init__(self):
        """Initialize the predictor with empty containers."""
        self.models: Dict[str, Any] = {}
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.best_model = None
        self.feature_names: List[str] = []
        self.results: Dict[str, Any] = {}

    def create_sample_data(self) -> pd.DataFrame:
        """
        Create realistic sample diabetes dataset for demonstration.

        Generates synthetic data that mimics the real diabetes prediction dataset
        with realistic distributions and correlations between features.

        Returns:
            pd.DataFrame: Synthetic diabetes dataset with 5000 samples
        """
        np.random.seed(42)
        n_samples = 5000

        # Generate base features
        gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.48, 0.52])
        age = np.random.normal(45, 15, n_samples).clip(18, 85)

        # Generate correlated features
        bmi = np.random.normal(27, 6, n_samples).clip(15, 50)

        # HbA1c and blood glucose are key diabetes indicators
        # Higher values indicate higher diabetes risk
        base_hba1c = np.random.normal(5.5, 1.0, n_samples)
        base_glucose = np.random.normal(100, 30, n_samples)

        # Add correlation with BMI and age
        hba1c_level = (base_hba1c + 0.02 * (bmi - 25) + 0.01 * (age - 45)).clip(4, 12)
        blood_glucose_level = (base_glucose + 0.5 * (bmi - 25) + 0.3 * (age - 45)).clip(60, 250)

        # Hypertension and heart disease probabilities
        hypertension_prob = 0.1 + 0.005 * (age - 40) + 0.01 * (bmi - 25)
        hypertension = (np.random.random(n_samples) < hypertension_prob.clip(0.05, 0.5)).astype(int)

        heart_disease_prob = 0.05 + 0.003 * (age - 40) + 0.005 * (bmi - 25)
        heart_disease = (np.random.random(n_samples) < heart_disease_prob.clip(0.02, 0.3)).astype(int)

        smoking_history = np.random.choice(
            ['never', 'former', 'current', 'not current', 'ever', 'No Info'],
            n_samples,
            p=[0.35, 0.15, 0.15, 0.10, 0.10, 0.15]
        )

        # Generate diabetes outcome based on risk factors
        diabetes_prob = (
            0.05 +  # base rate
            0.15 * (hba1c_level > 6.5) +
            0.1 * (blood_glucose_level > 140) +
            0.05 * (bmi > 30) +
            0.03 * (age > 50) +
            0.05 * hypertension +
            0.03 * heart_disease +
            0.02 * (smoking_history == 'current')
        )
        diabetes = (np.random.random(n_samples) < diabetes_prob.clip(0, 0.9)).astype(int)

        data = {
            'gender': gender,
            'age': age.round(1),
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'smoking_history': smoking_history,
            'bmi': bmi.round(2),
            'HbA1c_level': hba1c_level.round(1),
            'blood_glucose_level': blood_glucose_level.round(0).astype(int),
            'diabetes': diabetes
        }

        return pd.DataFrame(data)

    def preprocess_data(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Comprehensive data preprocessing with feature engineering.

        Args:
            df (pd.DataFrame): Raw input dataframe
            is_training (bool): Whether this is training data

        Returns:
            pd.DataFrame: Processed dataframe with engineered features
        """
        df = df.copy()

        # Encode categorical variables
        if is_training:
            self.label_encoders['gender'] = LabelEncoder()
            df['gender_encoded'] = self.label_encoders['gender'].fit_transform(df['gender'])

            self.label_encoders['smoking'] = LabelEncoder()
            df['smoking_encoded'] = self.label_encoders['smoking'].fit_transform(df['smoking_history'])
        else:
            df['gender_encoded'] = self.label_encoders['gender'].transform(df['gender'])
            df['smoking_encoded'] = self.label_encoders['smoking'].transform(df['smoking_history'])

        # Feature engineering
        # BMI categories
        df['bmi_category'] = pd.cut(
            df['bmi'],
            bins=[0, 18.5, 25, 30, 35, 100],
            labels=['underweight', 'normal', 'overweight', 'obese', 'severely_obese']
        )
        df['bmi_cat_encoded'] = LabelEncoder().fit_transform(df['bmi_category'])

        # Age groups
        df['age_group'] = pd.cut(
            df['age'],
            bins=[0, 30, 45, 60, 100],
            labels=['young', 'middle', 'senior', 'elderly']
        )
        df['age_group_encoded'] = LabelEncoder().fit_transform(df['age_group'])

        # Risk factor combinations
        df['comorbidity_count'] = df['hypertension'] + df['heart_disease']
        df['high_risk_glucose'] = (df['blood_glucose_level'] > 140).astype(int)
        df['high_risk_hba1c'] = (df['HbA1c_level'] > 6.5).astype(int)
        df['metabolic_risk'] = df['high_risk_glucose'] + df['high_risk_hba1c'] + (df['bmi'] > 30).astype(int)

        # Interaction features
        df['age_bmi_interaction'] = df['age'] * df['bmi'] / 100
        df['glucose_hba1c_interaction'] = df['blood_glucose_level'] * df['HbA1c_level'] / 100

        # Current smoker flag
        df['is_current_smoker'] = (df['smoking_history'] == 'current').astype(int)

        return df

    def plot_exploratory_analysis(self, df: pd.DataFrame, output_dir: str = '.') -> None:
        """
        Create comprehensive EDA visualizations for diabetes dataset.

        Args:
            df (pd.DataFrame): Input dataframe
            output_dir (str): Directory to save plots
        """
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Diabetes Dataset - Exploratory Data Analysis', fontsize=16, y=1.02)

        # 1. Diabetes distribution
        df['diabetes'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['green', 'red'])
        axes[0, 0].set_title('Diabetes Distribution')
        axes[0, 0].set_xticklabels(['No Diabetes', 'Diabetes'], rotation=0)
        axes[0, 0].set_ylabel('Count')

        # 2. Age distribution by diabetes status
        df[df['diabetes'] == 0]['age'].hist(bins=30, ax=axes[0, 1], alpha=0.5, label='No Diabetes', color='green')
        df[df['diabetes'] == 1]['age'].hist(bins=30, ax=axes[0, 1], alpha=0.5, label='Diabetes', color='red')
        axes[0, 1].set_title('Age Distribution by Diabetes Status')
        axes[0, 1].set_xlabel('Age')
        axes[0, 1].legend()

        # 3. BMI distribution
        df['bmi'].hist(bins=30, ax=axes[0, 2], color='purple', alpha=0.7)
        axes[0, 2].set_title('BMI Distribution')
        axes[0, 2].set_xlabel('BMI')
        axes[0, 2].axvline(x=25, color='orange', linestyle='--', label='Overweight threshold')
        axes[0, 2].axvline(x=30, color='red', linestyle='--', label='Obese threshold')
        axes[0, 2].legend()

        # 4. HbA1c vs Blood Glucose scatter
        scatter = axes[1, 0].scatter(
            df['HbA1c_level'], df['blood_glucose_level'],
            c=df['diabetes'], cmap='RdYlGn_r', alpha=0.5
        )
        axes[1, 0].set_title('HbA1c vs Blood Glucose Level')
        axes[1, 0].set_xlabel('HbA1c Level')
        axes[1, 0].set_ylabel('Blood Glucose Level')
        plt.colorbar(scatter, ax=axes[1, 0], label='Diabetes')

        # 5. Diabetes rate by gender
        gender_diabetes = df.groupby('gender')['diabetes'].mean()
        gender_diabetes.plot(kind='bar', ax=axes[1, 1], color=['pink', 'lightblue'])
        axes[1, 1].set_title('Diabetes Rate by Gender')
        axes[1, 1].set_ylabel('Diabetes Rate')
        axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=0)

        # 6. Diabetes rate by smoking history
        smoking_diabetes = df.groupby('smoking_history')['diabetes'].mean().sort_values(ascending=False)
        smoking_diabetes.plot(kind='bar', ax=axes[1, 2], color='coral')
        axes[1, 2].set_title('Diabetes Rate by Smoking History')
        axes[1, 2].set_ylabel('Diabetes Rate')
        axes[1, 2].tick_params(axis='x', rotation=45)

        # 7. Comorbidity analysis
        comorbidity_diabetes = df.groupby(['hypertension', 'heart_disease'])['diabetes'].mean().unstack()
        comorbidity_diabetes.plot(kind='bar', ax=axes[2, 0])
        axes[2, 0].set_title('Diabetes Rate by Comorbidities')
        axes[2, 0].set_xlabel('Hypertension')
        axes[2, 0].set_ylabel('Diabetes Rate')
        axes[2, 0].legend(title='Heart Disease')
        axes[2, 0].set_xticklabels(['No', 'Yes'], rotation=0)

        # 8. HbA1c level by diabetes status (box plot)
        df.boxplot(column='HbA1c_level', by='diabetes', ax=axes[2, 1])
        axes[2, 1].set_title('HbA1c Level by Diabetes Status')
        axes[2, 1].set_xlabel('Diabetes (0=No, 1=Yes)')
        plt.suptitle('')  # Remove automatic title

        # 9. Feature correlation heatmap
        numeric_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level',
                       'hypertension', 'heart_disease', 'diabetes']
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[2, 2])
        axes[2, 2].set_title('Feature Correlation Matrix')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/diabetes_eda.png', dpi=300, bbox_inches='tight')
        print(f"EDA saved to {output_dir}/diabetes_eda.png")
        plt.close()

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare feature matrix and target variable.

        Args:
            df (pd.DataFrame): Processed dataframe

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target
        """
        feature_cols = [
            'age', 'bmi', 'HbA1c_level', 'blood_glucose_level',
            'hypertension', 'heart_disease', 'gender_encoded', 'smoking_encoded',
            'bmi_cat_encoded', 'age_group_encoded', 'comorbidity_count',
            'high_risk_glucose', 'high_risk_hba1c', 'metabolic_risk',
            'age_bmi_interaction', 'glucose_hba1c_interaction', 'is_current_smoker'
        ]

        self.feature_names = feature_cols
        X = df[feature_cols]
        y = df['diabetes']

        return X, y

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                    use_smote: bool = True) -> None:
        """
        Train multiple machine learning models.

        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
            use_smote (bool): Whether to use SMOTE for class balancing
        """
        # Handle class imbalance
        if use_smote and SMOTE_AVAILABLE:
            print("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        else:
            X_train_balanced, y_train_balanced = X_train, y_train

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_balanced)

        print("\nTraining multiple models...")

        # 1. Logistic Regression
        print("  - Training Logistic Regression...")
        self.models['Logistic Regression'] = LogisticRegression(
            max_iter=1000, random_state=42, class_weight='balanced'
        )
        self.models['Logistic Regression'].fit(X_train_scaled, y_train_balanced)

        # 2. Random Forest
        print("  - Training Random Forest...")
        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1,
            class_weight='balanced'
        )
        self.models['Random Forest'].fit(X_train_scaled, y_train_balanced)

        # 3. Gradient Boosting
        print("  - Training Gradient Boosting...")
        self.models['Gradient Boosting'] = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5,
            random_state=42
        )
        self.models['Gradient Boosting'].fit(X_train_scaled, y_train_balanced)

        # 4. XGBoost
        if XGBOOST_AVAILABLE:
            print("  - Training XGBoost...")
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            self.models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5,
                random_state=42, scale_pos_weight=scale_pos_weight,
                use_label_encoder=False, eval_metric='logloss'
            )
            self.models['XGBoost'].fit(X_train_scaled, y_train_balanced)

        # 5. LightGBM
        if LIGHTGBM_AVAILABLE:
            print("  - Training LightGBM...")
            self.models['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5,
                random_state=42, is_unbalance=True, verbose=-1
            )
            self.models['LightGBM'].fit(X_train_scaled, y_train_balanced)

        print(f"\nTrained {len(self.models)} models successfully!")

    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Evaluate all trained models with focus on medical metrics.

        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels

        Returns:
            pd.DataFrame: Model comparison results
        """
        X_test_scaled = self.scaler.transform(X_test)
        results = []

        print("\n=== Model Evaluation Results ===")
        print("(Note: For medical diagnosis, high Recall is crucial)")

        for name, model in self.models.items():
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC-AUC': roc_auc
            })

            print(f"\n{name}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  ROC-AUC: {roc_auc:.4f}")

        results_df = pd.DataFrame(results)
        # Sort by recall (important for medical diagnosis)
        results_df = results_df.sort_values('Recall', ascending=False)

        # Select best model based on recall
        best_model_name = results_df.iloc[0]['Model']
        self.best_model = self.models[best_model_name]

        return results_df

    def plot_model_comparison(self, results_df: pd.DataFrame, output_dir: str = '.') -> None:
        """
        Visualize model performance comparison.

        Args:
            results_df (pd.DataFrame): Evaluation results
            output_dir (str): Directory to save plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performance Comparison - Diabetes Prediction', fontsize=14)

        metrics = ['Accuracy', 'Recall', 'Precision', 'ROC-AUC']
        colors = ['steelblue', 'coral', 'green', 'purple']

        for idx, (metric, color) in enumerate(zip(metrics, colors)):
            ax = axes[idx // 2, idx % 2]
            data = results_df.sort_values(metric, ascending=True)
            ax.barh(data['Model'], data[metric], color=color)
            ax.set_xlabel(metric)
            ax.set_title(f'{metric} by Model')
            ax.set_xlim([0.6, 1.0])

        plt.tight_layout()
        plt.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Model comparison saved to {output_dir}/model_comparison.png")
        plt.close()

    def plot_confusion_matrix(self, X_test: pd.DataFrame, y_test: pd.Series,
                             output_dir: str = '.') -> None:
        """
        Plot confusion matrix for the best model.

        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels
            output_dir (str): Directory to save plot
        """
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.best_model.predict(X_test_scaled)

        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Diabetes', 'Diabetes'],
                   yticklabels=['No Diabetes', 'Diabetes'])
        plt.title('Confusion Matrix - Best Model')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')

        # Add metrics text
        tn, fp, fn, tp = cm.ravel()
        plt.text(0.5, -0.15,
                f'True Negatives: {tn} | False Positives: {fp} | False Negatives: {fn} | True Positives: {tp}',
                ha='center', transform=plt.gca().transAxes, fontsize=10)

        plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {output_dir}/confusion_matrix.png")
        plt.close()

    def plot_roc_curves(self, X_test: pd.DataFrame, y_test: pd.Series,
                       output_dir: str = '.') -> None:
        """
        Plot ROC curves for all models.

        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels
            output_dir (str): Directory to save plot
        """
        X_test_scaled = self.scaler.transform(X_test)

        plt.figure(figsize=(10, 8))

        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate (Recall)')
        plt.title('ROC Curves - Diabetes Prediction Models')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)

        plt.savefig(f'{output_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {output_dir}/roc_curves.png")
        plt.close()

    def plot_feature_importance(self, output_dir: str = '.') -> None:
        """
        Plot feature importance from tree-based models.

        Args:
            output_dir (str): Directory to save plot
        """
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(12, 8))
            plt.title('Feature Importance - Diabetes Prediction')
            plt.bar(range(len(importances)), importances[indices], color='steelblue')
            plt.xticks(range(len(importances)),
                      [self.feature_names[i] for i in indices],
                      rotation=45, ha='right')
            plt.ylabel('Importance Score')
            plt.tight_layout()

            plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
            print(f"Feature importance saved to {output_dir}/feature_importance.png")
            plt.close()

    def generate_risk_report(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a risk assessment report for a single patient.

        Args:
            patient_data (Dict): Patient health indicators

        Returns:
            Dict[str, Any]: Risk assessment report
        """
        # Create dataframe from patient data
        df = pd.DataFrame([patient_data])
        df_processed = self.preprocess_data(df, is_training=False)
        X, _ = self.prepare_features(df_processed)
        X_scaled = self.scaler.transform(X)

        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            prob = model.predict_proba(X_scaled)[0, 1]
            predictions[name] = prob

        avg_risk = np.mean(list(predictions.values()))

        report = {
            'patient_data': patient_data,
            'risk_scores': predictions,
            'average_risk': avg_risk,
            'risk_level': 'High' if avg_risk > 0.5 else ('Medium' if avg_risk > 0.3 else 'Low'),
            'recommendations': self._generate_recommendations(patient_data, avg_risk)
        }

        return report

    def _generate_recommendations(self, patient_data: Dict, risk: float) -> List[str]:
        """Generate health recommendations based on risk factors."""
        recommendations = []

        if risk > 0.5:
            recommendations.append("Schedule a comprehensive diabetes screening immediately")

        if patient_data.get('bmi', 0) > 30:
            recommendations.append("Consider weight management program")

        if patient_data.get('HbA1c_level', 0) > 6.5:
            recommendations.append("Monitor blood sugar levels closely")

        if patient_data.get('smoking_history') == 'current':
            recommendations.append("Consider smoking cessation program")

        if patient_data.get('hypertension', 0) == 1:
            recommendations.append("Manage blood pressure with lifestyle changes or medication")

        if not recommendations:
            recommendations.append("Maintain healthy lifestyle and regular check-ups")

        return recommendations


def main():
    """Main execution function."""
    print("=" * 80)
    print("DIABETES PREDICTION - COMPREHENSIVE ML SOLUTION")
    print("=" * 80)

    # Initialize model
    predictor = DiabetesPredictionModel()

    # Create sample data
    print("\nCreating sample dataset...")
    df = predictor.create_sample_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Diabetes rate: {df['diabetes'].mean():.2%}")

    # Preprocess data
    print("\nPreprocessing data with feature engineering...")
    df_processed = predictor.preprocess_data(df)

    # Generate EDA
    print("\nGenerating exploratory data analysis...")
    predictor.plot_exploratory_analysis(df)

    # Prepare features
    X, y = predictor.prepare_features(df_processed)
    print(f"\nFeature matrix shape: {X.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Train models
    predictor.train_models(X_train, y_train)

    # Evaluate models
    results_df = predictor.evaluate_models(X_test, y_test)
    print(f"\n{results_df.to_string(index=False)}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    predictor.plot_model_comparison(results_df)
    predictor.plot_confusion_matrix(X_test, y_test)
    predictor.plot_roc_curves(X_test, y_test)
    predictor.plot_feature_importance()

    # Generate sample risk report
    print("\n" + "=" * 80)
    print("SAMPLE RISK ASSESSMENT")
    print("=" * 80)

    sample_patient = {
        'gender': 'Male',
        'age': 55,
        'hypertension': 1,
        'heart_disease': 0,
        'smoking_history': 'former',
        'bmi': 32.5,
        'HbA1c_level': 6.8,
        'blood_glucose_level': 160
    }

    report = predictor.generate_risk_report(sample_patient)
    print(f"\nPatient Profile: {report['patient_data']}")
    print(f"Average Risk Score: {report['average_risk']:.2%}")
    print(f"Risk Level: {report['risk_level']}")
    print(f"Recommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nBest Model: {results_df.iloc[0]['Model']}")
    print(f"Best Recall: {results_df.iloc[0]['Recall']:.4f}")
    print(f"Best ROC-AUC: {results_df.iloc[0]['ROC-AUC']:.4f}")


if __name__ == "__main__":
    main()
