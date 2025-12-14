"""
Student Performance Prediction - Education Data Analysis

This module predicts student academic performance based on demographic
and educational factors.

Dataset: https://www.kaggle.com/datasets/spscientist/students-performance-in-exams
Difficulty: ⭐⭐ Intermediate Level
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, classification_report, confusion_matrix
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class StudentPerformancePredictor:
    """Student Academic Performance Prediction Model."""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.best_model = None
        self.feature_names: List[str] = []

    def create_sample_data(self) -> pd.DataFrame:
        """Create realistic student performance dataset."""
        np.random.seed(42)
        n_students = 1000

        # Gender
        gender = np.random.choice(['female', 'male'], n_students, p=[0.52, 0.48])

        # Race/ethnicity
        race_ethnicity = np.random.choice(
            ['group A', 'group B', 'group C', 'group D', 'group E'],
            n_students, p=[0.15, 0.20, 0.30, 0.25, 0.10]
        )

        # Parental education
        parental_education = np.random.choice(
            ["some high school", "high school", "some college",
             "associate's degree", "bachelor's degree", "master's degree"],
            n_students, p=[0.15, 0.20, 0.25, 0.20, 0.15, 0.05]
        )

        # Lunch type
        lunch = np.random.choice(['standard', 'free/reduced'], n_students, p=[0.65, 0.35])

        # Test preparation
        test_preparation_course = np.random.choice(
            ['none', 'completed'], n_students, p=[0.64, 0.36]
        )

        # Generate base scores with various influences
        base_math = np.random.normal(66, 15, n_students)
        base_reading = np.random.normal(69, 14, n_students)
        base_writing = np.random.normal(68, 15, n_students)

        # Apply effects
        for i in range(n_students):
            # Gender effect
            if gender[i] == 'female':
                base_reading[i] += 5
                base_writing[i] += 5
                base_math[i] -= 2

            # Parental education effect
            edu_boost = {
                "some high school": -5, "high school": 0, "some college": 3,
                "associate's degree": 5, "bachelor's degree": 8, "master's degree": 12
            }
            boost = edu_boost[parental_education[i]]
            base_math[i] += boost
            base_reading[i] += boost
            base_writing[i] += boost

            # Lunch type effect (socioeconomic indicator)
            if lunch[i] == 'free/reduced':
                base_math[i] -= 8
                base_reading[i] -= 8
                base_writing[i] -= 8

            # Test prep effect
            if test_preparation_course[i] == 'completed':
                base_math[i] += 5
                base_reading[i] += 6
                base_writing[i] += 7

        # Clip scores to valid range
        math_score = np.clip(base_math, 0, 100).round(0).astype(int)
        reading_score = np.clip(base_reading, 0, 100).round(0).astype(int)
        writing_score = np.clip(base_writing, 0, 100).round(0).astype(int)

        return pd.DataFrame({
            'gender': gender,
            'race/ethnicity': race_ethnicity,
            'parental level of education': parental_education,
            'lunch': lunch,
            'test preparation course': test_preparation_course,
            'math score': math_score,
            'reading score': reading_score,
            'writing score': writing_score
        })

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess and engineer features."""
        df = df.copy()

        # Encode categorical variables
        cat_cols = ['gender', 'race/ethnicity', 'parental level of education',
                   'lunch', 'test preparation course']

        for col in cat_cols:
            self.label_encoders[col] = LabelEncoder()
            df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])

        # Feature engineering
        df['total_score'] = df['math score'] + df['reading score'] + df['writing score']
        df['average_score'] = df['total_score'] / 3

        # Performance grade
        df['grade'] = pd.cut(
            df['average_score'],
            bins=[0, 50, 60, 70, 80, 90, 100],
            labels=['F', 'D', 'C', 'B', 'A', 'A+']
        )
        df['grade_encoded'] = LabelEncoder().fit_transform(df['grade'])

        # Subject strength indicators
        df['math_strength'] = df['math score'] - df['average_score']
        df['reading_strength'] = df['reading score'] - df['average_score']
        df['writing_strength'] = df['writing score'] - df['average_score']

        # Reading-Writing correlation strength
        df['literacy_score'] = (df['reading score'] + df['writing score']) / 2

        # Pass/fail flags
        df['math_pass'] = (df['math score'] >= 50).astype(int)
        df['reading_pass'] = (df['reading score'] >= 50).astype(int)
        df['writing_pass'] = (df['writing score'] >= 50).astype(int)
        df['all_pass'] = df['math_pass'] & df['reading_pass'] & df['writing_pass']

        return df

    def plot_exploratory_analysis(self, df: pd.DataFrame, output_dir: str = '.') -> None:
        """Generate EDA visualizations."""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Student Performance Analysis - EDA', fontsize=16)

        # Score distributions
        df['math score'].hist(bins=20, ax=axes[0, 0], color='steelblue', alpha=0.7)
        axes[0, 0].set_title('Math Score Distribution')
        axes[0, 0].axvline(x=df['math score'].mean(), color='red', linestyle='--', label='Mean')
        axes[0, 0].legend()

        df['reading score'].hist(bins=20, ax=axes[0, 1], color='coral', alpha=0.7)
        axes[0, 1].set_title('Reading Score Distribution')

        df['writing score'].hist(bins=20, ax=axes[0, 2], color='green', alpha=0.7)
        axes[0, 2].set_title('Writing Score Distribution')

        # Gender comparison
        df.groupby('gender')[['math score', 'reading score', 'writing score']].mean().plot(
            kind='bar', ax=axes[1, 0]
        )
        axes[1, 0].set_title('Average Scores by Gender')
        axes[1, 0].tick_params(axis='x', rotation=0)

        # Parental education effect
        edu_order = ["some high school", "high school", "some college",
                    "associate's degree", "bachelor's degree", "master's degree"]
        df_edu = df.groupby('parental level of education')['average_score'].mean()
        df_edu = df_edu.reindex(edu_order)
        df_edu.plot(kind='bar', ax=axes[1, 1], color='purple')
        axes[1, 1].set_title('Average Score by Parental Education')
        axes[1, 1].tick_params(axis='x', rotation=45)

        # Test preparation effect
        df.groupby('test preparation course')[['math score', 'reading score', 'writing score']].mean().plot(
            kind='bar', ax=axes[1, 2]
        )
        axes[1, 2].set_title('Scores by Test Preparation')
        axes[1, 2].tick_params(axis='x', rotation=0)

        # Lunch type effect
        df.groupby('lunch')['average_score'].mean().plot(kind='bar', ax=axes[2, 0], color='orange')
        axes[2, 0].set_title('Average Score by Lunch Type')
        axes[2, 0].tick_params(axis='x', rotation=0)

        # Score correlations
        score_cols = ['math score', 'reading score', 'writing score']
        sns.heatmap(df[score_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=axes[2, 1])
        axes[2, 1].set_title('Score Correlations')

        # Grade distribution
        df['grade'].value_counts().sort_index().plot(kind='bar', ax=axes[2, 2], color='teal')
        axes[2, 2].set_title('Grade Distribution')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/student_eda.png', dpi=300, bbox_inches='tight')
        print(f"EDA saved to {output_dir}/student_eda.png")
        plt.close()

    def train_regression_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train regression models for score prediction."""
        X_train_scaled = self.scaler.fit_transform(X_train)

        print("\nTraining regression models...")

        self.models['Linear Regression'] = LinearRegression()
        self.models['Linear Regression'].fit(X_train_scaled, y_train)

        self.models['Random Forest'] = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        self.models['Random Forest'].fit(X_train_scaled, y_train)

        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            )
            self.models['XGBoost'].fit(X_train_scaled, y_train)

        print(f"Trained {len(self.models)} models!")

    def evaluate_regression_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """Evaluate regression models."""
        X_test_scaled = self.scaler.transform(X_test)
        results = []

        print("\n=== Regression Model Evaluation ===")

        for name, model in self.models.items():
            y_pred = model.predict(X_test_scaled)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results.append({
                'Model': name,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            })

            print(f"{name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}")

        results_df = pd.DataFrame(results).sort_values('R2', ascending=False)
        self.best_model = self.models[results_df.iloc[0]['Model']]
        return results_df

    def plot_results(self, results_df: pd.DataFrame, X_test: pd.DataFrame,
                    y_test: pd.Series, output_dir: str = '.') -> None:
        """Visualize model results."""
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.best_model.predict(X_test_scaled)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Model comparison
        results_df.set_index('Model')[['R2']].plot(kind='bar', ax=axes[0, 0], color='steelblue')
        axes[0, 0].set_title('Model R² Comparison')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Actual vs Predicted
        axes[0, 1].scatter(y_test, y_pred, alpha=0.5)
        axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        axes[0, 1].set_title('Actual vs Predicted Scores')
        axes[0, 1].set_xlabel('Actual')
        axes[0, 1].set_ylabel('Predicted')

        # Residuals
        residuals = y_test - y_pred
        axes[1, 0].hist(residuals, bins=30, color='coral', alpha=0.7)
        axes[1, 0].set_title('Residual Distribution')
        axes[1, 0].set_xlabel('Residual')

        # Feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            indices = np.argsort(importance)[-10:]
            axes[1, 1].barh(range(len(indices)), importance[indices], color='green')
            axes[1, 1].set_yticks(range(len(indices)))
            axes[1, 1].set_yticklabels([self.feature_names[i] for i in indices])
            axes[1, 1].set_title('Top Feature Importance')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/student_results.png', dpi=300, bbox_inches='tight')
        print(f"Results saved to {output_dir}/student_results.png")
        plt.close()


def main():
    """Main execution."""
    print("=" * 70)
    print("STUDENT PERFORMANCE PREDICTION")
    print("=" * 70)

    predictor = StudentPerformancePredictor()

    # Create and preprocess data
    df = predictor.create_sample_data()
    print(f"\nDataset: {df.shape}")

    df = predictor.preprocess_data(df)
    predictor.plot_exploratory_analysis(df)

    # Prepare features for predicting average score
    feature_cols = ['gender_encoded', 'race/ethnicity_encoded',
                   'parental level of education_encoded', 'lunch_encoded',
                   'test preparation course_encoded']
    predictor.feature_names = feature_cols

    X = df[feature_cols]
    y = df['average_score']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train and evaluate
    predictor.train_regression_models(X_train, y_train)
    results = predictor.evaluate_regression_models(X_test, y_test)

    print(f"\n{results.to_string(index=False)}")

    predictor.plot_results(results, X_test, y_test)

    print("\n" + "=" * 70)
    print(f"Best Model: {results.iloc[0]['Model']}")
    print(f"Best R²: {results.iloc[0]['R2']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
