"""
Athlete Performance Prediction

Predict athlete overall rating using training data, physical metrics
and historical performance for sports analytics.

Dataset: https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset
Difficulty: ⭐⭐⭐ Advanced Level
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class AthletePerformanceModel:
    """Athlete Performance Prediction Model."""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.best_model = None

    def create_sample_data(self, n_samples: int = 2000) -> pd.DataFrame:
        """Create synthetic athlete dataset."""
        np.random.seed(42)

        positions = ['ST', 'CF', 'LW', 'RW', 'CAM', 'CM', 'CDM', 'LB', 'RB', 'CB', 'GK']
        position_probs = [0.08, 0.05, 0.08, 0.08, 0.08, 0.12, 0.08, 0.1, 0.1, 0.15, 0.08]

        data = []

        for _ in range(n_samples):
            position = np.random.choice(positions, p=position_probs)

            # Age distribution (peak around 27)
            age = int(np.clip(np.random.normal(27, 4), 17, 40))

            # Physical attributes based on position
            if position == 'GK':
                height = int(np.random.normal(188, 5))
                weight = int(np.random.normal(85, 8))
            elif position in ['CB', 'ST']:
                height = int(np.random.normal(185, 6))
                weight = int(np.random.normal(80, 8))
            else:
                height = int(np.random.normal(178, 6))
                weight = int(np.random.normal(73, 7))

            height = np.clip(height, 160, 205)
            weight = np.clip(weight, 55, 100)

            # Base skill level (age curve)
            age_factor = 1 - 0.02 * abs(age - 27)
            base_skill = np.random.normal(65, 15) * age_factor

            # Skills based on position
            if position == 'GK':
                pace = int(np.random.normal(45, 10))
                shooting = int(np.random.normal(15, 8))
                passing = int(np.random.normal(40, 12))
                dribbling = int(np.random.normal(20, 8))
                defending = int(np.random.normal(15, 8))
                physic = int(np.random.normal(70, 10))
                gk_skills = int(base_skill + np.random.normal(10, 5))
            elif position in ['ST', 'CF', 'LW', 'RW']:
                pace = int(base_skill + np.random.normal(5, 8))
                shooting = int(base_skill + np.random.normal(8, 8))
                passing = int(base_skill + np.random.normal(-5, 8))
                dribbling = int(base_skill + np.random.normal(5, 8))
                defending = int(base_skill + np.random.normal(-25, 10))
                physic = int(base_skill + np.random.normal(-5, 8))
                gk_skills = int(np.random.normal(10, 5))
            elif position in ['CB', 'LB', 'RB']:
                pace = int(base_skill + np.random.normal(-5, 8))
                shooting = int(base_skill + np.random.normal(-20, 10))
                passing = int(base_skill + np.random.normal(-10, 8))
                dribbling = int(base_skill + np.random.normal(-15, 8))
                defending = int(base_skill + np.random.normal(10, 8))
                physic = int(base_skill + np.random.normal(5, 8))
                gk_skills = int(np.random.normal(10, 5))
            else:  # Midfielders
                pace = int(base_skill + np.random.normal(0, 8))
                shooting = int(base_skill + np.random.normal(-5, 8))
                passing = int(base_skill + np.random.normal(10, 8))
                dribbling = int(base_skill + np.random.normal(5, 8))
                defending = int(base_skill + np.random.normal(-10, 10))
                physic = int(base_skill + np.random.normal(0, 8))
                gk_skills = int(np.random.normal(10, 5))

            # Clip skills
            pace = np.clip(pace, 30, 99)
            shooting = np.clip(shooting, 10, 99)
            passing = np.clip(passing, 20, 99)
            dribbling = np.clip(dribbling, 20, 99)
            defending = np.clip(defending, 10, 99)
            physic = np.clip(physic, 30, 99)
            gk_skills = np.clip(gk_skills, 5, 99)

            # Calculate overall rating
            if position == 'GK':
                overall = int(0.7 * gk_skills + 0.1 * physic + 0.1 * pace + 0.1 * passing)
            elif position in ['ST', 'CF']:
                overall = int(0.25 * pace + 0.35 * shooting + 0.15 * dribbling +
                            0.1 * passing + 0.15 * physic)
            elif position in ['LW', 'RW']:
                overall = int(0.25 * pace + 0.2 * shooting + 0.25 * dribbling +
                            0.15 * passing + 0.15 * physic)
            elif position in ['CB']:
                overall = int(0.1 * pace + 0.35 * defending + 0.2 * physic +
                            0.15 * passing + 0.2 * (height - 150) / 50 * 20 + 40)
            elif position in ['LB', 'RB']:
                overall = int(0.2 * pace + 0.25 * defending + 0.2 * physic +
                            0.15 * passing + 0.2 * dribbling)
            else:  # Midfielders
                overall = int(0.15 * pace + 0.1 * shooting + 0.3 * passing +
                            0.25 * dribbling + 0.1 * defending + 0.1 * physic)

            overall = np.clip(overall + np.random.normal(0, 3), 45, 99)

            data.append({
                'age': age,
                'height_cm': height,
                'weight_kg': weight,
                'position': position,
                'pace': pace,
                'shooting': shooting,
                'passing': passing,
                'dribbling': dribbling,
                'defending': defending,
                'physic': physic,
                'gk_skills': gk_skills,
                'overall': int(overall)
            })

        return pd.DataFrame(data)

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features."""
        df = df.copy()

        # BMI
        df['bmi'] = df['weight_kg'] / (df['height_cm'] / 100) ** 2

        # Skill combinations
        df['attack_skills'] = (df['pace'] + df['shooting'] + df['dribbling']) / 3
        df['defense_skills'] = (df['defending'] + df['physic']) / 2
        df['technical_skills'] = (df['passing'] + df['dribbling']) / 2

        # Age groups
        df['age_group'] = pd.cut(df['age'], bins=[16, 23, 28, 33, 41],
                                  labels=['Young', 'Prime', 'Experienced', 'Veteran'])

        # Position categories
        position_type = {
            'ST': 'Forward', 'CF': 'Forward', 'LW': 'Forward', 'RW': 'Forward',
            'CAM': 'Midfielder', 'CM': 'Midfielder', 'CDM': 'Midfielder',
            'LB': 'Defender', 'RB': 'Defender', 'CB': 'Defender',
            'GK': 'Goalkeeper'
        }
        df['position_type'] = df['position'].map(position_type)

        return df

    def analyze_data(self, df: pd.DataFrame, output_dir: str = '.') -> None:
        """Perform exploratory data analysis."""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Athlete Performance Analysis', fontsize=16)

        # Overall distribution
        axes[0, 0].hist(df['overall'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 0].set_title('Overall Rating Distribution')
        axes[0, 0].set_xlabel('Overall Rating')
        axes[0, 0].set_ylabel('Count')

        # Age vs Overall
        axes[0, 1].scatter(df['age'], df['overall'], alpha=0.5, c=df['overall'], cmap='viridis')
        axes[0, 1].set_title('Age vs Overall Rating')
        axes[0, 1].set_xlabel('Age')
        axes[0, 1].set_ylabel('Overall Rating')

        # Position distribution
        df['position'].value_counts().plot(kind='bar', ax=axes[0, 2], color='steelblue')
        axes[0, 2].set_title('Position Distribution')
        axes[0, 2].tick_params(axis='x', rotation=45)

        # Skills radar (average by position type)
        skill_cols = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
        pos_skills = df.groupby('position_type')[skill_cols].mean()
        pos_skills.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Average Skills by Position Type')
        axes[1, 0].legend(loc='upper right', fontsize=8)
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Age curve
        age_overall = df.groupby('age')['overall'].mean()
        axes[1, 1].plot(age_overall.index, age_overall.values, 'b-', linewidth=2)
        axes[1, 1].fill_between(age_overall.index, age_overall.values, alpha=0.3)
        axes[1, 1].set_title('Performance Age Curve')
        axes[1, 1].set_xlabel('Age')
        axes[1, 1].set_ylabel('Average Overall')
        axes[1, 1].grid(True, alpha=0.3)

        # Height vs Weight by position
        for pos_type in df['position_type'].unique():
            subset = df[df['position_type'] == pos_type]
            axes[1, 2].scatter(subset['height_cm'], subset['weight_kg'],
                              alpha=0.5, label=pos_type)
        axes[1, 2].set_title('Height vs Weight by Position')
        axes[1, 2].set_xlabel('Height (cm)')
        axes[1, 2].set_ylabel('Weight (kg)')
        axes[1, 2].legend()

        # Correlation matrix
        corr_cols = ['overall', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
        corr = df[corr_cols].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[2, 0], center=0)
        axes[2, 0].set_title('Skill Correlations')

        # BMI distribution by position
        df.boxplot(column='bmi', by='position_type', ax=axes[2, 1])
        axes[2, 1].set_title('BMI by Position Type')
        plt.suptitle('')

        # Overall by position type
        df.boxplot(column='overall', by='position_type', ax=axes[2, 2])
        axes[2, 2].set_title('Overall Rating by Position Type')
        plt.suptitle('')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/athlete_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Analysis saved to {output_dir}/athlete_analysis.png")
        plt.close()

    def prepare_features(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for modeling."""
        # Numeric features
        numeric_cols = ['age', 'height_cm', 'weight_kg', 'pace', 'shooting',
                       'passing', 'dribbling', 'defending', 'physic',
                       'bmi', 'attack_skills', 'defense_skills', 'technical_skills']

        X_numeric = df[numeric_cols].values

        # Encode categoricals
        for col in ['position', 'position_type', 'age_group']:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                encoded = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                encoded = self.label_encoders[col].transform(df[col].astype(str))
            X_numeric = np.column_stack([X_numeric, encoded])

        if fit:
            X = self.scaler.fit_transform(X_numeric)
        else:
            X = self.scaler.transform(X_numeric)

        y = df['overall'].values

        return X, y

    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train regression models."""
        print("\nTraining models...")

        self.models['Linear Regression'] = LinearRegression()
        self.models['Linear Regression'].fit(X_train, y_train)

        self.models['Ridge Regression'] = Ridge(alpha=1.0)
        self.models['Ridge Regression'].fit(X_train, y_train)

        self.models['Random Forest'] = RandomForestRegressor(
            n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
        )
        self.models['Random Forest'].fit(X_train, y_train)

        self.models['Gradient Boosting'] = GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )
        self.models['Gradient Boosting'].fit(X_train, y_train)

        print(f"Trained {len(self.models)} models!")

    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """Evaluate all models."""
        results = []

        for name, model in self.models.items():
            y_pred = model.predict(X_test)

            results.append({
                'Model': name,
                'MAE': mean_absolute_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'R²': r2_score(y_test, y_pred)
            })

        results_df = pd.DataFrame(results).sort_values('MAE')
        self.best_model = self.models[results_df.iloc[0]['Model']]

        return results_df

    def plot_results(self, results: pd.DataFrame, X_test: np.ndarray,
                    y_test: np.ndarray, output_dir: str = '.') -> None:
        """Visualize results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Athlete Performance Prediction Results', fontsize=16)

        # Model comparison
        colors = ['steelblue', 'green', 'orange', 'red']
        results.set_index('Model')[['MAE', 'RMSE']].plot(kind='bar', ax=axes[0, 0], color=colors[:2])
        axes[0, 0].set_title('Model Error Comparison')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_ylabel('Error')

        # R² comparison
        results.set_index('Model')['R²'].plot(kind='bar', ax=axes[0, 1], color='steelblue')
        axes[0, 1].set_title('R² Score Comparison')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].set_ylim(0, 1)

        # Predictions vs Actual
        y_pred = self.best_model.predict(X_test)
        axes[1, 0].scatter(y_test, y_pred, alpha=0.5, c='steelblue')
        axes[1, 0].plot([y_test.min(), y_test.max()],
                        [y_test.min(), y_test.max()], 'r--', linewidth=2)
        axes[1, 0].set_xlabel('Actual Overall')
        axes[1, 0].set_ylabel('Predicted Overall')
        axes[1, 0].set_title('Actual vs Predicted (Best Model)')

        # Residual distribution
        residuals = y_test - y_pred
        axes[1, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[1, 1].axvline(x=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Residual')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Residual Distribution')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/athlete_results.png', dpi=300, bbox_inches='tight')
        print(f"Results saved to {output_dir}/athlete_results.png")
        plt.close()


def main():
    """Main execution."""
    print("=" * 70)
    print("ATHLETE PERFORMANCE PREDICTION")
    print("=" * 70)

    model = AthletePerformanceModel()

    # Create data
    print("\nCreating synthetic dataset...")
    df = model.create_sample_data(n_samples=2000)
    df = model.feature_engineering(df)
    print(f"Dataset shape: {df.shape}")

    # Analysis
    model.analyze_data(df)

    # Prepare features
    X, y = model.prepare_features(df, fit=True)
    print(f"\nFeature matrix shape: {X.shape}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train and evaluate
    model.train_models(X_train, y_train)
    results = model.evaluate_models(X_test, y_test)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(results.to_string(index=False))

    # Visualize
    model.plot_results(results, X_test, y_test)

    print("\n" + "=" * 70)
    best = results.iloc[0]
    print(f"Best Model: {best['Model']}")
    print(f"MAE: {best['MAE']:.2f}")
    print(f"R²: {best['R²']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
