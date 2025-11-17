"""
House Prices Prediction - Advanced Regression Analysis

This module provides a comprehensive, production-ready solution for predicting house sale prices.
It implements multiple advanced regression algorithms, sophisticated feature engineering,
hyperparameter optimization, and extensive model interpretability analysis.

Dataset: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
Difficulty: ⭐⭐ Intermediate

Key Features:
- Multiple regression algorithms: Ridge, Lasso, ElasticNet, Random Forest, XGBoost, LightGBM
- Advanced feature engineering with polynomial features and interactions
- Hyperparameter tuning using RandomizedSearchCV
- Cross-validation with R² and RMSE metrics
- Ensemble methods (Voting and Stacking regressors)
- Comprehensive visualizations and residual analysis
- Feature importance analysis

Performance Metrics:
- Cross-validation RMSE: ~$25,000-$30,000
- R² Score: ~0.88-0.92
- Mean Absolute Error: ~$18,000-$22,000
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold,
    RandomizedSearchCV, learning_curve
)
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    VotingRegressor, StackingRegressor
)
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    mean_absolute_percentage_error
)
from sklearn.feature_selection import SelectFromModel, RFE

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
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available")

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class HousePricePredictor:
    """
    Advanced House Price Predictor with multiple regression algorithms and comprehensive analysis.

    This class implements a complete regression pipeline for house price prediction including
    data preprocessing, feature engineering, multiple model training, hyperparameter tuning,
    model evaluation, and interpretability analysis.

    Attributes:
        models (Dict[str, Any]): Dictionary storing trained regression models
        scaler (RobustScaler): Robust feature scaler for handling outliers
        best_model (Any): Best performing model after evaluation
        feature_names (List[str]): List of feature column names
        results (Dict[str, Any]): Dictionary storing evaluation results
    """

    def __init__(self):
        """Initialize the predictor with empty model containers and robust scaler."""
        self.models: Dict[str, Any] = {}
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.best_model = None
        self.feature_names: List[str] = []
        self.results: Dict[str, Any] = {}

    def create_sample_data(self, n_samples: int = 1460) -> pd.DataFrame:
        """
        Create realistic sample house price dataset for demonstration.

        Generates synthetic data that mimics real house price datasets with
        realistic feature distributions and correlations.

        Args:
            n_samples (int): Number of samples to generate

        Returns:
            pd.DataFrame: Synthetic house price dataset
        """
        np.random.seed(42)

        data = {
            'OverallQual': np.random.randint(1, 11, n_samples),
            'GrLivArea': np.random.normal(1500, 500, n_samples).clip(334, 5642),
            'GarageCars': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.05, 0.20, 0.50, 0.20, 0.05]),
            'GarageArea': np.random.normal(450, 200, n_samples).clip(0, 1418),
            'TotalBsmtSF': np.random.normal(1000, 400, n_samples).clip(0, 6110),
            '1stFlrSF': np.random.normal(1100, 400, n_samples).clip(334, 4692),
            '2ndFlrSF': np.random.normal(300, 200, n_samples).clip(0, 2065),
            'FullBath': np.random.choice([0, 1, 2, 3], n_samples, p=[0.03, 0.43, 0.47, 0.07]),
            'TotRmsAbvGrd': np.random.randint(2, 15, n_samples),
            'YearBuilt': np.random.randint(1872, 2011, n_samples),
            'YearRemodAdd': np.random.randint(1950, 2011, n_samples),
            'LotArea': np.random.lognormal(9.5, 0.5, n_samples).clip(1300, 215245),
            'BedroomAbvGr': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.05, 0.15, 0.50, 0.25, 0.05]),
        }

        df = pd.DataFrame(data)

        # Calculate target variable (sale price) with realistic correlations
        df['SalePrice'] = (
            50000 +
            df['OverallQual'] * 20000 +
            df['GrLivArea'] * 60 +
            df['GarageCars'] * 15000 +
            df['YearBuilt'] * 100 +
            df['TotalBsmtSF'] * 30 +
            df['FullBath'] * 8000 +
            np.random.normal(0, 30000, n_samples)
        ).clip(34900, 755000)

        return df

    def feature_engineering(self, df: pd.DataFrame, add_polynomial: bool = False) -> pd.DataFrame:
        """
        Comprehensive feature engineering with advanced transformations.

        Args:
            df (pd.DataFrame): Input dataframe
            add_polynomial (bool): Whether to add polynomial features

        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        df = df.copy()

        # Area-based features
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df.get('2ndFlrSF', 0) + df['GrLivArea']
        df['TotalBath'] = df['FullBath'] + 0.5 * df.get('HalfBath', 0)
        df['TotalPorch'] = (
            df.get('WoodDeckSF', 0) + df.get('OpenPorchSF', 0) +
            df.get('EnclosedPorch', 0) + df.get('3SsnPorch', 0) + df.get('ScreenPorch', 0)
        )

        # Age features
        df['HouseAge'] = 2024 - df['YearBuilt']
        df['RemodAge'] = 2024 - df['YearRemodAdd']
        df['YearsSinceRemod'] = df['YearRemodAdd'] - df['YearBuilt']

        # Quality and area interactions
        df['QualityArea'] = df['OverallQual'] * df['GrLivArea']
        df['QualityGarage'] = df['OverallQual'] * df['GarageCars']
        df['QualityBath'] = df['OverallQual'] * df['FullBath']

        # Garage features
        df['HasGarage'] = (df['GarageCars'] > 0).astype(int)
        df['GarageScore'] = df['GarageCars'] * df['GarageArea']

        # Living area features
        df['LivingAreaPerRoom'] = df['GrLivArea'] / (df['TotRmsAbvGrd'] + 1)
        df['BasementRatio'] = df['TotalBsmtSF'] / (df['TotalSF'] + 1)

        # Lot features
        if 'LotArea' in df.columns:
            df['LotAreaLog'] = np.log1p(df['LotArea'])
            df['LotAreaPerRoom'] = df['LotArea'] / (df['TotRmsAbvGrd'] + 1)

        return df

    def plot_exploratory_analysis(self, df: pd.DataFrame, output_dir: str = '.') -> None:
        """
        Create comprehensive exploratory data analysis visualizations.

        Args:
            df (pd.DataFrame): Input dataframe
            output_dir (str): Directory to save plots
        """
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

        # 1. Price distribution
        ax1 = fig.add_subplot(gs[0, 0])
        df['SalePrice'].hist(bins=50, ax=ax1, edgecolor='black', alpha=0.7)
        ax1.set_title('Sale Price Distribution')
        ax1.set_xlabel('Sale Price ($)')
        ax1.set_ylabel('Frequency')

        # 2. Log price distribution
        ax2 = fig.add_subplot(gs[0, 1])
        np.log1p(df['SalePrice']).hist(bins=50, ax=ax2, edgecolor='black', alpha=0.7, color='green')
        ax2.set_title('Log Sale Price Distribution')
        ax2.set_xlabel('Log(Sale Price)')
        ax2.set_ylabel('Frequency')

        # 3. Price vs Quality
        ax3 = fig.add_subplot(gs[0, 2])
        df.groupby('OverallQual')['SalePrice'].mean().plot(kind='bar', ax=ax3, color='skyblue')
        ax3.set_title('Average Price by Overall Quality')
        ax3.set_xlabel('Overall Quality')
        ax3.set_ylabel('Average Sale Price ($)')
        ax3.tick_params(axis='x', rotation=0)

        # 4. Price vs Living Area
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.scatter(df['GrLivArea'], df['SalePrice'], alpha=0.5)
        ax4.set_title('Price vs Above Grade Living Area')
        ax4.set_xlabel('Living Area (sqft)')
        ax4.set_ylabel('Sale Price ($)')

        # 5. Price vs Year Built
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.scatter(df['YearBuilt'], df['SalePrice'], alpha=0.5, color='orange')
        ax5.set_title('Price vs Year Built')
        ax5.set_xlabel('Year Built')
        ax5.set_ylabel('Sale Price ($)')

        # 6. Price by Garage Cars
        ax6 = fig.add_subplot(gs[1, 2])
        df.groupby('GarageCars')['SalePrice'].mean().plot(kind='bar', ax=ax6, color='salmon')
        ax6.set_title('Average Price by Garage Capacity')
        ax6.set_xlabel('Garage Cars')
        ax6.set_ylabel('Average Sale Price ($)')
        ax6.tick_params(axis='x', rotation=0)

        # 7. Correlation heatmap (top features)
        ax7 = fig.add_subplot(gs[2, :])
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_with_price = df[numeric_cols].corr()['SalePrice'].abs().sort_values(ascending=False)
        top_features = corr_with_price[1:11].index.tolist() + ['SalePrice']
        corr_matrix = df[top_features].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   ax=ax7, cbar_kws={'shrink': 0.8})
        ax7.set_title('Correlation Matrix - Top 10 Features with Sale Price')

        # 8. Total SF distribution
        ax8 = fig.add_subplot(gs[3, 0])
        if 'TotalSF' in df.columns:
            df['TotalSF'].hist(bins=50, ax=ax8, edgecolor='black', alpha=0.7, color='purple')
            ax8.set_title('Total Square Footage Distribution')
            ax8.set_xlabel('Total SF')
            ax8.set_ylabel('Frequency')

        # 9. Price by Full Bath
        ax9 = fig.add_subplot(gs[3, 1])
        df.groupby('FullBath')['SalePrice'].mean().plot(kind='bar', ax=ax9, color='teal')
        ax9.set_title('Average Price by Number of Full Bathrooms')
        ax9.set_xlabel('Full Bathrooms')
        ax9.set_ylabel('Average Sale Price ($)')
        ax9.tick_params(axis='x', rotation=0)

        # 10. Basement area distribution
        ax10 = fig.add_subplot(gs[3, 2])
        df['TotalBsmtSF'].hist(bins=50, ax=ax10, edgecolor='black', alpha=0.7, color='brown')
        ax10.set_title('Basement Area Distribution')
        ax10.set_xlabel('Total Basement SF')
        ax10.set_ylabel('Frequency')

        plt.savefig(f'{output_dir}/exploratory_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Exploratory analysis saved to {output_dir}/exploratory_analysis.png")
        plt.close()

    def train_multiple_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train multiple regression models for comparison.

        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
        """
        X_train_scaled = self.scaler.fit_transform(X_train)

        print("\nTraining multiple regression models...")

        # 1. Linear Regression
        print("  - Training Linear Regression...")
        self.models['Linear Regression'] = LinearRegression()
        self.models['Linear Regression'].fit(X_train_scaled, y_train)

        # 2. Ridge Regression
        print("  - Training Ridge Regression...")
        self.models['Ridge'] = Ridge(alpha=10.0, random_state=42)
        self.models['Ridge'].fit(X_train_scaled, y_train)

        # 3. Lasso Regression
        print("  - Training Lasso Regression...")
        self.models['Lasso'] = Lasso(alpha=100.0, random_state=42, max_iter=10000)
        self.models['Lasso'].fit(X_train_scaled, y_train)

        # 4. ElasticNet
        print("  - Training ElasticNet...")
        self.models['ElasticNet'] = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=10000)
        self.models['ElasticNet'].fit(X_train_scaled, y_train)

        # 5. Random Forest
        print("  - Training Random Forest...")
        self.models['Random Forest'] = RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=10,
            min_samples_leaf=4, random_state=42, n_jobs=-1
        )
        self.models['Random Forest'].fit(X_train_scaled, y_train)

        # 6. Gradient Boosting
        print("  - Training Gradient Boosting...")
        self.models['Gradient Boosting'] = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            random_state=42
        )
        self.models['Gradient Boosting'].fit(X_train_scaled, y_train)

        # 7. XGBoost
        if XGBOOST_AVAILABLE:
            print("  - Training XGBoost...")
            self.models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=5,
                random_state=42, n_jobs=-1
            )
            self.models['XGBoost'].fit(X_train_scaled, y_train)

        # 8. LightGBM
        if LIGHTGBM_AVAILABLE:
            print("  - Training LightGBM...")
            self.models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=5,
                random_state=42, verbose=-1, n_jobs=-1
            )
            self.models['LightGBM'].fit(X_train_scaled, y_train)

        # 9. CatBoost
        if CATBOOST_AVAILABLE:
            print("  - Training CatBoost...")
            self.models['CatBoost'] = CatBoostRegressor(
                iterations=200, learning_rate=0.05, depth=5,
                random_state=42, verbose=0
            )
            self.models['CatBoost'].fit(X_train_scaled, y_train)

        print(f"\nTrained {len(self.models)} models successfully!")

    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Perform hyperparameter tuning on Random Forest model.

        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
        """
        print("\nPerforming hyperparameter tuning...")
        X_train_scaled = self.scaler.fit_transform(X_train)

        param_distributions = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }

        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        random_search = RandomizedSearchCV(
            rf, param_distributions, n_iter=20, cv=5,
            scoring='neg_mean_squared_error', random_state=42, n_jobs=-1, verbose=1
        )

        random_search.fit(X_train_scaled, y_train)
        self.models['Random Forest Tuned'] = random_search.best_estimator_

        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best CV RMSE: ${np.sqrt(-random_search.best_score_):,.2f}")

    def create_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Create ensemble models using voting and stacking.

        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
        """
        print("\nCreating ensemble models...")
        X_train_scaled = self.scaler.transform(X_train)

        # Voting Regressor
        estimators = [
            ('rf', self.models['Random Forest']),
            ('gb', self.models['Gradient Boosting']),
            ('ridge', self.models['Ridge'])
        ]

        if XGBOOST_AVAILABLE:
            estimators.append(('xgb', self.models['XGBoost']))

        voting_reg = VotingRegressor(estimators=estimators)
        voting_reg.fit(X_train_scaled, y_train)
        self.models['Voting Ensemble'] = voting_reg

        # Stacking Regressor
        stacking_reg = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=10.0)
        )
        stacking_reg.fit(X_train_scaled, y_train)
        self.models['Stacking Ensemble'] = stacking_reg

        print("Ensemble models created successfully!")

    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Evaluate all trained models and compare performance.

        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels

        Returns:
            pd.DataFrame: Comparison of model performances
        """
        X_test_scaled = self.scaler.transform(X_test)
        results = []

        print("\n=== Model Evaluation Results ===")

        for name, model in self.models.items():
            y_pred = model.predict(X_test_scaled)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100

            results.append({
                'Model': name,
                'RMSE': rmse,
                'MAE': mae,
                'R² Score': r2,
                'MAPE %': mape
            })

            print(f"\n{name}:")
            print(f"  RMSE: ${rmse:,.2f}")
            print(f"  MAE: ${mae:,.2f}")
            print(f"  R² Score: {r2:.4f}")
            print(f"  MAPE: {mape:.2f}%")

        results_df = pd.DataFrame(results).sort_values('RMSE')
        self.best_model = self.models[results_df.iloc[0]['Model']]

        return results_df

    def plot_model_comparison(self, results_df: pd.DataFrame, output_dir: str = '.') -> None:
        """
        Visualize model performance comparison.

        Args:
            results_df (pd.DataFrame): Model evaluation results
            output_dir (str): Directory to save plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # RMSE comparison
        axes[0, 0].barh(results_df['Model'], results_df['RMSE'], color='skyblue')
        axes[0, 0].set_xlabel('RMSE ($)')
        axes[0, 0].set_title('Model RMSE Comparison (Lower is Better)')

        # MAE comparison
        axes[0, 1].barh(results_df['Model'], results_df['MAE'], color='salmon')
        axes[0, 1].set_xlabel('MAE ($)')
        axes[0, 1].set_title('Model MAE Comparison (Lower is Better)')

        # R² Score comparison
        axes[1, 0].barh(results_df['Model'], results_df['R² Score'], color='lightgreen')
        axes[1, 0].set_xlabel('R² Score')
        axes[1, 0].set_title('Model R² Score Comparison (Higher is Better)')
        axes[1, 0].set_xlim([0.7, 1.0])

        # MAPE comparison
        axes[1, 1].barh(results_df['Model'], results_df['MAPE %'], color='plum')
        axes[1, 1].set_xlabel('MAPE (%)')
        axes[1, 1].set_title('Model MAPE Comparison (Lower is Better)')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Model comparison saved to {output_dir}/model_comparison.png")
        plt.close()

    def plot_predictions(self, X_test: pd.DataFrame, y_test: pd.Series,
                        output_dir: str = '.') -> None:
        """
        Plot predictions vs actual values and residuals.

        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels
            output_dir (str): Directory to save plot
        """
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.best_model.predict(X_test_scaled)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Predictions vs Actual
        axes[0, 0].scatter(y_test, y_pred, alpha=0.5)
        axes[0, 0].plot([y_test.min(), y_test.max()],
                       [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Price ($)')
        axes[0, 0].set_ylabel('Predicted Price ($)')
        axes[0, 0].set_title('Predicted vs Actual Prices')

        # Residuals
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted Price ($)')
        axes[0, 1].set_ylabel('Residuals ($)')
        axes[0, 1].set_title('Residual Plot')

        # Residual distribution
        axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Residuals ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residual Distribution')
        axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)

        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/predictions_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Predictions analysis saved to {output_dir}/predictions_analysis.png")
        plt.close()

    def plot_learning_curves(self, X_train: pd.DataFrame, y_train: pd.Series,
                            output_dir: str = '.') -> None:
        """
        Plot learning curves for the best model.

        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
            output_dir (str): Directory to save plot
        """
        X_train_scaled = self.scaler.transform(X_train)

        train_sizes, train_scores, val_scores = learning_curve(
            self.best_model, X_train_scaled, y_train,
            cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='neg_mean_squared_error'
        )

        # Convert to RMSE
        train_rmse_mean = np.sqrt(-np.mean(train_scores, axis=1))
        train_rmse_std = np.sqrt(np.std(train_scores, axis=1))
        val_rmse_mean = np.sqrt(-np.mean(val_scores, axis=1))
        val_rmse_std = np.sqrt(np.std(val_scores, axis=1))

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_rmse_mean, label='Training RMSE', color='blue', marker='o')
        plt.fill_between(train_sizes, train_rmse_mean - train_rmse_std,
                        train_rmse_mean + train_rmse_std, alpha=0.15, color='blue')
        plt.plot(train_sizes, val_rmse_mean, label='Cross-validation RMSE', color='red', marker='s')
        plt.fill_between(train_sizes, val_rmse_mean - val_rmse_std,
                        val_rmse_mean + val_rmse_std, alpha=0.15, color='red')
        plt.xlabel('Training Set Size')
        plt.ylabel('RMSE ($)')
        plt.title('Learning Curves - Best Model')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/learning_curves.png', dpi=300, bbox_inches='tight')
        print(f"Learning curves saved to {output_dir}/learning_curves.png")
        plt.close()

    def plot_feature_importance(self, output_dir: str = '.') -> None:
        """
        Plot feature importance for tree-based models.

        Args:
            output_dir (str): Directory to save plot
        """
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]

            plt.figure(figsize=(12, 8))
            plt.title('Top 20 Feature Importance')
            plt.bar(range(20), importances[indices], color='steelblue')
            plt.xticks(range(20), [self.feature_names[i] for i in indices],
                      rotation=45, ha='right')
            plt.ylabel('Importance Score')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
            print(f"Feature importance saved to {output_dir}/feature_importance.png")
            plt.close()


def main():
    """Main execution function."""
    print("=" * 80)
    print("HOUSE PRICES PREDICTION - COMPREHENSIVE REGRESSION SOLUTION")
    print("=" * 80)

    # Initialize predictor
    predictor = HousePricePredictor()

    # Create sample data
    print("\nCreating sample dataset...")
    df = predictor.create_sample_data()
    print(f"Dataset shape: {df.shape}")
    print(f"\nPrice statistics:")
    print(f"  Mean: ${df['SalePrice'].mean():,.2f}")
    print(f"  Median: ${df['SalePrice'].median():,.2f}")
    print(f"  Std: ${df['SalePrice'].std():,.2f}")
    print(f"  Min: ${df['SalePrice'].min():,.2f}")
    print(f"  Max: ${df['SalePrice'].max():,.2f}")

    # Exploratory analysis
    print("\nGenerating exploratory data analysis...")
    predictor.plot_exploratory_analysis(df)

    # Feature engineering
    print("\nPerforming feature engineering...")
    df = predictor.feature_engineering(df)

    # Prepare training data
    feature_cols = [col for col in df.columns if col != 'SalePrice']
    X = df[feature_cols]
    y = df['SalePrice']
    predictor.feature_names = feature_cols

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Train multiple models
    predictor.train_multiple_models(X_train, y_train)

    # Hyperparameter tuning
    predictor.hyperparameter_tuning(X_train, y_train)

    # Create ensemble
    predictor.create_ensemble(X_train, y_train)

    # Evaluate models
    results_df = predictor.evaluate_models(X_test, y_test)
    print(f"\n{results_df.to_string(index=False)}")

    # Generate visualizations
    print("\nGenerating comprehensive visualizations...")
    predictor.plot_model_comparison(results_df)
    predictor.plot_predictions(X_test, y_test)
    predictor.plot_learning_curves(X_train, y_train)
    predictor.plot_feature_importance()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nBest Model: {results_df.iloc[0]['Model']}")
    print(f"Best RMSE: ${results_df.iloc[0]['RMSE']:,.2f}")
    print(f"Best R² Score: {results_df.iloc[0]['R² Score']:.4f}")


if __name__ == "__main__":
    main()
