"""
Context-Aware Recommendation System
====================================

This solution demonstrates context-aware recommendations that consider
contextual information (time, location, device, mood) alongside user-item
interactions to provide more personalized recommendations.

Author: Kaggle Solutions Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class FactorizationMachine:
    """Factorization Machine for context-aware recommendations"""

    def __init__(self, n_factors: int = 10, learning_rate: float = 0.01,
                 reg_lambda: float = 0.01, n_epochs: int = 50):
        """
        Initialize Factorization Machine

        Args:
            n_factors: Number of latent factors
            learning_rate: Learning rate for SGD
            reg_lambda: Regularization parameter
            n_epochs: Number of training epochs
        """
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.n_epochs = n_epochs
        self.w0 = 0.0
        self.w = None
        self.V = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FactorizationMachine':
        """
        Train the factorization machine

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)

        Returns:
            self
        """
        n_samples, n_features = X.shape

        # Initialize parameters
        self.w0 = np.mean(y)
        self.w = np.zeros(n_features)
        self.V = np.random.normal(0, 0.1, (n_features, self.n_factors))

        # SGD training
        for epoch in range(self.n_epochs):
            total_loss = 0.0
            for i in range(n_samples):
                # Predict
                pred = self._predict_single(X[i])
                error = y[i] - pred

                # Update parameters
                self.w0 += self.learning_rate * error

                for j in range(n_features):
                    if X[i, j] != 0:
                        self.w[j] += self.learning_rate * (error * X[i, j] - self.reg_lambda * self.w[j])

                        for f in range(self.n_factors):
                            sum_vx = np.sum(self.V[:, f] * X[i]) - self.V[j, f] * X[i, j]
                            self.V[j, f] += self.learning_rate * (error * X[i, j] * sum_vx - self.reg_lambda * self.V[j, f])

                total_loss += error ** 2

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, MSE: {total_loss / n_samples:.4f}")

        return self

    def _predict_single(self, x: np.ndarray) -> float:
        """Predict for a single sample"""
        # Linear term
        linear = self.w0 + np.dot(x, self.w)

        # Interaction term
        interaction = 0.0
        for f in range(self.n_factors):
            sum_vx = np.sum(self.V[:, f] * x)
            sum_vx_sq = np.sum((self.V[:, f] * x) ** 2)
            interaction += sum_vx ** 2 - sum_vx_sq
        interaction *= 0.5

        return linear + interaction

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict for multiple samples"""
        return np.array([self._predict_single(x) for x in X])


class ContextAwareRecommender:
    """Context-aware recommendation system with multiple models"""

    def __init__(self, model_type: str = 'fm', n_factors: int = 10):
        """
        Initialize context-aware recommender

        Args:
            model_type: Type of model ('fm', 'rf', 'gb')
            n_factors: Number of latent factors (for FM)
        """
        self.model_type = model_type
        self.n_factors = n_factors
        self.model = None
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.context_encoders = {}
        self.scaler = StandardScaler()

    def prepare_features(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Prepare features including context

        Args:
            df: DataFrame with user, item, and context columns
            fit: Whether to fit encoders

        Returns:
            Feature matrix
        """
        features = []

        # Encode user and item
        if fit:
            user_encoded = self.user_encoder.fit_transform(df['user_id'])
            item_encoded = self.item_encoder.fit_transform(df['item_id'])
        else:
            user_encoded = self.user_encoder.transform(df['user_id'])
            item_encoded = self.item_encoder.transform(df['item_id'])

        features.append(user_encoded.reshape(-1, 1))
        features.append(item_encoded.reshape(-1, 1))

        # Encode context features
        context_cols = ['hour', 'day_of_week', 'device_type', 'location', 'mood']
        for col in context_cols:
            if col in df.columns:
                if fit:
                    if col not in self.context_encoders:
                        self.context_encoders[col] = LabelEncoder()
                    encoded = self.context_encoders[col].fit_transform(df[col])
                else:
                    encoded = self.context_encoders[col].transform(df[col])
                features.append(encoded.reshape(-1, 1))

        # Combine all features
        X = np.hstack(features)

        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        return X

    def fit(self, df: pd.DataFrame, ratings: np.ndarray) -> 'ContextAwareRecommender':
        """
        Train the recommender

        Args:
            df: DataFrame with user, item, and context columns
            ratings: Rating values

        Returns:
            self
        """
        X = self.prepare_features(df, fit=True)

        if self.model_type == 'fm':
            self.model = FactorizationMachine(n_factors=self.n_factors)
        elif self.model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == 'gb':
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)

        self.model.fit(X, ratings)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict ratings"""
        X = self.prepare_features(df, fit=False)
        return self.model.predict(X)


def generate_context_aware_data(n_users: int = 500, n_items: int = 200,
                                n_interactions: int = 10000) -> pd.DataFrame:
    """
    Generate synthetic context-aware interaction data

    Args:
        n_users: Number of users
        n_items: Number of items
        n_interactions: Number of interactions

    Returns:
        DataFrame with interactions and context
    """
    np.random.seed(42)

    # Generate user and item preferences
    user_time_pref = np.random.choice(['morning', 'afternoon', 'evening', 'night'], n_users)
    user_device_pref = np.random.choice(['mobile', 'tablet', 'desktop'], n_users)
    item_time_match = np.random.choice(['morning', 'afternoon', 'evening', 'night'], n_items)

    interactions = []
    for _ in range(n_interactions):
        user_id = np.random.randint(0, n_users)
        item_id = np.random.randint(0, n_items)

        # Context features
        hour = np.random.randint(0, 24)
        day_of_week = np.random.randint(0, 7)
        device_type = np.random.choice(['mobile', 'tablet', 'desktop'])
        location = np.random.choice(['home', 'work', 'commute', 'other'])
        mood = np.random.choice(['happy', 'sad', 'neutral', 'excited'])

        # Determine time of day
        if hour < 6:
            time_of_day = 'night'
        elif hour < 12:
            time_of_day = 'morning'
        elif hour < 18:
            time_of_day = 'afternoon'
        else:
            time_of_day = 'evening'

        # Base rating
        base_rating = np.random.uniform(2, 4)

        # Context effects
        time_match = 1.0 if user_time_pref[user_id] == time_of_day else 0.5
        device_match = 1.0 if user_device_pref[user_id] == device_type else 0.7
        item_time_match_score = 1.0 if item_time_match[item_id] == time_of_day else 0.6

        # Weekend effect
        weekend_boost = 0.5 if day_of_week >= 5 else 0.0

        # Location effect
        location_boost = {'home': 0.5, 'work': 0.0, 'commute': -0.3, 'other': 0.2}[location]

        # Mood effect
        mood_boost = {'happy': 0.5, 'sad': -0.3, 'neutral': 0.0, 'excited': 0.7}[mood]

        # Calculate final rating
        rating = base_rating * time_match * device_match * item_time_match_score
        rating += weekend_boost + location_boost + mood_boost
        rating = np.clip(rating + np.random.normal(0, 0.3), 1, 5)

        interactions.append({
            'user_id': user_id,
            'item_id': item_id,
            'hour': hour,
            'day_of_week': day_of_week,
            'device_type': device_type,
            'location': location,
            'mood': mood,
            'rating': rating
        })

    return pd.DataFrame(interactions)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate recommendation model

    Args:
        y_true: True ratings
        y_pred: Predicted ratings

    Returns:
        Dictionary of metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {
        'MAE': mae,
        'RMSE': rmse
    }


def plot_context_effects(df: pd.DataFrame, save_path: str = None):
    """Plot the effect of different contexts on ratings"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Hour of day effect
    hour_ratings = df.groupby('hour')['rating'].mean()
    axes[0, 0].plot(hour_ratings.index, hour_ratings.values, marker='o')
    axes[0, 0].set_title('Rating vs Hour of Day', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Hour')
    axes[0, 0].set_ylabel('Average Rating')
    axes[0, 0].grid(True, alpha=0.3)

    # Day of week effect
    day_ratings = df.groupby('day_of_week')['rating'].mean()
    axes[0, 1].bar(day_ratings.index, day_ratings.values, color='skyblue')
    axes[0, 1].set_title('Rating vs Day of Week', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Day of Week')
    axes[0, 1].set_ylabel('Average Rating')
    axes[0, 1].set_xticks(range(7))
    axes[0, 1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

    # Device type effect
    device_ratings = df.groupby('device_type')['rating'].mean().sort_values(ascending=False)
    axes[0, 2].barh(device_ratings.index, device_ratings.values, color='coral')
    axes[0, 2].set_title('Rating vs Device Type', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Average Rating')

    # Location effect
    location_ratings = df.groupby('location')['rating'].mean().sort_values(ascending=False)
    axes[1, 0].barh(location_ratings.index, location_ratings.values, color='lightgreen')
    axes[1, 0].set_title('Rating vs Location', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Average Rating')

    # Mood effect
    mood_ratings = df.groupby('mood')['rating'].mean().sort_values(ascending=False)
    axes[1, 1].barh(mood_ratings.index, mood_ratings.values, color='plum')
    axes[1, 1].set_title('Rating vs Mood', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Average Rating')

    # Rating distribution
    axes[1, 2].hist(df['rating'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1, 2].set_title('Rating Distribution', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Rating')
    axes[1, 2].set_ylabel('Frequency')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_model_comparison(results: Dict[str, Dict[str, float]], save_path: str = None):
    """Plot comparison of different models"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    models = list(results.keys())
    mae_scores = [results[m]['MAE'] for m in models]
    rmse_scores = [results[m]['RMSE'] for m in models]

    x = np.arange(len(models))
    width = 0.35

    axes[0].bar(x, mae_scores, width, label='MAE', color='skyblue')
    axes[0].set_ylabel('Mean Absolute Error')
    axes[0].set_title('Model Comparison - MAE', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45)
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(x, rmse_scores, width, label='RMSE', color='coral')
    axes[1].set_ylabel('Root Mean Squared Error')
    axes[1].set_title('Model Comparison - RMSE', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=45)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_context_vs_no_context(with_context_pred: np.ndarray, without_context_pred: np.ndarray,
                               y_true: np.ndarray, save_path: str = None):
    """Compare predictions with and without context"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot with context
    axes[0].scatter(y_true, with_context_pred, alpha=0.5, s=10)
    axes[0].plot([1, 5], [1, 5], 'r--', lw=2)
    axes[0].set_xlabel('True Rating')
    axes[0].set_ylabel('Predicted Rating')
    axes[0].set_title('With Context', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Scatter plot without context
    axes[1].scatter(y_true, without_context_pred, alpha=0.5, s=10, color='coral')
    axes[1].plot([1, 5], [1, 5], 'r--', lw=2)
    axes[1].set_xlabel('True Rating')
    axes[1].set_ylabel('Predicted Rating')
    axes[1].set_title('Without Context', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_context_heatmap(df: pd.DataFrame, save_path: str = None):
    """Plot heatmap of context interactions"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Device type vs time of day
    df['time_of_day'] = pd.cut(df['hour'], bins=[0, 6, 12, 18, 24],
                                labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    pivot1 = df.pivot_table(values='rating', index='device_type',
                            columns='time_of_day', aggfunc='mean')
    sns.heatmap(pivot1, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[0, 0])
    axes[0, 0].set_title('Device Type vs Time of Day', fontsize=12, fontweight='bold')

    # Location vs mood
    pivot2 = df.pivot_table(values='rating', index='location',
                            columns='mood', aggfunc='mean')
    sns.heatmap(pivot2, annot=True, fmt='.2f', cmap='YlGnBu', ax=axes[0, 1])
    axes[0, 1].set_title('Location vs Mood', fontsize=12, fontweight='bold')

    # Device type vs location
    pivot3 = df.pivot_table(values='rating', index='device_type',
                            columns='location', aggfunc='mean')
    sns.heatmap(pivot3, annot=True, fmt='.2f', cmap='RdPu', ax=axes[1, 0])
    axes[1, 0].set_title('Device Type vs Location', fontsize=12, fontweight='bold')

    # Day of week vs mood
    pivot4 = df.pivot_table(values='rating', index='day_of_week',
                            columns='mood', aggfunc='mean')
    sns.heatmap(pivot4, annot=True, fmt='.2f', cmap='viridis', ax=axes[1, 1])
    axes[1, 1].set_title('Day of Week vs Mood', fontsize=12, fontweight='bold')
    axes[1, 1].set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_prediction_errors(errors_with: np.ndarray, errors_without: np.ndarray,
                          save_path: str = None):
    """Plot distribution of prediction errors"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].hist(errors_with, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Prediction Error')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Error Distribution - With Context', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(errors_without, bins=50, alpha=0.7, color='coral', edgecolor='black')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Prediction Error')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Error Distribution - Without Context', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function"""
    print("=" * 80)
    print("Context-Aware Recommendation System")
    print("=" * 80)

    # Generate data
    print("\n1. Generating context-aware interaction data...")
    df = generate_context_aware_data(n_users=500, n_items=200, n_interactions=10000)
    print(f"Generated {len(df)} interactions")
    print(f"Number of users: {df['user_id'].nunique()}")
    print(f"Number of items: {df['item_id'].nunique()}")

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    y_train = train_df['rating'].values
    y_test = test_df['rating'].values

    print(f"Training set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")

    # Plot context effects
    print("\n2. Analyzing context effects...")
    plot_context_effects(df)
    plot_context_heatmap(df)

    # Train models with context
    print("\n3. Training context-aware models...")
    results = {}

    # Factorization Machine
    print("\n   Training Factorization Machine...")
    fm_model = ContextAwareRecommender(model_type='fm', n_factors=10)
    fm_model.fit(train_df, y_train)
    fm_pred = fm_model.predict(test_df)
    results['FM'] = evaluate_model(y_test, fm_pred)
    print(f"   FM - MAE: {results['FM']['MAE']:.4f}, RMSE: {results['FM']['RMSE']:.4f}")

    # Random Forest
    print("\n   Training Random Forest...")
    rf_model = ContextAwareRecommender(model_type='rf')
    rf_model.fit(train_df, y_train)
    rf_pred = rf_model.predict(test_df)
    results['RF'] = evaluate_model(y_test, rf_pred)
    print(f"   RF - MAE: {results['RF']['MAE']:.4f}, RMSE: {results['RF']['RMSE']:.4f}")

    # Gradient Boosting
    print("\n   Training Gradient Boosting...")
    gb_model = ContextAwareRecommender(model_type='gb')
    gb_model.fit(train_df, y_train)
    gb_pred = gb_model.predict(test_df)
    results['GB'] = evaluate_model(y_test, gb_pred)
    print(f"   GB - MAE: {results['GB']['MAE']:.4f}, RMSE: {results['GB']['RMSE']:.4f}")

    # Train baseline without context
    print("\n4. Training baseline model (without context)...")
    train_df_no_context = train_df[['user_id', 'item_id']].copy()
    test_df_no_context = test_df[['user_id', 'item_id']].copy()

    baseline_model = ContextAwareRecommender(model_type='rf')
    baseline_model.fit(train_df_no_context, y_train)
    baseline_pred = baseline_model.predict(test_df_no_context)
    results['Baseline (No Context)'] = evaluate_model(y_test, baseline_pred)
    print(f"   Baseline - MAE: {results['Baseline (No Context)']['MAE']:.4f}, "
          f"RMSE: {results['Baseline (No Context)']['RMSE']:.4f}")

    # Plot comparisons
    print("\n5. Visualizing results...")
    plot_model_comparison(results)
    plot_context_vs_no_context(rf_pred, baseline_pred, y_test)

    # Error analysis
    errors_with = y_test - rf_pred
    errors_without = y_test - baseline_pred
    plot_prediction_errors(errors_with, errors_without)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nModel Performance:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

    # Calculate improvement
    improvement_mae = ((results['Baseline (No Context)']['MAE'] - results['RF']['MAE']) /
                      results['Baseline (No Context)']['MAE'] * 100)
    improvement_rmse = ((results['Baseline (No Context)']['RMSE'] - results['RF']['RMSE']) /
                       results['Baseline (No Context)']['RMSE'] * 100)

    print(f"\nImprovement with context (RF vs Baseline):")
    print(f"  MAE improvement: {improvement_mae:.2f}%")
    print(f"  RMSE improvement: {improvement_rmse:.2f}%")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
