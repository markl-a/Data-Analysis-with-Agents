"""
Used Car Price Prediction

Predict used car market prices based on various features like brand,
age, mileage to help buyers and sellers make informed decisions.

Dataset: https://www.kaggle.com/datasets/vijayaadithyanvg/car-price-predictionused-cars
Difficulty: ⭐⭐ Intermediate Level
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


class UsedCarPriceModel:
    """Used Car Price Prediction Model."""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.best_model = None

    def create_sample_data(self, n_samples: int = 3000) -> pd.DataFrame:
        """Create synthetic used car dataset."""
        np.random.seed(42)

        brands = {
            'Toyota': {'base_price': 25000, 'depreciation': 0.12},
            'Honda': {'base_price': 24000, 'depreciation': 0.13},
            'BMW': {'base_price': 45000, 'depreciation': 0.18},
            'Mercedes': {'base_price': 50000, 'depreciation': 0.20},
            'Ford': {'base_price': 22000, 'depreciation': 0.15},
            'Hyundai': {'base_price': 20000, 'depreciation': 0.14},
            'Volkswagen': {'base_price': 28000, 'depreciation': 0.16},
            'Audi': {'base_price': 42000, 'depreciation': 0.19},
            'Nissan': {'base_price': 23000, 'depreciation': 0.14},
            'Kia': {'base_price': 19000, 'depreciation': 0.13}
        }

        fuel_types = ['Petrol', 'Diesel', 'CNG', 'Electric']
        fuel_probs = [0.50, 0.35, 0.10, 0.05]
        fuel_multiplier = {'Petrol': 1.0, 'Diesel': 1.15, 'CNG': 0.9, 'Electric': 1.25}

        transmissions = ['Manual', 'Automatic']
        trans_probs = [0.55, 0.45]
        trans_multiplier = {'Manual': 1.0, 'Automatic': 1.12}

        seller_types = ['Individual', 'Dealer', 'Trustmark Dealer']
        seller_probs = [0.40, 0.45, 0.15]

        owners = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above']
        owner_probs = [0.45, 0.35, 0.15, 0.05]
        owner_multiplier = {'First Owner': 1.0, 'Second Owner': 0.92,
                           'Third Owner': 0.85, 'Fourth & Above': 0.75}

        current_year = 2024
        data = []

        for _ in range(n_samples):
            brand = np.random.choice(list(brands.keys()))
            brand_info = brands[brand]

            # Year (2005-2024)
            year = np.random.randint(2005, 2025)
            age = current_year - year

            # Kilometers driven (based on age)
            avg_km_per_year = np.random.normal(12000, 3000)
            km_driven = int(max(1000, age * avg_km_per_year + np.random.normal(0, 5000)))

            fuel = np.random.choice(fuel_types, p=fuel_probs)
            transmission = np.random.choice(transmissions, p=trans_probs)
            seller_type = np.random.choice(seller_types, p=seller_probs)
            owner = np.random.choice(owners, p=owner_probs)

            # Engine and power
            engine = np.random.choice([1000, 1200, 1400, 1500, 1600, 1800, 2000, 2500, 3000])
            max_power = engine * np.random.uniform(0.06, 0.09)
            seats = np.random.choice([4, 5, 6, 7], p=[0.05, 0.70, 0.10, 0.15])

            # Calculate price
            base = brand_info['base_price']
            depreciation = brand_info['depreciation']

            # Depreciation formula (exponential decay)
            price = base * ((1 - depreciation) ** age)

            # Adjustments
            price *= fuel_multiplier[fuel]
            price *= trans_multiplier[transmission]
            price *= owner_multiplier[owner]

            # Mileage adjustment
            expected_km = age * 12000
            km_diff = (km_driven - expected_km) / expected_km if expected_km > 0 else 0
            price *= (1 - 0.1 * km_diff)

            # Engine size bonus
            if engine >= 2000:
                price *= 1.1
            elif engine <= 1200:
                price *= 0.95

            # Add noise
            price *= np.random.uniform(0.9, 1.1)
            price = max(1000, price)

            data.append({
                'brand': brand,
                'year': year,
                'km_driven': km_driven,
                'fuel': fuel,
                'seller_type': seller_type,
                'transmission': transmission,
                'owner': owner,
                'engine': engine,
                'max_power': round(max_power, 1),
                'seats': seats,
                'selling_price': int(price)
            })

        return pd.DataFrame(data)

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features."""
        df = df.copy()
        current_year = 2024

        # Car age
        df['age'] = current_year - df['year']

        # Km per year
        df['km_per_year'] = df['km_driven'] / df['age'].replace(0, 1)

        # Power to weight proxy (using engine as proxy)
        df['power_ratio'] = df['max_power'] / df['engine'] * 1000

        # Brand tier
        premium_brands = ['BMW', 'Mercedes', 'Audi']
        df['is_premium'] = df['brand'].isin(premium_brands).astype(int)

        # Age categories
        df['age_category'] = pd.cut(df['age'],
                                    bins=[-1, 3, 7, 12, 100],
                                    labels=['New', 'Used', 'Old', 'Classic'])

        # High mileage flag
        df['high_mileage'] = (df['km_driven'] > 100000).astype(int)

        return df

    def analyze_data(self, df: pd.DataFrame, output_dir: str = '.') -> None:
        """Perform exploratory data analysis."""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Used Car Price Analysis', fontsize=16)

        # Price distribution
        axes[0, 0].hist(df['selling_price'] / 1000, bins=50, edgecolor='black',
                       alpha=0.7, color='steelblue')
        axes[0, 0].set_title('Price Distribution')
        axes[0, 0].set_xlabel('Price (thousands)')
        axes[0, 0].set_ylabel('Count')

        # Price by brand
        brand_price = df.groupby('brand')['selling_price'].median().sort_values()
        brand_price.plot(kind='barh', ax=axes[0, 1], color='steelblue')
        axes[0, 1].set_title('Median Price by Brand')
        axes[0, 1].set_xlabel('Price')

        # Age vs Price
        axes[0, 2].scatter(df['age'], df['selling_price'] / 1000, alpha=0.5)
        axes[0, 2].set_title('Age vs Price')
        axes[0, 2].set_xlabel('Age (years)')
        axes[0, 2].set_ylabel('Price (thousands)')

        # Km driven vs Price
        axes[1, 0].scatter(df['km_driven'] / 1000, df['selling_price'] / 1000, alpha=0.5)
        axes[1, 0].set_title('Mileage vs Price')
        axes[1, 0].set_xlabel('Km Driven (thousands)')
        axes[1, 0].set_ylabel('Price (thousands)')

        # Price by fuel type
        df.boxplot(column='selling_price', by='fuel', ax=axes[1, 1])
        axes[1, 1].set_title('Price by Fuel Type')
        plt.suptitle('')

        # Price by transmission
        df.boxplot(column='selling_price', by='transmission', ax=axes[1, 2])
        axes[1, 2].set_title('Price by Transmission')
        plt.suptitle('')

        # Price by owner
        owner_order = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above']
        df['owner'] = pd.Categorical(df['owner'], categories=owner_order, ordered=True)
        df.boxplot(column='selling_price', by='owner', ax=axes[2, 0])
        axes[2, 0].set_title('Price by Owner History')
        axes[2, 0].tick_params(axis='x', rotation=45)
        plt.suptitle('')

        # Correlation heatmap
        numeric_cols = ['selling_price', 'year', 'km_driven', 'engine', 'max_power', 'seats', 'age']
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[2, 1], center=0)
        axes[2, 1].set_title('Feature Correlations')

        # Depreciation curve
        age_price = df.groupby('age')['selling_price'].mean() / 1000
        axes[2, 2].plot(age_price.index, age_price.values, 'b-', linewidth=2, marker='o')
        axes[2, 2].set_title('Depreciation Curve')
        axes[2, 2].set_xlabel('Age (years)')
        axes[2, 2].set_ylabel('Avg Price (thousands)')
        axes[2, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/used_car_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Analysis saved to {output_dir}/used_car_analysis.png")
        plt.close()

    def prepare_features(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for modeling."""
        numeric_cols = ['year', 'km_driven', 'engine', 'max_power', 'seats',
                       'age', 'km_per_year', 'power_ratio', 'is_premium', 'high_mileage']

        X_numeric = df[numeric_cols].values

        # Encode categoricals
        categorical_cols = ['brand', 'fuel', 'seller_type', 'transmission', 'owner', 'age_category']

        for col in categorical_cols:
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

        y = df['selling_price'].values

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

            # MAPE
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

            results.append({
                'Model': name,
                'MAE': mean_absolute_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'R²': r2_score(y_test, y_pred),
                'MAPE (%)': mape
            })

        results_df = pd.DataFrame(results).sort_values('MAE')
        self.best_model = self.models[results_df.iloc[0]['Model']]

        return results_df

    def plot_results(self, results: pd.DataFrame, X_test: np.ndarray,
                    y_test: np.ndarray, output_dir: str = '.') -> None:
        """Visualize results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Used Car Price Prediction Results', fontsize=16)

        # Model comparison
        results.set_index('Model')[['MAE', 'RMSE']].plot(kind='bar', ax=axes[0, 0])
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
        axes[1, 0].scatter(y_test / 1000, y_pred / 1000, alpha=0.5)
        max_val = max(y_test.max(), y_pred.max()) / 1000
        axes[1, 0].plot([0, max_val], [0, max_val], 'r--', linewidth=2)
        axes[1, 0].set_xlabel('Actual Price (thousands)')
        axes[1, 0].set_ylabel('Predicted Price (thousands)')
        axes[1, 0].set_title('Actual vs Predicted (Best Model)')

        # Error distribution
        errors = (y_test - y_pred) / 1000
        axes[1, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[1, 1].axvline(x=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Prediction Error (thousands)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Error Distribution')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/used_car_results.png', dpi=300, bbox_inches='tight')
        print(f"Results saved to {output_dir}/used_car_results.png")
        plt.close()


def main():
    """Main execution."""
    print("=" * 70)
    print("USED CAR PRICE PREDICTION")
    print("=" * 70)

    model = UsedCarPriceModel()

    # Create data
    print("\nCreating synthetic dataset...")
    df = model.create_sample_data(n_samples=3000)
    df = model.feature_engineering(df)
    print(f"Dataset shape: {df.shape}")
    print(f"\nPrice statistics:")
    print(df['selling_price'].describe())

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
    print(f"MAE: ${best['MAE']:,.0f}")
    print(f"MAPE: {best['MAPE (%)']:.1f}%")
    print(f"R²: {best['R²']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
