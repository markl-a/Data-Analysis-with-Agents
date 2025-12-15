"""
Supply Chain Inventory Forecasting

Predict product demand to optimize inventory levels,
reduce costs and minimize stockout risks.

Dataset: https://www.kaggle.com/c/demand-forecasting-kernels-only
Difficulty: ⭐⭐⭐ Advanced Level
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class InventoryForecaster:
    """Supply Chain Inventory Forecasting Model."""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.best_model = None

    def create_sample_data(self, n_stores: int = 5, n_items: int = 10,
                          n_days: int = 730) -> pd.DataFrame:
        """Create synthetic inventory demand dataset."""
        np.random.seed(42)

        dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')

        # Store characteristics
        store_base = {i: np.random.uniform(50, 150) for i in range(1, n_stores + 1)}
        store_weekend_factor = {i: np.random.uniform(1.1, 1.5) for i in range(1, n_stores + 1)}

        # Item characteristics
        item_base = {i: np.random.uniform(20, 100) for i in range(1, n_items + 1)}
        item_seasonal = {i: np.random.choice([0, 1, 2, 3]) for i in range(1, n_items + 1)}
        # 0: no seasonality, 1: summer peak, 2: winter peak, 3: holiday peak

        # Holidays
        holidays = pd.to_datetime([
            '2022-01-01', '2022-02-14', '2022-04-17', '2022-07-04',
            '2022-11-24', '2022-12-25', '2023-01-01', '2023-02-14',
            '2023-04-09', '2023-07-04', '2023-11-23', '2023-12-25'
        ])

        data = []

        for store in range(1, n_stores + 1):
            for item in range(1, n_items + 1):
                for date in dates:
                    # Base demand
                    base = store_base[store] * item_base[item] / 100

                    # Day of week effect
                    dow = date.dayofweek
                    if dow >= 5:  # Weekend
                        dow_factor = store_weekend_factor[store]
                    else:
                        dow_factor = 1.0

                    # Monthly seasonality
                    month = date.month
                    season_type = item_seasonal[item]

                    if season_type == 1:  # Summer peak
                        seasonal_factor = 1 + 0.3 * np.sin(np.pi * (month - 3) / 6)
                    elif season_type == 2:  # Winter peak
                        seasonal_factor = 1 + 0.3 * np.sin(np.pi * (month - 9) / 6)
                    elif season_type == 3:  # Holiday peak
                        if month in [11, 12, 1]:
                            seasonal_factor = 1.5
                        else:
                            seasonal_factor = 1.0
                    else:
                        seasonal_factor = 1.0

                    # Holiday effect
                    is_holiday = date in holidays
                    holiday_factor = 2.0 if is_holiday else 1.0

                    # Promotion (random 10% of days)
                    is_promotion = np.random.random() < 0.1
                    promo_factor = 1.5 if is_promotion else 1.0

                    # Weather effect (simplified)
                    if month in [12, 1, 2]:  # Winter
                        weather = np.random.choice(['Cold', 'Snow', 'Normal'], p=[0.4, 0.2, 0.4])
                    elif month in [6, 7, 8]:  # Summer
                        weather = np.random.choice(['Hot', 'Rain', 'Normal'], p=[0.4, 0.2, 0.4])
                    else:
                        weather = np.random.choice(['Rain', 'Normal'], p=[0.3, 0.7])

                    weather_factor = {'Cold': 0.8, 'Snow': 0.6, 'Hot': 1.1,
                                     'Rain': 0.9, 'Normal': 1.0}[weather]

                    # Calculate sales
                    sales = base * dow_factor * seasonal_factor * holiday_factor * promo_factor * weather_factor

                    # Add noise
                    sales *= np.random.uniform(0.8, 1.2)
                    sales = max(0, int(sales))

                    data.append({
                        'date': date,
                        'store': store,
                        'item': item,
                        'sales': sales,
                        'is_holiday': int(is_holiday),
                        'is_promotion': int(is_promotion),
                        'weather': weather
                    })

        return pd.DataFrame(data)

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time series features."""
        df = df.copy()

        # Date features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Quarter
        df['quarter'] = df['date'].dt.quarter

        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Lag features (by store-item)
        df = df.sort_values(['store', 'item', 'date'])
        for lag in [1, 7, 14, 28]:
            df[f'sales_lag_{lag}'] = df.groupby(['store', 'item'])['sales'].shift(lag)

        # Rolling statistics
        for window in [7, 14, 28]:
            df[f'sales_rolling_mean_{window}'] = df.groupby(['store', 'item'])['sales'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f'sales_rolling_std_{window}'] = df.groupby(['store', 'item'])['sales'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )

        # Same day last week
        df['sales_same_dow_4w_ago'] = df.groupby(['store', 'item'])['sales'].shift(28)

        # Drop rows with NaN
        df = df.dropna()

        return df

    def analyze_data(self, df: pd.DataFrame, output_dir: str = '.') -> None:
        """Perform exploratory data analysis."""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Inventory Demand Analysis', fontsize=16)

        # Overall sales trend
        daily_sales = df.groupby('date')['sales'].sum()
        axes[0, 0].plot(daily_sales.index, daily_sales.values, 'b-', alpha=0.7, linewidth=0.5)
        axes[0, 0].set_title('Total Daily Sales Trend')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Total Sales')

        # Sales by store
        store_sales = df.groupby('store')['sales'].sum()
        store_sales.plot(kind='bar', ax=axes[0, 1], color='steelblue')
        axes[0, 1].set_title('Total Sales by Store')
        axes[0, 1].set_xlabel('Store')
        axes[0, 1].set_ylabel('Total Sales')

        # Sales by day of week
        dow_sales = df.groupby('day_of_week')['sales'].mean()
        dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[0, 2].bar(dow_labels, dow_sales.values, color='steelblue')
        axes[0, 2].set_title('Average Sales by Day of Week')
        axes[0, 2].set_ylabel('Avg Sales')

        # Monthly sales pattern
        monthly_sales = df.groupby('month')['sales'].mean()
        axes[1, 0].bar(range(1, 13), monthly_sales.values, color='steelblue')
        axes[1, 0].set_title('Average Sales by Month')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Avg Sales')
        axes[1, 0].set_xticks(range(1, 13))

        # Holiday effect
        holiday_sales = df.groupby('is_holiday')['sales'].mean()
        axes[1, 1].bar(['Regular', 'Holiday'], holiday_sales.values, color=['steelblue', 'orange'])
        axes[1, 1].set_title('Average Sales: Holiday vs Regular')
        axes[1, 1].set_ylabel('Avg Sales')

        # Promotion effect
        promo_sales = df.groupby('is_promotion')['sales'].mean()
        axes[1, 2].bar(['No Promo', 'Promo'], promo_sales.values, color=['steelblue', 'green'])
        axes[1, 2].set_title('Average Sales: Promotion Effect')
        axes[1, 2].set_ylabel('Avg Sales')

        # Weather effect
        weather_sales = df.groupby('weather')['sales'].mean().sort_values(ascending=False)
        weather_sales.plot(kind='bar', ax=axes[2, 0], color='steelblue')
        axes[2, 0].set_title('Average Sales by Weather')
        axes[2, 0].tick_params(axis='x', rotation=45)

        # Sales distribution
        axes[2, 1].hist(df['sales'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[2, 1].set_title('Sales Distribution')
        axes[2, 1].set_xlabel('Sales')
        axes[2, 1].set_ylabel('Frequency')

        # Top items
        item_sales = df.groupby('item')['sales'].sum().sort_values(ascending=True)
        item_sales.plot(kind='barh', ax=axes[2, 2], color='steelblue')
        axes[2, 2].set_title('Total Sales by Item')
        axes[2, 2].set_xlabel('Total Sales')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/inventory_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Analysis saved to {output_dir}/inventory_analysis.png")
        plt.close()

    def prepare_features(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for modeling."""
        exclude_cols = ['date', 'sales', 'weather']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Encode weather
        if fit:
            self.label_encoders['weather'] = LabelEncoder()
            weather_encoded = self.label_encoders['weather'].fit_transform(df['weather'])
        else:
            weather_encoded = self.label_encoders['weather'].transform(df['weather'])

        X = df[feature_cols].values
        X = np.column_stack([X, weather_encoded])

        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        y = df['sales'].values

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
            y_pred = np.maximum(0, y_pred)  # Sales can't be negative

            # SMAPE
            smape = 100 * np.mean(2 * np.abs(y_test - y_pred) /
                                  (np.abs(y_test) + np.abs(y_pred) + 1e-8))

            results.append({
                'Model': name,
                'MAE': mean_absolute_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'R²': r2_score(y_test, y_pred),
                'SMAPE (%)': smape
            })

        results_df = pd.DataFrame(results).sort_values('MAE')
        self.best_model = self.models[results_df.iloc[0]['Model']]

        return results_df

    def plot_results(self, results: pd.DataFrame, X_test: np.ndarray,
                    y_test: np.ndarray, output_dir: str = '.') -> None:
        """Visualize results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Inventory Forecasting Results', fontsize=16)

        # Model comparison
        results.set_index('Model')[['MAE', 'RMSE']].plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Model Error Comparison')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_ylabel('Error')

        # SMAPE comparison
        results.set_index('Model')['SMAPE (%)'].plot(kind='bar', ax=axes[0, 1], color='steelblue')
        axes[0, 1].set_title('SMAPE Comparison (Lower is Better)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].set_ylabel('SMAPE (%)')

        # Predictions vs Actual
        y_pred = self.best_model.predict(X_test)
        y_pred = np.maximum(0, y_pred)

        axes[1, 0].scatter(y_test, y_pred, alpha=0.3, s=10)
        max_val = max(y_test.max(), y_pred.max())
        axes[1, 0].plot([0, max_val], [0, max_val], 'r--', linewidth=2)
        axes[1, 0].set_xlabel('Actual Sales')
        axes[1, 0].set_ylabel('Predicted Sales')
        axes[1, 0].set_title('Actual vs Predicted (Best Model)')

        # Error distribution
        errors = y_test - y_pred
        axes[1, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[1, 1].axvline(x=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Prediction Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Error Distribution')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/inventory_results.png', dpi=300, bbox_inches='tight')
        print(f"Results saved to {output_dir}/inventory_results.png")
        plt.close()


def main():
    """Main execution."""
    print("=" * 70)
    print("SUPPLY CHAIN INVENTORY FORECASTING")
    print("=" * 70)

    forecaster = InventoryForecaster()

    # Create data
    print("\nCreating synthetic dataset...")
    df = forecaster.create_sample_data(n_stores=5, n_items=10, n_days=730)
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Stores: {df['store'].nunique()}, Items: {df['item'].nunique()}")

    # Analysis
    forecaster.analyze_data(df)

    # Feature engineering
    df_fe = forecaster.feature_engineering(df)
    print(f"\nFeatures created: {df_fe.shape[1]} columns")

    # Time series split
    train_size = int(len(df_fe) * 0.8)
    df_train = df_fe.iloc[:train_size]
    df_test = df_fe.iloc[train_size:]

    # Prepare features
    X_train, y_train = forecaster.prepare_features(df_train, fit=True)
    X_test, y_test = forecaster.prepare_features(df_test, fit=False)

    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

    # Train and evaluate
    forecaster.train_models(X_train, y_train)
    results = forecaster.evaluate_models(X_test, y_test)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(results.to_string(index=False))

    # Visualize
    forecaster.plot_results(results, X_test, y_test)

    print("\n" + "=" * 70)
    best = results.iloc[0]
    print(f"Best Model: {best['Model']}")
    print(f"MAE: {best['MAE']:.2f}")
    print(f"SMAPE: {best['SMAPE (%)']:.1f}%")
    print(f"R²: {best['R²']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
