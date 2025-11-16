"""
保險費用預測
醫療保險費用預測
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

class InsurancePremiumPredictor:
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)

    def create_data(self, n=1338):
        np.random.seed(42)
        df = pd.DataFrame({
            'age': np.random.randint(18, 65, n),
            'bmi': np.random.normal(30, 6, n).clip(15, 55),
            'children': np.random.randint(0, 6, n),
            'smoker': np.random.choice([0, 1], n, p=[0.8, 0.2]),
            'region': np.random.choice([0, 1, 2, 3], n)
        })
        # 保險費用計算
        df['charges'] = (
            1000 +
            df['age'] * 250 +
            df['bmi'] * 30 +
            df['children'] * 500 +
            df['smoker'] * 23000 +
            np.random.normal(0, 3000, n)
        ).clip(1000, 65000)
        return df

    def train_evaluate(self, df):
        X = df.drop('charges', axis=1)
        y = df['charges']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_test)

        print("=== 保險費用預測結果 ===")
        print(f"MAE: ${mean_absolute_error(y_test, pred):,.2f}")
        print(f"R²: {r2_score(y_test, pred):.4f}")

        # 特徵重要性
        print("\n特徵重要性:")
        for feature, importance in zip(X.columns, self.model.feature_importances_):
            print(f"{feature}: {importance:.4f}")

if __name__ == "__main__":
    print("保險費用預測")
    predictor = InsurancePremiumPredictor()
    df = predictor.create_data()
    print(f"數據形狀: {df.shape}")
    print(f"平均保費: ${df['charges'].mean():,.2f}\n")
    predictor.train_evaluate(df)
