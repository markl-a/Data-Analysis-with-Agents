"""
紅酒品質預測
基於化學特性預測紅酒品質
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class WineQualityPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def create_data(self, n=1599):
        np.random.seed(42)
        df = pd.DataFrame({
            'fixed_acidity': np.random.uniform(4.6, 15.9, n),
            'volatile_acidity': np.random.uniform(0.12, 1.58, n),
            'citric_acid': np.random.uniform(0, 1, n),
            'residual_sugar': np.random.uniform(0.9, 15.5, n),
            'chlorides': np.random.uniform(0.012, 0.611, n),
            'alcohol': np.random.uniform(8.4, 14.9, n),
        })
        df['quality'] = (df['alcohol'] * 0.5 + np.random.normal(0, 1, n)).clip(3, 8).round()
        return df

    def train_evaluate(self, df):
        X = df.drop('quality', axis=1)
        y = df['quality']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_test)
        print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred)):.4f}")
        print(f"R²: {r2_score(y_test, pred):.4f}")

if __name__ == "__main__":
    print("紅酒品質預測")
    wqp = WineQualityPredictor()
    df = wqp.create_data()
    print(f"平均品質: {df['quality'].mean():.2f}")
    wqp.train_evaluate(df)
