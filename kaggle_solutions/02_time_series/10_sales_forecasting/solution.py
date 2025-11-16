"""
銷售預測
預測商店未來銷售額
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

class SalesForecaster:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def create_data(self, n=1000):
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=n, freq='D')
        df = pd.DataFrame({'Date': dates})
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['Sales'] = (
            1000 +
            df['DayOfWeek'] * 50 +
            df['Month'] * 30 +
            np.random.normal(0, 100, n)
        ).clip(0)
        return df

    def feature_engineering(self, df):
        df = df.copy()
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        return df

    def train_predict(self, df):
        df = self.feature_engineering(df)
        features = ['DayOfWeek', 'Month', 'Quarter', 'IsWeekend']
        X = df[features]
        y = df['Sales']

        split = int(0.8 * len(df))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        print(f"MAE: ${mae:.2f}")

if __name__ == "__main__":
    print("銷售預測")
    sf = SalesForecaster()
    df = sf.create_data()
    print(f"平均銷售額: ${df['Sales'].mean():.2f}")
    sf.train_predict(df)
