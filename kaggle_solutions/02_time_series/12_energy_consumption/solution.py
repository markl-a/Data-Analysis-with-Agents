"""
能源消耗預測
預測建築物能源使用
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

class EnergyPredictor:
    def __init__(self):
        self.model = GradientBoostingRegressor(random_state=42)

    def create_data(self, n=8760):  # 一年的小時數
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=n, freq='H')
        df = pd.DataFrame({'DateTime': dates})
        df['Hour'] = df['DateTime'].dt.hour
        df['Month'] = df['DateTime'].dt.month
        df['Temperature'] = 20 + 10*np.sin(df['Month']*np.pi/6) + np.random.normal(0, 2, n)
        df['Energy'] = (
            100 +
            (df['Hour'] - 12)**2 * 2 +
            (df['Temperature'] - 20)**2 * 0.5 +
            np.random.normal(0, 10, n)
        ).clip(0)
        return df

    def train_predict(self, df):
        features = ['Hour', 'Month', 'Temperature']
        X = df[features]
        y = df['Energy']

        split = int(0.8 * len(df))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, pred))
        print(f"RMSE: {rmse:.2f} kWh")

if __name__ == "__main__":
    print("能源消耗預測")
    ep = EnergyPredictor()
    df = ep.create_data()
    print(f"平均能耗: {df['Energy'].mean():.2f} kWh")
    ep.train_predict(df)
