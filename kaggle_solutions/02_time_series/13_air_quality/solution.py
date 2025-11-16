"""
空氣品質預測
預測空氣品質指標 (AQI)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

class AirQualityPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def create_data(self, n=1000):
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=n, freq='D')
        df = pd.DataFrame({
            'Date': dates,
            'Temperature': np.random.normal(25, 10, n),
            'Humidity': np.random.uniform(30, 90, n),
            'WindSpeed': np.random.uniform(0, 20, n),
        })
        df['PM25'] = (
            50 +
            df['Temperature'] * 0.5 -
            df['WindSpeed'] * 2 +
            df['Humidity'] * 0.3 +
            np.random.normal(0, 10, n)
        ).clip(0, 500)
        return df

    def train_predict(self, df):
        features = ['Temperature', 'Humidity', 'WindSpeed']
        X = df[features]
        y = df['PM25']

        split = int(0.8 * len(df))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        print(f"MAE: {mae:.2f}")

        # AQI分級
        def classify_aqi(pm25):
            if pm25 < 35: return '良好'
            elif pm25 < 75: return '中等'
            else: return '不健康'

        print(f"\n空氣品質分布:")
        print(df['PM25'].apply(classify_aqi).value_counts())

if __name__ == "__main__":
    print("空氣品質預測")
    aqp = AirQualityPredictor()
    df = aqp.create_data()
    print(f"平均PM2.5: {df['PM25'].mean():.2f}")
    aqp.train_predict(df)
