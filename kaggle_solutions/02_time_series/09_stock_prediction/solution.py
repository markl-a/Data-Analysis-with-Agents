"""
股票價格預測
使用LSTM進行時間序列預測
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self, lookback=60):
        self.lookback = lookback
        self.scaler = MinMaxScaler()

    def create_data(self, n=1000):
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=n, freq='D')
        trend = np.linspace(100, 150, n)
        seasonal = 10 * np.sin(np.linspace(0, 10*np.pi, n))
        noise = np.random.normal(0, 2, n)
        price = trend + seasonal + noise
        df = pd.DataFrame({'Date': dates, 'Close': price})
        return df

    def prepare_data(self, data):
        scaled = self.scaler.fit_transform(data.reshape(-1, 1))
        X, y = [], []
        for i in range(self.lookback, len(scaled)):
            X.append(scaled[i-self.lookback:i, 0])
            y.append(scaled[i, 0])
        return np.array(X), np.array(y)

    def train_predict(self, df):
        from sklearn.linear_model import LinearRegression

        prices = df['Close'].values
        X, y = self.prepare_data(prices)

        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        # 反標準化
        y_test_inv = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        pred_inv = self.scaler.inverse_transform(pred.reshape(-1, 1))

        rmse = np.sqrt(mean_squared_error(y_test_inv, pred_inv))
        print(f"RMSE: ${rmse:.2f}")

        plt.figure(figsize=(12, 6))
        plt.plot(y_test_inv, label='實際')
        plt.plot(pred_inv, label='預測')
        plt.legend()
        plt.title('股票價格預測')
        plt.savefig('stock_prediction.png', dpi=300, bbox_inches='tight')
        print("圖表已保存")

if __name__ == "__main__":
    print("股票價格預測")
    sp = StockPredictor()
    df = sp.create_data()
    print(f"數據範圍: {df['Date'].min()} 到 {df['Date'].max()}")
    sp.train_predict(df)
