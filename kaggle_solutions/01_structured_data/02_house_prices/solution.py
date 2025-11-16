"""
House Prices 房價預測
預測房屋的銷售價格

數據集: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
難度: ⭐⭐ 初級
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class HousePricePredictor:
    """房價預測器"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def create_sample_data(self, n_samples=1460):
        """創建範例數據"""
        np.random.seed(42)

        data = {
            'OverallQual': np.random.randint(1, 11, n_samples),
            'GrLivArea': np.random.normal(1500, 500, n_samples).clip(334, 5642),
            'GarageCars': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.05, 0.20, 0.50, 0.20, 0.05]),
            'GarageArea': np.random.normal(450, 200, n_samples).clip(0, 1418),
            'TotalBsmtSF': np.random.normal(1000, 400, n_samples).clip(0, 6110),
            '1stFlrSF': np.random.normal(1100, 400, n_samples).clip(334, 4692),
            'FullBath': np.random.choice([0, 1, 2, 3], n_samples, p=[0.03, 0.43, 0.47, 0.07]),
            'TotRmsAbvGrd': np.random.randint(2, 15, n_samples),
            'YearBuilt': np.random.randint(1872, 2011, n_samples),
            'YearRemodAdd': np.random.randint(1950, 2011, n_samples),
        }

        df = pd.DataFrame(data)

        # 計算目標變量（價格）
        df['SalePrice'] = (
            50000 +
            df['OverallQual'] * 20000 +
            df['GrLivArea'] * 60 +
            df['GarageCars'] * 15000 +
            df['YearBuilt'] * 100 +
            np.random.normal(0, 30000, n_samples)
        ).clip(34900, 755000)

        return df

    def feature_engineering(self, df):
        """特徵工程"""
        df = df.copy()

        # 創建新特徵
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['GrLivArea']
        df['HouseAge'] = 2024 - df['YearBuilt']
        df['RemodAge'] = 2024 - df['YearRemodAdd']
        df['QualityArea'] = df['OverallQual'] * df['GrLivArea']

        return df

    def train(self, X_train, y_train, model_type='ridge'):
        """訓練模型"""
        # 選擇模型
        if model_type == 'ridge':
            self.model = Ridge(alpha=10.0)
        elif model_type == 'lasso':
            self.model = Lasso(alpha=1.0)
        elif model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'gbm':
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)

        # 標準化
        X_train_scaled = self.scaler.fit_transform(X_train)

        # 訓練
        self.model.fit(X_train_scaled, y_train)

        # 交叉驗證
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train,
                                    cv=5, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-cv_scores)
        print(f"交叉驗證 RMSE: {rmse_scores.mean():.2f} (+/- {rmse_scores.std():.2f})")

    def predict(self, X):
        """預測"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, X_test, y_test):
        """評估模型"""
        predictions = self.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print("\n=== 模型評估結果 ===")
        print(f"RMSE: ${rmse:,.2f}")
        print(f"MAE: ${mae:,.2f}")
        print(f"R² Score: {r2:.4f}")

        # 預測 vs 實際
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, predictions, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('實際價格')
        plt.ylabel('預測價格')
        plt.title('預測價格 vs 實際價格')
        plt.savefig('prediction_vs_actual.png', dpi=300, bbox_inches='tight')
        print("\n預測圖已保存為 prediction_vs_actual.png")

        # 殘差圖
        residuals = y_test - predictions
        plt.figure(figsize=(10, 6))
        plt.scatter(predictions, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.xlabel('預測價格')
        plt.ylabel('殘差')
        plt.title('殘差圖')
        plt.savefig('residuals.png', dpi=300, bbox_inches='tight')
        print("殘差圖已保存為 residuals.png")


def main():
    """主函數"""
    print("=" * 60)
    print("House Prices 房價預測")
    print("=" * 60)

    predictor = HousePricePredictor()

    # 創建數據
    print("\n正在創建範例數據...")
    df = predictor.create_sample_data()
    print(f"數據集大小: {df.shape}")
    print(f"\n數據摘要:\n{df.describe()}")

    # 特徵工程
    print("\n正在進行特徵工程...")
    df = predictor.feature_engineering(df)

    # 準備數據
    feature_cols = [col for col in df.columns if col != 'SalePrice']
    X = df[feature_cols]
    y = df['SalePrice']

    # 分割數據
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 訓練模型
    print("\n正在訓練 Ridge 回歸模型...")
    predictor.train(X_train, y_train, model_type='ridge')

    # 評估
    print("\n正在評估模型...")
    predictor.evaluate(X_test, y_test)

    # 示例預測
    print("\n=== 示例預測 ===")
    sample = X_test.iloc[0:1]
    pred_price = predictor.predict(sample)[0]
    actual_price = y_test.iloc[0]
    print(f"預測價格: ${pred_price:,.2f}")
    print(f"實際價格: ${actual_price:,.2f}")
    print(f"誤差: ${abs(pred_price - actual_price):,.2f}")

    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
