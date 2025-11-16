"""
異常檢測
使用Isolation Forest檢測異常
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

class AnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.1, random_state=42)

    def create_data(self, n=1000):
        np.random.seed(42)
        # 正常數據
        normal = pd.DataFrame({
            'feature1': np.random.normal(0, 1, int(n*0.9)),
            'feature2': np.random.normal(0, 1, int(n*0.9))
        })
        # 異常數據
        anomaly = pd.DataFrame({
            'feature1': np.random.normal(5, 1, int(n*0.1)),
            'feature2': np.random.normal(5, 1, int(n*0.1))
        })
        df = pd.concat([normal, anomaly], ignore_index=True)
        df['is_anomaly'] = [0]*int(n*0.9) + [1]*int(n*0.1)
        return df.sample(frac=1, random_state=42).reset_index(drop=True)

    def detect(self, df):
        X = df[['feature1', 'feature2']]
        predictions = self.model.fit_predict(X)
        # Isolation Forest: 1=normal, -1=anomaly
        predictions = (predictions == -1).astype(int)

        print("=== 異常檢測結果 ===")
        print(classification_report(df['is_anomaly'], predictions,
                                   target_names=['正常', '異常']))

        # 可視化
        plt.figure(figsize=(10, 6))
        plt.scatter(df[predictions==0]['feature1'], df[predictions==0]['feature2'],
                   c='blue', label='正常', alpha=0.5)
        plt.scatter(df[predictions==1]['feature1'], df[predictions==1]['feature2'],
                   c='red', label='異常', alpha=0.5)
        plt.legend()
        plt.title('異常檢測')
        plt.savefig('anomaly_detection.png', dpi=300, bbox_inches='tight')
        print("圖表已保存")

if __name__ == "__main__":
    print("異常檢測 - Isolation Forest")
    detector = AnomalyDetector()
    df = detector.create_data()
    print(f"數據形狀: {df.shape}")
    print(f"異常比例: {df['is_anomaly'].mean():.2%}\n")
    detector.detect(df)
