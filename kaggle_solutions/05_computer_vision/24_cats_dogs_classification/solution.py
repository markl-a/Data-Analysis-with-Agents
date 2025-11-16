"""
貓狗圖像識別
使用CNN進行貓狗分類
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class CatDogClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def create_sample_data(self, n_samples=1000):
        """創建範例數據（簡化版）"""
        np.random.seed(42)
        # 模擬圖像特徵
        X = np.random.randint(0, 256, (n_samples, 100))  # 100個特徵
        y = np.random.randint(0, 2, n_samples)  # 0=貓, 1=狗
        return X, y

    def train_evaluate(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 標準化
        X_train = X_train / 255.0
        X_test = X_test / 255.0

        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_test)

        print(f"準確率: {accuracy_score(y_test, pred):.4f}")
        print("\n分類報告:")
        print(classification_report(y_test, pred, target_names=['貓', '狗']))

if __name__ == "__main__":
    print("貓狗圖像識別")
    classifier = CatDogClassifier()

    # 創建數據
    X, y = classifier.create_sample_data()
    print(f"數據形狀: {X.shape}")
    print(f"類別分布: 貓={np.sum(y==0)}, 狗={np.sum(y==1)}")

    # 訓練評估
    classifier.train_evaluate(X, y)

    print("\n注意: 實際應用需使用CNN和真實圖像數據")
