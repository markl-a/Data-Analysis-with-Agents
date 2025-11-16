"""
手寫數字識別 - MNIST
使用神經網絡識別手寫數字
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

class DigitRecognizer:
    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=20, random_state=42)

    def load_data(self):
        """使用sklearn內建的數字數據集"""
        digits = load_digits()
        return digits.data, digits.target, digits.images

    def train_evaluate(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 標準化
        X_train = X_train / 16.0
        X_test = X_test / 16.0

        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_test)

        print(f"準確率: {accuracy_score(y_test, pred):.4f}")
        print("\n分類報告:")
        print(classification_report(y_test, pred))

        return X_test, y_test, pred

    def visualize_predictions(self, X_test, y_test, pred, n=10):
        """可視化預測結果"""
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        for i, ax in enumerate(axes.flat):
            if i < n:
                ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
                ax.set_title(f'真實: {y_test[i]}, 預測: {pred[i]}')
                ax.axis('off')
        plt.tight_layout()
        plt.savefig('digit_predictions.png', dpi=300, bbox_inches='tight')
        print("\n預測可視化已保存")

if __name__ == "__main__":
    print("手寫數字識別 - MNIST")
    recognizer = DigitRecognizer()

    # 加載數據
    X, y, images = recognizer.load_data()
    print(f"數據集大小: {X.shape}")
    print(f"類別數: {len(np.unique(y))}")

    # 訓練評估
    X_test, y_test, pred = recognizer.train_evaluate(X, y)

    # 可視化
    recognizer.visualize_predictions(X_test, y_test, pred)
