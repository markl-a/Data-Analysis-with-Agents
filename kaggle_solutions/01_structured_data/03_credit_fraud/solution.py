"""
信用卡詐欺偵測
處理高度不平衡數據集，檢測欺詐交易

數據集: https://www.kaggle.com/mlg-ulb/creditcardfraud
難度: ⭐⭐⭐ 中級
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')


class CreditFraudDetector:
    """信用卡詐欺檢測器"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def create_sample_data(self, n_samples=10000):
        """創建範例數據（模擬高度不平衡）"""
        np.random.seed(42)

        # 正常交易 (99.8%)
        n_normal = int(n_samples * 0.998)
        normal_data = {
            'Time': np.random.uniform(0, 172800, n_normal),
            'V1': np.random.normal(0, 1.5, n_normal),
            'V2': np.random.normal(0, 1.5, n_normal),
            'V3': np.random.normal(0, 1.5, n_normal),
            'V4': np.random.normal(0, 1.5, n_normal),
            'Amount': np.random.lognormal(4, 1.5, n_normal).clip(0, 2000),
            'Class': np.zeros(n_normal)
        }

        # 欺詐交易 (0.2%)
        n_fraud = n_samples - n_normal
        fraud_data = {
            'Time': np.random.uniform(0, 172800, n_fraud),
            'V1': np.random.normal(3, 2, n_fraud),
            'V2': np.random.normal(-3, 2, n_fraud),
            'V3': np.random.normal(2, 2, n_fraud),
            'V4': np.random.normal(-2, 2, n_fraud),
            'Amount': np.random.uniform(0, 500, n_fraud),
            'Class': np.ones(n_fraud)
        }

        df_normal = pd.DataFrame(normal_data)
        df_fraud = pd.DataFrame(fraud_data)
        df = pd.concat([df_normal, df_fraud], ignore_index=True)

        return df.sample(frac=1, random_state=42).reset_index(drop=True)

    def balance_data(self, X, y, method='smote'):
        """平衡數據集"""
        if method == 'smote':
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
        elif method == 'undersample':
            rus = RandomUnderSampler(random_state=42)
            X_balanced, y_balanced = rus.fit_resample(X, y)
        else:
            X_balanced, y_balanced = X, y

        return X_balanced, y_balanced

    def train(self, X_train, y_train, balance_method='smote'):
        """訓練模型"""
        # 平衡數據
        print(f"原始數據分布: {pd.Series(y_train).value_counts().to_dict()}")
        X_train_balanced, y_train_balanced = self.balance_data(
            X_train, y_train, method=balance_method
        )
        print(f"平衡後數據分布: {pd.Series(y_train_balanced).value_counts().to_dict()}")

        # 標準化
        X_train_scaled = self.scaler.fit_transform(X_train_balanced)

        # 訓練模型
        self.model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train_balanced)

    def predict(self, X):
        """預測"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        """預測概率"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def evaluate(self, X_test, y_test):
        """評估模型"""
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)

        print("\n=== 模型評估結果 ===")
        print("\n分類報告:")
        print(classification_report(y_test, predictions,
                                   target_names=['正常', '欺詐']))

        # AUC-ROC
        auc = roc_auc_score(y_test, probabilities)
        print(f"\nAUC-ROC Score: {auc:.4f}")

        # 混淆矩陣
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
        plt.title('混淆矩陣')
        plt.ylabel('實際')
        plt.xlabel('預測')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')

        # ROC曲線
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')


def main():
    print("=" * 60)
    print("信用卡詐欺偵測")
    print("=" * 60)

    detector = CreditFraudDetector()

    # 創建數據
    print("\n正在創建範例數據...")
    df = detector.create_sample_data()
    print(f"數據集大小: {df.shape}")
    print(f"欺詐比例: {df['Class'].mean():.2%}")

    # 準備數據
    X = df.drop('Class', axis=1)
    y = df['Class']

    # 分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 訓練
    print("\n正在訓練模型（使用SMOTE平衡數據）...")
    detector.train(X_train, y_train, balance_method='smote')

    # 評估
    print("\n正在評估模型...")
    detector.evaluate(X_test, y_test)

    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
