"""
客戶流失預測 - Telco Customer Churn
預測電信客戶是否會流失

數據集: https://www.kaggle.com/blastchar/telco-customer-churn
難度: ⭐⭐ 初級
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


class ChurnPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoders = {}

    def create_sample_data(self, n=2000):
        np.random.seed(42)
        data = {
            'tenure': np.random.randint(1, 73, n),
            'MonthlyCharges': np.random.uniform(18, 118, n),
            'TotalCharges': np.random.uniform(18, 8500, n),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.5, 0.25, 0.25]),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.35, 0.45, 0.2]),
            'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n),
        }
        df = pd.DataFrame(data)
        # 簡化的流失邏輯
        churn_prob = (1 - df['tenure'] / 72) * 0.3 + (df['MonthlyCharges'] / 118) * 0.3
        df['Churn'] = (np.random.random(n) < churn_prob).astype(int)
        return df

    def preprocess(self, df):
        df = df.copy()
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col] = self.label_encoders[col].transform(df[col])
        return df

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        pred = self.model.predict(X_test)
        prob = self.model.predict_proba(X_test)[:, 1]
        print("\n=== 評估結果 ===")
        print(classification_report(y_test, pred, target_names=['留存', '流失']))
        print(f"AUC: {roc_auc_score(y_test, prob):.4f}")


def main():
    print("=" * 50)
    print("客戶流失預測")
    print("=" * 50)

    predictor = ChurnPredictor()
    df = predictor.create_sample_data()
    print(f"\n數據形狀: {df.shape}")
    print(f"流失率: {df['Churn'].mean():.2%}")

    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X = predictor.preprocess(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    predictor.train(X_train, y_train)
    predictor.evaluate(X_test, y_test)

    print("\n完成！")


if __name__ == "__main__":
    main()
