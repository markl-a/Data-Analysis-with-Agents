"""
醫療診斷
糖尿病/心臟病預測
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

class MedicalDiagnosisPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def create_diabetes_data(self, n=768):
        """創建糖尿病數據"""
        np.random.seed(42)
        df = pd.DataFrame({
            'Glucose': np.random.normal(120, 30, n).clip(0, 200),
            'BMI': np.random.normal(32, 7, n).clip(0, 70),
            'Age': np.random.randint(21, 81, n),
            'BloodPressure': np.random.normal(72, 12, n).clip(0, 122),
            'Insulin': np.random.normal(80, 115, n).clip(0, 846)
        })
        # 簡化的糖尿病邏輯
        diabetes_risk = (
            (df['Glucose'] > 140) * 0.4 +
            (df['BMI'] > 30) * 0.3 +
            (df['Age'] > 50) * 0.3
        )
        df['Outcome'] = (np.random.random(n) < diabetes_risk).astype(int)
        return df

    def train_evaluate(self, df):
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_test)
        prob = self.model.predict_proba(X_test)[:, 1]

        print("=== 醫療診斷結果 ===")
        print(f"準確率: {accuracy_score(y_test, pred):.4f}")
        print(f"AUC: {roc_auc_score(y_test, prob):.4f}")
        print("\n分類報告:")
        print(classification_report(y_test, pred, target_names=['健康', '糖尿病']))

        # 特徵重要性
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\n特徵重要性:")
        print(feature_importance)

if __name__ == "__main__":
    print("醫療診斷 - 糖尿病預測")
    predictor = MedicalDiagnosisPredictor()
    df = predictor.create_diabetes_data()
    print(f"數據形狀: {df.shape}")
    print(f"糖尿病患病率: {df['Outcome'].mean():.2%}\n")
    predictor.train_evaluate(df)
