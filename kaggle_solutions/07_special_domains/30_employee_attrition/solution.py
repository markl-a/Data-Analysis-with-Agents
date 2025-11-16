"""
員工離職預測
HR分析：預測員工流失
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

class EmployeeAttritionPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def create_data(self, n=1470):
        np.random.seed(42)
        df = pd.DataFrame({
            'Age': np.random.randint(18, 60, n),
            'MonthlyIncome': np.random.normal(6500, 4500, n).clip(1000, 20000),
            'YearsAtCompany': np.random.randint(0, 40, n),
            'JobSatisfaction': np.random.randint(1, 5, n),
            'WorkLifeBalance': np.random.randint(1, 5, n),
            'DistanceFromHome': np.random.randint(1, 30, n),
            'OverTime': np.random.choice([0, 1], n, p=[0.7, 0.3])
        })
        # 離職邏輯
        attrition_risk = (
            (df['JobSatisfaction'] <= 2) * 0.3 +
            (df['WorkLifeBalance'] <= 2) * 0.2 +
            (df['OverTime'] == 1) * 0.2 +
            (df['YearsAtCompany'] < 2) * 0.2 +
            (df['MonthlyIncome'] < 3000) * 0.1
        )
        df['Attrition'] = (np.random.random(n) < attrition_risk).astype(int)
        return df

    def train_evaluate(self, df):
        X = df.drop('Attrition', axis=1)
        y = df['Attrition']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_test)
        prob = self.model.predict_proba(X_test)[:, 1]

        print("=== 員工離職預測結果 ===")
        print(f"準確率: {accuracy_score(y_test, pred):.4f}")
        print(f"AUC: {roc_auc_score(y_test, prob):.4f}")
        print("\n分類報告:")
        print(classification_report(y_test, pred, target_names=['留任', '離職']))

        # 最重要的影響因素
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\n離職影響因素:")
        print(feature_importance.head())

if __name__ == "__main__":
    print("員工離職預測 - HR分析")
    predictor = EmployeeAttritionPredictor()
    df = predictor.create_data()
    print(f"數據形狀: {df.shape}")
    print(f"離職率: {df['Attrition'].mean():.2%}\n")
    predictor.train_evaluate(df)
