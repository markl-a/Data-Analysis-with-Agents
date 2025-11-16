"""
收入預測 - Adult Income
預測個人年收入是否超過50K
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

class IncomePredictor:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, random_state=42)

    def create_data(self, n=32561):
        np.random.seed(42)
        df = pd.DataFrame({
            'age': np.random.randint(17, 90, n),
            'education_num': np.random.randint(1, 17, n),
            'hours_per_week': np.random.randint(1, 100, n),
            'occupation': np.random.choice(['Tech-support', 'Craft-repair', 'Sales', 'Exec-managerial'], n),
        })
        df['income'] = ((df['education_num'] > 12) & (df['hours_per_week'] > 40)).astype(int)
        return df

    def train_evaluate(self, df):
        X = df.drop('income', axis=1)
        y = df['income']
        for col in X.select_dtypes(include='object'):
            X[col] = LabelEncoder().fit_transform(X[col])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_test)
        print(f"準確率: {accuracy_score(y_test, pred):.4f}")
        print(classification_report(y_test, pred, target_names=['<=50K', '>50K']))

if __name__ == "__main__":
    print("收入預測")
    ip = IncomePredictor()
    df = ip.create_data()
    print(f">50K比例: {df['income'].mean():.2%}")
    ip.train_evaluate(df)
