"""
銀行營銷預測
預測客戶是否會訂閱定期存款
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score

class BankMarketingPredictor:
    def __init__(self):
        self.model = GradientBoostingClassifier(random_state=42)
        self.encoders = {}

    def create_data(self, n=5000):
        np.random.seed(42)
        df = pd.DataFrame({
            'age': np.random.randint(18, 95, n),
            'job': np.random.choice(['admin', 'technician', 'services', 'management'], n),
            'marital': np.random.choice(['married', 'single', 'divorced'], n),
            'education': np.random.choice(['primary', 'secondary', 'tertiary'], n),
            'balance': np.random.randint(-5000, 50000, n),
            'duration': np.random.randint(0, 5000, n),
            'campaign': np.random.randint(1, 50, n),
        })
        df['y'] = (df['duration'] > 500).astype(int)
        return df

    def train_evaluate(self, df):
        X = df.drop('y', axis=1)
        y = df['y']
        for col in X.select_dtypes(include='object'):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_test)
        print(classification_report(y_test, pred))
        print(f"AUC: {roc_auc_score(y_test, self.model.predict_proba(X_test)[:, 1]):.4f}")

if __name__ == "__main__":
    print("銀行營銷預測")
    bmp = BankMarketingPredictor()
    df = bmp.create_data()
    print(f"訂閱率: {df['y'].mean():.2%}")
    bmp.train_evaluate(df)
