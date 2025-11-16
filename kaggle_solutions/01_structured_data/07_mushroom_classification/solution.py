"""
蘑菇分類 - 判斷蘑菇是否有毒
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

class MushroomClassifier:
    def __init__(self):
        self.model = DecisionTreeClassifier(max_depth=10, random_state=42)

    def create_data(self, n=8124):
        np.random.seed(42)
        df = pd.DataFrame({
            'cap_shape': np.random.choice(['b', 'c', 'x', 'f', 'k', 's'], n),
            'cap_color': np.random.choice(['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'], n),
            'odor': np.random.choice(['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'], n),
            'gill_color': np.random.choice(['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y'], n),
            'stalk_shape': np.random.choice(['e', 't'], n),
        })
        df['class'] = np.random.choice(['e', 'p'], n, p=[0.52, 0.48])
        return df

    def train_evaluate(self, df):
        X = df.drop('class', axis=1)
        y = df['class']
        for col in X.columns:
            X[col] = LabelEncoder().fit_transform(X[col])
        y = LabelEncoder().fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_test)
        print(f"準確率: {accuracy_score(y_test, pred):.4f}")
        print(classification_report(y_test, pred, target_names=['可食用', '有毒']))

if __name__ == "__main__":
    print("蘑菇分類")
    mc = MushroomClassifier()
    df = mc.create_data()
    print(f"有毒比例: {(df['class'] == 'p').mean():.2%}")
    mc.train_evaluate(df)
