"""
災難推文識別
識別真實災難相關推文
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class DisasterTweetClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=3000)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def create_data(self, n=1000):
        np.random.seed(42)
        disaster_words = ['fire', 'earthquake', 'flood', 'emergency', 'disaster', 'rescue']
        normal_words = ['good', 'day', 'happy', 'love', 'great', 'nice']

        texts, labels = [], []
        for i in range(n):
            if i % 2 == 0:  # disaster
                text = ' '.join(np.random.choice(disaster_words, 5))
                label = 1
            else:  # normal
                text = ' '.join(np.random.choice(normal_words, 5))
                label = 0
            texts.append(text)
            labels.append(label)

        return pd.DataFrame({'text': texts, 'target': labels})

    def train_evaluate(self, df):
        X = self.vectorizer.fit_transform(df['text'])
        y = df['target']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_test)

        print(f"準確率: {accuracy_score(y_test, pred):.4f}")
        print(classification_report(y_test, pred, target_names=['正常', '災難']))

if __name__ == "__main__":
    print("災難推文識別")
    dtc = DisasterTweetClassifier()
    df = dtc.create_data()
    print(f"災難推文比例: {df['target'].mean():.2%}")
    dtc.train_evaluate(df)
