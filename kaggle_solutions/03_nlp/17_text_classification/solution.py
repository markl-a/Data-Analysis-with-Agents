"""
文本分類 - 新聞主題
將新聞文章分類到不同主題
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

class TextClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=2000)
        self.model = LogisticRegression(max_iter=1000, random_state=42)

    def create_data(self, n=1200):
        np.random.seed(42)
        categories = {
            'sports': ['football', 'basketball', 'player', 'game', 'score', 'team'],
            'politics': ['government', 'president', 'election', 'vote', 'policy'],
            'tech': ['technology', 'software', 'computer', 'digital', 'internet'],
            'business': ['market', 'stock', 'economy', 'company', 'finance']
        }

        texts, labels = [], []
        for cat, words in categories.items():
            for _ in range(n // 4):
                text = ' '.join(np.random.choice(words, 7))
                texts.append(text)
                labels.append(cat)

        return pd.DataFrame({'text': texts, 'category': labels})

    def train_evaluate(self, df):
        X = self.vectorizer.fit_transform(df['text'])
        y = df['category']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_test)

        print(classification_report(y_test, pred))

if __name__ == "__main__":
    print("文本分類 - 新聞主題")
    tc = TextClassifier()
    df = tc.create_data()
    print(f"類別分布:\n{df['category'].value_counts()}")
    tc.train_evaluate(df)
