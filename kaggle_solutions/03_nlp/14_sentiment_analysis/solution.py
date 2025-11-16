"""
情感分析 - 電影評論
分析文本情感（正面/負面）
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = LogisticRegression(max_iter=1000, random_state=42)

    def create_data(self, n=1000):
        np.random.seed(42)
        positive_words = ['great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'best']
        negative_words = ['terrible', 'awful', 'horrible', 'worst', 'hate', 'bad', 'poor']

        texts, labels = [], []
        for i in range(n):
            if i % 2 == 0:  # positive
                text = ' '.join(np.random.choice(positive_words, 5))
                label = 1
            else:  # negative
                text = ' '.join(np.random.choice(negative_words, 5))
                label = 0
            texts.append(text)
            labels.append(label)

        return pd.DataFrame({'text': texts, 'sentiment': labels})

    def train_evaluate(self, df):
        X = self.vectorizer.fit_transform(df['text'])
        y = df['sentiment']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_test)

        print(f"準確率: {accuracy_score(y_test, pred):.4f}")
        print(classification_report(y_test, pred, target_names=['負面', '正面']))

    def predict_sentiment(self, text):
        X = self.vectorizer.transform([text])
        return '正面' if self.model.predict(X)[0] == 1 else '負面'

if __name__ == "__main__":
    print("情感分析 - 電影評論")
    sa = SentimentAnalyzer()
    df = sa.create_data()
    print(f"正面評論比例: {df['sentiment'].mean():.2%}")
    sa.train_evaluate(df)

    # 測試
    test_text = "great movie love it"
    print(f"\n測試文本: '{test_text}'")
    print(f"預測情感: {sa.predict_sentiment(test_text)}")
