"""
假新聞偵測
識別假新聞文章
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=3000)
        self.model = MultinomialNB()

    def create_data(self, n=1000):
        np.random.seed(42)
        real_keywords = ['official', 'reported', 'according', 'statement', 'confirmed']
        fake_keywords = ['shocking', 'unbelievable', 'secret', 'exposed', 'they dont want']

        texts, labels = [], []
        for i in range(n):
            if i % 2 == 0:  # real
                text = ' '.join(np.random.choice(real_keywords, 8))
                label = 0
            else:  # fake
                text = ' '.join(np.random.choice(fake_keywords, 8))
                label = 1
            texts.append(text)
            labels.append(label)

        return pd.DataFrame({'text': texts, 'label': labels})

    def train_evaluate(self, df):
        X = self.vectorizer.fit_transform(df['text'])
        y = df['label']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_test)

        print(classification_report(y_test, pred, target_names=['真實', '假新聞']))

if __name__ == "__main__":
    print("假新聞偵測")
    fnd = FakeNewsDetector()
    df = fnd.create_data()
    print(f"假新聞比例: {df['label'].mean():.2%}")
    fnd.train_evaluate(df)
