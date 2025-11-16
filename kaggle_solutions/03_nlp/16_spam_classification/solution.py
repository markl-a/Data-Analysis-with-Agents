"""
垃圾郵件分類
過濾垃圾郵件/短信
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

class SpamClassifier:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()

    def create_data(self, n=1000):
        np.random.seed(42)
        spam_words = ['win', 'free', 'prize', 'click', 'urgent', 'congratulations']
        ham_words = ['meeting', 'tomorrow', 'thanks', 'please', 'schedule']

        texts, labels = [], []
        for i in range(n):
            if i % 3 == 0:  # spam
                text = ' '.join(np.random.choice(spam_words, 6))
                label = 1
            else:  # ham
                text = ' '.join(np.random.choice(ham_words, 6))
                label = 0
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

        print(f"準確率: {accuracy_score(y_test, pred):.4f}")
        print(classification_report(y_test, pred, target_names=['正常', '垃圾']))

if __name__ == "__main__":
    print("垃圾郵件分類")
    sc = SpamClassifier()
    df = sc.create_data()
    print(f"垃圾郵件比例: {df['label'].mean():.2%}")
    sc.train_evaluate(df)
