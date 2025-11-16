"""
書籍推薦系統
基於內容的書籍推薦
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class BookRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.similarity_matrix = None

    def create_data(self, n=100):
        np.random.seed(42)
        genres = ['fiction', 'mystery', 'romance', 'scifi', 'history']
        authors = ['AuthorA', 'AuthorB', 'AuthorC', 'AuthorD']

        data = {
            'book_id': range(n),
            'title': [f'Book_{i}' for i in range(n)],
            'author': np.random.choice(authors, n),
            'genre': np.random.choice(genres, n),
            'rating': np.random.uniform(3, 5, n)
        }
        df = pd.DataFrame(data)
        df['description'] = df['genre'] + ' ' + df['author']
        return df

    def build_model(self, df):
        tfidf_matrix = self.vectorizer.fit_transform(df['description'])
        self.similarity_matrix = cosine_similarity(tfidf_matrix)

    def recommend(self, book_id, n=5):
        similarities = self.similarity_matrix[book_id]
        similar_indices = np.argsort(similarities)[::-1][1:n+1]
        return similar_indices.tolist()

if __name__ == "__main__":
    print("書籍推薦系統")
    recommender = BookRecommender()
    df = recommender.create_data()
    print(f"書籍數量: {len(df)}")

    recommender.build_model(df)

    book_id = 5
    recs = recommender.recommend(book_id, n=5)
    print(f"\n與 '{df.iloc[book_id]['title']}' ({df.iloc[book_id]['genre']}) 相似的書籍:")
    for rec_id in recs:
        print(f"- {df.iloc[rec_id]['title']} ({df.iloc[rec_id]['genre']})")
