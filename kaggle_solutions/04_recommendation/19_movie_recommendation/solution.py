"""
電影推薦系統
使用協同過濾推薦電影
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self):
        self.user_item_matrix = None
        self.similarity_matrix = None

    def create_data(self, n_users=100, n_movies=50):
        np.random.seed(42)
        data = []
        for user in range(n_users):
            # 每個用戶評分 5-15 部電影
            n_ratings = np.random.randint(5, 16)
            movies = np.random.choice(n_movies, n_ratings, replace=False)
            ratings = np.random.randint(1, 6, n_ratings)
            for movie, rating in zip(movies, ratings):
                data.append({'user_id': user, 'movie_id': movie, 'rating': rating})
        return pd.DataFrame(data)

    def build_model(self, df):
        # 創建用戶-電影評分矩陣
        self.user_item_matrix = df.pivot(
            index='user_id', columns='movie_id', values='rating'
        ).fillna(0)

        # 計算電影相似度
        self.similarity_matrix = cosine_similarity(self.user_item_matrix.T)

    def recommend(self, movie_id, n=5):
        """基於電影相似度推薦"""
        if movie_id >= len(self.similarity_matrix):
            return []

        similarities = self.similarity_matrix[movie_id]
        similar_indices = np.argsort(similarities)[::-1][1:n+1]

        recommendations = [
            {'movie_id': idx, 'similarity': similarities[idx]}
            for idx in similar_indices
        ]
        return recommendations

if __name__ == "__main__":
    print("電影推薦系統")
    recommender = MovieRecommender()

    # 創建數據
    df = recommender.create_data()
    print(f"總評分數: {len(df)}")
    print(f"用戶數: {df['user_id'].nunique()}")
    print(f"電影數: {df['movie_id'].nunique()}")

    # 建立模型
    recommender.build_model(df)

    # 推薦
    movie_id = 5
    recs = recommender.recommend(movie_id, n=5)
    print(f"\n與電影 {movie_id} 相似的電影:")
    for rec in recs:
        print(f"電影 {rec['movie_id']}: 相似度 {rec['similarity']:.4f}")
