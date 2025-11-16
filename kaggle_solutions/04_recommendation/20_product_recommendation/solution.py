"""
產品推薦系統
電商產品推薦
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ProductRecommender:
    def __init__(self):
        self.product_features = None

    def create_data(self, n_products=100):
        np.random.seed(42)
        data = {
            'product_id': range(n_products),
            'price': np.random.uniform(10, 1000, n_products),
            'category': np.random.choice(['電子', '服飾', '食品', '家具'], n_products),
            'rating': np.random.uniform(3, 5, n_products),
            'popularity': np.random.randint(0, 10000, n_products)
        }
        df = pd.DataFrame(data)
        # One-hot encode category
        df = pd.get_dummies(df, columns=['category'])
        return df

    def recommend_similar(self, df, product_id, n=5):
        """推薦相似產品"""
        feature_cols = [col for col in df.columns if col != 'product_id']
        features = df[feature_cols].values

        similarities = cosine_similarity([features[product_id]], features)[0]
        similar_indices = np.argsort(similarities)[::-1][1:n+1]

        return df.iloc[similar_indices]['product_id'].tolist()

if __name__ == "__main__":
    print("產品推薦系統")
    recommender = ProductRecommender()
    df = recommender.create_data()
    print(f"產品數量: {len(df)}")

    product_id = 10
    recs = recommender.recommend_similar(df, product_id, n=5)
    print(f"\n與產品 {product_id} 相似的產品: {recs}")
