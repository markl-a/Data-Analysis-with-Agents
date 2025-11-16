"""
購物籃分析
使用關聯規則挖掘
"""
import pandas as pd
import numpy as np
from itertools import combinations

class MarketBasketAnalyzer:
    def __init__(self, min_support=0.1):
        self.min_support = min_support

    def create_data(self, n_transactions=1000):
        np.random.seed(42)
        items = ['Milk', 'Bread', 'Butter', 'Beer', 'Diapers', 'Eggs', 'Cheese']

        transactions = []
        for _ in range(n_transactions):
            # 每次購買2-5個商品
            n_items = np.random.randint(2, 6)
            transaction = list(np.random.choice(items, n_items, replace=False))
            transactions.append(transaction)

        return transactions

    def find_frequent_itemsets(self, transactions):
        """找出頻繁項集"""
        # 計算單個商品的支持度
        item_counts = {}
        total = len(transactions)

        for transaction in transactions:
            for item in transaction:
                item_counts[item] = item_counts.get(item, 0) + 1

        # 找出頻繁單項
        frequent_items = {item: count/total
                         for item, count in item_counts.items()
                         if count/total >= self.min_support}

        print("=== 頻繁單項 ===")
        for item, support in sorted(frequent_items.items(), key=lambda x: x[1], reverse=True):
            print(f"{item}: {support:.2%}")

        # 找出頻繁項對
        pair_counts = {}
        for transaction in transactions:
            for pair in combinations(transaction, 2):
                pair = tuple(sorted(pair))
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

        frequent_pairs = {pair: count/total
                         for pair, count in pair_counts.items()
                         if count/total >= self.min_support}

        print("\n=== 頻繁項對 ===")
        for pair, support in sorted(frequent_pairs.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"{pair}: {support:.2%}")

if __name__ == "__main__":
    print("購物籃分析 - 關聯規則挖掘")
    analyzer = MarketBasketAnalyzer(min_support=0.15)
    transactions = analyzer.create_data()
    print(f"交易數量: {len(transactions)}")
    print(f"範例交易: {transactions[:3]}\n")
    analyzer.find_frequent_itemsets(transactions)
