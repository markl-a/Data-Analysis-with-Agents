"""
Titanic 存活預測 - Kaggle經典入門項目
預測鐵達尼號乘客的存活率

數據集: https://www.kaggle.com/c/titanic
難度: ⭐ 入門級
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class TitanicSurvivalPredictor:
    """鐵達尼號存活預測器"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def create_sample_data(self):
        """創建範例數據（用於演示）"""
        np.random.seed(42)
        n_samples = 891

        data = {
            'PassengerId': range(1, n_samples + 1),
            'Survived': np.random.choice([0, 1], n_samples, p=[0.62, 0.38]),
            'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55]),
            'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
            'Age': np.random.normal(30, 14, n_samples).clip(0.42, 80),
            'SibSp': np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.68, 0.23, 0.05, 0.02, 0.01, 0.01]),
            'Parch': np.random.choice([0, 1, 2, 3, 4, 5, 6], n_samples, p=[0.76, 0.13, 0.08, 0.01, 0.01, 0.003, 0.001]),
            'Fare': np.random.lognormal(3, 1, n_samples).clip(0, 512),
            'Embarked': np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.72, 0.19, 0.09])
        }

        df = pd.DataFrame(data)

        # 增加現實性：女性和高等艙位存活率更高
        for idx in df.index:
            if df.loc[idx, 'Sex'] == 'female':
                df.loc[idx, 'Survived'] = np.random.choice([0, 1], p=[0.26, 0.74])
            if df.loc[idx, 'Pclass'] == 1:
                df.loc[idx, 'Survived'] = np.random.choice([0, 1], p=[0.37, 0.63])
            elif df.loc[idx, 'Pclass'] == 3:
                df.loc[idx, 'Survived'] = np.random.choice([0, 1], p=[0.76, 0.24])

        # 添加一些缺失值
        df.loc[np.random.choice(df.index, 177, replace=False), 'Age'] = np.nan
        df.loc[np.random.choice(df.index, 2, replace=False), 'Embarked'] = np.nan

        return df

    def preprocess_data(self, df):
        """數據預處理"""
        df = df.copy()

        # 處理缺失值
        df['Age'].fillna(df['Age'].median(), inplace=True)
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
        df['Fare'].fillna(df['Fare'].median(), inplace=True)

        # 特徵工程
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        df['Age_Group'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100],
                                  labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
        df['Fare_Group'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Med', 'High', 'VeryHigh'])

        # 編碼分類變量
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        df = pd.get_dummies(df, columns=['Embarked', 'Age_Group', 'Fare_Group'], drop_first=True)

        return df

    def train(self, X_train, y_train):
        """訓練模型"""
        # 使用隨機森林分類器
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            random_state=42
        )

        # 標準化特徵
        X_train_scaled = self.scaler.fit_transform(X_train)

        # 訓練模型
        self.model.fit(X_train_scaled, y_train)

        # 交叉驗證
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"交叉驗證準確率: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    def predict(self, X):
        """預測"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, X_test, y_test):
        """評估模型"""
        predictions = self.predict(X_test)

        print("\n=== 模型評估結果 ===")
        print(f"準確率: {accuracy_score(y_test, predictions):.4f}")
        print("\n分類報告:")
        print(classification_report(y_test, predictions,
                                   target_names=['未存活', '存活']))

        # 混淆矩陣
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('混淆矩陣')
        plt.ylabel('實際值')
        plt.xlabel('預測值')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\n混淆矩陣已保存為 confusion_matrix.png")

    def plot_feature_importance(self, feature_names):
        """繪製特徵重要性"""
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]

        plt.figure(figsize=(10, 6))
        plt.title('Top 10 特徵重要性')
        plt.bar(range(10), importances[indices])
        plt.xticks(range(10), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("特徵重要性圖已保存為 feature_importance.png")


def main():
    """主函數"""
    print("=" * 60)
    print("Titanic 存活預測 - Kaggle經典入門項目")
    print("=" * 60)

    # 初始化預測器
    predictor = TitanicSurvivalPredictor()

    # 創建範例數據（實際使用時應從Kaggle下載真實數據）
    print("\n正在創建範例數據...")
    df = predictor.create_sample_data()
    print(f"數據集大小: {df.shape}")
    print(f"\n前5行數據:\n{df.head()}")

    # 數據探索
    print(f"\n存活率: {df['Survived'].mean():.2%}")
    print(f"\n各艙位存活率:\n{df.groupby('Pclass')['Survived'].mean()}")
    print(f"\n性別存活率:\n{df.groupby('Sex')['Survived'].mean()}")

    # 數據預處理
    print("\n正在進行數據預處理...")
    df_processed = predictor.preprocess_data(df)

    # 準備訓練數據
    feature_cols = [col for col in df_processed.columns
                   if col not in ['PassengerId', 'Survived']]
    X = df_processed[feature_cols]
    y = df['Survived']

    # 分割數據
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"訓練集大小: {X_train.shape}")
    print(f"測試集大小: {X_test.shape}")

    # 訓練模型
    print("\n正在訓練隨機森林模型...")
    predictor.train(X_train, y_train)

    # 評估模型
    print("\n正在評估模型...")
    predictor.evaluate(X_test, y_test)

    # 特徵重要性
    print("\n正在分析特徵重要性...")
    predictor.plot_feature_importance(feature_cols)

    # 示例預測
    print("\n=== 示例預測 ===")
    sample_passenger = X_test.iloc[0:1]
    prediction = predictor.predict(sample_passenger)
    print(f"乘客特徵: {sample_passenger.to_dict('records')[0]}")
    print(f"預測結果: {'存活' if prediction[0] == 1 else '未存活'}")
    print(f"實際結果: {'存活' if y_test.iloc[0] == 1 else '未存活'}")

    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
