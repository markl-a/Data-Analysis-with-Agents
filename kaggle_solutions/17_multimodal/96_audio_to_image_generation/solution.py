"""
音頻生成圖像 - Kaggle 解決方案

Audio2Image

作者: AI Assistant
日期: 2025
版本: 2.0
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class AudioToImageGeneration:
    """
    音頻生成圖像解決方案類
    
    實現Audio2Image的完整機器學習流程。
    
    主要功能:
    - 數據加載和驗證
    - 特徵預處理和工程
    - 模型訓練和優化
    - 模型評估和驗證
    - 結果可視化
    
    屬性:
        model: 訓練好的模型
        scaler: 特徵標準化器
        is_trained: 訓練狀態
        results: 結果字典
    """
    
    def __init__(self, random_state: int = 42):
        """初始化解決方案"""
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.random_state = random_state
        self.feature_names = []
        self.results = {}
        np.random.seed(random_state)
        print(f"✓ 音頻生成圖像解決方案已初始化")
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """加載數據"""
        try:
            print(f"\n============================================================")
            print(f"加載數據: {data_path}")
            print(f"============================================================")
            df = pd.read_csv(data_path)
            print(f"✓ 數據加載成功: {df.shape}")
            return df
        except Exception as e:
            raise ValueError(f"數據加載失敗: {str(e)}")
    
    def preprocess(self, df: pd.DataFrame, target_col: str = 'target') -> Tuple[np.ndarray, np.ndarray]:
        """數據預處理"""
        print(f"\n============================================================")
        print("數據預處理")
        print(f"============================================================")
        
        df_processed = df.copy()
        
        # 處理缺失值
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
        
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != target_col]
        for col in categorical_cols:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
        
        # 分離特徵和目標
        if target_col in df_processed.columns:
            y = df_processed[target_col].values
            X = df_processed.drop(columns=[target_col])
        else:
            raise ValueError(f"目標列 '{target_col}' 不存在")
        
        # 編碼分類特徵
        for col in categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        self.feature_names = X.columns.tolist()
        X = X.values
        
        print(f"✓ 預處理完成: {X.shape[1]} 特徵, {X.shape[0]} 樣本")
        return X, y
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        """訓練模型"""
        print(f"\n============================================================")
        print("模型訓練")
        print(f"============================================================")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            **kwargs
        )
        
        self.model.fit(X_train_scaled, y_train)
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        self.is_trained = True
        self.results['cv_scores'] = cv_scores
        print(f"✓ 訓練完成 CV: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """評估模型"""
        if not self.is_trained:
            raise ValueError("模型尚未訓練")
        
        print(f"\n============================================================")
        print("模型評估")
        print(f"============================================================")
        
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"✓ 準確率: {accuracy:.4f}")
        print(f"\n{classification_report(y_test, y_pred)}")
        
        metrics = {'accuracy': accuracy, 'predictions': y_pred}
        self.results.update(metrics)
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """預測"""
        if not self.is_trained:
            raise ValueError("模型尚未訓練")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def visualize(self, results: Optional[Dict] = None):
        """可視化結果"""
        if results is None:
            results = self.results
        
        print(f"\n============================================================")
        print("結果可視化")
        print(f"============================================================")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{name}分析結果', fontsize=16, fontweight='bold')
        
        # CV得分
        if 'cv_scores' in results:
            ax = axes[0, 0]
            cv_scores = results['cv_scores']
            ax.bar(range(len(cv_scores)), cv_scores, color='skyblue')
            ax.axhline(y=cv_scores.mean(), color='red', linestyle='--')
            ax.set_xlabel('折數')
            ax.set_ylabel('得分')
            ax.set_title('交叉驗證得分')
            ax.grid(True, alpha=0.3)
        
        # 特徵重要性
        if hasattr(self.model, 'feature_importances_'):
            ax = axes[0, 1]
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[-10:]
            ax.barh(range(len(indices)), importances[indices], color='coral')
            ax.set_xlabel('重要性')
            ax.set_title('特徵重要性 (Top 10)')
        
        plt.tight_layout()
        plt.savefig(f'{name.replace(" ", "_")}_results.png', dpi=300, bbox_inches='tight')
        print(f"✓ 可視化完成")
        plt.show()
    
    def run_pipeline(self, data_path: str, target_col: str = 'target', test_size: float = 0.2, **model_kwargs):
        """運行完整流程"""
        print(f"\n============================================================")
        print(f"音頻生成圖像 - 完整流程")
        print(f"============================================================\n")
        
        df = self.load_data(data_path)
        X, y = self.preprocess(df, target_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=self.random_state)
        
        print(f"\n數據集劃分: 訓練{X_train.shape} 測試{X_test.shape}")
        
        self.train(X_train, y_train, **model_kwargs)
        metrics = self.evaluate(X_test, y_test)
        self.visualize()
        
        print(f"\n============================================================")
        print("✓ 流程完成！")
        print(f"============================================================")
        return metrics


def main():
    """主函數"""
    print(f"""
    ╔══════════════════════════════════════════════════════╗
    ║          音頻生成圖像                    ║
    ║          Kaggle Solution                            ║
    ║  描述: Audio2Image                     ║
    ╚══════════════════════════════════════════════════════╝
    """)
    
    solver = AudioToImageGeneration(random_state=42)
    print("\n使用示例:")
    print("  solver.run_pipeline('data.csv', target_col='target')")


if __name__ == "__main__":
    main()
