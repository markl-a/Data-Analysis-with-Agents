"""
市場響應模型 - Kaggle 解決方案

營銷活動響應預測

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


class MarketResponseModel:
    """
    市場響應模型解決方案類
    
    這個類實現了營銷活動響應預測的完整機器學習流程，
    包括數據加載、預處理、特徵工程、模型訓練、評估和可視化。
    
    主要功能:
    - 自動數據加載和驗證
    - 智能特徵預處理
    - 模型訓練和優化
    - 多維度模型評估
    - 結果可視化
    
    屬性:
        model: 訓練好的機器學習模型
        scaler: 數據標準化器
        label_encoder: 標籤編碼器
        is_trained (bool): 模型是否已訓練
        feature_names: 特徵名稱列表
        results (dict): 訓練和評估結果
    
    示例:
        >>> solver = MarketResponseModel()
        >>> solver.run_pipeline('data.csv')
    """
    
    def __init__(self, random_state: int = 42):
        """
        初始化市場響應模型解決方案
        
        Args:
            random_state: 隨機種子，用於結果可復現
        """
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.random_state = random_state
        self.feature_names = []
        self.results = {}
        
        # 設置隨機種子
        np.random.seed(random_state)
        
        print(f"✓ 市場響應模型解決方案已初始化")
        print(f"  描述: 營銷活動響應預測")
        print(f"  隨機種子: {random_state}")
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        加載和初步驗證數據
        
        Args:
            data_path: 數據文件路徑（CSV格式）
        
        Returns:
            加載的DataFrame
        
        Raises:
            FileNotFoundError: 數據文件不存在
            ValueError: 數據格式不正確
        """
        try:
            print(f"\n============================================================")
            print(f"加載數據: {data_path}")
            print(f"============================================================")
            
            df = pd.read_csv(data_path)
            
            print(f"✓ 數據加載成功")
            print(f"  形狀: {df.shape}")
            print(f"  列數: {len(df.columns)}")
            print(f"  行數: {len(df)}")
            print(f"\n列信息:")
            print(df.dtypes)
            print(f"\n缺失值:")
            print(df.isnull().sum())
            
            return df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"數據文件不存在: {data_path}")
        except Exception as e:
            raise ValueError(f"數據加載失敗: {str(e)}")
    
    def preprocess(self, df: pd.DataFrame, target_col: str = 'target') -> Tuple[np.ndarray, np.ndarray]:
        """
        數據預處理和特徵工程
        
        Args:
            df: 原始數據DataFrame
            target_col: 目標列名稱
        
        Returns:
            (X, y): 特徵矩陣和目標向量
        """
        print(f"\n============================================================")
        print("數據預處理")
        print(f"============================================================")
        
        df_processed = df.copy()
        
        # 處理缺失值
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(
            df_processed[numeric_cols].median()
        )
        
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != target_col]
        
        for col in categorical_cols:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
        
        # 特徵工程
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
        
        print(f"✓ 預處理完成")
        print(f"  特徵數量: {X.shape[1]}")
        print(f"  樣本數量: {X.shape[0]}")
        print(f"  目標值範圍: {y.min()} - {y.max()}")
        
        return X, y
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        """
        訓練模型
        
        Args:
            X_train: 訓練特徵
            y_train: 訓練標籤
            **kwargs: 模型額外參數
        """
        print(f"\n============================================================")
        print("模型訓練")
        print(f"============================================================")
        
        # 數據標準化
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # 這裡使用簡單模型作為示例，實際應用中應根據任務選擇合適模型
        from sklearn.ensemble import RandomForestClassifier
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            **kwargs
        )
        
        print(f"開始訓練...")
        self.model.fit(X_train_scaled, y_train)
        
        # 交叉驗證
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        self.is_trained = True
        self.results['cv_scores'] = cv_scores
        
        print(f"✓ 訓練完成")
        print(f"  交叉驗證得分: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        評估模型性能
        
        Args:
            X_test: 測試特徵
            y_test: 測試標籤
        
        Returns:
            評估指標字典
        """
        if not self.is_trained:
            raise ValueError("模型尚未訓練，請先調用 train() 方法")
        
        print(f"\n============================================================")
        print("模型評估")
        print(f"============================================================")
        
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✓ 評估完成")
        print(f"  準確率: {accuracy:.4f}")
        print(f"\n分類報告:")
        print(classification_report(y_test, y_pred))
        
        metrics = {
            'accuracy': accuracy,
            'predictions': y_pred
        }
        
        self.results.update(metrics)
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用訓練好的模型進行預測
        
        Args:
            X: 特徵矩陣
        
        Returns:
            預測結果
        """
        if not self.is_trained:
            raise ValueError("模型尚未訓練，請先調用 train() 方法")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        print(f"✓ 預測完成，生成 {len(predictions)} 個預測結果")
        return predictions
    
    def visualize(self, results: Optional[Dict] = None):
        """
        可視化結果
        
        Args:
            results: 結果字典，如果為None則使用self.results
        """
        if results is None:
            results = self.results
        
        print(f"\n============================================================")
        print("結果可視化")
        print(f"============================================================")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{name}分析結果', fontsize=16, fontweight='bold')
        
        # 1. 交叉驗證得分
        if 'cv_scores' in results:
            ax = axes[0, 0]
            cv_scores = results['cv_scores']
            ax.bar(range(len(cv_scores)), cv_scores, color='skyblue', edgecolor='navy')
            ax.axhline(y=cv_scores.mean(), color='red', linestyle='--', 
                      label=f'平均: {cv_scores.mean():.4f}')
            ax.set_xlabel('折數')
            ax.set_ylabel('得分')
            ax.set_title('交叉驗證得分')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2. 特徵重要性
        if hasattr(self.model, 'feature_importances_') and self.feature_names:
            ax = axes[0, 1]
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[-10:]  # 前10個重要特徵
            
            feature_names_plot = [self.feature_names[i] if i < len(self.feature_names) 
                                 else f'Feature {i}' for i in indices]
            
            ax.barh(range(len(indices)), importances[indices], color='coral')
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels(feature_names_plot)
            ax.set_xlabel('重要性')
            ax.set_title('特徵重要性 (Top 10)')
            ax.grid(True, alpha=0.3, axis='x')
        
        # 3. 預測分佈
        if 'predictions' in results:
            ax = axes[1, 0]
            predictions = results['predictions']
            ax.hist(predictions, bins=30, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
            ax.set_xlabel('預測值')
            ax.set_ylabel('頻率')
            ax.set_title('預測值分佈')
            ax.grid(True, alpha=0.3)
        
        # 4. 性能指標
        ax = axes[1, 1]
        ax.axis('off')
        
        metrics_text = f"""
        ╔══════════════════════════════╗
        ║     模型性能總結              ║
        ╠══════════════════════════════╣
        ║                              ║
        """
        
        if 'accuracy' in results:
            metrics_text += f"║  準確率: {results['accuracy']:.4f}         ║\n"
        if 'cv_scores' in results:
            metrics_text += f"║  CV均值: {results['cv_scores'].mean():.4f}         ║\n"
        
        metrics_text += """        ║                              ║
        ╚══════════════════════════════╝
        """
        
        ax.text(0.5, 0.5, metrics_text, fontsize=12, ha='center', va='center',
               family='monospace', transform=ax.transAxes)
        
        plt.tight_layout()
        plt.savefig(f'{name.replace(" ", "_")}_results.png', dpi=300, bbox_inches='tight')
        print(f"✓ 可視化完成，圖表已保存")
        plt.show()
    
    def run_pipeline(self, data_path: str, target_col: str = 'target', 
                    test_size: float = 0.2, **model_kwargs):
        """
        運行完整的機器學習流程
        
        Args:
            data_path: 數據文件路徑
            target_col: 目標列名稱
            test_size: 測試集比例
            **model_kwargs: 模型額外參數
        """
        print(f"\n============================================================")
        print(f"市場響應模型 - 完整流程")
        print(f"============================================================\n")
        
        # 1. 加載數據
        df = self.load_data(data_path)
        
        # 2. 預處理
        X, y = self.preprocess(df, target_col)
        
        # 3. 劃分數據集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        print(f"\n數據集劃分:")
        print(f"  訓練集: {X_train.shape}")
        print(f"  測試集: {X_test.shape}")
        
        # 4. 訓練模型
        self.train(X_train, y_train, **model_kwargs)
        
        # 5. 評估模型
        metrics = self.evaluate(X_test, y_test)
        
        # 6. 可視化
        self.visualize()
        
        print(f"\n============================================================")
        print("✓ 流程完成！")
        print(f"============================================================")
        
        return metrics


def main():
    """
    主函數 - 演示如何使用MarketResponseModel
    """
    print(f"""
    ╔══════════════════════════════════════════════════════╗
    ║                                                      ║
    ║          市場響應模型                    ║
    ║          Kaggle Solution                            ║
    ║                                                      ║
    ║  描述: 營銷活動響應預測                        ║
    ║                                                      ║
    ╚══════════════════════════════════════════════════════╝
    """)
    
    # 創建解決方案實例
    solver = MarketResponseModel(random_state=42)
    
    # 示例用法
    print("\n使用示例:")
    print("1. 基本用法:")
    print(f"   solver = MarketResponseModel()")
    print("   solver.run_pipeline('your_data.csv', target_col='target')")
    print("\n2. 自定義流程:")
    print("   df = solver.load_data('data.csv')")
    print("   X, y = solver.preprocess(df)")
    print("   solver.train(X_train, y_train)")
    print("   metrics = solver.evaluate(X_test, y_test)")
    print("   predictions = solver.predict(X_new)")
    
    print("\n" + "="*60)
    print("注意: 請準備好數據文件後運行 run_pipeline() 方法")
    print("="*60)


if __name__ == "__main__":
    main()
