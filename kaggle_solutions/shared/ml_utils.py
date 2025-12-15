"""
機器學習工具模組

提供防止數據洩漏的標準化 ML 功能：
- 智能 train_test_split（自動處理分類問題的分層抽樣）
- 安全的 Scaler（防止訓練/測試數據洩漏）
- 統一的評估指標
"""

from typing import Tuple, Optional, Dict, Any, Union, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
)
import warnings


def safe_train_test_split(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    test_size: float = 0.2,
    random_state: int = 42,
    task_type: Optional[str] = None,
    min_samples_per_class: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    安全的訓練/測試數據分割，自動處理分層抽樣。

    對於分類問題，自動使用 stratify=y 來維持類別分布。
    對於回歸問題或類別樣本不足時，使用普通分割。

    Parameters
    ----------
    X : array-like
        特徵矩陣
    y : array-like
        目標變量
    test_size : float
        測試集比例 (0-1)
    random_state : int
        隨機種子
    task_type : str, optional
        任務類型 ('classification' 或 'regression')，自動檢測如果未指定
    min_samples_per_class : int
        分層抽樣所需的每類最小樣本數

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple
        分割後的數據

    Examples
    --------
    >>> X_train, X_test, y_train, y_test = safe_train_test_split(X, y)
    """
    y_array = np.asarray(y)

    # 自動檢測任務類型
    if task_type is None:
        task_type = _detect_task_type(y_array)

    stratify = None

    if task_type == 'classification':
        # 檢查是否可以進行分層抽樣
        unique, counts = np.unique(y_array, return_counts=True)
        min_count = counts.min()

        if min_count >= min_samples_per_class:
            stratify = y
        else:
            warnings.warn(
                f"某些類別樣本數不足 ({min_count} < {min_samples_per_class})，"
                "無法進行分層抽樣，使用普通分割。"
            )

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )


def _detect_task_type(y: np.ndarray) -> str:
    """
    自動檢測任務類型（分類或回歸）。

    基於目標變量的特性判斷：
    - 少量唯一值 → 分類
    - 整數且有限值 → 分類
    - 浮點數且連續 → 回歸
    """
    unique_values = np.unique(y)
    n_unique = len(unique_values)
    n_samples = len(y)

    # 二元或少量類別 → 分類
    if n_unique <= 20:
        return 'classification'

    # 唯一值比例很低 → 分類
    if n_unique / n_samples < 0.05:
        return 'classification'

    # 都是整數 → 可能是分類
    if np.issubdtype(y.dtype, np.integer):
        if n_unique <= 50:
            return 'classification'

    return 'regression'


class SafeScaler:
    """
    安全的特徵縮放器，防止數據洩漏。

    確保縮放參數只從訓練數據學習，然後應用到測試數據。
    這是防止數據洩漏的關鍵最佳實踐。

    Parameters
    ----------
    scaler_type : str
        縮放器類型: 'standard' (StandardScaler) 或 'minmax' (MinMaxScaler)

    Examples
    --------
    >>> scaler = SafeScaler('standard')
    >>> X_train_scaled, X_test_scaled = scaler.fit_transform_split(X_train, X_test)
    """

    def __init__(self, scaler_type: str = 'standard'):
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"未知的縮放器類型: {scaler_type}")

        self._is_fitted = False

    def fit(self, X_train: Union[np.ndarray, pd.DataFrame]) -> 'SafeScaler':
        """只在訓練數據上擬合縮放器。"""
        self.scaler.fit(X_train)
        self._is_fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """使用已擬合的參數轉換數據。"""
        if not self._is_fitted:
            raise RuntimeError("必須先調用 fit() 或 fit_transform()")
        return self.scaler.transform(X)

    def fit_transform(self, X_train: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """擬合並轉換訓練數據。"""
        self._is_fitted = True
        return self.scaler.fit_transform(X_train)

    def fit_transform_split(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        X_test: Union[np.ndarray, pd.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        正確處理訓練和測試數據的縮放。

        - 在訓練數據上擬合並轉換
        - 使用訓練數據的參數轉換測試數據（防止數據洩漏）

        Returns
        -------
        X_train_scaled, X_test_scaled : tuple
        """
        X_train_scaled = self.fit_transform(X_train)
        X_test_scaled = self.transform(X_test)
        return X_train_scaled, X_test_scaled


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    評估分類模型的完整指標。

    Parameters
    ----------
    y_true : array-like
        真實標籤
    y_pred : array-like
        預測標籤
    y_proba : array-like, optional
        預測概率（用於 ROC-AUC）
    average : str
        多類別平均方式: 'weighted', 'macro', 'micro'

    Returns
    -------
    dict
        包含各項指標的字典
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    # ROC-AUC（需要概率預測）
    if y_proba is not None:
        try:
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                # 二元分類
                if y_proba.ndim == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            else:
                # 多類別
                metrics['roc_auc'] = roc_auc_score(
                    y_true, y_proba, multi_class='ovr', average=average
                )
        except ValueError:
            # 某些情況下無法計算 ROC-AUC
            metrics['roc_auc'] = None

    return metrics


def evaluate_regressor(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    評估回歸模型的完整指標。

    Parameters
    ----------
    y_true : array-like
        真實值
    y_pred : array-like
        預測值

    Returns
    -------
    dict
        包含各項指標的字典
    """
    # 避免除以零
    mask = y_true != 0

    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
    }

    # MAPE（避免除以零）
    if mask.sum() > 0:
        metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        metrics['mape'] = None

    return metrics


def cross_validate_model(
    model,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    cv: int = 5,
    task_type: Optional[str] = None,
    scoring: Optional[str] = None
) -> Dict[str, Any]:
    """
    使用交叉驗證評估模型。

    對於分類問題自動使用分層交叉驗證。

    Parameters
    ----------
    model : estimator
        sklearn 兼容的模型
    X : array-like
        特徵矩陣
    y : array-like
        目標變量
    cv : int
        交叉驗證折數
    task_type : str, optional
        任務類型，自動檢測如果未指定
    scoring : str, optional
        評分指標

    Returns
    -------
    dict
        交叉驗證結果
    """
    y_array = np.asarray(y)

    if task_type is None:
        task_type = _detect_task_type(y_array)

    # 選擇交叉驗證策略
    if task_type == 'classification':
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        default_scoring = 'f1_weighted'
    else:
        cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=42)
        default_scoring = 'neg_mean_squared_error'

    if scoring is None:
        scoring = default_scoring

    scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=scoring)

    return {
        'scores': scores,
        'mean': scores.mean(),
        'std': scores.std(),
        'cv_strategy': type(cv_strategy).__name__,
        'scoring': scoring
    }
