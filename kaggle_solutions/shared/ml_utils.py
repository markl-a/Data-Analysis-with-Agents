"""
機器學習工具模組

提供防止數據洩漏的標準化 ML 功能：
- 智能 train_test_split（自動處理分類問題的分層抽樣）
- 安全的 Scaler（防止訓練/測試數據洩漏）
- 統一的評估指標
"""

from typing import Tuple, Optional, Dict, Any, Union, List, Callable
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
)
from sklearn.base import clone
import warnings
import time


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


class EarlyStopping:
    """
    早停回調工具，用於防止過擬合。

    監控驗證集上的性能指標，當連續多個 epoch 沒有改善時停止訓練。

    Parameters
    ----------
    patience : int
        容忍的連續無改善輪數
    min_delta : float
        被認為是改善的最小變化量
    mode : str
        'min' (越小越好，如損失) 或 'max' (越大越好，如準確率)
    restore_best : bool
        是否在停止時恢復最佳參數
    verbose : bool
        是否打印信息

    Attributes
    ----------
    best_score : float
        記錄的最佳分數
    counter : int
        無改善計數器
    best_params : dict
        最佳模型參數（如果 restore_best=True）
    stopped_epoch : int
        停止的輪數

    Examples
    --------
    >>> early_stop = EarlyStopping(patience=5, mode='max')
    >>> for epoch in range(100):
    ...     train_model(model, X_train, y_train)
    ...     val_score = evaluate_model(model, X_val, y_val)
    ...     if early_stop.step(val_score, model):
    ...         print(f"Early stopping at epoch {epoch}")
    ...         break
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'max',
        restore_best: bool = True,
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.best_params = None
        self.stopped_epoch = 0
        self.should_stop = False

        if mode not in ['min', 'max']:
            raise ValueError("mode must be 'min' or 'max'")

    def step(self, score: float, model: Optional[Any] = None) -> bool:
        """
        檢查是否應該停止訓練。

        Parameters
        ----------
        score : float
            當前驗證分數
        model : object, optional
            當前模型（用於保存最佳參數）

        Returns
        -------
        bool
            如果應該停止訓練則返回 True
        """
        if self.best_score is None:
            self.best_score = score
            if model is not None and self.restore_best:
                self.best_params = self._get_model_params(model)
            return False

        # 檢查是否有改善
        if self._is_improvement(score):
            if self.verbose:
                print(f"Validation score improved: {self.best_score:.6f} -> {score:.6f}")
            self.best_score = score
            self.counter = 0
            if model is not None and self.restore_best:
                self.best_params = self._get_model_params(model)
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement for {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"Early stopping triggered! Best score: {self.best_score:.6f}")
                return True

        return False

    def _is_improvement(self, score: float) -> bool:
        """檢查分數是否有改善。"""
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta

    def _get_model_params(self, model: Any) -> Dict[str, Any]:
        """獲取模型參數（如果可能）。"""
        try:
            if hasattr(model, 'get_params'):
                return model.get_params()
            elif hasattr(model, 'state_dict'):  # PyTorch 模型
                import copy
                return copy.deepcopy(model.state_dict())
            else:
                return None
        except Exception:
            return None

    def restore_best_weights(self, model: Any) -> None:
        """恢復最佳模型參數。"""
        if self.best_params is None:
            if self.verbose:
                print("No best parameters saved, cannot restore")
            return

        try:
            if hasattr(model, 'set_params'):
                model.set_params(**self.best_params)
            elif hasattr(model, 'load_state_dict'):  # PyTorch 模型
                model.load_state_dict(self.best_params)
            if self.verbose:
                print("Best weights restored")
        except Exception as e:
            if self.verbose:
                print(f"Failed to restore best weights: {e}")


def model_comparison(
    models: Dict[str, Any],
    X_train: Union[np.ndarray, pd.DataFrame],
    y_train: Union[np.ndarray, pd.Series],
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: Union[np.ndarray, pd.Series],
    cv: int = 5,
    task_type: Optional[str] = None,
    scoring: Optional[str] = None,
    return_predictions: bool = False
) -> pd.DataFrame:
    """
    比較多個模型的性能。

    訓練並評估多個模型，生成性能對比表格。

    Parameters
    ----------
    models : dict
        模型字典，格式 {'模型名稱': 模型實例}
    X_train : array-like
        訓練特徵
    y_train : array-like
        訓練標籤
    X_test : array-like
        測試特徵
    y_test : array-like
        測試標籤
    cv : int
        交叉驗證折數
    task_type : str, optional
        任務類型 ('classification' 或 'regression')
    scoring : str, optional
        評分指標
    return_predictions : bool
        是否返回預測結果

    Returns
    -------
    pd.DataFrame
        模型性能對比表格

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.linear_model import LogisticRegression
    >>> models = {
    ...     'Random Forest': RandomForestClassifier(),
    ...     'Logistic Regression': LogisticRegression()
    ... }
    >>> results = model_comparison(models, X_train, y_train, X_test, y_test)
    >>> print(results.sort_values('test_score', ascending=False))
    """
    y_array = np.asarray(y_train)
    if task_type is None:
        task_type = _detect_task_type(y_array)

    results = []
    predictions = {}

    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")

        # 訓練計時
        start_time = time.time()
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        train_time = time.time() - start_time

        # 預測
        y_pred = model_clone.predict(X_test)

        # 交叉驗證
        cv_results = cross_validate_model(
            model_clone, X_train, y_train, cv=cv,
            task_type=task_type, scoring=scoring
        )

        # 評估
        if task_type == 'classification':
            y_proba = None
            if hasattr(model_clone, 'predict_proba'):
                y_proba = model_clone.predict_proba(X_test)

            metrics = evaluate_classifier(y_test, y_pred, y_proba)
            test_score = metrics['accuracy']
        else:
            metrics = evaluate_regressor(y_test, y_pred)
            test_score = metrics['r2']

        # 收集結果
        result = {
            'Model': name,
            'CV Mean': cv_results['mean'],
            'CV Std': cv_results['std'],
            'Test Score': test_score,
            'Train Time (s)': train_time
        }

        # 添加詳細指標
        result.update({f'test_{k}': v for k, v in metrics.items()})

        results.append(result)

        if return_predictions:
            predictions[name] = {
                'y_pred': y_pred,
                'model': model_clone
            }

        # 打印結果
        print(f"CV Score: {cv_results['mean']:.4f} (+/- {cv_results['std']:.4f})")
        print(f"Test Score: {test_score:.4f}")
        print(f"Training Time: {train_time:.2f}s")

    # 創建結果 DataFrame
    df_results = pd.DataFrame(results)

    # 排序（測試分數降序）
    df_results = df_results.sort_values('Test Score', ascending=False).reset_index(drop=True)

    print(f"\n{'='*60}")
    print("Model Comparison Summary")
    print(f"{'='*60}")
    print(df_results[['Model', 'CV Mean', 'CV Std', 'Test Score', 'Train Time (s)']].to_string(index=False))

    if return_predictions:
        return df_results, predictions
    return df_results


def generate_hyperparameter_grid(
    model_type: str,
    search_type: str = 'grid',
    size: str = 'medium'
) -> Dict[str, List[Any]]:
    """
    生成常用模型的超參數網格。

    為流行的機器學習模型生成預定義的超參數搜索空間。

    Parameters
    ----------
    model_type : str
        模型類型，支持：
        - 'random_forest'
        - 'gradient_boosting'
        - 'xgboost'
        - 'lightgbm'
        - 'logistic_regression'
        - 'svm'
        - 'knn'
        - 'mlp'
    search_type : str
        搜索類型: 'grid' (網格搜索) 或 'random' (隨機搜索)
    size : str
        搜索空間大小: 'small', 'medium', 'large'

    Returns
    -------
    dict
        超參數網格字典

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.model_selection import GridSearchCV
    >>> param_grid = generate_hyperparameter_grid('random_forest', size='small')
    >>> grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    >>> grid_search.fit(X_train, y_train)

    Notes
    -----
    - 'small': 快速搜索，適合探索性分析
    - 'medium': 平衡的搜索空間，適合大多數情況
    - 'large': 詳盡搜索，適合最終調優
    """
    grids = {
        'random_forest': {
            'small': {
                'n_estimators': [50, 100],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5],
            },
            'medium': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2'],
            },
            'large': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 5, 10, 20, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False],
            }
        },
        'gradient_boosting': {
            'small': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
            },
            'medium': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'min_samples_split': [2, 5],
            },
            'large': {
                'n_estimators': [50, 100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9],
                'subsample': [0.6, 0.8, 1.0],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            }
        },
        'xgboost': {
            'small': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
            },
            'medium': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'gamma': [0, 0.1],
            },
            'large': {
                'n_estimators': [50, 100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'gamma': [0, 0.1, 0.2],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [0, 0.1, 1],
            }
        },
        'lightgbm': {
            'small': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'num_leaves': [31, 63],
            },
            'medium': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 63, 127],
                'max_depth': [-1, 5, 10],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
            },
            'large': {
                'n_estimators': [50, 100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'num_leaves': [31, 63, 127, 255],
                'max_depth': [-1, 5, 10, 20],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [0, 0.1, 1],
                'min_child_samples': [5, 10, 20],
            }
        },
        'logistic_regression': {
            'small': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l2'],
            },
            'medium': {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000],
            },
            'large': {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000, 2000],
                'l1_ratio': [0.3, 0.5, 0.7],
            }
        },
        'svm': {
            'small': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['rbf'],
            },
            'medium': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto'],
            },
            'large': {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'kernel': ['rbf', 'linear', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'degree': [2, 3, 4],
            }
        },
        'knn': {
            'small': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance'],
            },
            'medium': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan'],
            },
            'large': {
                'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski'],
                'p': [1, 2, 3],
                'algorithm': ['auto', 'ball_tree', 'kd_tree'],
            }
        },
        'mlp': {
            'small': {
                'hidden_layer_sizes': [(50,), (100,)],
                'activation': ['relu'],
                'alpha': [0.0001, 0.001],
            },
            'medium': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive'],
            },
            'large': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100), (100, 50, 25)],
                'activation': ['relu', 'tanh', 'logistic'],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'learning_rate': ['constant', 'invscaling', 'adaptive'],
                'learning_rate_init': [0.001, 0.01],
                'max_iter': [500, 1000],
            }
        },
    }

    if model_type not in grids:
        available_models = ', '.join(grids.keys())
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Available models: {available_models}"
        )

    if size not in ['small', 'medium', 'large']:
        raise ValueError("size must be 'small', 'medium', or 'large'")

    param_grid = grids[model_type][size]

    print(f"\n{'='*60}")
    print(f"Hyperparameter Grid for {model_type.replace('_', ' ').title()}")
    print(f"Search Type: {search_type.upper()}, Size: {size.upper()}")
    print(f"{'='*60}")

    # 計算總組合數（僅網格搜索）
    if search_type == 'grid':
        total_combinations = 1
        for param_values in param_grid.values():
            total_combinations *= len(param_values)
        print(f"Total combinations: {total_combinations}")

    print("\nParameters:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")

    return param_grid
