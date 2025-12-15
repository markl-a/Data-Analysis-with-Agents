"""
數據處理工具模組

提供數據預處理和驗證功能：
- 任務類型檢測
- 類別平衡檢查
- 缺失值處理
- 範例數據生成
"""

from typing import Dict, Any, Optional, Tuple, Union, List
import numpy as np
import pandas as pd
import warnings


def detect_task_type(y: Union[np.ndarray, pd.Series]) -> str:
    """
    自動檢測機器學習任務類型。

    Parameters
    ----------
    y : array-like
        目標變量

    Returns
    -------
    str
        'classification' 或 'regression'

    Examples
    --------
    >>> task_type = detect_task_type(y)
    >>> print(f"Task type: {task_type}")
    """
    y_array = np.asarray(y)
    unique_values = np.unique(y_array)
    n_unique = len(unique_values)
    n_samples = len(y_array)

    # 二元或少量類別 → 分類
    if n_unique <= 20:
        return 'classification'

    # 唯一值比例很低 → 分類
    if n_unique / n_samples < 0.05:
        return 'classification'

    # 都是整數且有限值 → 分類
    if np.issubdtype(y_array.dtype, np.integer):
        if n_unique <= 50:
            return 'classification'

    return 'regression'


def check_class_balance(
    y: Union[np.ndarray, pd.Series],
    threshold: float = 0.1
) -> Dict[str, Any]:
    """
    檢查分類問題的類別平衡情況。

    Parameters
    ----------
    y : array-like
        目標變量
    threshold : float
        不平衡閾值（最小類別佔比低於此值則認為不平衡）

    Returns
    -------
    dict
        類別分布信息

    Examples
    --------
    >>> balance_info = check_class_balance(y_train)
    >>> if balance_info['is_imbalanced']:
    ...     print("Warning: Class imbalance detected!")
    """
    y_array = np.asarray(y)
    unique, counts = np.unique(y_array, return_counts=True)

    total = len(y_array)
    proportions = counts / total
    min_proportion = proportions.min()

    return {
        'classes': unique.tolist(),
        'counts': dict(zip(unique.tolist(), counts.tolist())),
        'proportions': dict(zip(unique.tolist(), proportions.tolist())),
        'min_proportion': float(min_proportion),
        'is_imbalanced': min_proportion < threshold,
        'imbalance_ratio': float(counts.max() / counts.min()),
        'recommendation': _get_balance_recommendation(min_proportion, threshold)
    }


def _get_balance_recommendation(min_proportion: float, threshold: float) -> str:
    """生成類別平衡建議。"""
    if min_proportion >= threshold:
        return "類別分布相對平衡，無需特殊處理。"
    elif min_proportion >= 0.05:
        return "輕度不平衡，建議使用 class_weight='balanced' 或 SMOTE。"
    elif min_proportion >= 0.01:
        return "中度不平衡，建議使用 SMOTE、過採樣或欠採樣技術。"
    else:
        return "嚴重不平衡，建議使用組合採樣技術或異常檢測方法。"


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'auto',
    fill_value: Optional[Any] = None,
    drop_threshold: float = 0.5
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    處理數據框中的缺失值。

    Parameters
    ----------
    df : pd.DataFrame
        輸入數據框
    strategy : str
        處理策略:
        - 'auto': 自動選擇（數值用中位數，類別用眾數）
        - 'mean': 均值填充（僅數值）
        - 'median': 中位數填充（僅數值）
        - 'mode': 眾數填充
        - 'drop_rows': 刪除含缺失值的行
        - 'drop_cols': 刪除缺失值超過閾值的列
        - 'constant': 使用 fill_value 填充
    fill_value : any, optional
        constant 策略的填充值
    drop_threshold : float
        drop_cols 策略的缺失值比例閾值

    Returns
    -------
    df_cleaned : pd.DataFrame
        清理後的數據框
    report : dict
        處理報告

    Examples
    --------
    >>> df_clean, report = handle_missing_values(df, strategy='auto')
    >>> print(f"Filled {report['total_filled']} missing values")
    """
    df_clean = df.copy()
    report = {
        'original_missing': df.isnull().sum().to_dict(),
        'total_missing_before': int(df.isnull().sum().sum()),
        'strategy': strategy,
        'actions': []
    }

    if strategy == 'drop_rows':
        df_clean = df_clean.dropna()
        report['rows_dropped'] = len(df) - len(df_clean)
        report['actions'].append(f"Dropped {report['rows_dropped']} rows with missing values")

    elif strategy == 'drop_cols':
        missing_ratio = df_clean.isnull().sum() / len(df_clean)
        cols_to_drop = missing_ratio[missing_ratio > drop_threshold].index.tolist()
        df_clean = df_clean.drop(columns=cols_to_drop)
        report['columns_dropped'] = cols_to_drop
        report['actions'].append(f"Dropped columns: {cols_to_drop}")

    elif strategy == 'constant':
        if fill_value is None:
            raise ValueError("fill_value is required for 'constant' strategy")
        df_clean = df_clean.fillna(fill_value)
        report['actions'].append(f"Filled all missing with: {fill_value}")

    elif strategy == 'auto':
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                if df_clean[col].dtype in ['int64', 'float64']:
                    fill_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(fill_val)
                    report['actions'].append(f"{col}: filled with median ({fill_val:.2f})")
                else:
                    fill_val = df_clean[col].mode().iloc[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
                    df_clean[col] = df_clean[col].fillna(fill_val)
                    report['actions'].append(f"{col}: filled with mode ({fill_val})")

    elif strategy in ['mean', 'median', 'mode']:
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                if df_clean[col].dtype in ['int64', 'float64']:
                    if strategy == 'mean':
                        fill_val = df_clean[col].mean()
                    elif strategy == 'median':
                        fill_val = df_clean[col].median()
                    else:
                        fill_val = df_clean[col].mode().iloc[0]
                    df_clean[col] = df_clean[col].fillna(fill_val)
                    report['actions'].append(f"{col}: filled with {strategy} ({fill_val:.2f})")
                elif strategy == 'mode':
                    fill_val = df_clean[col].mode().iloc[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
                    df_clean[col] = df_clean[col].fillna(fill_val)
                    report['actions'].append(f"{col}: filled with mode ({fill_val})")

    report['total_missing_after'] = int(df_clean.isnull().sum().sum())
    report['total_filled'] = report['total_missing_before'] - report['total_missing_after']

    return df_clean, report


def create_sample_data(
    task: str = 'classification',
    n_samples: int = 1000,
    n_features: int = 10,
    n_classes: int = 2,
    noise: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    創建範例數據集用於測試和演示。

    Parameters
    ----------
    task : str
        'classification' 或 'regression'
    n_samples : int
        樣本數
    n_features : int
        特徵數
    n_classes : int
        類別數（僅分類）
    noise : float
        噪聲水平
    random_state : int
        隨機種子

    Returns
    -------
    X, y : tuple
        特徵矩陣和目標變量

    Examples
    --------
    >>> X, y = create_sample_data('classification', n_samples=500, n_classes=3)
    >>> print(f"Data shape: {X.shape}, Classes: {np.unique(y)}")
    """
    np.random.seed(random_state)

    if task == 'classification':
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(2, n_features // 2),
            n_redundant=n_features // 4,
            n_classes=n_classes,
            n_clusters_per_class=1,
            flip_y=noise,
            random_state=random_state
        )
    else:
        from sklearn.datasets import make_regression
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(2, n_features // 2),
            noise=noise * 10,
            random_state=random_state
        )

    return X, y


def validate_data(
    X: Union[np.ndarray, pd.DataFrame],
    y: Optional[Union[np.ndarray, pd.Series]] = None,
    check_nan: bool = True,
    check_inf: bool = True,
    check_constant: bool = True
) -> Dict[str, Any]:
    """
    驗證數據質量。

    Parameters
    ----------
    X : array-like
        特徵矩陣
    y : array-like, optional
        目標變量
    check_nan : bool
        檢查 NaN 值
    check_inf : bool
        檢查無窮值
    check_constant : bool
        檢查常數列

    Returns
    -------
    dict
        驗證報告

    Examples
    --------
    >>> report = validate_data(X_train, y_train)
    >>> if report['has_issues']:
    ...     print(report['issues'])
    """
    X_array = np.asarray(X)
    report = {
        'n_samples': X_array.shape[0],
        'n_features': X_array.shape[1],
        'has_issues': False,
        'issues': []
    }

    # 檢查 NaN
    if check_nan:
        nan_count = np.isnan(X_array).sum()
        if nan_count > 0:
            report['has_issues'] = True
            report['nan_count'] = int(nan_count)
            report['issues'].append(f"Found {nan_count} NaN values in features")

    # 檢查無窮值
    if check_inf:
        inf_count = np.isinf(X_array).sum()
        if inf_count > 0:
            report['has_issues'] = True
            report['inf_count'] = int(inf_count)
            report['issues'].append(f"Found {inf_count} infinite values in features")

    # 檢查常數列
    if check_constant:
        constant_cols = []
        for i in range(X_array.shape[1]):
            if np.std(X_array[:, i]) == 0:
                constant_cols.append(i)
        if constant_cols:
            report['has_issues'] = True
            report['constant_columns'] = constant_cols
            report['issues'].append(f"Found {len(constant_cols)} constant columns: {constant_cols}")

    # 檢查目標變量
    if y is not None:
        y_array = np.asarray(y)
        report['target_shape'] = y_array.shape

        if check_nan and np.isnan(y_array.astype(float)).any():
            report['has_issues'] = True
            report['issues'].append("Found NaN values in target variable")

        # 樣本數匹配檢查
        if len(y_array) != X_array.shape[0]:
            report['has_issues'] = True
            report['issues'].append(
                f"Sample count mismatch: X has {X_array.shape[0]}, y has {len(y_array)}"
            )

    return report
