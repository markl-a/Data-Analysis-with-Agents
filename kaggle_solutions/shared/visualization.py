"""
可視化工具模組

提供常用的機器學習可視化功能：
- 混淆矩陣
- ROC 曲線
- 特徵重要性
- 學習曲線
- 殘差分析
"""

from typing import Optional, List, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import learning_curve


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    normalize: bool = False,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = 'Blues',
    title: str = 'Confusion Matrix',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    繪製混淆矩陣熱力圖。

    Parameters
    ----------
    y_true : array-like
        真實標籤
    y_pred : array-like
        預測標籤
    labels : list, optional
        類別名稱
    normalize : bool
        是否歸一化顯示百分比
    figsize : tuple
        圖像大小
    cmap : str
        顏色映射
    title : str
        標題
    save_path : str, optional
        保存路徑

    Returns
    -------
    matplotlib.figure.Figure
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    figsize: Tuple[int, int] = (8, 6),
    title: str = 'ROC Curve',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    繪製 ROC 曲線。

    Parameters
    ----------
    y_true : array-like
        真實標籤（二元）
    y_proba : array-like
        正類的預測概率
    figsize : tuple
        圖像大小
    title : str
        標題
    save_path : str, optional
        保存路徑

    Returns
    -------
    matplotlib.figure.Figure
    """
    # 處理多維概率數組
    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1]

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    title: str = 'Feature Importance',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    繪製特徵重要性條形圖。

    Parameters
    ----------
    feature_names : list
        特徵名稱
    importances : array-like
        特徵重要性分數
    top_n : int
        顯示前 N 個重要特徵
    figsize : tuple
        圖像大小
    title : str
        標題
    save_path : str, optional
        保存路徑

    Returns
    -------
    matplotlib.figure.Figure
    """
    # 排序並取前 N 個
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_importances, align='center', color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()  # 最重要的在上面
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_learning_curve(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
    figsize: Tuple[int, int] = (10, 6),
    title: str = 'Learning Curve',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    繪製學習曲線，用於診斷過擬合/欠擬合。

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
    train_sizes : array-like
        訓練集大小比例
    figsize : tuple
        圖像大小
    title : str
        標題
    save_path : str, optional
        保存路徑

    Returns
    -------
    matplotlib.figure.Figure
    """
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1,
        train_sizes=train_sizes, random_state=42
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    fig, ax = plt.subplots(figsize=figsize)

    # 訓練分數
    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                    alpha=0.2, color='blue')
    ax.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')

    # 驗證分數
    ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std,
                    alpha=0.2, color='orange')
    ax.plot(train_sizes_abs, val_mean, 'o-', color='orange', label='Validation Score')

    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figsize: Tuple[int, int] = (14, 5),
    title: str = 'Residual Analysis',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    繪製殘差分析圖（用於回歸模型）。

    包含：
    - 預測 vs 實際散點圖
    - 殘差分布直方圖
    - 殘差 vs 預測值散點圖

    Parameters
    ----------
    y_true : array-like
        真實值
    y_pred : array-like
        預測值
    figsize : tuple
        圖像大小
    title : str
        標題
    save_path : str, optional
        保存路徑

    Returns
    -------
    matplotlib.figure.Figure
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # 1. 預測 vs 實際
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.5, s=20)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax1.set_xlabel('Actual', fontsize=11)
    ax1.set_ylabel('Predicted', fontsize=11)
    ax1.set_title('Predicted vs Actual')
    ax1.grid(True, alpha=0.3)

    # 2. 殘差分布
    ax2 = axes[1]
    ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.axvline(x=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Residual', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Residual Distribution')
    ax2.grid(True, alpha=0.3)

    # 3. 殘差 vs 預測值
    ax3 = axes[2]
    ax3.scatter(y_pred, residuals, alpha=0.5, s=20)
    ax3.axhline(y=0, color='r', linestyle='--', lw=2)
    ax3.set_xlabel('Predicted', fontsize=11)
    ax3.set_ylabel('Residual', fontsize=11)
    ax3.set_title('Residuals vs Predicted')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
