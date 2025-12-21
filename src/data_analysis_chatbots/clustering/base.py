from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from loguru import logger

class BaseClusterer(ABC):
    """所有聚類算法的統一基類"""

    def __init__(self, normalize: bool = True, random_state: int = 42):
        self.normalize = normalize
        self.random_state = random_state
        self.scaler: Optional[StandardScaler] = None
        self.labels_: Optional[np.ndarray] = None
        self.feature_columns: Optional[List[str]] = None
        self._X_fitted: Optional[np.ndarray] = None

    def _prepare_features(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        fit_scaler: bool = True
    ) -> np.ndarray:
        """提取並標準化特徵(共用邏輯)"""
        from ..exceptions import ValidationError

        if df.empty:
            raise ValidationError("DataFrame is empty")

        missing_cols = set(feature_columns) - set(df.columns)
        if missing_cols:
            raise ValidationError(f"Missing columns: {missing_cols}")

        X = df[feature_columns].values

        if np.isnan(X).any():
            raise ValidationError("Feature data contains NaN values")

        if self.normalize:
            if fit_scaler:
                self.scaler = StandardScaler()
                X = self.scaler.fit_transform(X)
            else:
                if self.scaler is None:
                    raise ValueError("Scaler not fitted")
                X = self.scaler.transform(X)

        return X

    @abstractmethod
    def fit(self, df: pd.DataFrame, feature_columns: List[str]) -> 'BaseClusterer':
        """訓練聚類模型"""
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """預測新數據的聚類標籤"""
        pass

    def fit_predict(self, df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
        """訓練並預測"""
        self.fit(df, feature_columns)
        return self.labels_

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """評估聚類質量"""
        pass

    @abstractmethod
    def get_cluster_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """獲取聚類摘要統計"""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """導出為字典(用於序列化)"""
        return {
            'algorithm': self.__class__.__name__,
            'normalize': self.normalize,
            'random_state': self.random_state,
            'n_clusters': getattr(self, 'n_clusters', None),
        }
