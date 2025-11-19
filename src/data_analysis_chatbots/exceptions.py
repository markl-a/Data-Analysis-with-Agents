"""自定義異常類

此模塊定義了專案中使用的所有自定義異常類,提供更精確的錯誤處理和更好的錯誤消息。
"""


class DataAnalysisError(Exception):
    """數據分析基礎異常

    所有自定義異常的基類
    """
    pass


class DataLoadError(DataAnalysisError):
    """數據加載異常

    當數據文件無法找到或加載失敗時拋出

    Examples:
        >>> raise DataLoadError("無法加載數據集 'mall_customers'")
    """
    def __init__(self, message: str, dataset_name: str = None, file_path: str = None):
        self.dataset_name = dataset_name
        self.file_path = file_path
        super().__init__(message)


class DataDownloadError(DataAnalysisError):
    """數據下載異常

    當從遠程源下載數據失敗時拋出
    """
    def __init__(self, message: str, url: str = None, dataset_name: str = None):
        self.url = url
        self.dataset_name = dataset_name
        super().__init__(message)


class ValidationError(DataAnalysisError):
    """數據驗證異常

    當數據驗證失敗時拋出(例如:缺失值、數據類型錯誤等)

    Examples:
        >>> raise ValidationError("數據包含過多缺失值", missing_percentage=45.2)
    """
    def __init__(self, message: str, column: str = None, missing_percentage: float = None):
        self.column = column
        self.missing_percentage = missing_percentage
        super().__init__(message)


class ClusteringError(DataAnalysisError):
    """聚類分析異常

    當聚類算法執行失敗時拋出

    Examples:
        >>> raise ClusteringError("聚類失敗: 數據維度不匹配")
    """
    def __init__(self, message: str, algorithm: str = None, n_clusters: int = None):
        self.algorithm = algorithm
        self.n_clusters = n_clusters
        super().__init__(message)


class RFMAnalysisError(DataAnalysisError):
    """RFM分析異常

    當RFM分析失敗時拋出

    Examples:
        >>> raise RFMAnalysisError("缺少必要的列: InvoiceDate")
    """
    def __init__(self, message: str, missing_columns: list = None):
        self.missing_columns = missing_columns or []
        super().__init__(message)


class CLVPredictionError(DataAnalysisError):
    """CLV預測異常

    當客戶終身價值預測失敗時拋出

    Examples:
        >>> raise CLVPredictionError("預測失敗: 負值折扣率")
    """
    def __init__(self, message: str, discount_rate: float = None):
        self.discount_rate = discount_rate
        super().__init__(message)


class ConfigurationError(DataAnalysisError):
    """配置錯誤異常

    當配置文件無效或缺失時拋出

    Examples:
        >>> raise ConfigurationError("配置文件不存在: config/config.yaml")
    """
    def __init__(self, message: str, config_file: str = None, missing_keys: list = None):
        self.config_file = config_file
        self.missing_keys = missing_keys or []
        super().__init__(message)


class VisualizationError(DataAnalysisError):
    """可視化錯誤異常

    當圖表生成失敗時拋出

    Examples:
        >>> raise VisualizationError("無法生成散點圖: 缺少Y軸數據")
    """
    def __init__(self, message: str, plot_type: str = None):
        self.plot_type = plot_type
        super().__init__(message)


class PreprocessingError(DataAnalysisError):
    """數據預處理異常

    當數據清洗或預處理失敗時拋出

    Examples:
        >>> raise PreprocessingError("文本清洗失敗")
    """
    pass


class ModelSaveError(DataAnalysisError):
    """模型保存異常

    當保存訓練好的模型失敗時拋出

    Examples:
        >>> raise ModelSaveError("無法保存模型到指定路徑")
    """
    def __init__(self, message: str, file_path: str = None):
        self.file_path = file_path
        super().__init__(message)


class ModelLoadError(DataAnalysisError):
    """模型加載異常

    當加載已保存的模型失敗時拋出

    Examples:
        >>> raise ModelLoadError("模型文件損壞或版本不兼容")
    """
    def __init__(self, message: str, file_path: str = None):
        self.file_path = file_path
        super().__init__(message)


class FeatureEngineeringError(DataAnalysisError):
    """特徵工程異常

    當特徵提取或轉換失敗時拋出
    """
    def __init__(self, message: str, feature_name: str = None):
        self.feature_name = feature_name
        super().__init__(message)


class CampaignError(DataAnalysisError):
    """營銷活動異常

    當營銷活動創建或管理失敗時拋出
    """
    def __init__(self, message: str, campaign_name: str = None):
        self.campaign_name = campaign_name
        super().__init__(message)


# 便捷函數用於常見錯誤場景

def raise_if_file_not_found(file_path: str, dataset_name: str = None) -> None:
    """如果文件不存在則拋出DataLoadError

    Args:
        file_path: 文件路徑
        dataset_name: 數據集名稱

    Raises:
        DataLoadError: 當文件不存在時
    """
    from pathlib import Path

    if not Path(file_path).exists():
        message = f"數據文件未找到: {file_path}"
        if dataset_name:
            message += f"\n提示: 運行 'python -m data_analysis_chatbots.data_downloader --dataset {dataset_name}' 下載數據集"
        raise DataLoadError(message, dataset_name=dataset_name, file_path=file_path)


def raise_if_columns_missing(df, required_columns: list, analysis_type: str = "分析") -> None:
    """如果DataFrame缺少必要列則拋出ValidationError

    Args:
        df: Pandas DataFrame
        required_columns: 必需的列名列表
        analysis_type: 分析類型描述

    Raises:
        ValidationError: 當缺少必要列時
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValidationError(
            f"{analysis_type}需要的列缺失: {', '.join(missing)}\n"
            f"可用列: {', '.join(df.columns.tolist())}"
        )


def raise_if_empty_dataframe(df, context: str = "操作") -> None:
    """如果DataFrame為空則拋出ValidationError

    Args:
        df: Pandas DataFrame
        context: 操作上下文描述

    Raises:
        ValidationError: 當DataFrame為空時
    """
    if df is None or len(df) == 0:
        raise ValidationError(f"{context}失敗: DataFrame為空")


def raise_if_invalid_parameter(condition: bool, message: str, parameter_name: str = None) -> None:
    """如果參數無效則拋出ConfigurationError

    Args:
        condition: 無效條件(True時拋出異常)
        message: 錯誤消息
        parameter_name: 參數名稱

    Raises:
        ConfigurationError: 當條件為True時
    """
    if condition:
        if parameter_name:
            message = f"參數 '{parameter_name}' 無效: {message}"
        raise ConfigurationError(message)
