"""
Data Analysis with Chatbots
===========================

A comprehensive framework for customer analytics and segmentation using AI-powered chatbots.

Modules:
    - preprocessing: Data cleaning and preprocessing utilities
    - clustering: Customer segmentation and clustering algorithms
    - visualization: Data visualization tools
    - marketing: Marketing strategy and campaign management
    - exceptions: Custom exception classes
    - init: Project initialization utilities
"""

__version__ = "1.0.0"
__author__ = "賴祺清"

from .config_loader import ConfigLoader
from .data_loader import DataLoader
from .utils import setup_logging, ensure_dir

# Import exceptions for convenient access
from .exceptions import (
    DataAnalysisError,
    DataLoadError,
    DataDownloadError,
    ValidationError,
    ClusteringError,
    RFMAnalysisError,
    CLVPredictionError,
    ConfigurationError,
    VisualizationError,
    PreprocessingError,
    ModelSaveError,
    ModelLoadError,
    FeatureEngineeringError,
    CampaignError,
)

# Import initialization utilities
from .init import initialize_project, validate_project_structure, get_project_root

__all__ = [
    # Core utilities
    "ConfigLoader",
    "DataLoader",
    "setup_logging",
    "ensure_dir",
    # Exceptions
    "DataAnalysisError",
    "DataLoadError",
    "DataDownloadError",
    "ValidationError",
    "ClusteringError",
    "RFMAnalysisError",
    "CLVPredictionError",
    "ConfigurationError",
    "VisualizationError",
    "PreprocessingError",
    "ModelSaveError",
    "ModelLoadError",
    "FeatureEngineeringError",
    "CampaignError",
    # Initialization
    "initialize_project",
    "validate_project_structure",
    "get_project_root",
]
