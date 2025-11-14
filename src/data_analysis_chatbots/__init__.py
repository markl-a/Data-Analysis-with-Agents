"""
Data Analysis with Chatbots
===========================

A comprehensive framework for customer analytics and segmentation using AI-powered chatbots.

Modules:
    - preprocessing: Data cleaning and preprocessing utilities
    - clustering: Customer segmentation and clustering algorithms
    - visualization: Data visualization tools
    - marketing: Marketing strategy and campaign management
"""

__version__ = "1.0.0"
__author__ = "賴祺清"

from .config_loader import ConfigLoader
from .data_loader import DataLoader
from .utils import setup_logging, ensure_dir

__all__ = [
    "ConfigLoader",
    "DataLoader",
    "setup_logging",
    "ensure_dir",
]
