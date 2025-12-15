"""
共享工具模組 - Kaggle 解決方案的通用功能

提供標準化的機器學習實用工具，包含最佳實踐：
- 防止數據洩漏的數據分割
- 統一的評估指標
- 可視化工具
"""

from .ml_utils import (
    safe_train_test_split,
    SafeScaler,
    evaluate_classifier,
    evaluate_regressor,
    cross_validate_model,
)
from .visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    plot_learning_curve,
    plot_residuals,
)
from .data_utils import (
    detect_task_type,
    check_class_balance,
    handle_missing_values,
    create_sample_data,
)

__all__ = [
    # ML utilities
    'safe_train_test_split',
    'SafeScaler',
    'evaluate_classifier',
    'evaluate_regressor',
    'cross_validate_model',
    # Visualization
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_feature_importance',
    'plot_learning_curve',
    'plot_residuals',
    # Data utilities
    'detect_task_type',
    'check_class_balance',
    'handle_missing_values',
    'create_sample_data',
]
