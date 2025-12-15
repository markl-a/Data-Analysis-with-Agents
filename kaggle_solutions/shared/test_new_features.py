"""
測試共享工具庫的新功能
"""
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 導入共享工具
from ml_utils import (
    EarlyStopping,
    model_comparison,
    generate_hyperparameter_grid,
)
from data_utils import create_sample_data

print("="*70)
print("測試共享工具庫新功能")
print("="*70)

# 創建測試數據
print("\n1. 創建測試數據...")
X, y = create_sample_data('classification', n_samples=500, n_classes=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✓ 數據集: {X.shape[0]} 樣本, {X.shape[1]} 特徵, {len(np.unique(y))} 類別")

# 測試 1: EarlyStopping
print("\n2. 測試 EarlyStopping 類...")
try:
    early_stop = EarlyStopping(patience=3, mode='max', verbose=False)
    scores = [0.7, 0.75, 0.76, 0.76, 0.75, 0.74, 0.74]

    for i, score in enumerate(scores):
        if early_stop.step(score):
            print(f"✓ Early stopping 在第 {i+1} 輪觸發")
            print(f"  最佳分數: {early_stop.best_score:.4f}")
            break

    print("✓ EarlyStopping 類測試通過")
except Exception as e:
    print(f"✗ EarlyStopping 測試失敗: {e}")
    sys.exit(1)

# 測試 2: model_comparison
print("\n3. 測試 model_comparison 函數...")
try:
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }

    results = model_comparison(
        models, X_train, y_train, X_test, y_test,
        cv=3, return_predictions=False
    )

    print(f"\n✓ model_comparison 測試通過")
    print(f"  比較了 {len(results)} 個模型")
    print(f"  最佳模型: {results.iloc[0]['Model']}")
    print(f"  最佳測試分數: {results.iloc[0]['Test Score']:.4f}")

except Exception as e:
    print(f"✗ model_comparison 測試失敗: {e}")
    sys.exit(1)

# 測試 3: generate_hyperparameter_grid
print("\n4. 測試 generate_hyperparameter_grid 函數...")
try:
    # 測試不同模型和大小
    test_cases = [
        ('random_forest', 'small'),
        ('gradient_boosting', 'medium'),
        ('logistic_regression', 'small'),
    ]

    for model_type, size in test_cases:
        param_grid = generate_hyperparameter_grid(model_type, size=size)
        print(f"\n✓ {model_type} ({size}): {len(param_grid)} 個參數")

        # 計算組合數
        total_combinations = 1
        for param_values in param_grid.values():
            total_combinations *= len(param_values)
        print(f"  總組合數: {total_combinations}")

    print("\n✓ generate_hyperparameter_grid 測試通過")

except Exception as e:
    print(f"✗ generate_hyperparameter_grid 測試失敗: {e}")
    sys.exit(1)

# 測試完成
print("\n" + "="*70)
print("所有新功能測試通過！✓")
print("="*70)
