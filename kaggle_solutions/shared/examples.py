"""
共享工具庫使用示例

演示如何正確使用 kaggle_solutions.shared 中的工具，
包括防止數據洩漏的最佳實踐。
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

# 導入共享工具
from data_utils import create_sample_data, check_class_balance, detect_task_type
from ml_utils import (
    safe_train_test_split,
    SafeScaler,
    evaluate_classifier,
    evaluate_regressor,
    cross_validate_model,
    EarlyStopping,
    model_comparison,
    generate_hyperparameter_grid
)
from visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    plot_learning_curve,
    plot_residuals
)


def example_classification_pipeline():
    """
    示例 1: 完整的分類問題流程
    展示如何正確使用工具避免數據洩漏
    """
    print("=" * 80)
    print("示例 1: 分類問題完整流程")
    print("=" * 80)

    # 1. 創建範例數據
    print("\n步驟 1: 創建範例數據")
    X, y = create_sample_data(
        task='classification',
        n_samples=1000,
        n_features=20,
        n_classes=3,
        noise=0.1,
        random_state=42
    )
    print(f"數據形狀: X={X.shape}, y={y.shape}")
    print(f"類別分布: {np.bincount(y)}")

    # 2. 檢查任務類型
    print("\n步驟 2: 自動檢測任務類型")
    task_type = detect_task_type(y)
    print(f"檢測到的任務類型: {task_type}")

    # 3. 檢查類別平衡
    print("\n步驟 3: 檢查類別平衡")
    balance_info = check_class_balance(y, threshold=0.1)
    print(f"類別比例: {balance_info['proportions']}")
    print(f"是否不平衡: {balance_info['is_imbalanced']}")
    print(f"建議: {balance_info['recommendation']}")

    # 4. 使用 safe_train_test_split 分割數據
    # 這個函數會自動為分類問題使用分層抽樣
    print("\n步驟 4: 分割訓練集和測試集（自動分層抽樣）")
    X_train, X_test, y_train, y_test = safe_train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )
    print(f"訓練集: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"測試集: X_test={X_test.shape}, y_test={y_test.shape}")
    print(f"訓練集類別分布: {np.bincount(y_train)}")
    print(f"測試集類別分布: {np.bincount(y_test)}")

    # 5. 使用 SafeScaler 正確地縮放數據（防止數據洩漏）
    print("\n步驟 5: 特徵縮放（防止數據洩漏）")
    print("關鍵點: 縮放參數只從訓練集學習，然後應用到測試集")
    scaler = SafeScaler(scaler_type='standard')
    X_train_scaled, X_test_scaled = scaler.fit_transform_split(X_train, X_test)
    print(f"縮放後訓練集: mean={X_train_scaled.mean():.4f}, std={X_train_scaled.std():.4f}")
    print(f"縮放後測試集: mean={X_test_scaled.mean():.4f}, std={X_test_scaled.std():.4f}")
    print("注意: 測試集的均值和標準差略有不同，這是正確的！")

    # 6. 訓練模型
    print("\n步驟 6: 訓練分類模型")
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train_scaled, y_train)
    print("模型訓練完成")

    # 7. 使用 evaluate_classifier 進行完整評估
    print("\n步驟 7: 評估模型性能")
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    metrics = evaluate_classifier(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        average='weighted'
    )

    print("分類指標:")
    for metric_name, value in metrics.items():
        if value is not None:
            print(f"  {metric_name}: {value:.4f}")

    # 8. 交叉驗證
    print("\n步驟 8: 交叉驗證（使用原始數據）")
    cv_results = cross_validate_model(
        model=RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10),
        X=X,
        y=y,
        cv=5,
        task_type='classification'
    )
    print(f"交叉驗證策略: {cv_results['cv_strategy']}")
    print(f"評分指標: {cv_results['scoring']}")
    print(f"平均分數: {cv_results['mean']:.4f} ± {cv_results['std']:.4f}")
    print(f"各折分數: {cv_results['scores']}")

    # 9. 可視化
    print("\n步驟 9: 生成可視化圖表")

    # 混淆矩陣
    fig1 = plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        labels=['Class 0', 'Class 1', 'Class 2'],
        normalize=True,
        title='分類混淆矩陣（歸一化）'
    )

    # ROC 曲線（僅對二元分類展示單個曲線）
    # 對於多類別，我們展示 one-vs-rest 的第一個類別
    fig2 = plot_roc_curve(
        y_true=(y_test == 0).astype(int),
        y_proba=y_proba[:, 0],
        title='ROC 曲線 - Class 0 vs Rest'
    )

    # 特徵重要性
    feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    importances = model.feature_importances_
    fig3 = plot_feature_importance(
        feature_names=feature_names,
        importances=importances,
        top_n=15,
        title='前 15 個最重要的特徵'
    )

    # 學習曲線
    fig4 = plot_learning_curve(
        model=RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10),
        X=X,
        y=y,
        cv=5,
        title='學習曲線 - 檢測過擬合'
    )

    print("可視化圖表已生成")
    print("\n示例 1 完成！")

    return fig1, fig2, fig3, fig4


def example_regression_pipeline():
    """
    示例 2: 完整的回歸問題流程
    """
    print("\n" + "=" * 80)
    print("示例 2: 回歸問題完整流程")
    print("=" * 80)

    # 1. 創建回歸數據
    print("\n步驟 1: 創建回歸數據")
    X, y = create_sample_data(
        task='regression',
        n_samples=800,
        n_features=15,
        noise=0.2,
        random_state=42
    )
    print(f"數據形狀: X={X.shape}, y={y.shape}")
    print(f"目標變量範圍: [{y.min():.2f}, {y.max():.2f}]")

    # 2. 自動檢測任務類型
    print("\n步驟 2: 自動檢測任務類型")
    task_type = detect_task_type(y)
    print(f"檢測到的任務類型: {task_type}")

    # 3. 數據分割（回歸問題不需要分層）
    print("\n步驟 3: 分割數據")
    X_train, X_test, y_train, y_test = safe_train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )
    print(f"訓練集大小: {X_train.shape[0]}")
    print(f"測試集大小: {X_test.shape[0]}")

    # 4. 特徵縮放
    print("\n步驟 4: 特徵縮放（使用 MinMax Scaler）")
    scaler = SafeScaler(scaler_type='minmax')
    X_train_scaled, X_test_scaled = scaler.fit_transform_split(X_train, X_test)
    print(f"訓練集範圍: [{X_train_scaled.min():.4f}, {X_train_scaled.max():.4f}]")
    print(f"測試集範圍: [{X_test_scaled.min():.4f}, {X_test_scaled.max():.4f}]")
    print("注意: 測試集可能略微超出 [0,1] 範圍，這是正確的！")

    # 5. 訓練回歸模型
    print("\n步驟 5: 訓練回歸模型")
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train_scaled, y_train)
    print("模型訓練完成")

    # 6. 評估回歸模型
    print("\n步驟 6: 評估回歸性能")
    y_pred = model.predict(X_test_scaled)

    metrics = evaluate_regressor(y_true=y_test, y_pred=y_pred)

    print("回歸指標:")
    for metric_name, value in metrics.items():
        if value is not None:
            if metric_name == 'mape':
                print(f"  {metric_name}: {value:.2f}%")
            else:
                print(f"  {metric_name}: {value:.4f}")

    # 7. 交叉驗證
    print("\n步驟 7: 交叉驗證")
    cv_results = cross_validate_model(
        model=RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10),
        X=X,
        y=y,
        cv=5,
        task_type='regression'
    )
    print(f"交叉驗證策略: {cv_results['cv_strategy']}")
    print(f"評分指標: {cv_results['scoring']}")
    print(f"平均分數: {cv_results['mean']:.4f} ± {cv_results['std']:.4f}")

    # 8. 可視化
    print("\n步驟 8: 生成可視化圖表")

    # 殘差分析
    fig1 = plot_residuals(
        y_true=y_test,
        y_pred=y_pred,
        title='殘差分析 - 檢查模型假設'
    )

    # 特徵重要性
    feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    importances = model.feature_importances_
    fig2 = plot_feature_importance(
        feature_names=feature_names,
        importances=importances,
        top_n=10,
        title='回歸模型的特徵重要性'
    )

    # 學習曲線
    fig3 = plot_learning_curve(
        model=RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10),
        X=X,
        y=y,
        cv=5,
        title='回歸模型學習曲線'
    )

    print("可視化圖表已生成")
    print("\n示例 2 完成！")

    return fig1, fig2, fig3


def example_scaler_comparison():
    """
    示例 3: 比較正確和錯誤的縮放方法
    演示數據洩漏的問題
    """
    print("\n" + "=" * 80)
    print("示例 3: 正確 vs 錯誤的數據縮放（數據洩漏演示）")
    print("=" * 80)

    # 創建數據
    X, y = create_sample_data(
        task='classification',
        n_samples=500,
        n_features=10,
        n_classes=2,
        random_state=42
    )

    # 分割數據
    X_train, X_test, y_train, y_test = safe_train_test_split(X, y, test_size=0.2)

    print("\n方法 1: 錯誤的方式（數據洩漏）")
    print("-" * 40)
    print("問題: 在分割前就縮放所有數據")
    from sklearn.preprocessing import StandardScaler

    # 錯誤示範（僅用於教學）
    wrong_scaler = StandardScaler()
    X_wrong = wrong_scaler.fit_transform(X)  # 在分割前縮放！
    X_train_wrong = X_wrong[:len(X_train)]
    X_test_wrong = X_wrong[len(X_train):]

    print("結果: 測試集的信息洩漏到了縮放參數中")
    print(f"  訓練集 mean: {X_train_wrong.mean():.6f}")
    print(f"  測試集 mean: {X_test_wrong.mean():.6f}")
    print("  警告: 這會導致模型性能被高估！")

    print("\n方法 2: 正確的方式（使用 SafeScaler）")
    print("-" * 40)
    print("正確: 只從訓練集學習縮放參數")

    scaler = SafeScaler('standard')
    X_train_right, X_test_right = scaler.fit_transform_split(X_train, X_test)

    print("結果: 測試集完全獨立")
    print(f"  訓練集 mean: {X_train_right.mean():.6f}")
    print(f"  測試集 mean: {X_test_right.mean():.6f}")
    print("  正確: 測試集的統計量與訓練集不同")

    # 比較模型性能
    print("\n性能比較:")
    print("-" * 40)

    model = LogisticRegression(random_state=42, max_iter=1000)

    # 錯誤方法的性能
    model.fit(X_train_wrong, y_train)
    y_pred_wrong = model.predict(X_test_wrong)
    metrics_wrong = evaluate_classifier(y_test, y_pred_wrong)

    # 正確方法的性能
    model.fit(X_train_right, y_train)
    y_pred_right = model.predict(X_test_right)
    metrics_right = evaluate_classifier(y_test, y_pred_right)

    print(f"錯誤方法準確率: {metrics_wrong['accuracy']:.4f}")
    print(f"正確方法準確率: {metrics_right['accuracy']:.4f}")
    print(f"差異: {abs(metrics_wrong['accuracy'] - metrics_right['accuracy']):.4f}")

    print("\n關鍵要點:")
    print("  1. 永遠不要在分割數據前進行任何依賴數據統計的操作")
    print("  2. 使用 SafeScaler 確保正確的縮放流程")
    print("  3. 縮放參數必須只從訓練集學習")

    print("\n示例 3 完成！")


def example_quick_start():
    """
    示例 4: 快速開始模板
    最簡單的使用方式
    """
    print("\n" + "=" * 80)
    print("示例 4: 快速開始模板（最少代碼）")
    print("=" * 80)

    print("\n快速分類流程（5 步完成）:")
    print("-" * 40)

    # 1. 創建數據
    X, y = create_sample_data('classification', n_samples=500)

    # 2. 分割數據（自動分層）
    X_train, X_test, y_train, y_test = safe_train_test_split(X, y)

    # 3. 縮放數據（防止洩漏）
    scaler = SafeScaler('standard')
    X_train_scaled, X_test_scaled = scaler.fit_transform_split(X_train, X_test)

    # 4. 訓練和預測
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    # 5. 評估
    metrics = evaluate_classifier(y_test, y_pred, y_proba)

    print(f"準確率: {metrics['accuracy']:.4f}")
    print(f"F1 分數: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    # 6. 可視化（可選）
    fig = plot_confusion_matrix(y_test, y_pred)

    print("\n只需 6 行核心代碼即可完成完整的 ML 流程！")
    print("\n示例 4 完成！")

    return fig


def main():
    """
    運行所有示例
    """
    print("\n" + "=" * 80)
    print("Kaggle 共享工具庫使用示例")
    print("=" * 80)
    print("\n本文件展示如何正確使用共享工具庫，包括：")
    print("  - safe_train_test_split: 自動分層抽樣")
    print("  - SafeScaler: 防止數據洩漏的縮放器")
    print("  - evaluate_classifier/regressor: 統一的評估接口")
    print("  - 可視化工具: 混淆矩陣、ROC、特徵重要性等")
    print("\n")

    # 運行所有示例
    try:
        # 示例 1: 分類問題完整流程
        figs1 = example_classification_pipeline()

        # 示例 2: 回歸問題完整流程
        figs2 = example_regression_pipeline()

        # 示例 3: 數據洩漏演示
        example_scaler_comparison()

        # 示例 4: 快速開始
        fig4 = example_quick_start()

        print("\n" + "=" * 80)
        print("所有示例運行完成！")
        print("=" * 80)
        print("\n提示: 如果要保存圖表，請使用 save_path 參數")
        print("示例: plot_confusion_matrix(..., save_path='confusion_matrix.png')")

        # 顯示所有圖表
        plt.show()

    except Exception as e:
        print(f"\n錯誤: {e}")
        import traceback
        traceback.print_exc()


def example_early_stopping():
    """
    示例 5: 使用 EarlyStopping 防止過擬合
    """
    print("\n" + "=" * 80)
    print("示例 5: EarlyStopping 回調使用")
    print("=" * 80)

    print("\n演示: 使用 early stopping 監控訓練過程")

    # 模擬訓練過程中的驗證分數
    validation_scores = [0.65, 0.72, 0.78, 0.82, 0.84, 0.85, 0.85, 0.84, 0.84, 0.83]

    early_stop = EarlyStopping(
        patience=3,
        min_delta=0.001,
        mode='max',
        verbose=True
    )

    print("\n模擬訓練循環:")
    for epoch, score in enumerate(validation_scores, 1):
        print(f"\nEpoch {epoch}: Validation Score = {score:.4f}")
        if early_stop.step(score):
            print(f"\n✓ 訓練在 epoch {epoch} 停止")
            print(f"✓ 最佳分數: {early_stop.best_score:.4f}")
            break

    print("\n關鍵要點:")
    print("  1. patience: 容忍連續無改善的輪數")
    print("  2. min_delta: 被認為是改善的最小變化")
    print("  3. mode: 'max' 用於準確率, 'min' 用於損失")
    print("  4. restore_best: 可以保存和恢復最佳模型參數")

    print("\n示例 5 完成！")


def example_model_comparison():
    """
    示例 6: 使用 model_comparison 比較多個模型
    """
    print("\n" + "=" * 80)
    print("示例 6: 模型比較工具")
    print("=" * 80)

    # 創建數據
    print("\n步驟 1: 準備數據")
    X, y = create_sample_data('classification', n_samples=800, n_classes=2)
    X_train, X_test, y_train, y_test = safe_train_test_split(X, y)
    print(f"數據集: {X.shape[0]} 樣本, {X.shape[1]} 特徵")

    # 定義多個模型
    print("\n步驟 2: 定義待比較的模型")
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.svm import SVC

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }

    # 比較模型
    print("\n步驟 3: 訓練和比較所有模型")
    results_df = model_comparison(
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        cv=5
    )

    print("\n最終排名:")
    print(results_df[['Model', 'CV Mean', 'Test Score', 'Train Time (s)']].to_string(index=False))

    print("\n關鍵要點:")
    print("  1. 自動訓練和評估所有模型")
    print("  2. 包含交叉驗證和測試集性能")
    print("  3. 記錄訓練時間以便選擇")
    print("  4. 結果自動按測試分數排序")

    print("\n示例 6 完成！")

    return results_df


def example_hyperparameter_grid():
    """
    示例 7: 使用 generate_hyperparameter_grid 進行超參數調優
    """
    print("\n" + "=" * 80)
    print("示例 7: 超參數網格生成器")
    print("=" * 80)

    print("\n支持的模型類型:")
    model_types = [
        'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm',
        'logistic_regression', 'svm', 'knn', 'mlp'
    ]
    for mt in model_types:
        print(f"  - {mt}")

    # 示例 1: Random Forest 小型網格
    print("\n示例 1: Random Forest - 快速搜索")
    param_grid_small = generate_hyperparameter_grid('random_forest', size='small')

    # 示例 2: Gradient Boosting 中型網格
    print("\n示例 2: Gradient Boosting - 標準搜索")
    param_grid_medium = generate_hyperparameter_grid('gradient_boosting', size='medium')

    # 示例 3: 使用 GridSearchCV
    print("\n示例 3: 結合 GridSearchCV 使用")
    from sklearn.model_selection import GridSearchCV

    X, y = create_sample_data('classification', n_samples=500, n_classes=2)
    X_train, X_test, y_train, y_test = safe_train_test_split(X, y)

    param_grid = generate_hyperparameter_grid('random_forest', size='small')

    print("\n執行網格搜索...")
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    print(f"\n最佳參數: {grid_search.best_params_}")
    print(f"最佳交叉驗證分數: {grid_search.best_score_:.4f}")

    # 測試集評估
    y_pred = grid_search.predict(X_test)
    metrics = evaluate_classifier(y_test, y_pred)
    print(f"測試集準確率: {metrics['accuracy']:.4f}")

    print("\n關鍵要點:")
    print("  1. 'small' 適合快速實驗")
    print("  2. 'medium' 適合標準調優")
    print("  3. 'large' 適合最終優化（需要更多時間）")
    print("  4. 支持 8 種常用模型類型")

    print("\n示例 7 完成！")

    return grid_search


def example_new_features_showcase():
    """
    示例 8: 新功能綜合展示
    """
    print("\n" + "=" * 80)
    print("示例 8: 新功能綜合展示")
    print("=" * 80)

    print("\n場景: 完整的模型開發工作流")

    # 1. 準備數據
    print("\n步驟 1: 準備數據")
    X, y = create_sample_data('classification', n_samples=1000, n_classes=2)
    X_train, X_test, y_train, y_test = safe_train_test_split(X, y)
    scaler = SafeScaler('standard')
    X_train_scaled, X_test_scaled = scaler.fit_transform_split(X_train, X_test)

    # 2. 快速比較多個模型
    print("\n步驟 2: 快速模型比較")
    models = {
        'RF': RandomForestClassifier(n_estimators=50, random_state=42),
        'GB': LogisticRegression(max_iter=1000, random_state=42)
    }
    results = model_comparison(models, X_train_scaled, y_train, X_test_scaled, y_test, cv=3)
    best_model_name = results.iloc[0]['Model']
    print(f"\n最佳基礎模型: {best_model_name}")

    # 3. 超參數調優
    print("\n步驟 3: 對最佳模型進行超參數調優")
    from sklearn.model_selection import GridSearchCV

    param_grid = generate_hyperparameter_grid('random_forest', size='small')
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train_scaled, y_train)

    print(f"最佳參數: {grid_search.best_params_}")
    print(f"優化後 CV 分數: {grid_search.best_score_:.4f}")

    # 4. 使用 Early Stopping 進行進一步訓練（模擬）
    print("\n步驟 4: 使用 Early Stopping 監控訓練")
    print("（在實際的迭代訓練場景中使用）")

    early_stop = EarlyStopping(patience=5, mode='max', verbose=False)
    print("✓ Early Stopping 已配置")

    # 5. 最終評估
    print("\n步驟 5: 最終模型評估")
    y_pred = grid_search.predict(X_test_scaled)
    y_proba = grid_search.predict_proba(X_test_scaled)
    final_metrics = evaluate_classifier(y_test, y_pred, y_proba)

    print("\n最終性能:")
    for metric_name, value in final_metrics.items():
        if value is not None:
            print(f"  {metric_name}: {value:.4f}")

    # 6. 可視化
    print("\n步驟 6: 生成可視化報告")
    fig1 = plot_confusion_matrix(y_test, y_pred, normalize=True)
    fig2 = plot_roc_curve(y_test, y_proba)

    print("✓ 可視化圖表已生成")

    print("\n" + "=" * 80)
    print("新功能工作流完成！")
    print("=" * 80)
    print("\n工作流程總結:")
    print("  1. model_comparison: 快速比較多個模型")
    print("  2. generate_hyperparameter_grid: 生成調優參數")
    print("  3. GridSearchCV: 執行超參數搜索")
    print("  4. EarlyStopping: 防止過擬合")
    print("  5. 完整評估和可視化")

    print("\n示例 8 完成！")

    return grid_search, results, final_metrics


if __name__ == '__main__':
    main()
