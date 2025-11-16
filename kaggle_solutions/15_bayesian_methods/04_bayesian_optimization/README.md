# Bayesian Optimization - 貝葉斯優化

## 項目概述

使用貝葉斯優化進行超參數調優和黑盒函數優化。特別適合評估成本高昂的優化問題。

**難度**: ⭐⭐⭐ 高級
**數據集**: 模擬優化問題

## 核心概念

### 問題設定
尋找函數最大值（或最小值）:
```
x* = argmax_x f(x)
```

其中 f(x) 是昂貴的黑盒函數（如模型訓練）。

### 關鍵優勢
1. **樣本效率**: 評估次數少
2. **全局優化**: 平衡探索與利用
3. **不確定性**: 利用預測不確定性
4. **無梯度**: 適用於黑盒函數

## 方法論

### 1. 高斯過程（Surrogate Model）
建模目標函數: f(x) ~ GP(μ, k)
- 提供預測均值和方差
- 量化不確定性

### 2. 採集函數（Acquisition Function）

#### 期望改進（EI）
```
EI(x) = E[max(f(x) - f(x_best), 0)]
```

#### 上置信界（UCB）
```
UCB(x) = μ(x) + κ × σ(x)
```

#### 改進概率（PI）
```
PI(x) = P(f(x) > f(x_best))
```

### 3. 優化流程
1. 初始隨機採樣
2. 擬合高斯過程
3. 最大化採集函數
4. 評估新點
5. 更新 GP，重複

## 文件結構

```
04_bayesian_optimization/
├── solution.py          # 貝葉斯優化實現
└── README.md           # 本文件
```

## 核心功能

### BayesianOptimizer 類
```python
optimizer = BayesianOptimizer(
    objective_function=f,
    bounds=[(low1, high1), (low2, high2)],
    n_initial=5
)
```

**主要方法**:
- `optimize()`: 執行優化
- `acquisition_EI()`: 期望改進
- `acquisition_UCB()`: 上置信界
- `acquisition_PI()`: 改進概率
- `propose_location()`: 提議下一個點
- `plot_optimization_1d()`: 1D 可視化
- `plot_convergence()`: 收斂曲線

## 使用方法

### 基本使用
```bash
python solution.py
```

### 自定義優化
```python
from solution import BayesianOptimizer

# 定義目標函數
def my_function(x):
    return -(x[0] - 2)**2 + 5

# 創建優化器
optimizer = BayesianOptimizer(
    objective_function=my_function,
    bounds=[(-5, 5)],
    n_initial=3
)

# 執行優化
best_x, best_y = optimizer.optimize(
    n_iterations=20,
    acquisition='EI'
)
```

### 超參數調優
```python
def model_score(params):
    model = Model(param1=params[0], param2=params[1])
    return cross_val_score(model, X, y).mean()

optimizer = BayesianOptimizer(
    objective_function=model_score,
    bounds=[(0, 100), (1, 10)]
)

best_params, best_score = optimizer.optimize(n_iterations=30)
```

## 示例

### 1. 一維函數優化
優化複雜的 1D 函數，展示 GP 如何逼近真實函數。

### 2. 二維 Branin 函數
經典優化基準測試函數。

### 3. 隨機森林超參數調優
自動調優:
- n_estimators
- max_depth
- min_samples_split

## 可視化輸出

### 1D 優化圖
- 真實函數 vs GP 預測
- 95% 置信區間
- 觀測點和最優點
- 採集函數值
- 下一個建議採樣點

### 收斂曲線
每次迭代的最優值變化

## 採集函數比較

### Expected Improvement (EI)
- **平衡**: 探索與利用的良好平衡
- **推薦**: 通用選擇

### Upper Confidence Bound (UCB)
- **探索**: κ 越大越探索
- **理論**: 有理論保證（GP-UCB）

### Probability of Improvement (PI)
- **保守**: 更傾向於利用
- **收斂**: 可能過早收斂

## 進階主題

### 1. 核函數選擇
```python
from sklearn.gaussian_process.kernels import Matern, RBF

# Matérn 核（更通用）
kernel = Matern(nu=2.5)

# RBF 核（無限可微）
kernel = RBF(length_scale=1.0)
```

### 2. 並行優化
同時評估多個點，加速優化。

### 3. 約束優化
處理約束條件的優化問題。

### 4. 多目標優化
同時優化多個目標（帕累托前沿）。

## 應用場景

### 機器學習
- 超參數調優
- 神經架構搜索
- AutoML

### 實驗設計
- A/B 測試參數
- 實驗條件優化
- 配方優化

### 工程優化
- 控制系統參數
- 製程參數優化
- 材料設計

## 與其他方法比較

### vs. 網格搜索
- **網格**: 窮舉，樣本多
- **BO**: 智能採樣，樣本少

### vs. 隨機搜索
- **隨機**: 無策略
- **BO**: 有策略，利用歷史

### vs. 梯度優化
- **梯度**: 需要梯度，局部
- **BO**: 無梯度，全局

## 計算複雜度

### 時間複雜度
- GP 訓練: O(n³)
- 預測: O(n)

### 可擴展性
大規模問題考慮:
- 稀疏 GP
- 局部 GP
- 變分稀疏 GP

## 依賴項

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
scikit-learn>=1.0.0
```

## 參考資料

### 教材
1. "Gaussian Processes for Machine Learning" - Rasmussen & Williams
2. "Bayesian Optimization" - Brochu et al.

### 論文
1. Snoek et al. (2012) - "Practical Bayesian Optimization"
2. Shahriari et al. (2016) - "Taking the Human Out of the Loop"

### 工具庫
- [Scikit-Optimize](https://scikit-optimize.github.io/)
- [GPyOpt](https://github.com/SheffieldML/GPyOpt)
- [BayesOpt](https://github.com/rmcantin/bayesopt)

## 作者

Kaggle Solutions - Bayesian Methods Series

## 許可證

MIT License
