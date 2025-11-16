# Bayesian Linear Regression - 貝葉斯線性回歸

## 項目概述

使用貝葉斯方法進行線性回歸，提供參數不確定性量化和預測區間。與頻率學派方法不同，貝葉斯回歸提供完整的參數後驗分佈。

**難度**: ⭐⭐ 中級
**數據集**: 模擬多項式回歸數據

## 貝葉斯回歸優勢

### 相比傳統線性回歸
1. **不確定性量化**: 提供參數的完整後驗分佈
2. **預測區間**: 自然包含參數和觀測不確定性
3. **正則化**: 先驗自然引入正則化
4. **小樣本**: 在數據稀少時表現更好

## 方法論

### 1. 貝葉斯線性模型
```
y = Xw + ε
ε ~ N(0, β⁻¹I)
```

### 2. 共軛先驗
- **權重先驗**: w ~ N(0, α⁻¹I)
- **似然**: p(y|X,w) ~ N(Xw, β⁻¹I)
- **後驗**: w|X,y ~ N(μ_post, Σ_post)

### 3. 後驗參數
```
Σ_post = (αI + βX^T X)⁻¹
μ_post = β Σ_post X^T y
```

### 4. 預測分佈
```
p(y*|x*, X, y) ~ N(x*^T μ_post, σ²_pred)
σ²_pred = x*^T Σ_post x* + β⁻¹
```

## 文件結構

```
02_bayesian_regression/
├── solution.py          # 完整的貝葉斯回歸實現
└── README.md           # 本文件
```

## 核心功能

### 1. BayesianLinearRegression 類
```python
blr = BayesianLinearRegression(alpha_prior=1.0, beta_prior=1.0)
```

**主要方法**:
- `create_sample_data()`: 生成多項式回歸數據
- `fit(X, y)`: 擬合貝葉斯線性回歸
- `predict(X, return_std=True)`: 預測均值和標準差
- `predict_distribution(X)`: 預測完整分佈
- `plot_results()`: 可視化結果
- `compute_metrics()`: 計算評估指標

### 2. BayesianRidgeRegression 類
```python
ridge = BayesianRidgeRegression(n_iterations=100)
```

使用證據近似（Evidence Approximation）自動估計超參數 α 和 β。

**主要方法**:
- `fit(X, y)`: 擬合並自動調整超參數
- `plot_hyperparameter_evolution()`: 繪製超參數演化

## 使用方法

### 基本使用
```bash
python solution.py
```

### 自定義分析
```python
from solution import BayesianLinearRegression

# 初始化
blr = BayesianLinearRegression(alpha_prior=1.0, beta_prior=10.0)

# 創建數據
X, y, X_poly, true_weights = blr.create_sample_data(
    n_samples=100,
    noise_level=0.3
)

# 擬合
blr.fit(X_poly, y)

# 預測
y_pred, y_std = blr.predict(X_poly, return_std=True)

# 評估
metrics = blr.compute_metrics(X_poly, y)
```

### 使用貝葉斯嶺回歸
```python
from solution import BayesianRidgeRegression

ridge = BayesianRidgeRegression(n_iterations=100)
ridge.fit(X_train, y_train)
ridge.plot_hyperparameter_evolution()
```

## 理論基礎

### 共軛性
高斯先驗 + 高斯似然 → 高斯後驗

這使得後驗計算解析可得，無需數值積分。

### 證據近似（Type-II Maximum Likelihood）
通過最大化邊際似然自動估計超參數：
```
p(y|X, α, β) = ∫ p(y|X, w, β) p(w|α) dw
```

### 預測不確定性
1. **認識不確定性**（Epistemic）: 來自參數不確定性
2. **偶然不確定性**（Aleatoric）: 來自觀測噪聲

貝葉斯方法自然分離這兩種不確定性。

## 可視化輸出

### 1. 預測結果與不確定性
- 訓練數據散點
- 預測均值曲線
- 95% 預測區間
- 真實函數（如果已知）

### 2. 後驗樣本
從後驗分佈採樣的 50 條回歸曲線，展示參數不確定性。

### 3. 權重後驗分佈
每個權重參數的後驗分佈直方圖，對比真實值。

### 4. 殘差 Q-Q 圖
檢驗殘差的正態性假設。

### 5. 超參數演化（嶺回歸）
α 和 β 隨迭代次數的變化。

## 評估指標

```
MSE: 均方誤差
RMSE: 均方根誤差
R²: 決定係數
負對數似然: 預測對數概率
```

## 實際應用

### 1. 經濟預測
- GDP 預測
- 股票價格建模
- 需求預測

### 2. 科學研究
- 實驗數據擬合
- 物理定律驗證
- 生物統計

### 3. 工程應用
- 質量控制
- 傳感器校準
- 系統建模

## 與其他方法比較

### vs. 普通最小二乘法（OLS）
- **OLS**: 點估計，無不確定性量化
- **貝葉斯**: 完整後驗分佈，自然量化不確定性

### vs. 嶺回歸
- **嶺回歸**: 固定正則化參數 λ
- **貝葉斯嶺**: 自動估計 α，β

### vs. LASSO
- **LASSO**: L1 正則化，稀疏解
- **貝葉斯 LASSO**: 使用 Laplace 先驗

## 進階話題

### 1. 非共軛先驗
對於非共軛先驗（如 Laplace、Student-t），需要使用：
- 變分推斷（Variational Inference）
- MCMC 採樣

### 2. 貝葉斯模型選擇
使用邊際似然（Evidence）進行模型比較：
```
Bayes Factor = p(y|M1) / p(y|M2)
```

### 3. 在線學習
隨著新數據到來，遞歸更新後驗：
```
後驗_new = 似然_new × 後驗_old
```

### 4. 異方差性
建模 β 隨輸入變化：
```
β(x) = f(x)
```

## 超參數選擇

### 1. 無信息先驗
```python
alpha_prior = 1e-6  # 近似均勻分佈
beta_prior = 1e-6
```

### 2. 弱信息先驗
```python
alpha_prior = 1.0
beta_prior = 1.0
```

### 3. 信息先驗
基於領域知識或歷史數據設置。

### 4. 自動估計
使用證據近似（BayesianRidgeRegression）。

## 數學推導

### 後驗推導
```
p(w|X, y, α, β) ∝ p(y|X, w, β) p(w|α)

似然: p(y|X, w, β) = N(y | Xw, β⁻¹I)
先驗: p(w|α) = N(w | 0, α⁻¹I)

對數後驗:
log p(w|X,y,α,β) = -β/2 ||y - Xw||² - α/2 ||w||² + const

最大化對數後驗等價於嶺回歸！

完整後驗:
w|X,y ~ N(μ_post, Σ_post)
其中:
Σ_post = (αI + βX^T X)⁻¹
μ_post = β Σ_post X^T y
```

## 計算複雜度

- **時間複雜度**: O(n³) - 矩陣求逆
- **空間複雜度**: O(n²) - 協方差矩陣

對於大規模問題，考慮：
- 稀疏矩陣技巧
- 變分推斷
- 隨機梯度下降

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
1. "Pattern Recognition and Machine Learning" - Bishop, Chapter 3
2. "Bayesian Data Analysis" - Gelman et al.
3. "Machine Learning: A Probabilistic Perspective" - Murphy

### 論文
1. Tipping, M. E. (2001). "Sparse Bayesian Learning and the Relevance Vector Machine"
2. MacKay, D. J. (1992). "Bayesian Interpolation"

### 在線資源
- [Scikit-learn BayesianRidge Documentation](https://scikit-learn.org/stable/modules/linear_model.html#bayesian-ridge-regression)
- [Kevin Murphy's Machine Learning Book](https://probml.github.io/pml-book/)

## 作者

Kaggle Solutions - Bayesian Methods Series

## 許可證

MIT License
