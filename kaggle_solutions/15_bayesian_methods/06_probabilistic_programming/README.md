# Probabilistic Programming - 概率編程

## 項目概述

使用純 NumPy/SciPy 實現概率編程的基本概念，包括隨機變量、概率模型和貝葉斯推斷。

**難度**: ⭐⭐⭐ 高級
**數據集**: 模擬概率模型

## 核心概念

### 概率編程
將概率模型表達為程序，進行自動推斷。

### 關鍵組件
1. **隨機變量**: 概率分佈的實例
2. **概率模型**: 隨機變量的組合
3. **推斷**: 從數據學習參數分佈

## 實現的模型

### 1. Beta-Binomial 模型
**應用**: 硬幣投擲、點擊率估計

**數學**:
```
p ~ Beta(α, β)
x ~ Binomial(n, p)
```

### 2. 貝葉斯線性回歸
**應用**: 預測建模、趨勢分析

**數學**:
```
w ~ N(0, σ_w²I)
y ~ N(Xw, σ²)
```

### 3. 貝葉斯混合模型
**應用**: 聚類、異常檢測

**數學**:
```
π ~ Dirichlet(α)
μₖ ~ N(μ₀, σ₀²)
xᵢ ~ Σₖ πₖ N(μₖ, σₖ²)
```

## 文件結構

```
06_probabilistic_programming/
├── solution.py          # 概率編程實現
└── README.md           # 本文件
```

## 核心類

### RandomVariable
表示隨機變量：
```python
rv = RandomVariable('theta', 'beta', alpha=2, beta=5)
samples = rv.sample(1000)
log_prob = rv.log_prob(0.5)
```

### ProbabilisticModel
組合多個隨機變量：
```python
model = ProbabilisticModel()
model.add_variable(rv)
model.observe('y', data)
```

### BayesianLinearModel
貝葉斯線性回歸：
```python
model = BayesianLinearModel()
model.fit(X, y)
y_pred, y_std = model.predict(X_new)
```

### BayesianMixtureModel
貝葉斯混合模型：
```python
model = BayesianMixtureModel(n_components=3)
model.fit(X)
proba = model.predict_proba(X_new)
```

## 使用方法

### 運行所有示例
```bash
python solution.py
```

### 自定義示例
```python
# 硬幣投擲
alpha_prior, beta_prior = 1, 1
flips = np.random.binomial(1, 0.7, 100)
alpha_post = alpha_prior + flips.sum()
beta_post = beta_prior + len(flips) - flips.sum()

# 貝葉斯回歸
model = BayesianLinearModel()
model.fit(X_train, y_train, n_samples=2000)
y_pred, y_std = model.predict(X_test)
```

## 示例

### 1. 硬幣投擲
- Beta-Binomial 共軛更新
- 先驗到後驗的演化
- 不確定性隨數據減少

### 2. 貝葉斯線性回歸
- 使用 MCMC 採樣
- 預測不確定性
- 外推時不確定性增加

### 3. 混合模型
- EM 算法
- 自動聚類
- 密度估計

## 推斷方法

### 共軛更新
當先驗和後驗屬於同一分佈族時，可解析更新。

### MCMC 採樣
對於複雜模型，使用 Metropolis-Hastings 採樣。

### 變分推斷
近似後驗為簡單分佈（如高斯）。

## 與專用庫比較

### PyMC
功能更全，自動微分，HMC 採樣。

### Stan
高性能，NUTS 採樣器。

### 本實現
教育目的，理解底層原理。

## 擴展方向

### 1. 更多分佈
- Poisson、Exponential
- Student-t、Cauchy
- Dirichlet、Wishart

### 2. 更複雜模型
- 階層模型
- 隱馬爾可夫模型
- 狀態空間模型

### 3. 更好的推斷
- HMC 採樣
- NUTS 採樣器
- 變分貝葉斯

## 依賴項

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
```

## 參考資料

### 工具庫
- [PyMC](https://www.pymc.io/)
- [Stan](https://mc-stan.org/)
- [TensorFlow Probability](https://www.tensorflow.org/probability)

### 教材
1. "Probabilistic Programming & Bayesian Methods for Hackers"
2. "Bayesian Methods for Hackers" - Cameron Davidson-Pilon

## 作者

Kaggle Solutions - Bayesian Methods Series

## 許可證

MIT License
