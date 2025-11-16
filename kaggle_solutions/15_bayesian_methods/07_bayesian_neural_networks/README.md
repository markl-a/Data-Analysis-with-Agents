# Bayesian Neural Networks - 貝葉斯神經網絡

## 項目概述

實現貝葉斯神經網絡，使用變分推斷進行訓練。提供預測不確定性量化，區分認識不確定性和偶然不確定性。

**難度**: ⭐⭐⭐ 高級
**數據集**: 模擬回歸和分類數據

## 核心思想

### 傳統神經網絡
權重是點估計：w = w*

### 貝葉斯神經網絡
權重是分佈：w ~ p(w|D)

## 不確定性類型

### 1. 認識不確定性（Epistemic）
- **來源**: 模型參數不確定性
- **特點**: 隨數據增加而減少
- **量化**: 權重後驗分佈的方差

### 2. 偶然不確定性（Aleatoric）
- **來源**: 數據固有噪聲
- **特點**: 數據增加不減少
- **量化**: 似然方差

## 變分推斷

### 問題
後驗 p(w|D) 難以計算

### 解決方案
用簡單分佈 q(w|θ) 近似 p(w|D)

### ELBO 目標
```
ELBO = E_q[log p(y|X,w)] - KL(q(w)||p(w))
     = 似然期望 - KL 散度
```

### 重參數化技巧
```
w = μ + σ × ε, ε ~ N(0,1)
```

## 文件結構

```
07_bayesian_neural_networks/
├── solution.py          # BNN 實現
└── README.md           # 本文件
```

## 核心類

### BayesianNeuralNetwork
```python
bnn = BayesianNeuralNetwork(
    layer_sizes=[input_dim, hidden_dim, output_dim],
    prior_std=1.0,
    noise_std=0.1
)
```

**主要方法**:
- `fit()`: 變分推斷訓練
- `predict()`: 預測均值和標準差
- `sample_weights()`: 從後驗採樣
- `kl_divergence()`: 計算 KL 散度
- `elbo()`: 計算證據下界

## 使用方法

### 基本使用
```bash
python solution.py
```

### 回歸任務
```python
from solution import BayesianNeuralNetwork

# 創建 BNN
bnn = BayesianNeuralNetwork(
    layer_sizes=[1, 50, 50, 1],
    prior_std=1.0,
    noise_std=0.1
)

# 訓練
bnn.fit(X_train, y_train, n_iterations=1000)

# 預測
y_mean, y_std = bnn.predict(X_test, n_samples=100)
```

### 分類任務
```python
# 二分類
bnn = BayesianNeuralNetwork(
    layer_sizes=[2, 20, 1],
    prior_std=1.0,
    noise_std=0.1
)

bnn.fit(X_train, y_train, n_iterations=1000)
y_pred, y_std = bnn.predict(X_test, n_samples=100)
y_class = (y_pred > 0.5).astype(int)
```

## 示例

### 1. 回歸任務
- 非線性函數擬合
- 異方差噪聲
- 不確定性量化
- 外推區域高不確定性

### 2. 分類任務
- Moons 數據集
- 決策邊界
- 不確定性地圖
- 邊界處高不確定性

## 可視化輸出

### 回歸
1. **預測曲線**: 均值 + 2σ 區間
2. **不確定性分析**: 標準差隨 x 變化
3. **後驗樣本**: 50 條可能的函數

### 分類
1. **決策邊界**: 類別概率熱圖
2. **不確定性地圖**: 預測標準差
3. **訓練數據**: 疊加在圖上

## 優缺點

### 優點
1. **不確定性**: 自然量化預測不確定性
2. **正則化**: 貝葉斯正則化防止過擬合
3. **小數據**: 在數據稀少時表現好
4. **可解釋**: 不確定性有明確含義

### 缺點
1. **計算**: 比普通 NN 慢
2. **內存**: 需存儲均值和方差
3. **調參**: 需設置先驗參數

## 進階主題

### 1. 更好的近似
- **Dropout**: 作為貝葉斯近似
- **Concrete Dropout**: 學習 dropout 率
- **MC Dropout**: 測試時使用 dropout

### 2. 不同先驗
- **Spike-and-Slab**: 稀疏性
- **Horseshoe**: 自適應稀疏
- **Group Lasso**: 結構化稀疏

### 3. 全協方差
使用完整協方差矩陣而非對角。

### 4. 自然梯度
使用自然梯度加速收斂。

## 應用場景

### 主動學習
選擇不確定性最高的樣本標註。

### 強化學習
探索-利用權衡。

### 醫療診斷
高風險決策需要不確定性。

### 異常檢測
高不確定性 → 異常。

## 與其他方法比較

### vs. Dropout
- **Dropout**: 近似貝葉斯，更快
- **BNN**: 完整貝葉斯，更準

### vs. Ensemble
- **Ensemble**: 訓練多個模型
- **BNN**: 單個模型，參數分佈

### vs. Gaussian Processes
- **GP**: 函數空間先驗
- **BNN**: 權重空間先驗

## 實現細節

### 變分參數
每個權重有兩個參數：
- **μ**: 均值
- **log σ**: 對數標準差

### 梯度計算
使用重參數化技巧使梯度可計算。

### ELBO 估計
使用單樣本估計加速訓練。

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

### 論文
1. Graves (2011) - "Practical Variational Inference for Neural Networks"
2. Blundell et al. (2015) - "Weight Uncertainty in Neural Networks"
3. Gal & Ghahramani (2016) - "Dropout as a Bayesian Approximation"

### 教材
1. "Deep Learning" - Goodfellow et al.
2. "Bayesian Deep Learning" - Wilson

### 工具
- [TensorFlow Probability](https://www.tensorflow.org/probability)
- [Pyro](https://pyro.ai/)
- [Edward](http://edwardlib.org/)

## 作者

Kaggle Solutions - Bayesian Methods Series

## 許可證

MIT License
