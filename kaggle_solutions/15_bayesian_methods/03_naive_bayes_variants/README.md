# Naive Bayes Variants - 樸素貝葉斯變體

## 項目概述

實現和比較三種樸素貝葉斯分類器變體：高斯、多項式和伯努利。每種變體適用於不同類型的數據。

**難度**: ⭐⭐ 中級
**數據集**: 模擬混合特徵數據

## 樸素貝葉斯原理

### 核心假設
**條件獨立性**: 給定類別，特徵之間相互獨立
```
P(X|y) = P(x₁|y) × P(x₂|y) × ... × P(xₙ|y)
```

### 貝葉斯定理
```
P(y|X) = P(X|y) × P(y) / P(X)
       ∝ P(X|y) × P(y)
```

### 決策規則
```
ŷ = argmax_y P(y|X) = argmax_y [P(X|y) × P(y)]
```

## 三種變體

### 1. 高斯樸素貝葉斯
**適用**: 連續特徵，假設每個類別的特徵服從高斯分佈

**模型**:
```
P(xᵢ|y) ~ N(μᵧ, σᵧ²)
```

**應用**:
- 傳感器數據分類
- 生物信號分析
- 一般連續數據

### 2. 多項式樸素貝葉斯
**適用**: 計數特徵（詞頻）

**模型**:
```
P(xᵢ|y) = θᵧᵢ （多項式分佈）
```

**應用**:
- 文本分類
- 詞頻分析
- 文檔分類

### 3. 伯努利樸素貝葉斯
**適用**: 二值特徵（出現/不出現）

**模型**:
```
P(xᵢ|y) = θᵧᵢ^xᵢ × (1-θᵧᵢ)^(1-xᵢ)
```

**應用**:
- 文檔-詞矩陣
- 特徵存在性檢測
- 二值化數據

## 文件結構

```
03_naive_bayes_variants/
├── solution.py          # 三種樸素貝葉斯實現
└── README.md           # 本文件
```

## 核心功能

### 實現的類

1. **GaussianNaiveBayes**: 高斯樸素貝葉斯
2. **MultinomialNaiveBayes**: 多項式樸素貝葉斯
3. **BernoulliNaiveBayes**: 伯努利樸素貝葉斯

### 主要方法
```python
fit(X, y)              # 訓練模型
predict(X)             # 預測類別
predict_proba(X)       # 預測概率
```

## 使用方法

### 運行完整分析
```bash
python solution.py
```

### 自定義使用
```python
from solution import GaussianNaiveBayes

# 創建和訓練模型
gnb = GaussianNaiveBayes()
gnb.fit(X_train, y_train)

# 預測
y_pred = gnb.predict(X_test)
y_proba = gnb.predict_proba(X_test)
```

## 比較結果

### 可視化輸出
1. **準確率比較**: 條形圖顯示各模型性能
2. **混淆矩陣**: 詳細的分類結果
3. **概率校準**: 預測概率的可靠性
4. **概率分佈**: 兩個類別的概率分佈

## 平滑技術

### Laplace 平滑
防止零概率問題：
```
P(xᵢ|y) = (count(xᵢ, y) + α) / (count(y) + α × n_features)
```

常用 α = 1（Laplace 平滑）或 α = 0.5（Lidstone 平滑）

## 優缺點

### 優點
1. **訓練快速**: O(n × d) 時間複雜度
2. **可解釋性強**: 概率模型，易於理解
3. **少量數據**: 在小數據集上表現良好
4. **多分類**: 自然處理多分類問題

### 缺點
1. **獨立性假設**: 現實中特徵通常相關
2. **零頻率問題**: 需要平滑處理
3. **表達能力**: 模型較簡單，複雜決策邊界受限

## 實際應用

### 文本分類
- 垃圾郵件過濾
- 情感分析
- 主題分類

### 醫療診斷
- 疾病預測
- 症狀分析

### 推薦系統
- 用戶偏好預測
- 內容推薦

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

1. "Pattern Recognition and Machine Learning" - Bishop
2. "Machine Learning" - Tom Mitchell
3. [Scikit-learn Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)

## 作者

Kaggle Solutions - Bayesian Methods Series

## 許可證

MIT License
