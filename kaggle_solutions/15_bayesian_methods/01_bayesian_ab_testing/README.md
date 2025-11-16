# Bayesian A/B Testing - 貝葉斯 A/B 測試

## 項目概述

使用貝葉斯統計方法進行 A/B 測試分析，提供比傳統頻率學派方法更直觀的結果解釋。

**難度**: ⭐⭐ 中級
**數據集**: 模擬網站轉換率數據

## 貝葉斯 A/B 測試優勢

### 相比頻率學派方法
1. **直接概率陳述**: 可以直接說"B 優於 A 的概率是 95%"
2. **無需固定樣本量**: 可以隨時停止測試並得出結論
3. **量化不確定性**: 提供完整的後驗分佈
4. **融合先驗知識**: 可以利用歷史數據

## 方法論

### 1. Beta-Binomial 模型
- **似然函數**: Binomial(n, p)
- **先驗分佈**: Beta(α, β)
- **後驗分佈**: Beta(α + successes, β + failures)

### 2. 關鍵指標
- **P(B > A)**: B 優於 A 的概率
- **Expected Loss**: 選擇次優方案的期望損失
- **Credible Interval**: 貝葉斯可信區間（不同於置信區間）

### 3. 決策準則
- P(B > A) > 0.95: 高信心選擇 B
- P(B > A) > 0.90: 中等信心選擇 B
- 否則: 繼續收集數據

## 文件結構

```
01_bayesian_ab_testing/
├── solution.py          # 完整的貝葉斯 A/B 測試實現
└── README.md           # 本文件
```

## 核心功能

### 1. BayesianABTesting 類
```python
ab_test = BayesianABTesting(alpha_prior=1, beta_prior=1)
```

**主要方法**:
- `create_sample_data()`: 生成模擬 A/B 測試數據
- `analyze_variant()`: 分析單個變體的後驗分佈
- `compare_variants()`: 比較兩個變體並計算勝率
- `plot_posteriors()`: 可視化後驗分佈
- `sequential_analysis()`: 序列分析觀察統計量變化

### 2. 輸出結果

#### 後驗統計量
- 轉換率後驗均值和眾數
- 95% 可信區間
- 後驗分佈參數

#### 比較指標
- P(B > A): B 優於 A 的概率
- Expected Loss: 期望損失
- Relative Lift: 相對提升百分比

#### 可視化
1. **後驗分佈比較**: 兩個變體的 Beta 分佈
2. **累積分佈函數**: CDF 比較
3. **差異分佈**: 轉換率差異的分佈
4. **可信區間**: 95% 可信區間視覺比較
5. **序列分析**: P(B > A) 隨樣本量變化

## 使用方法

### 運行完整分析
```bash
python solution.py
```

### 自定義分析
```python
from solution import BayesianABTesting

# 初始化（使用信息先驗）
ab_test = BayesianABTesting(alpha_prior=2, beta_prior=20)

# 創建數據
data = ab_test.create_sample_data(
    n_visitors_A=1000,
    n_visitors_B=1000,
    true_rate_A=0.10,
    true_rate_B=0.12
)

# 分析
results = ab_test.compare_variants(data)
ab_test.plot_posteriors()
ab_test.sequential_analysis(data)
```

## 理論基礎

### Beta 分佈
Beta 分佈是 [0, 1] 區間上的連續概率分佈，適合建模比例/概率：

```
Beta(α, β) ∝ θ^(α-1) × (1-θ)^(β-1)
```

### 貝葉斯更新
```
後驗 ∝ 似然 × 先驗
Beta(α_post, β_post) = Beta(α_prior + successes, β_prior + failures)
```

### 決策理論
選擇期望損失最小的方案：
```
Expected Loss = E[max(θ_B - θ_A, 0)]
```

## 實際應用場景

### 1. 網站優化
- 按鈕顏色/位置測試
- 著陸頁設計比較
- 郵件標題優化

### 2. 產品功能
- 新功能採納率
- 用戶參與度測試
- 定價策略測試

### 3. 營銷活動
- 廣告創意比較
- 促銷策略效果
- 渠道效能測試

## 結果解讀

### 示例輸出
```
變體 A:
  轉換率（後驗均值）: 0.1010
  95% 可信區間: [0.0820, 0.1220]

變體 B:
  轉換率（後驗均值）: 0.1190
  95% 可信區間: [0.0990, 0.1410]

比較結果:
  P(B > A) = 0.9642 (96.42%)
  相對提升: 17.82%
  選擇 A 的期望損失: 0.014231
  選擇 B 的期望損失: 0.000512

結論: B 變體有 96.42% 的概率優於 A（高信心）
```

### 決策建議
- **P(B > A) > 95%**: 採用 B，高信心
- **90% < P(B > A) < 95%**: 傾向 B，但可繼續測試
- **P(B > A) < 90%**: 證據不足，繼續測試或採用 A

## 進階話題

### 1. 先驗選擇
- **均勻先驗**: Beta(1, 1) - 無信息先驗
- **Jeffreys 先驗**: Beta(0.5, 0.5) - 不變性先驗
- **信息先驗**: Beta(α, β) - 基於歷史數據

### 2. 多變體測試
擴展到 A/B/C/D... 測試：
- 使用 Dirichlet 分佈
- 計算所有配對概率
- 控制多重比較問題

### 3. 轉換價值
考慮轉換的貨幣價值：
- Gamma 分佈建模收入
- Beta-Gamma 模型
- 計算期望收入差異

## 統計特性

### 1. 先驗-後驗更新
Beta 分佈是 Binomial 似然的共軛先驗，使得後驗計算解析可得。

### 2. 樣本量考慮
雖然可以隨時停止，但仍需考慮：
- 最小可檢測效應 (MDE)
- 統計功效
- 實際意義的提升幅度

### 3. 序列測試
貝葉斯方法自然支持序列測試，無需修正 α。

## 依賴項

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
```

## 參考資料

### 論文
1. VWO SmartStats Whitepaper
2. "Bayesian A/B Testing at VWO" - Chris Stucchio
3. "Easy Evaluation of Decision Rules in Bayesian A/B testing" - Kamil Bartocha

### 書籍
1. "Bayesian Data Analysis" - Gelman et al.
2. "Doing Bayesian Data Analysis" - John Kruschke

### 在線資源
- [Chris Stucchio's Blog](https://www.chrisstucchio.com/blog/2014/bayesian_ab_decision_rule.html)
- [Evan Miller's A/B Testing Guide](https://www.evanmiller.org/bayesian-ab-testing.html)

## 作者

Kaggle Solutions - Bayesian Methods Series

## 許可證

MIT License
