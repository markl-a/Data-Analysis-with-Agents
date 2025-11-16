# Markov Chain Monte Carlo (MCMC) - 馬可夫鏈蒙特卡洛

## 項目概述

實現三種主流 MCMC 採樣算法：Metropolis-Hastings、Gibbs Sampling 和 Hamiltonian Monte Carlo (HMC)，用於貝葉斯推斷。

**難度**: ⭐⭐⭐ 高級
**數據集**: 模擬貝葉斯推斷問題

## MCMC 核心概念

### 目標
從複雜的後驗分佈 p(θ|D) 中採樣，當直接採樣困難時。

### 基本思想
構造馬可夫鏈，其平穩分佈為目標分佈。

## 三種算法

### 1. Metropolis-Hastings (MH)
最通用的 MCMC 算法。

**步驟**:
1. 從提議分佈採樣 θ' ~ q(θ'|θ)
2. 計算接受概率 α = min(1, p(θ')/p(θ))
3. 以概率 α 接受新狀態

**優點**: 通用，易實現
**缺點**: 需要調整提議分佈

### 2. Gibbs Sampling
逐個維度採樣。

**步驟**:
1. 固定其他維度
2. 從條件分佈採樣當前維度
3. 循環所有維度

**優點**: 接受率 100%
**缺點**: 需要條件分佈

### 3. Hamiltonian Monte Carlo (HMC)
利用梯度信息的高效採樣。

**步驟**:
1. 引入輔助動量變量
2. 使用 Hamiltonian 動力學
3. Leapfrog 積分
4. Metropolis 接受

**優點**: 高效，低自相關
**缺點**: 需要梯度，需調參

## 文件結構

```
05_mcmc/
├── solution.py          # 三種 MCMC 實現
└── README.md           # 本文件
```

## 核心功能

### 實現的類

1. **MetropolisHastings**: MH 採樣器
2. **GibbsSampler**: Gibbs 採樣器
3. **HamiltonianMonteCarlo**: HMC 採樣器

### 主要方法
```python
sample(n_samples, initial_state, burn_in)
```

## 使用方法

### 運行示例
```bash
python solution.py
```

### 自定義 MH 採樣
```python
from solution import MetropolisHastings

def target_log_prob(x):
    return -0.5 * np.sum(x**2)

mh = MetropolisHastings(target_log_prob, proposal_std=0.5)
samples = mh.sample(
    n_samples=5000,
    initial_state=np.zeros(2),
    burn_in=1000
)
```

## MCMC 診斷

### 1. 軌跡圖（Trace Plot）
觀察鏈是否收斂、是否混合良好。

### 2. 自相關圖
評估樣本獨立性，低自相關更好。

### 3. 接受率
- **MH**: 23% - 40% 較理想
- **HMC**: 60% - 90% 較理想

### 4. 有效樣本量（ESS）
```
ESS = N / (1 + 2 * Σ ρₖ)
```

## 調參建議

### Metropolis-Hastings
- **proposal_std**: 調整接受率至 20-40%

### HMC
- **step_size**: 較小 → 高接受率，較慢
- **n_leapfrog**: 較多 → 低自相關，較慢

### 通用
- **burn_in**: 通常 1000-5000
- **thin**: 減少自相關，增加獨立性

## 進階話題

### 1. 收斂診斷
- Gelman-Rubin 統計量
- Geweke 診斷
- Heidelberger-Welch 測試

### 2. 自適應 MCMC
自動調整提議分佈或步長。

### 3. 並行 MCMC
多條鏈並行，加速收斂診斷。

### 4. No-U-Turn Sampler (NUTS)
自動調整 HMC 參數的算法。

## 應用場景

### 貝葉斯推斷
- 參數後驗採樣
- 預測分佈
- 模型比較

### 統計物理
- 配分函數計算
- 相變研究

### 機器學習
- 貝葉斯神經網絡
- 隱變量模型

## 依賴項

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
```

## 參考資料

### 書籍
1. "Monte Carlo Statistical Methods" - Robert & Casella
2. "Bayesian Data Analysis" - Gelman et al.

### 論文
1. Metropolis et al. (1953) - 原始論文
2. Hastings (1970) - 推廣版本
3. Neal (2011) - "MCMC using Hamiltonian dynamics"

### 工具
- [PyMC](https://www.pymc.io/)
- [Stan](https://mc-stan.org/)
- [emcee](https://emcee.readthedocs.io/)

## 作者

Kaggle Solutions - Bayesian Methods Series

## 許可證

MIT License
