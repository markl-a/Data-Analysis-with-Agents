# 116. 信用風險評估

## 項目概述

這是一個金融領域的信用風險評估項目，基於申請人的個人資料和財務狀況預測貸款違約風險，幫助金融機構做出貸款決策。

**Kaggle 數據集**: [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)

**難度**: ⭐⭐⭐ 進階

## 目標

預測貸款申請人的違約概率（二元分類問題）

## 數據集描述

### 特徵說明

| 特徵 | 描述 | 類型 |
|------|------|------|
| SK_ID_CURR | 申請人ID | ID |
| TARGET | 是否違約 (0=正常, 1=違約) | 目標變量 |
| AMT_INCOME_TOTAL | 年收入 | 數值 |
| AMT_CREDIT | 信用額度 | 數值 |
| AMT_ANNUITY | 年金金額 | 數值 |
| DAYS_BIRTH | 年齡(天數) | 數值 |
| DAYS_EMPLOYED | 工作年限(天數) | 數值 |
| EXT_SOURCE_1/2/3 | 外部信用評分 | 數值 |
| ... | 其他特徵 | 多種類型 |

## 關鍵洞察

1. **外部評分**: EXT_SOURCE 是最強預測因子
2. **收入負債比**: 高負債收入比增加違約風險
3. **工作穩定性**: 工作年限與違約率負相關
4. **年齡因素**: 年輕申請人違約率較高

## 技術方法

### 模型
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- 神經網絡

### 評估指標
- AUC-ROC
- Gini 係數
- KS 統計量
- Precision-Recall

## 使用方法

```bash
python solution.py
```

---

**難度**: ⭐⭐⭐ 進階
**預計完成時間**: 4-5 小時
**推薦給**: 對金融風控感興趣的學習者
