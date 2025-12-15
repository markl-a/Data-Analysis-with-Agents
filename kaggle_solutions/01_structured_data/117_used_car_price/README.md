# 117. 二手車價格預測

## 項目概述

這是一個回歸項目，根據二手車的各種特徵（品牌、車齡、里程等）預測其市場價格，幫助買賣雙方做出明智決策。

**Kaggle 數據集**: [Used Cars Price Prediction](https://www.kaggle.com/datasets/vijayaadithyanvg/car-price-predictionused-cars)

**難度**: ⭐⭐ 中級

## 目標

預測二手車的合理市場價格（回歸問題）

## 數據集描述

### 特徵說明

| 特徵 | 描述 | 類型 |
|------|------|------|
| brand | 汽車品牌 | 類別 |
| model | 車型 | 類別 |
| year | 出廠年份 | 數值 |
| km_driven | 行駛里程 | 數值 |
| fuel | 燃料類型 | 類別 |
| seller_type | 賣家類型 | 類別 |
| transmission | 變速箱類型 | 類別 |
| owner | 過戶次數 | 類別 |
| engine | 引擎容量 | 數值 |
| max_power | 最大馬力 | 數值 |
| seats | 座位數 | 數值 |
| selling_price | 售價 | 目標變量 |

## 關鍵洞察

1. **品牌效應**: 豪華品牌保值率更高
2. **車齡影響**: 新車折舊最快
3. **里程重要**: 低里程車價格更高
4. **燃油類型**: 柴油車通常價格更高

## 技術方法

### 特徵工程
- 車齡計算
- 每年平均里程
- 品牌分級
- 異常值處理

### 模型
- Linear Regression
- Random Forest
- XGBoost
- LightGBM

### 評估指標
- MAE
- RMSE
- R² Score
- MAPE

## 使用方法

```bash
python solution.py
```

---

**難度**: ⭐⭐ 中級
**預計完成時間**: 3-4 小時
**推薦給**: 對價格預測感興趣的學習者
