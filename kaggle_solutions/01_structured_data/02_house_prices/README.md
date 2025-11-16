# 02. House Prices 房價預測

## 📋 項目概述

預測房屋銷售價格的迴歸問題，使用79個特徵變量來預測房屋的最終售價。

**Kaggle競賽**: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

**難度**: ⭐⭐ 初級

## 🎯 目標

預測房屋的銷售價格（迴歸問題）

## 📊 數據集描述

### 關鍵特徵

- **OverallQual**: 整體材料和完成質量
- **GrLivArea**: 地上居住面積（平方英尺）
- **GarageCars**: 車庫容量
- **GarageArea**: 車庫面積
- **TotalBsmtSF**: 地下室總面積
- **1stFlrSF**: 一樓面積
- **FullBath**: 全套浴室數量
- **YearBuilt**: 建造年份
- **YearRemodAdd**: 改建年份

## 🛠️ 技術方法

### 特徵工程
```python
TotalSF = TotalBsmtSF + 1stFlrSF + GrLivArea
HouseAge = 2024 - YearBuilt
QualityArea = OverallQual * GrLivArea
```

### 模型選擇
- Ridge 回歸（主要）
- Lasso 回歸
- Random Forest
- Gradient Boosting

### 評估指標
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

## 🚀 使用方法

```bash
python solution.py
```

## 📈 預期結果

- RMSE: ~$30,000
- R²: ~0.85-0.90

## 💡 改進建議

1. 處理異常值
2. Log轉換價格
3. 更多特徵工程
4. 集成學習
5. 超參數調優

## 📚 學習要點

- ✅ 迴歸問題建模
- ✅ 特徵工程技巧
- ✅ 處理連續型變量
- ✅ 模型評估指標

---

**難度**: ⭐⭐ 初級
**預計時間**: 3-4 小時
