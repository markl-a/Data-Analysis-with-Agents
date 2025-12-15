# 126. 運動員表現預測

## 項目概述

這是一個體育分析項目，通過分析運動員的訓練數據、生理指標和歷史表現來預測比賽成績，幫助教練優化訓練計劃。

**Kaggle 數據集**: [FIFA Player Stats](https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset)

**難度**: ⭐⭐⭐ 進階

## 目標

預測運動員的整體評分和表現（回歸問題）

## 數據集描述

### 特徵說明

| 特徵 | 描述 | 類型 |
|------|------|------|
| age | 年齡 | 數值 |
| height_cm | 身高 | 數值 |
| weight_kg | 體重 | 數值 |
| pace | 速度 | 數值 |
| shooting | 射門 | 數值 |
| passing | 傳球 | 數值 |
| dribbling | 盤帶 | 數值 |
| defending | 防守 | 數值 |
| physic | 體能 | 數值 |
| position | 位置 | 類別 |

## 關鍵洞察

1. **位置影響**: 不同位置需要不同技能組合
2. **年齡曲線**: 球員表現隨年齡呈倒 U 形
3. **身體素質**: 身高體重與位置高度相關
4. **技能平衡**: 頂級球員通常技能更均衡

## 技術方法

### 特徵工程
- 位置編碼
- 年齡分組
- 技能組合特徵
- BMI 計算

### 模型
- Linear Regression
- Random Forest
- XGBoost
- Neural Network

### 評估指標
- MAE
- RMSE
- R² Score

## 使用方法

```bash
python solution.py
```

---

**難度**: ⭐⭐⭐ 進階
**預計完成時間**: 3-4 小時
**推薦給**: 對體育分析感興趣的學習者
