# 130. 天氣預測

## 項目概述

這是一個時間序列預測項目，基於歷史氣象數據預測未來天氣狀況，應用於農業規劃、能源管理和日常生活決策。

**Kaggle 數據集**: [Weather Prediction Dataset](https://www.kaggle.com/datasets/ananthr1/weather-prediction)

**難度**: ⭐⭐⭐ 進階

## 目標

預測未來的溫度、降水量等氣象指標

## 數據集描述

### 特徵說明

| 特徵 | 描述 | 類型 |
|------|------|------|
| date | 日期 | 時間 |
| temperature | 溫度 (°C) | 數值 |
| humidity | 濕度 (%) | 數值 |
| pressure | 氣壓 (hPa) | 數值 |
| wind_speed | 風速 (km/h) | 數值 |
| precipitation | 降水量 (mm) | 數值 |
| weather_type | 天氣類型 | 類別 |

### 天氣類型

| 類別 | 描述 |
|------|------|
| Sunny | 晴天 |
| Cloudy | 多雲 |
| Rainy | 雨天 |
| Stormy | 暴風雨 |
| Snowy | 下雪 |

## 關鍵洞察

1. **季節性**: 明顯的年度週期模式
2. **日變化**: 日溫差模式
3. **氣壓趨勢**: 氣壓下降預示天氣變化
4. **滯後效應**: 前期天氣影響後期

## 技術方法

### 時間特徵
- 月份、季節
- 日週期特徵
- 滾動統計量
- 滯後特徵

### 模型
- ARIMA
- Prophet
- LSTM
- XGBoost (回歸)

### 評估指標
- MAE
- RMSE
- MAPE

## 使用方法

```bash
python solution.py
```

---

**難度**: ⭐⭐⭐ 進階
**預計完成時間**: 4-5 小時
**推薦給**: 對時間序列預測感興趣的學習者
