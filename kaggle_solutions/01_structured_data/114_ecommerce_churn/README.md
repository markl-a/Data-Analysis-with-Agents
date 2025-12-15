# 114. 電商客戶流失預測

## 項目概述

這是一個電子商務客戶流失預測項目，通過分析客戶行為數據，預測哪些客戶可能停止使用平台服務，幫助企業制定客戶保留策略。

**Kaggle 數據集**: [E-Commerce Churn Dataset](https://www.kaggle.com/datasets/nabihazahid/e-commerce-customer-insights-and-churn-dataset)

**難度**: ⭐⭐ 中級

## 目標

預測電商平台客戶是否會流失（二元分類問題）

## 數據集描述

### 特徵說明

| 特徵 | 描述 | 類型 |
|------|------|------|
| customer_id | 客戶唯一識別碼 | ID |
| tenure | 客戶使用平台時長（月） | 數值 |
| preferred_login_device | 偏好登錄設備 | 分類 |
| city_tier | 城市等級 | 分類 |
| warehouse_to_home | 倉庫到家距離 | 數值 |
| preferred_payment_mode | 偏好支付方式 | 分類 |
| gender | 性別 | 分類 |
| hour_spend_on_app | 每日使用APP時長 | 數值 |
| number_of_device_registered | 註冊設備數量 | 數值 |
| preferred_order_cat | 偏好訂單類別 | 分類 |
| satisfaction_score | 滿意度評分 | 數值 |
| marital_status | 婚姻狀況 | 分類 |
| number_of_address | 地址數量 | 數值 |
| complain | 是否有投訴 | 二元 |
| order_amount_hike_from_last_year | 訂單金額同比增長 | 數值 |
| coupon_used | 使用優惠券數量 | 數值 |
| order_count | 訂單數量 | 數值 |
| day_since_last_order | 距上次訂單天數 | 數值 |
| cashback_amount | 返現金額 | 數值 |
| churn | 是否流失 (0=否, 1=是) | 目標變量 |

## 關鍵洞察

1. **投訴歷史**: 有投訴記錄的客戶流失率顯著升高
2. **活躍度**: 距上次訂單時間越長，流失風險越高
3. **滿意度**: 低滿意度評分與高流失率強相關
4. **使用時長**: 新客戶（tenure < 6個月）流失風險較高
5. **優惠依賴**: 高度依賴優惠券的客戶可能更容易流失

## 技術方法

### 數據預處理
- 缺失值填充
- 異常值處理
- 類別變量編碼
- 特徵標準化

### 特徵工程
- 客戶生命週期階段
- 活躍度指標
- 消費行為特徵
- RFM 相關特徵

### 模型
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

### 評估指標
- 準確率
- 精確率
- 召回率
- F1 分數
- ROC-AUC

## 使用方法

```bash
python solution.py
```

## 預期結果

- 準確率: ~90-93%
- ROC-AUC: ~0.92-0.95
- 召回率: ~85%+

## 商業價值

1. **主動挽留**: 識別高風險客戶並主動干預
2. **資源優化**: 將營銷資源集中在高價值客戶
3. **策略優化**: 了解流失原因以改進服務
4. **成本節省**: 降低客戶獲取成本

---

**難度**: ⭐⭐ 中級
**預計完成時間**: 3-4 小時
**推薦給**: 對商業分析感興趣的學習者
