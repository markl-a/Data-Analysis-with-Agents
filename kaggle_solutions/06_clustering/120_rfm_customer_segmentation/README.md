# 120. RFM 客戶細分分析

## 項目概述

這是一個基於 RFM (Recency, Frequency, Monetary) 模型的客戶細分項目，通過聚類分析識別不同價值的客戶群體，支持精準營銷策略制定。

**Kaggle 數據集**: [Customer Segmentation Dataset](https://www.kaggle.com/datasets/yasserh/customer-segmentation-dataset)

**難度**: ⭐⭐ 中級

## 目標

基於客戶交易行為，將客戶分為不同價值群體（無監督學習/聚類問題）

## 數據集描述

### RFM 指標說明

| 指標 | 描述 | 計算方式 |
|------|------|----------|
| Recency (R) | 最近一次購買距今天數 | 當前日期 - 最後購買日期 |
| Frequency (F) | 購買頻率 | 訂單總數量 |
| Monetary (M) | 消費金額 | 訂單總金額 |

### 原始數據特徵

| 特徵 | 描述 | 類型 |
|------|------|------|
| CustomerID | 客戶唯一識別碼 | ID |
| InvoiceNo | 發票編號 | ID |
| InvoiceDate | 發票日期 | 日期 |
| StockCode | 產品代碼 | 分類 |
| Description | 產品描述 | 文本 |
| Quantity | 數量 | 數值 |
| UnitPrice | 單價 | 數值 |
| Country | 國家 | 分類 |

## 關鍵洞察

1. **高價值客戶**: R低、F高、M高的 VIP 客戶
2. **流失風險客戶**: R高（長時間未購買）的客戶
3. **潛力客戶**: F低但M高的客戶有開發潛力
4. **忠實客戶**: F高但M中等的常客

## 技術方法

### RFM 計算
- Recency: 計算最近購買日期與分析日期的差異
- Frequency: 計算唯一訂單數量
- Monetary: 計算總消費金額

### 聚類算法
- K-Means 聚類
- 層次聚類 (Hierarchical Clustering)
- DBSCAN
- 高斯混合模型 (GMM)

### 評估指標
- 輪廓係數 (Silhouette Score)
- Davies-Bouldin 指數
- Calinski-Harabasz 指數

## 客戶分群結果

典型的客戶分群包括：

1. **Champions (冠軍客戶)**: 最近購買、頻繁購買、高消費
2. **Loyal Customers (忠實客戶)**: 經常購買、高消費
3. **Potential Loyalists (潛力客戶)**: 最近購買、有潛力成為忠實客戶
4. **At Risk (風險客戶)**: 曾經頻繁購買但最近沒有購買
5. **Hibernating (休眠客戶)**: 很久沒購買的客戶
6. **Lost (流失客戶)**: 長期未購買的低價值客戶

## 使用方法

```bash
python solution.py
```

## 商業價值

1. **精準營銷**: 針對不同客群制定營銷策略
2. **客戶保留**: 識別高風險流失客戶
3. **資源優化**: 將資源集中在高價值客戶
4. **預測分析**: 預測客戶未來行為

---

**難度**: ⭐⭐ 中級
**預計完成時間**: 3-4 小時
**推薦給**: 對客戶分析和營銷感興趣的學習者
