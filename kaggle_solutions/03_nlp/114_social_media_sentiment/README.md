# 114. 社交媒體情緒分析

## 項目概述

這是一個 NLP 項目，分析社交媒體（如 Twitter）上的用戶評論情緒，用於品牌監控、市場研究和輿情分析。

**Kaggle 數據集**: [Twitter Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)

**難度**: ⭐⭐ 中級

## 目標

將社交媒體帖子分類為正面、負面或中性情緒

## 數據集描述

### 特徵說明

| 特徵 | 描述 | 類型 |
|------|------|------|
| text | 推文內容 | 文本 |
| entity | 提及的實體 | 類別 |
| sentiment | 情緒標籤 | 目標變量 |

### 情緒類別

| 類別 | 描述 |
|------|------|
| Positive | 正面情緒 |
| Negative | 負面情緒 |
| Neutral | 中性/無情緒 |
| Irrelevant | 不相關 |

## 關鍵洞察

1. **表情符號**: 表情符號是強烈的情緒指標
2. **否定詞**: 否定詞會反轉情緒
3. **俚語縮寫**: 社交媒體特有語言需特殊處理
4. **上下文**: 同一詞在不同語境含義不同

## 技術方法

### 文本預處理
- 去除 URL 和特殊字符
- 處理表情符號
- 小寫化
- 停用詞移除
- 詞形還原

### 特徵提取
- TF-IDF
- Word2Vec
- BERT Embeddings
- 情感詞典

### 模型
- Naive Bayes
- SVM
- LSTM
- BERT Fine-tuning

### 評估指標
- 準確率
- F1 分數
- 混淆矩陣

## 使用方法

```bash
python solution.py
```

---

**難度**: ⭐⭐ 中級
**預計完成時間**: 3-4 小時
**推薦給**: NLP 初學者和社交媒體分析愛好者
