# 113. AI 生成文本檢測

## 項目概述

這是一個 NLP 項目，旨在區分 AI 生成的文本（如 GPT、Gemini 等 LLM 生成）和人類撰寫的文本。隨著生成式 AI 的普及，這項技術對於學術誠信、內容審核等領域至關重要。

**Kaggle 競賽**: [LLM - Detect AI Generated Text](https://www.kaggle.com/competitions/llm-detect-ai-generated-text)

**難度**: ⭐⭐⭐ 進階

## 目標

判斷給定文本是 AI 生成還是人類撰寫（二元分類問題）

## 數據集描述

### 特徵說明

| 特徵 | 描述 | 類型 |
|------|------|------|
| id | 文本唯一識別碼 | ID |
| text | 文本內容 | 文本 |
| label | 0=人類撰寫, 1=AI生成 | 目標變量 |

## 關鍵洞察

1. **文本流暢度**: AI 文本通常過於流暢和一致
2. **詞彙多樣性**: 人類文本詞彙選擇更多變
3. **句式結構**: AI 傾向使用特定句式模式
4. **困惑度 (Perplexity)**: AI 文本的困惑度通常較低
5. **連貫性**: AI 文本的段落連貫性過於完美

## 技術方法

### 文本特徵
- TF-IDF 向量
- 詞彙多樣性指標
- 句子長度分布
- 困惑度計算
- N-gram 分析

### 模型
- BERT/RoBERTa 微調
- XGBoost + 文本特徵
- Ensemble 方法
- Logistic Regression 基線

### 評估指標
- AUC-ROC
- F1 分數
- 準確率

## 使用方法

```bash
python solution.py
```

---

**難度**: ⭐⭐⭐ 進階
**預計完成時間**: 5-6 小時
**推薦給**: 對 NLP 和 LLM 感興趣的學習者
