# 120. 語音情感識別

## 項目概述

這是一個語音情感識別項目，通過分析語音信號的聲學特徵來識別說話者的情感狀態，應用於客服質量監控、心理健康評估等領域。

**Kaggle 數據集**: [RAVDESS Emotional Speech Audio](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

**難度**: ⭐⭐⭐ 進階

## 目標

識別語音中的情感類別（多類分類問題）

## 數據集描述

### 情感類別

| 類別 | 描述 |
|------|------|
| neutral | 中性 |
| calm | 平靜 |
| happy | 高興 |
| sad | 悲傷 |
| angry | 憤怒 |
| fearful | 恐懼 |
| disgust | 厭惡 |
| surprised | 驚訝 |

### 聲學特徵

| 特徵 | 描述 |
|------|------|
| MFCC | 梅爾頻率倒譜係數 |
| Chroma | 色度特徵 |
| Mel Spectrogram | 梅爾頻譜圖 |
| Spectral Contrast | 頻譜對比度 |
| Zero Crossing Rate | 過零率 |
| RMS Energy | 均方根能量 |

## 技術方法

### 特徵提取
- MFCC (13-40 係數)
- Delta 和 Delta-Delta 特徵
- 統計特徵（均值、標準差）

### 模型
- SVM
- Random Forest
- CNN (1D/2D)
- LSTM
- Transformer

### 評估指標
- 準確率
- F1 分數
- 混淆矩陣

## 使用方法

```bash
python solution.py
```

---

**難度**: ⭐⭐⭐ 進階
**預計完成時間**: 5-6 小時
**推薦給**: 對語音處理感興趣的學習者
