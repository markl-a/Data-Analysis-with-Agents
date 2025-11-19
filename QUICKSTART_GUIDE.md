# 🚀 快速啟動指南

本指南幫助您在5分鐘內開始使用本專案的核心功能。

## 📋 前置需求

- Python 3.8+
- pip 包管理器
- (可選) Kaggle API憑證

## ⚡ 快速開始

### 1. 安裝依賴

```bash
# 安裝核心依賴
pip install -r requirements.txt

# 或使用開發模式安裝
pip install -e .
```

### 2. 運行快速測試

確保所有功能正常：

```bash
python scripts/quick_test.py
```

✅ 如果看到"所有測試通過"，說明環境配置成功！

### 3. 查看專案統計

```bash
python scripts/generate_statistics.py
```

這會生成完整的專案統計報告，包括：
- 1002個Kaggle解決方案的分布
- 各類別數量統計
- 文檔完整度

## 🎯 使用場景

### 場景 1: 探索Kaggle解決方案

```bash
# 瀏覽所有解決方案
python scripts/browse_solutions.py

# 按類別搜索
python scripts/browse_solutions.py --category 01_structured_data

# 按關鍵詞搜索
python scripts/browse_solutions.py --search "time series"
```

### 場景 2: 下載Kaggle數據集

```python
from data_analysis_chatbots import quick_download

# 快速下載流行數據集
data_path = quick_download('titanic')
print(f"數據已下載到: {data_path}")

# 支持的數據集包括:
# titanic, house-prices, digit-recognizer, credit-fraud等50+
```

### 場景 3: 使用聚類算法

```python
from data_analysis_chatbots import KMeansClusterer
import pandas as pd

# 加載數據
df = pd.read_csv('data/your_data.csv')

# 創建聚類器
clusterer = KMeansClusterer(n_clusters=3, random_state=42)

# 訓練模型
labels = clusterer.fit_predict(df[['feature1', 'feature2']])

# 可視化結果
clusterer.visualize()
```

### 場景 4: 運行具體的Kaggle解決方案

```bash
# 運行結構化數據解決方案
cd kaggle_solutions/01_structured_data/01_customer_churn_prediction
python solution.py

# 查看解決方案說明
cat README.md
```

## 📊 1002個Kaggle解決方案一覽

本專案包含1002個完整的Kaggle解決方案，分為17個類別：

| 類別 | 數量 | 主要內容 |
|------|------|----------|
| 結構化數據 | 51 | 客戶流失、信用評分、保險定價、員工留任等 |
| 時間序列 | 67 | 銷售預測、股價預測、能源消耗、多時間跨度預測等 |
| NLP | 51 | 情感分析、文本分類、BERT微調、GPT文本生成等 |
| 推薦系統 | 56 | 電影推薦、產品推薦、神經協同過濾、實時推薦等 |
| 計算機視覺 | 50 | 圖像分類、物體檢測、Vision Transformer、Stable Diffusion等 |
| 聚類分析 | 60 | 客戶分群、市場細分、深度聚類、多視圖聚類等 |
| 特殊領域 | 65 | 醫療診斷、金融風控、藥物發現、智能制造等 |
| 深度學習 | 65 | 神經架構搜索、元學習、少樣本學習、因果表示等 |
| 音訊信號 | 60 | 語音識別、Wav2Vec、Whisper轉錄、音頻深偽檢測等 |
| 異常檢測 | 59 | 欺詐檢測、質量控制、自編碼異常、在線異常檢測等 |
| 圖神經網絡 | 59 | 社交網絡、GAT注意力、GraphSAGE、時序GNN等 |
| 地理空間 | 59 | 地圖分析、衛星圖像、路徑優化、位置智能等 |
| 特徵工程 | 64 | 特徵選擇、自動特徵工程、目標編碼、實體嵌入等 |
| 集成學習 | 64 | Stacking、Boosting、動態集成、集成剪枝等 |
| 貝葉斯方法 | 59 | 概率建模、貝葉斯優化、高斯過程、MCMC採樣等 |
| 優化算法 | 59 | 遺傳算法、粒子群、多目標優化、約束優化等 |
| 多模態學習 | 54 | 圖文融合、視覺問答、圖像描述、跨模態檢索等 |

## 🛠️ 實用工具

### 驗證解決方案質量

```bash
python scripts/validate_solutions.py
```

這會檢查：
- Python語法正確性
- 文件完整性
- 代碼風格
- 文檔質量

### 生成缺失的README

```bash
python scripts/generate_missing_readmes.py
```

### 配置Kaggle API

```python
from data_analysis_chatbots import setup_kaggle_credentials

# 交互式設置Kaggle憑證
setup_kaggle_credentials()
```

## 📚 更多資源

- [完整文檔](README.md) - 詳細的專案說明
- [架構設計](ARCHITECTURE.md) - 系統架構文檔
- [FAQ](FAQ.md) - 常見問題解答
- [教程](TUTORIAL.md) - 詳細教程
- [Kaggle快速入門](docs/KAGGLE_QUICKSTART.md) - Kaggle數據集使用指南

## 🤝 需要幫助？

1. 查看 [FAQ.md](FAQ.md) 常見問題
2. 閱讀 [TUTORIAL.md](TUTORIAL.md) 詳細教程
3. 運行 `python scripts/quick_test.py` 診斷問題
4. 查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何貢獻

## 🎉 開始您的數據分析之旅！

選擇一個Kaggle解決方案開始：

```bash
# 推薦新手從這些開始
cd kaggle_solutions/01_structured_data/01_customer_churn_prediction
python solution.py
```

祝您使用愉快！ 🚀
