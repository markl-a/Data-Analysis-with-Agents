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
- 802個Kaggle解決方案的分布
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

## 📊 802個Kaggle解決方案一覽

本專案包含802個完整的Kaggle解決方案，分為17個類別：

| 類別 | 數量 | 主要內容 |
|------|------|----------|
| 結構化數據 | 39 | 客戶流失、信用評分、保險定價等 |
| 時間序列 | 55 | 銷售預測、股價預測、能源消耗等 |
| NLP | 39 | 情感分析、文本分類、機器翻譯等 |
| 推薦系統 | 44 | 電影推薦、產品推薦、新聞推薦等 |
| 計算機視覺 | 38 | 圖像分類、物體檢測、面部識別等 |
| 聚類分析 | 48 | 客戶分群、市場細分、基因聚類等 |
| 特殊領域 | 53 | 醫療診斷、金融風控、法律預測等 |
| 深度學習 | 53 | 神經網絡、元學習、遷移學習等 |
| 音訊信號 | 48 | 語音識別、音樂分類、說話人識別等 |
| 異常檢測 | 47 | 欺詐檢測、質量控制、網絡安全等 |
| 圖神經網絡 | 47 | 社交網絡、知識圖譜、分子預測等 |
| 地理空間 | 47 | 地圖分析、路徑規劃、災害預測等 |
| 特徵工程 | 52 | 特徵選擇、特徵轉換、嵌入表示等 |
| 集成學習 | 52 | Stacking、Boosting、Bagging等 |
| 貝葉斯方法 | 47 | 概率建模、A/B測試、因果推斷等 |
| 優化算法 | 47 | 遺傳算法、粒子群、模擬退火等 |
| 多模態學習 | 46 | 圖文融合、視聽結合、跨模態檢索等 |

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
