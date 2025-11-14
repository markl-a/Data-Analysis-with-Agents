# 📊 Data Analysis with Chatbots

> **一個利用AI聊天機器人(ChatGPT, Gemini, Claude)進行客戶分析與分群的完整框架**

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🎯 專案概述

本專案提供一個全面的客戶分析框架,結合傳統數據科學方法與最新的AI技術(2024-2025年最佳實踐),幫助企業深入了解客戶行為、進行精準分群,並制定有效的營銷策略。

### ✨ 核心特色

- 🤖 **AI輔助分析** - 整合ChatGPT、Gemini、Claude等AI工具進行智能分析
- 📈 **完整分析流程** - 從數據清洗到策略制定的端到端解決方案
- 🎨 **豐富可視化** - 多樣化的圖表和交互式儀表板
- 🔧 **模塊化設計** - 易於擴展和定制的架構
- 📚 **實戰案例** - 5個真實數據集的完整分析範例
- 🚀 **最佳實踐** - 採用2024-2025年行業最新標準

### 🆕 2024-2025年最新特性

根據最新行業研究:

- **AI驅動分群** - 使用AI工具可減少75%分析時間,提升95%準確度
- **動態實時分析** - 機器學習模型自動更新客戶分群
- **隱私優先** - 95%客戶更信任重視數據隱私的品牌
- **多算法支持** - 除K-means外,支持DBSCAN、GMM、Fuzzy C-Means等先進算法
- **進階CLV預測** - 結合RFM與機器學習的混合方法

## 📁 專案結構

```
Data-Analysis-with-Chatbots/
│
├── docs/                           # 📖 文檔
│   ├── 01_data_cleaning.md        # 數據清洗指南
│   ├── 02_customer_segmentation.md # 客戶分群分析
│   ├── 03_mall_customer_analysis.md # 購物中心會員分析
│   ├── 04_personality_analysis.md  # 客戶人格分析
│   └── 05_marketing_segmentation.md # 營銷分群策略
│
├── notebooks/                      # 📓 Jupyter Notebooks
│   └── (待添加互動式演示)
│
├── src/data_analysis_chatbots/    # 💻 源代碼
│   ├── config_loader.py           # 配置管理
│   ├── data_loader.py             # 數據加載器
│   ├── data_downloader.py         # 數據下載工具
│   ├── preprocessing/             # 數據預處理模塊
│   ├── clustering/                # 聚類分析模塊
│   ├── visualization/             # 可視化模塊
│   └── marketing/                 # 營銷分析模塊
│
├── data/                          # 📦 數據目錄
│   ├── raw/                       # 原始數據
│   ├── processed/                 # 處理後數據
│   └── outputs/                   # 分析結果
│
├── config/config.yaml             # ⚙️ 配置文件
├── requirements.txt               # Python依賴
├── setup.py                       # 專案安裝配置
└── README.md                      # 本文檔
```

## 🚀 快速開始

### 環境要求

- Python 3.8或更高版本
- pip包管理器
- (可選) Jupyter Notebook
- (可選) Kaggle API憑證

### 安裝步驟

1. **克隆專案**

```bash
git clone https://github.com/markl-a/Data-Analysis-with-Chatbots.git
cd Data-Analysis-with-Chatbots
```

2. **創建虛擬環境**(推薦)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **安裝依賴**

```bash
pip install -r requirements.txt
```

4. **安裝專案包**

```bash
pip install -e .
```

5. **下載數據集**(可選)

```bash
# 下載所有數據集
python -m data_analysis_chatbots.data_downloader --all

# 或下載特定數據集
python -m data_analysis_chatbots.data_downloader --dataset mall_customers

# 或創建範例數據用於測試
python -m data_analysis_chatbots.data_downloader --sample
```

### 基本使用

```python
from data_analysis_chatbots import DataLoader, KMeansClusterer, Plotter

# 加載數據
loader = DataLoader()
df = loader.load_mall_customers()

# 執行K-means聚類
clusterer = KMeansClusterer(n_clusters=5)
labels = clusterer.fit_predict(df, ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])

# 可視化結果
plotter = Plotter()
plotter.plot_clusters(df, 'Annual Income (k$)', 'Spending Score (1-100)', 'Cluster')
```

## 📊 數據集

專案包含5個真實世界的數據集:

| # | 數據集 | 來源 | 用途 |
|---|--------|------|------|
| 1 | [NLP災難推文](https://www.kaggle.com/datasets/vbmokin/nlp-with-disaster-tweets-cleaning-data) | Kaggle | 文本清洗與NLP |
| 2 | [電商交易](https://www.kaggle.com/datasets/carrie1/ecommerce-data) | Kaggle | RFM分析 |
| 3 | [購物中心客戶](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) | Kaggle | K-means聚類 |
| 4 | [客戶人格](https://github.com/g-aditi/customer-personality-analysis) | GitHub | CLV預測 |
| 5 | [營銷分群](https://www.kaggle.com/datasets/fahmidachowdhury/customer-segmentation-data-for-marketing-analysis) | Kaggle | 營銷策略 |

## 🎓 核心功能模塊

### 1. 數據預處理

```python
from data_analysis_chatbots.preprocessing import TextCleaner, DataValidator

# 文本清洗
cleaner = TextCleaner(lowercase=True, remove_urls=True)
clean_text = cleaner.clean_text("Check this out: http://example.com @user")

# 數據驗證
validator = DataValidator(df)
report = validator.generate_report()
```

### 2. RFM分析

```python
from data_analysis_chatbots.clustering import RFMAnalyzer

rfm = RFMAnalyzer(df, 'CustomerID', 'InvoiceDate', 'TotalAmount')
rfm_data = rfm.calculate_rfm()
segments = rfm.segment_customers()
summary = rfm.get_segment_summary()
```

### 3. K-Means聚類

```python
from data_analysis_chatbots.clustering import KMeansClusterer

clusterer = KMeansClusterer(n_clusters=5)
labels = clusterer.fit_predict(df, ['Age', 'Income', 'Spending'])
metrics = clusterer.evaluate_clustering()
```

### 4. CLV預測

```python
from data_analysis_chatbots.marketing import CLVPredictor

clv_predictor = CLVPredictor(discount_rate=0.1)
clv_results = clv_predictor.calculate_rfm_based_clv(rfm_data)
summary = clv_predictor.get_clv_summary(clv_results)
```

### 5. 營銷活動管理

```python
from data_analysis_chatbots.marketing import CampaignManager

campaign_mgr = CampaignManager(customer_df, 'CustomerID')
targeted = campaign_mgr.create_campaign(
    'VIP促銷',
    {'Income': {'min': 70}, 'Spending': {'min': 60}}
)
roi = campaign_mgr.calculate_campaign_roi('VIP促銷', 50, 0.15, 500)
```

## 📖 詳細文檔

請查看[docs/](docs/)目錄獲取完整文檔:

- [01_data_cleaning.md](docs/01_data_cleaning.md) - 數據清洗完整指南
- [02_customer_segmentation.md](docs/02_customer_segmentation.md) - RFM與電商分析
- [03_mall_customer_analysis.md](docs/03_mall_customer_analysis.md) - K-means聚類實戰
- [04_personality_analysis.md](docs/04_personality_analysis.md) - 客戶人格與CLV
- [05_marketing_segmentation.md](docs/05_marketing_segmentation.md) - 營銷策略制定

## 🔧 配置

編輯`config/config.yaml`以自定義分析參數:

```yaml
analysis:
  clustering:
    n_clusters_range: [2, 3, 4, 5, 6, 7, 8]
    random_state: 42
  rfm:
    recency_bins: [0, 30, 90, 180, 365, 9999]
  clv:
    discount_rate: 0.1
    time_horizon_years: 3
```

## 📚 參考資源

### 相關文章
- [利用集群分析掌握消費者輪廓](https://medium.com/finformation%E7%95%B6%E7%A8%8B%E5%BC%8F%E9%81%87%E4%B8%8A%E8%B2%A1%E5%8B%99%E9%87%91%E8%9E%8D/%E5%88%A9%E7%94%A8%E9%9B%86%E7%BE%A4%E5%88%86%E6%9E%90%E6%8E%8C%E6%8F%A1%E6%B6%88%E8%B2%BB%E8%80%85%E8%BC%AA%E5%BB%93-python%E5%AF%A6%E4%BD%9C-%E4%B8%80-7086082fbb2e)

### 學術研究
- Estimating Customer Lifetime Value Based on RFM Analysis
- Machine Learning Algorithms for Customer Relationship Management

### 行業最佳實踐(2024-2025)
- AI驅動分群減少75%分析時間,提升95%準確度
- 動態實時客戶分析
- 隱私優先的數據處理

## 📄 許可證

本專案採用MIT許可證 - 詳見[LICENSE](LICENSE)文件

## 👤 作者

**賴祺清**

## 🙏 致謝

- Kaggle社區提供的優質數據集
- Scikit-learn、Pandas、Matplotlib等開源專案
- ChatGPT、Gemini、Claude等AI工具的啟發

---

⭐ 如果這個專案對你有幫助,請給個星星! ⭐

**最後更新:** 2024年11月14日
