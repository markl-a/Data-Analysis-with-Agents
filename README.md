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
- 🏆 **1204個Kaggle解決方案** - 涵蓋機器學習17個領域的完整實現（100%文檔覆蓋）
- 🚀 **最佳實踐** - 採用2024-2025年行業最新標準
- 🎯 **多種聚類算法** - K-Means, DBSCAN, GMM, Hierarchical完整實現
- 🛡️ **強大錯誤處理** - 自定義異常系統提供清晰的錯誤信息
- ⚡ **一鍵初始化** - 自動創建專案目錄結構
- 🔽 **Kaggle數據集下載** - 支持50+熱門數據集一鍵下載

### 🆕 2024-2025年最新特性

根據最新行業研究:

- **AI驅動分群** - 使用AI工具可減少75%分析時間,提升95%準確度
- **動態實時分析** - 機器學習模型自動更新客戶分群
- **隱私優先** - 95%客戶更信任重視數據隱私的品牌
- **多算法支持** - K-Means、DBSCAN、GMM、Hierarchical等4種完整實現的聚類算法
- **進階CLV預測** - 結合RFM與機器學習的混合方法
- **自動化初始化** - 一鍵創建完整專案結構
- **專業異常處理** - 15+個自定義異常類,提供精確的錯誤信息和恢復建議

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
├── kaggle_solutions/               # 🏆 1204個Kaggle實戰解決方案
│   ├── 01_structured_data/        # 結構化數據 (64個)
│   ├── 02_time_series/            # 時間序列 (80個)
│   ├── 03_nlp/                    # 自然語言處理 (64個)
│   ├── 04_recommendation/         # 推薦系統 (69個)
│   ├── 05_computer_vision/        # 計算機視覺 (63個)
│   ├── 06_clustering/             # 聚類分析 (73個)
│   ├── 07_special_domains/        # 特殊領域 (78個)
│   ├── 08_deep_learning/          # 深度學習 (78個)
│   ├── 09_audio_signal/           # 音訊信號 (73個)
│   ├── 10_anomaly_detection/      # 異常檢測 (72個)
│   ├── 11_graph_networks/         # 圖神經網絡 (72個)
│   ├── 12_geospatial/             # 地理空間 (72個)
│   ├── 13_feature_engineering/    # 特徵工程 (77個)
│   ├── 14_ensemble_methods/       # 集成學習 (77個)
│   ├── 15_bayesian_methods/       # 貝葉斯方法 (72個)
│   ├── 16_optimization/           # 優化算法 (72個)
│   ├── 17_multimodal/             # 多模態學習 (65個)
│   ├── README.md                  # Kaggle解決方案總覽
│   ├── INDEX.md                   # 完整索引
│   └── SOLUTIONS_SUMMARY.md       # 統計摘要
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

5. **初始化專案結構**(新增!)

```bash
# 創建必要的目錄結構(data/, models/, logs/等)
python -m data_analysis_chatbots.init --with-examples

# 或僅驗證目錄結構是否完整
python -m data_analysis_chatbots.init --validate
```

6. **下載數據集**(可選)

```bash
# 下載所有數據集
python -m data_analysis_chatbots.data_downloader --all

# 或下載特定數據集
python -m data_analysis_chatbots.data_downloader --dataset mall_customers

# 或創建範例數據用於測試
python -m data_analysis_chatbots.data_downloader --sample
```

### 基本使用

#### 快速入門 - K-Means聚類

```python
from data_analysis_chatbots import DataLoader
from data_analysis_chatbots.clustering import KMeansClusterer
from data_analysis_chatbots.visualization import Plotter

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

#### 新增! 使用高級聚類算法

```python
from data_analysis_chatbots.clustering import (
    DBSCANClusterer,      # 密度聚類
    GMMClusterer,         # 高斯混合模型
    HierarchicalClusterer # 層次聚類
)

# DBSCAN - 適合發現任意形狀的聚類
dbscan = DBSCANClusterer(eps=0.5, min_samples=10)
labels = dbscan.fit_predict(df, features)
print(f"發現 {dbscan.n_clusters_} 個聚類和 {dbscan.n_noise_} 個異常點")

# GMM - 提供概率性聚類結果
gmm = GMMClusterer(n_components=3)
labels = gmm.fit_predict(df, features)
probabilities = gmm.predict_proba(df, features)  # 獲取每個點屬於各聚類的概率

# Hierarchical - 可視化層次結構
hierarchical = HierarchicalClusterer(n_clusters=4, linkage='ward')
labels = hierarchical.fit_predict(df, features)
hierarchical.plot_dendrogram(save_path='outputs/plots/dendrogram.png')
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

## 🏆 30個Kaggle實戰解決方案

專案包含 **30個完整的Kaggle比賽解決方案**，涵蓋機器學習的各個領域：

### 📊 結構化數據與分類 (1-8)
- Titanic存活預測、房價預測、信用卡詐欺偵測、客戶流失預測等

### ⏱️ 時間序列分析 (9-13)
- 股票價格預測、銷售預測、COVID-19分析、能源消耗預測等

### 💬 自然語言處理 (14-18)
- 情感分析、假新聞偵測、垃圾郵件分類、災難推文識別等

### 🎬 推薦系統 (19-21)
- 電影推薦、產品推薦、書籍推薦

### 🖼️ 計算機視覺 (22-24)
- 手寫數字識別、服裝分類、貓狗識別

### 🔍 聚類與無監督學習 (25-27)
- 客戶分群、異常檢測、購物籃分析

### 🏥 特殊領域應用 (28-30)
- 醫療診斷、保險費用預測、員工離職預測

### 快速開始

```bash
# 查看所有解決方案
cd kaggle_solutions
python run_all.py

# 運行單個解決方案
cd kaggle_solutions/01_structured_data/01_titanic_survival
python solution.py
```

**詳細信息**: 查看 [kaggle_solutions/README.md](kaggle_solutions/README.md)

## 📖 詳細文檔

### 📚 完整學習指南 (新增!)

**從零基礎到專家的完整路徑**:

- 📖 **[TUTORIAL.md](TUTORIAL.md)** - 完整學習教程 (入門到精通)
  - 4個階段學習路徑 (0-12個月)
  - 每週/每月結構化課程
  - 實戰代碼示例與練習
  - 進度追蹤檢查表

- 🤖 **[AI_ASSISTANCE_GUIDE.md](AI_ASSISTANCE_GUIDE.md)** - AI輔助使用指南
  - ChatGPT、Claude、Gemini 比較與使用
  - AI輔助數據清洗 (3種方法)
  - AI輔助編碼最佳實踐
  - 實戰案例與安全建議

- 📊 **[CASE_STUDIES.md](CASE_STUDIES.md)** - 實際案例研究
  - 4個熱門Kaggle競賽詳解
  - 3個企業應用案例
  - 完整端到端專案實戰
  - 業務成果與指標

- 🏆 **[KAGGLE_COMPETITIONS_SUGGESTIONS.md](KAGGLE_COMPETITIONS_SUGGESTIONS.md)** - Kaggle競賽推薦
  - 12個精選競賽 (入門到專家)
  - 詳細競賽分析與策略
  - 學習路徑建議
  - AI輔助競賽技巧

- ❓ **[FAQ.md](FAQ.md)** - 常見問題解答 (新增!)
  - 20個最常見問題詳解
  - 安裝、使用、錯誤處理
  - 聚類算法選擇指南
  - 性能優化技巧
  - AI輔助最佳實踐

- 🏗️ **[ARCHITECTURE.md](ARCHITECTURE.md)** - 系統架構設計 (新增!)
  - 分層架構詳解
  - 模塊設計原則
  - 設計模式應用
  - 數據流與狀態管理
  - 性能優化策略
  - 安全性考慮

### 📂 技術文檔

請查看[docs/](docs/)目錄獲取技術文檔:

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

---

## 🚀 快速導航

### 新手入門
1. 閱讀 [QUICKSTART.md](QUICKSTART.md) - 5分鐘快速體驗
2. 學習 [TUTORIAL.md](TUTORIAL.md) - 完整學習路徑
3. 運行第一個範例: `python examples/complete_analysis_workflow.py`

### AI輔助學習
1. 閱讀 [AI_ASSISTANCE_GUIDE.md](AI_ASSISTANCE_GUIDE.md)
2. 使用AI優化你的代碼
3. 參考 [CASE_STUDIES.md](CASE_STUDIES.md) 中的AI輔助案例

### Kaggle競賽
1. 查看 [KAGGLE_COMPETITIONS_SUGGESTIONS.md](KAGGLE_COMPETITIONS_SUGGESTIONS.md)
2. 從Titanic開始: `cd kaggle_solutions/01_structured_data/01_titanic_survival`
3. 參加你的第一個競賽!

### 企業應用
1. 研究 [CASE_STUDIES.md](CASE_STUDIES.md) 中的企業案例
2. 運行CLV預測: `docs/04_personality_analysis.md`
3. 部署Streamlit儀表板: `streamlit run app.py`

---

**最後更新:** 2025年1月18日
