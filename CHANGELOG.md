# Changelog

所有重要的變更都會記錄在這個文件中。

本格式基於 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.0.0/)，
並且本專案遵循 [Semantic Versioning](https://semver.org/lang/zh-TW/)。

## [Unreleased]

### Added
- 完整的 CI/CD pipeline（GitHub Actions）
- Docker 容器化支持（Dockerfile, docker-compose.yml）
- Pre-commit hooks 配置
- 擴充的測試套件（單元測試、集成測試）
- Sphinx API 文檔系統
- 專案管理文檔（CONTRIBUTING.md, SECURITY.md, CODE_OF_CONDUCT.md）
- Makefile 簡化開發流程
- pyproject.toml 配置

### Changed
- 更新 requirements.txt，添加開發工具
- 優化專案結構

### Fixed
- [待添加]

## [1.0.0] - 2024-11-14

### Added
- **核心功能**
  - 數據預處理模塊（文本清洗、數據驗證）
  - RFM 分析模塊
  - K-Means 聚類模塊
  - CLV（客戶終身價值）預測模塊
  - 營銷活動管理模塊
  - 數據可視化模塊

- **Kaggle 解決方案**
  - 500 個機器學習解決方案
  - 涵蓋 17 個主要類別
  - 每個解決方案都有完整文檔

- **數據集支持**
  - NLP 災難推文
  - 電商交易數據
  - 購物中心客戶數據
  - 客戶人格分析數據
  - 營銷分群數據

- **文檔**
  - 完整的 README.md
  - 5 個詳細的教程文檔
  - 安裝指南（INSTALLATION.md）
  - 快速開始指南（QUICKSTART.md）

- **工具**
  - Streamlit 交互式儀表板
  - 數據下載器
  - 配置管理系統
  - CLI 工具

### Technical Details
- Python 3.8+ 支持
- 使用 scikit-learn 進行機器學習
- 使用 pandas/numpy 進行數據處理
- 使用 matplotlib/seaborn/plotly 進行可視化
- 使用 streamlit 構建 Web 應用

## [0.9.0] - 2024-11-01 (Beta)

### Added
- 基礎專案結構
- 核心分析功能原型
- 初始測試套件

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

---

## 版本說明

### [Unreleased]
正在開發中的功能，尚未發布。

### [1.0.0] - 2024-11-14
首個正式發布版本，包含完整的客戶分析框架和 500 個 Kaggle 解決方案。

### 變更類型

- `Added`: 新增功能
- `Changed`: 現有功能的變更
- `Deprecated`: 即將移除的功能
- `Removed`: 已移除的功能
- `Fixed`: Bug 修復
- `Security`: 安全性修復

## 貢獻

如果你發現任何問題或有改進建議，請：
1. 查看 [Issues](https://github.com/markl-a/Data-Analysis-with-Chatbots/issues)
2. 閱讀 [貢獻指南](CONTRIBUTING.md)
3. 提交 Pull Request

## 連結

- [專案首頁](https://github.com/markl-a/Data-Analysis-with-Chatbots)
- [問題追蹤](https://github.com/markl-a/Data-Analysis-with-Chatbots/issues)
- [發布頁面](https://github.com/markl-a/Data-Analysis-with-Chatbots/releases)
