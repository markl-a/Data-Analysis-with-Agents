# 更新日誌 (Changelog)

所有重要更改都將記錄在此文件中。

格式基於 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/),
並且本專案遵循 [語義化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 新增 (Added)
- 🎯 **3個高級聚類算法**
  - DBSCAN聚類器 - 密度基礎聚類,自動檢測異常點
  - GMM聚類器 - 高斯混合模型,提供概率性軟聚類
  - Hierarchical聚類器 - 層次聚類,支持樹狀圖可視化

- 🛡️ **自定義異常系統** (15個異常類)
  - DataLoadError, ValidationError, ClusteringError等
  - 便捷的驗證函數 (raise_if_*)

- ⚡ **專案初始化工具** (init.py)
  - 一鍵創建完整目錄結構
  - 自動生成README說明文件

- 💾 **模型管理工具** (model_utils.py)
  - 統一的模型保存/加載接口
  - 模型註冊表管理
  - 元數據追蹤和版本控制

- 📊 **增強的CLI工具**
  - 支持所有4種聚類算法
  - 算法特定參數配置

- 📚 **擴充文檔**
  - FAQ.md - 20個常見問題
  - ARCHITECTURE.md - 系統架構設計
  - CHANGELOG.md - 更新日誌

### 修復 (Fixed)
- 🐛 修復 cli.py:47 的bug

## [1.0.0] - 2025-01-17

初始版本發布
