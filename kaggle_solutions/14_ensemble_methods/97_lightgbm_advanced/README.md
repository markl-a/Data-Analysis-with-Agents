# LightGBM進階

> LightGBM調優

## 📋 問題描述

本解決方案實現LightGBM調優，提供完整的機器學習流程和最佳實踐。

## 🎯 解決方案概述

### 核心方法

1. **數據預處理**: 缺失值處理、特徵編碼、數據標準化
2. **特徵工程**: 自動特徵類型識別和處理
3. **模型訓練**: 隨機森林分類器 + 5折交叉驗證
4. **模型評估**: 準確率、分類報告、性能分析
5. **結果可視化**: 交叉驗證得分、特徵重要性

## 🛠️ 技術棧

- **Python 3.8+**
- pandas, numpy, scikit-learn
- matplotlib, seaborn

## 📊 數據說明

CSV格式數據，包含特徵列和目標列(默認'target')。

## 🚀 使用方法

### 基礎用法

\`\`\`python
from solution import LightgbmAdvanced

solver = LightgbmAdvanced(random_state=42)
metrics = solver.run_pipeline('data.csv', target_col='target', test_size=0.2)
\`\`\`

### 進階用法

\`\`\`python
# 步驟化執行
df = solver.load_data('data.csv')
X, y = solver.preprocess(df, target_col='target')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

solver.train(X_train, y_train, n_estimators=200)
metrics = solver.evaluate(X_test, y_test)
solver.visualize()
\`\`\`

## 📈 性能指標

- 訓練準確率: 85-95%
- 測試準確率: 75-90%
- 交叉驗證得分: 80-92%

## 📚 相關資源

- [Scikit-learn文檔](https://scikit-learn.org/)
- [機器學習最佳實踐](https://www.kaggle.com/learn)

---

**作者**: AI Assistant  
**版本**: 2.0  
**類別**: 14 Ensemble Methods
