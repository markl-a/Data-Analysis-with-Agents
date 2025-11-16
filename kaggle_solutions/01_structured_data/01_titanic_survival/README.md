# 01. Titanic 存活預測

## 📋 項目概述

這是Kaggle最經典的入門競賽，目標是根據乘客的各種特徵預測他們在鐵達尼號災難中的存活情況。

**Kaggle競賽**: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic)

**難度**: ⭐ 入門級

## 🎯 目標

預測鐵達尼號乘客是否存活（二元分類問題）

## 📊 數據集描述

### 特徵說明

| 特徵 | 描述 | 類型 |
|------|------|------|
| PassengerId | 乘客ID | 數值 |
| Survived | 是否存活 (0=未存活, 1=存活) | 目標變量 |
| Pclass | 艙位等級 (1=頭等, 2=二等, 3=三等) | 分類 |
| Name | 姓名 | 文本 |
| Sex | 性別 | 分類 |
| Age | 年齡 | 數值 |
| SibSp | 同行的兄弟姐妹/配偶數量 | 數值 |
| Parch | 同行的父母/子女數量 | 數值 |
| Ticket | 票號 | 文本 |
| Fare | 票價 | 數值 |
| Cabin | 客艙號 | 文本 |
| Embarked | 登船港口 (C=Cherbourg, Q=Queenstown, S=Southampton) | 分類 |

### 數據集大小
- 訓練集: 891 筆
- 測試集: 418 筆

## 🔍 關鍵洞察

1. **女性優先**: 女性存活率顯著高於男性
2. **艙位等級**: 頭等艙乘客存活率更高
3. **年齡**: 兒童存活率較高
4. **家庭規模**: 中等家庭規模的存活率較高

## 🛠️ 技術方法

### 數據預處理
- 缺失值處理（Age, Embarked, Fare）
- 特徵工程：
  - FamilySize = SibSp + Parch + 1
  - IsAlone = (FamilySize == 1)
  - Age_Group: 年齡分組
  - Fare_Group: 票價分組

### 模型
- 隨機森林分類器（Random Forest Classifier）
- 超參數：
  - n_estimators=100
  - max_depth=5
  - min_samples_split=10

### 評估指標
- 準確率（Accuracy）
- 精確率（Precision）
- 召回率（Recall）
- F1分數

## 🚀 使用方法

### 運行分析

```bash
python solution.py
```

### 使用真實數據

1. 從Kaggle下載數據集
2. 修改代碼讀取實際CSV文件：

```python
# 替換 create_sample_data() 為：
df = pd.read_csv('train.csv')
```

## 📈 預期結果

- 準確率: ~78-82%
- 交叉驗證分數: ~0.80

## 💡 改進建議

1. **更多特徵工程**
   - 從Name中提取頭銜（Mr, Mrs, Miss等）
   - 使用Cabin的首字母作為特徵
   - 票價與艙位的交互特徵

2. **高級模型**
   - Gradient Boosting (XGBoost, LightGBM)
   - 神經網絡
   - 集成學習（Ensemble）

3. **超參數優化**
   - Grid Search
   - Random Search
   - Bayesian Optimization

## 📚 學習要點

通過這個項目，你將學會：
- ✅ 基本的數據探索與可視化
- ✅ 處理缺失值的方法
- ✅ 特徵工程的重要性
- ✅ 分類模型的訓練與評估
- ✅ 交叉驗證技術
- ✅ 模型性能解釋

## 🔗 資源

- [Kaggle競賽頁面](https://www.kaggle.com/c/titanic)
- [數據集下載](https://www.kaggle.com/c/titanic/data)
- [教程與Notebooks](https://www.kaggle.com/c/titanic/notebooks)

## 📝 注意事項

- 本代碼使用範例數據進行演示
- 實際競賽需要下載真實數據集
- 特徵工程是提高分數的關鍵
- 建議使用Kaggle Notebook進行實驗

---

**難度**: ⭐ 入門級
**預計完成時間**: 2-3 小時
**推薦給**: 機器學習初學者
