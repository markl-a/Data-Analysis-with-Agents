# 112. 糖尿病預測分析

## 項目概述

這是一個基於患者健康指標預測糖尿病風險的分類項目。使用機器學習方法分析患者的生理數據，預測其是否可能患有糖尿病。

**Kaggle 數據集**: [Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)

**難度**: ⭐⭐ 中級

## 目標

基於患者的健康指標（如血糖水平、BMI、年齡等）預測糖尿病風險（二元分類問題）

## 數據集描述

### 特徵說明

| 特徵 | 描述 | 類型 |
|------|------|------|
| gender | 性別 | 分類 |
| age | 年齡 | 數值 |
| hypertension | 是否有高血壓 (0=否, 1=是) | 二元 |
| heart_disease | 是否有心臟病 (0=否, 1=是) | 二元 |
| smoking_history | 吸煙史 | 分類 |
| bmi | 身體質量指數 | 數值 |
| HbA1c_level | 糖化血紅蛋白水平 | 數值 |
| blood_glucose_level | 血糖水平 | 數值 |
| diabetes | 是否患有糖尿病 (0=否, 1=是) | 目標變量 |

### 數據集大小
- 總樣本數: ~100,000 筆
- 特徵數: 8 個
- 目標變量: 1 個（二元分類）

## 關鍵洞察

1. **HbA1c 水平**: 糖化血紅蛋白水平是最強的預測因子
2. **血糖水平**: 空腹血糖水平與糖尿病風險高度相關
3. **BMI**: 肥胖（BMI > 30）顯著增加糖尿病風險
4. **年齡**: 糖尿病風險隨年齡增長而增加
5. **共病症**: 高血壓和心臟病與糖尿病有較強關聯

## 技術方法

### 數據預處理
- 缺失值處理
- 類別變量編碼（性別、吸煙史）
- 數值特徵標準化
- 處理類別不平衡（SMOTE）

### 特徵工程
- BMI 分類（正常、超重、肥胖）
- 年齡分組
- 風險因子組合特徵
- HbA1c 和血糖的交互特徵

### 模型
- Logistic Regression（基線模型）
- Random Forest Classifier
- XGBoost Classifier
- LightGBM Classifier
- 集成學習（Stacking）

### 評估指標
- 準確率（Accuracy）
- 精確率（Precision）
- 召回率（Recall）
- F1 分數
- ROC-AUC
- 混淆矩陣

## 使用方法

### 運行分析

```bash
python solution.py
```

### 使用真實數據

1. 從 Kaggle 下載數據集
2. 修改代碼讀取實際 CSV 文件：

```python
# 替換 create_sample_data() 為：
df = pd.read_csv('diabetes_prediction_dataset.csv')
```

## 預期結果

- 準確率: ~95-97%
- ROC-AUC: ~0.96-0.98
- 召回率（對糖尿病患者）: ~90%+

## 改進建議

1. **更多特徵工程**
   - 添加家族病史特徵
   - 運動習慣和飲食特徵
   - 實驗室檢測指標（如胰島素水平）

2. **高級模型**
   - 神經網絡（Deep Learning）
   - CatBoost（處理類別特徵）
   - 貝葉斯優化超參數

3. **模型解釋性**
   - SHAP 值分析
   - LIME 局部解釋
   - 特徵重要性可視化

## 學習要點

通過這個項目，你將學會：
- 醫療健康數據的預處理技巧
- 處理類別不平衡問題
- 多種分類算法的比較
- 模型評估與選擇
- 醫療診斷模型的特殊考慮（召回率優先）
- 可解釋性分析

## 資源

- [Kaggle 數據集頁面](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)
- [WHO 糖尿病指南](https://www.who.int/health-topics/diabetes)
- [美國糖尿病協會](https://diabetes.org/)

## 注意事項

- 本代碼使用範例數據進行演示
- 實際應用需要下載真實數據集
- 醫療診斷模型需要專業驗證
- 模型結果僅供參考，不能替代專業醫療診斷

---

**難度**: ⭐⭐ 中級
**預計完成時間**: 3-4 小時
**推薦給**: 對醫療健康數據分析感興趣的學習者
