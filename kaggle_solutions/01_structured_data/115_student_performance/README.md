# 115. 學生表現預測

## 項目概述

這是一個教育數據分析項目，基於學生的背景因素和學習行為預測學業表現，幫助識別可能需要額外支持的學生。

**Kaggle 數據集**: [Students Performance Dataset](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

**難度**: ⭐⭐ 中級

## 目標

預測學生的考試成績或學業表現等級（回歸/分類問題）

## 數據集描述

### 特徵說明

| 特徵 | 描述 | 類型 |
|------|------|------|
| gender | 性別 | 分類 |
| race/ethnicity | 種族/民族 | 分類 |
| parental level of education | 父母教育程度 | 分類 |
| lunch | 午餐類型 (標準/減免) | 分類 |
| test preparation course | 是否參加備考課程 | 分類 |
| math score | 數學成績 | 數值 |
| reading score | 閱讀成績 | 數值 |
| writing score | 寫作成績 | 數值 |

## 關鍵洞察

1. **備考課程**: 參加備考課程的學生平均成績更高
2. **父母教育**: 父母教育程度與學生成績正相關
3. **午餐類型**: 標準午餐學生成績普遍較高（家庭經濟指標）
4. **性別差異**: 女生在閱讀和寫作上表現更好
5. **科目相關性**: 三個科目成績高度相關

## 技術方法

### 數據預處理
- 類別變量編碼
- 特徵標準化
- 缺失值處理

### 特徵工程
- 總分和平均分
- 學業等級劃分
- 科目優勢指標
- 綜合表現評分

### 模型
- Linear Regression / Logistic Regression
- Random Forest
- XGBoost
- Support Vector Machine
- Neural Network

## 使用方法

```bash
python solution.py
```

## 預期結果

- R² (回歸): > 0.85
- 準確率 (分類): > 80%

---

**難度**: ⭐⭐ 中級
**預計完成時間**: 3 小時
**推薦給**: 對教育數據分析感興趣的學習者
