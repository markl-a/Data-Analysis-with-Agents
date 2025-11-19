# Kaggle 解決方案索引

本專案包含 **500 個完整的機器學習解決方案**，涵蓋 17 個主要類別。每個解決方案都包含完整的實現代碼和詳細文檔。

## 📊 統計概覽

- **總解決方案數**: 500
- **文檔完整度**: 100% (500/500 有 README)
- **類別數量**: 17
- **程式語言**: Python 3.8+
- **主要框架**: Scikit-learn, TensorFlow, PyTorch, XGBoost, LightGBM

## 🗂️ 類別導航

### 1️⃣ 結構化數據與分類 (01_structured_data) - 20 個解決方案

經典的結構化數據分類問題，涵蓋金融、醫療、電商等多個領域。

**代表性解決方案**:
- [01_titanic_survival](01_structured_data/01_titanic_survival/) - Titanic 生存預測（入門級）
- [02_house_prices](01_structured_data/02_house_prices/) - 房價預測
- [03_credit_fraud](01_structured_data/03_credit_fraud/) - 信用卡欺詐檢測
- [04_customer_churn](01_structured_data/04_customer_churn/) - 客戶流失預測
- [05_bank_marketing](01_structured_data/05_bank_marketing/) - 銀行營銷活動預測

**完整列表**: [查看全部 20 個解決方案 →](01_structured_data/)

---

### 2️⃣ 時間序列分析 (02_time_series) - 35 個解決方案

從基礎統計方法到深度學習，全面覆蓋時間序列預測技術。

**代表性解決方案**:
- [06_bitcoin_price](02_time_series/06_bitcoin_price/) - 比特幣價格預測
- [10_sales_forecasting](02_time_series/10_sales_forecasting/) - 銷售預測
- [16_arima_model_selection](02_time_series/16_arima_model_selection/) - ARIMA 模型選擇
- [18_prophet_forecasting](02_time_series/18_prophet_forecasting/) - Prophet 預測
- [20_lstm_sequence_prediction](02_time_series/20_lstm_sequence_prediction/) - LSTM 序列預測
- [23_transformer_timeseries](02_time_series/23_transformer_timeseries/) - Transformer 時間序列

**技術覆蓋**: ARIMA, SARIMA, Prophet, LSTM, GRU, Transformer, 動態時間規整

**完整列表**: [查看全部 35 個解決方案 →](02_time_series/)

---

### 3️⃣ 自然語言處理 (03_nlp) - 20 個解決方案

涵蓋文本分類、情感分析、命名實體識別、問答系統等 NLP 核心任務。

**代表性解決方案**:
- [14_sentiment_analysis](03_nlp/14_sentiment_analysis/) - 情感分析
- [06_question_answering](03_nlp/06_question_answering/) - 問答系統
- [07_named_entity_recognition](03_nlp/07_named_entity_recognition/) - 命名實體識別
- [08_text_summarization](03_nlp/08_text_summarization/) - 文本摘要
- [11_toxic_comment](03_nlp/11_toxic_comment/) - 有毒評論檢測
- [15_fake_news_detection](03_nlp/15_fake_news_detection/) - 假新聞檢測

**技術覆蓋**: BERT, Transformers, TF-IDF, Word2Vec, LSTM, 主題建模

**完整列表**: [查看全部 20 個解決方案 →](03_nlp/)

---

### 4️⃣ 推薦系統 (04_recommendation) - 25 個解決方案

從協同過濾到深度學習，全面覆蓋推薦系統技術棧。

**代表性解決方案**:
- [19_movie_recommendation](04_recommendation/19_movie_recommendation/) - 電影推薦
- [11_user_based_collaborative_filtering](04_recommendation/11_user_based_collaborative_filtering/) - 基於用戶的協同過濾
- [12_item_based_collaborative_filtering](04_recommendation/12_item_based_collaborative_filtering/) - 基於物品的協同過濾
- [13_matrix_factorization](04_recommendation/13_matrix_factorization/) - 矩陣分解
- [15_neural_collaborative_filtering](04_recommendation/15_neural_collaborative_filtering/) - 神經協同過濾
- [23_session_based_recommendations](04_recommendation/23_session_based_recommendations/) - 基於會話的推薦

**技術覆蓋**: 協同過濾, 矩陣分解, 內容過濾, 混合推薦, 神經網絡推薦

**完整列表**: [查看全部 25 個解決方案 →](04_recommendation/)

---

### 5️⃣ 計算機視覺 (05_computer_vision) - 20 個解決方案

圖像分類、目標檢測、圖像分割等計算機視覺核心任務。

**代表性解決方案**:
- [22_digit_recognition](05_computer_vision/22_digit_recognition/) - MNIST 數字識別
- [23_fashion_classification](05_computer_vision/23_fashion_classification/) - Fashion MNIST 分類
- [05_object_detection](05_computer_vision/05_object_detection/) - 目標檢測
- [06_image_segmentation](05_computer_vision/06_image_segmentation/) - 圖像分割
- [09_xray_pneumonia](05_computer_vision/09_xray_pneumonia/) - X光肺炎檢測
- [19_style_transfer](05_computer_vision/19_style_transfer/) - 風格遷移

**技術覆蓋**: CNN, ResNet, YOLO, U-Net, 遷移學習, GAN

**完整列表**: [查看全部 20 個解決方案 →](05_computer_vision/)

---

### 6️⃣ 聚類與無監督學習 (06_clustering) - 30 個解決方案

從 K-Means 到高級聚類算法的完整覆蓋。

**代表性解決方案**:
- [25_customer_segmentation](06_clustering/25_customer_segmentation/) - 客戶細分
- [11_kmeans_variants](06_clustering/11_kmeans_variants/) - K-Means 變體
- [19_dbscan_variants](06_clustering/19_dbscan_variants/) - DBSCAN 變體
- [21_hdbscan_clustering](06_clustering/21_hdbscan_clustering/) - HDBSCAN 聚類
- [23_gaussian_mixture](06_clustering/23_gaussian_mixture/) - 高斯混合模型
- [28_spectral_clustering](06_clustering/28_spectral_clustering/) - 譜聚類

**技術覆蓋**: K-Means, DBSCAN, GMM, 層次聚類, 譜聚類, 親和傳播

**完整列表**: [查看全部 30 個解決方案 →](06_clustering/)

---

### 7️⃣ 特殊領域應用 (07_special_domains) - 35 個解決方案

金融、醫療、氣候、量子計算等垂直領域的專業應用。

**代表性解決方案**:
- [04_fraud_detection](07_special_domains/04_fraud_detection/) - 欺詐檢測
- [21_portfolio_optimization](07_special_domains/21_portfolio_optimization/) - 投資組合優化
- [22_credit_risk_modeling](07_special_domains/22_credit_risk_modeling/) - 信用風險建模
- [26_protein_structure_prediction](07_special_domains/26_protein_structure_prediction/) - 蛋白質結構預測
- [27_climate_weather_forecasting](07_special_domains/27_climate_weather_forecasting/) - 氣候天氣預測
- [32_quantum_computing_simulation](07_special_domains/32_quantum_computing_simulation/) - 量子計算模擬

**領域覆蓋**: 金融科技, 醫療健康, 氣候科學, 生物信息, 量子計算, 社會科學

**完整列表**: [查看全部 35 個解決方案 →](07_special_domains/)

---

### 8️⃣ 深度學習 (08_deep_learning) - 35 個解決方案

深度學習前沿技術與高級架構。

**代表性解決方案**:
- [01_neural_style_transfer](08_deep_learning/01_neural_style_transfer/) - 神經風格遷移
- [02_gan_image_generation](08_deep_learning/02_gan_image_generation/) - GAN 圖像生成
- [16_resnet_skip_connections](08_deep_learning/16_resnet_skip_connections/) - ResNet 跳躍連接
- [20_vision_transformer](08_deep_learning/20_vision_transformer/) - Vision Transformer
- [26_knowledge_distillation](08_deep_learning/26_knowledge_distillation/) - 知識蒸餾
- [35_few_shot_learning](08_deep_learning/35_few_shot_learning/) - 少樣本學習

**技術覆蓋**: GAN, VAE, Transformer, 遷移學習, 自監督學習, 元學習

**完整列表**: [查看全部 35 個解決方案 →](08_deep_learning/)

---

### 9️⃣ 音訊與信號處理 (09_audio_signal) - 30 個解決方案

語音識別、音樂分類、聲音事件檢測等音訊處理任務。

**代表性解決方案**:
- [01_speech_emotion](09_audio_signal/01_speech_emotion/) - 語音情感識別
- [02_music_genre](09_audio_signal/02_music_genre/) - 音樂流派分類
- [11_mfcc_extraction](09_audio_signal/11_mfcc_extraction/) - MFCC 特徵提取
- [18_phoneme_recognition](09_audio_signal/18_phoneme_recognition/) - 音素識別
- [22_source_separation](09_audio_signal/22_source_separation/) - 音源分離
- [30_spatial_audio](09_audio_signal/30_spatial_audio/) - 空間音訊

**技術覆蓋**: MFCC, 譜特徵, CNN, RNN, WaveNet, 音訊增強

**完整列表**: [查看全部 30 個解決方案 →](09_audio_signal/)

---

### 🔟 異常檢測 (10_anomaly_detection) - 30 個解決方案

從統計方法到深度學習的異常檢測技術。

**代表性解決方案**:
- [11_zscore_modified_zscore_detection](10_anomaly_detection/11_zscore_modified_zscore_detection/) - Z-Score 檢測
- [16_lof_detection](10_anomaly_detection/16_lof_detection/) - 局部異常因子
- [20_isolation_forest_detection](10_anomaly_detection/20_isolation_forest_detection/) - 隔離森林
- [21_one_class_svm_detection](10_anomaly_detection/21_one_class_svm_detection/) - One-Class SVM
- [27_autoencoder_anomaly_detection](10_anomaly_detection/27_autoencoder_anomaly_detection/) - 自編碼器異常檢測
- [30_deep_svdd_detection](10_anomaly_detection/30_deep_svdd_detection/) - Deep SVDD

**技術覆蓋**: 統計方法, 基於距離, 基於密度, 孤立, 深度學習

**完整列表**: [查看全部 30 個解決方案 →](10_anomaly_detection/)

---

### 1️⃣1️⃣ 圖神經網絡 (11_graph_networks) - 30 個解決方案

圖結構數據的深度學習方法。

**代表性解決方案**:
- [11_gcn_node_classification](11_graph_networks/11_gcn_node_classification/) - GCN 節點分類
- [12_graph_attention_networks](11_graph_networks/12_graph_attention_networks/) - 圖注意力網絡
- [13_graphsage_inductive](11_graph_networks/13_graphsage_inductive/) - GraphSAGE 歸納學習
- [21_node2vec_embeddings](11_graph_networks/21_node2vec_embeddings/) - Node2Vec 嵌入
- [28_gnn_explainability](11_graph_networks/28_gnn_explainability/) - GNN 可解釋性
- [30_knowledge_graph_completion](11_graph_networks/30_knowledge_graph_completion/) - 知識圖譜補全

**技術覆蓋**: GCN, GAT, GraphSAGE, 圖嵌入, 時態圖網絡

**完整列表**: [查看全部 30 個解決方案 →](11_graph_networks/)

---

### 1️⃣2️⃣ 地理空間分析 (12_geospatial) - 30 個解決方案

空間數據索引、空間統計、地理可視化等 GIS 任務。

**代表性解決方案**:
- [11_rtree_spatial_indexing](12_geospatial/11_rtree_spatial_indexing/) - R-tree 空間索引
- [15_spatial_autocorrelation](12_geospatial/15_spatial_autocorrelation/) - 空間自相關
- [16_hotspot_analysis](12_geospatial/16_hotspot_analysis/) - 熱點分析
- [20_shortest_path](12_geospatial/20_shortest_path/) - 最短路徑
- [28_spatial_regression](12_geospatial/28_spatial_regression/) - 空間回歸
- [30_geospatial_deep_learning](12_geospatial/30_geospatial_deep_learning/) - 地理空間深度學習

**技術覆蓋**: 空間索引, 空間統計, 路徑規劃, DEM 分析, 地理編碼

**完整列表**: [查看全部 30 個解決方案 →](12_geospatial/)

---

### 1️⃣3️⃣ 特徵工程 (13_feature_engineering) - 35 個解決方案

數據預處理和特徵構建的系統化方法。

**代表性解決方案**:
- [09_polynomial_features](13_feature_engineering/09_polynomial_features/) - 多項式特徵
- [14_feature_interactions](13_feature_engineering/14_feature_interactions/) - 特徵交互
- [16_target_encoding](13_feature_engineering/16_target_encoding/) - 目標編碼
- [21_lag_rolling_features](13_feature_engineering/21_lag_rolling_features/) - 滯後滾動特徵
- [29_automated_featuretools](13_feature_engineering/29_automated_featuretools/) - 自動化特徵工程
- [35_autoencoder_features](13_feature_engineering/35_autoencoder_features/) - 自編碼器特徵

**技術覆蓋**: 編碼技術, 特徵變換, 時間特徵, 文本特徵, 自動化工具

**完整列表**: [查看全部 35 個解決方案 →](13_feature_engineering/)

---

### 1️⃣4️⃣ 集成學習方法 (14_ensemble_methods) - 35 個解決方案

從 Bagging 到 Stacking 的完整集成學習技術。

**代表性解決方案**:
- [09_extra_trees_analysis](14_ensemble_methods/09_extra_trees_analysis/) - 極度隨機樹
- [15_xgboost_advanced](14_ensemble_methods/15_xgboost_advanced/) - XGBoost 高級技術
- [16_lightgbm_optimization](14_ensemble_methods/16_lightgbm_optimization/) - LightGBM 優化
- [17_catboost_categorical](14_ensemble_methods/17_catboost_categorical/) - CatBoost 類別處理
- [19_multilayer_stacking](14_ensemble_methods/19_multilayer_stacking/) - 多層 Stacking
- [35_bayesian_model_averaging](14_ensemble_methods/35_bayesian_model_averaging/) - 貝葉斯模型平均

**技術覆蓋**: Bagging, Boosting, Stacking, Voting, 動態集成選擇

**完整列表**: [查看全部 35 個解決方案 →](14_ensemble_methods/)

---

### 1️⃣5️⃣ 貝葉斯方法 (15_bayesian_methods) - 30 個解決方案

貝葉斯統計和概率編程在機器學習中的應用。

**代表性解決方案**:
- [08_bayesian_linear_regression](15_bayesian_methods/08_bayesian_linear_regression/) - 貝葉斯線性回歸
- [11_hamiltonian_monte_carlo](15_bayesian_methods/11_hamiltonian_monte_carlo/) - 哈密頓蒙特卡洛
- [13_pymc3_hierarchical_models](15_bayesian_methods/13_pymc3_hierarchical_models/) - PyMC3 層次模型
- [18_gaussian_process_regression](15_bayesian_methods/18_gaussian_process_regression/) - 高斯過程回歸
- [22_variational_autoencoders](15_bayesian_methods/22_variational_autoencoders/) - 變分自編碼器
- [29_bayesian_deep_learning_uncertainty](15_bayesian_methods/29_bayesian_deep_learning_uncertainty/) - 貝葉斯深度學習不確定性

**技術覆蓋**: MCMC, 變分推斷, 高斯過程, 貝葉斯優化, 概率編程

**完整列表**: [查看全部 30 個解決方案 →](15_bayesian_methods/)

---

### 1️⃣6️⃣ 優化算法 (16_optimization) - 30 個解決方案

經典優化到元啟發式算法的全面覆蓋。

**代表性解決方案**:
- [08_simplex_method](16_optimization/08_simplex_method/) - 單純形法
- [13_genetic_algorithm](16_optimization/13_genetic_algorithm/) - 遺傳算法
- [14_particle_swarm](16_optimization/14_particle_swarm/) - 粒子群優化
- [18_gradient_descent_variants](16_optimization/18_gradient_descent_variants/) - 梯度下降變體
- [29_bayesian_optimization](16_optimization/29_bayesian_optimization/) - 貝葉斯優化
- [30_rl_optimization](16_optimization/30_rl_optimization/) - 強化學習優化

**技術覆蓋**: 線性規劃, 非線性優化, 進化算法, 梯度方法, 超參數優化

**完整列表**: [查看全部 30 個解決方案 →](16_optimization/)

---

### 1️⃣7️⃣ 多模態學習 (17_multimodal) - 30 個解決方案

融合視覺、語言、音訊等多種模態的深度學習。

**代表性解決方案**:
- [06_image_captioning_attention](17_multimodal/06_image_captioning_attention/) - 圖像描述生成
- [10_clip_style_pretraining](17_multimodal/10_clip_style_pretraining/) - CLIP 風格預訓練
- [11_audio_visual_speech_recognition](17_multimodal/11_audio_visual_speech_recognition/) - 視聽語音識別
- [16_multimodal_sentiment_analysis](17_multimodal/16_multimodal_sentiment_analysis/) - 多模態情感分析
- [21_multimodal_transformers](17_multimodal/21_multimodal_transformers/) - 多模態 Transformers
- [30_multimodal_representation_learning](17_multimodal/30_multimodal_representation_learning/) - 多模態表示學習

**技術覆蓋**: 視覺-語言, 音視頻融合, 跨模態檢索, 多模態 Transformer

**完整列表**: [查看全部 30 個解決方案 →](17_multimodal/)

---

## 🎯 快速開始

### 運行單個解決方案

```bash
# 進入解決方案目錄
cd kaggle_solutions/01_structured_data/01_titanic_survival

# 查看 README 了解詳情
cat README.md

# 運行解決方案
python solution.py
```

### 按類別瀏覽

每個類別目錄都包含該類別的所有解決方案：

```bash
# 查看時間序列所有解決方案
ls kaggle_solutions/02_time_series/

# 查看深度學習所有解決方案
ls kaggle_solutions/08_deep_learning/
```

## 📈 難度分級

- **入門級** (🟢): 適合初學者，基礎算法和簡單數據集
  - 例: 01_titanic_survival, 22_digit_recognition

- **中級** (🟡): 需要一定機器學習基礎，涉及特徵工程和調參
  - 例: 大部分解決方案屬於此級別

- **高級** (🔴): 需要深入理解算法原理，複雜模型架構
  - 例: 30_knowledge_graph_completion, 35_few_shot_learning

## 🛠️ 技術棧統計

### 常用庫
- **Scikit-learn**: 300+ 解決方案
- **Pandas/NumPy**: 500 個解決方案
- **TensorFlow/Keras**: 150+ 解決方案
- **PyTorch**: 100+ 解決方案
- **XGBoost/LightGBM**: 80+ 解決方案

### 主要算法類型
- 分類: 200+ 解決方案
- 回歸: 100+ 解決方案
- 聚類: 60+ 解決方案
- 深度學習: 120+ 解決方案
- 強化學習: 20+ 解決方案

## 📚 學習路徑建議

### 路徑 1: 機器學習入門
1. 01_structured_data/01_titanic_survival
2. 05_computer_vision/22_digit_recognition
3. 03_nlp/14_sentiment_analysis
4. 06_clustering/25_customer_segmentation
5. 02_time_series/10_sales_forecasting

### 路徑 2: 深度學習專精
1. 08_deep_learning/07_transfer_learning
2. 08_deep_learning/16_resnet_skip_connections
3. 08_deep_learning/20_vision_transformer
4. 08_deep_learning/26_knowledge_distillation
5. 08_deep_learning/35_few_shot_learning

### 路徑 3: 數據科學實務
1. 13_feature_engineering/14_feature_interactions
2. 14_ensemble_methods/15_xgboost_advanced
3. 07_special_domains/04_fraud_detection
4. 10_anomaly_detection/20_isolation_forest_detection
5. 02_time_series/29_probabilistic_forecasting

## 🔍 搜索技巧

### 按技術搜索
```bash
# 查找所有使用 LSTM 的解決方案
grep -r "lstm" kaggle_solutions/*/*/solution.py

# 查找所有使用 XGBoost 的解決方案
grep -r "xgboost" kaggle_solutions/*/*/solution.py
```

### 按應用領域搜索
- **金融**: 07_special_domains/21-25
- **醫療**: 07_special_domains/16-20, 05_computer_vision/09
- **電商**: 04_recommendation, 06_clustering/25
- **NLP**: 03_nlp 全部

## 💡 貢獻指南

想要添加新的解決方案？請參考：
- 解決方案應包含完整的 solution.py 和 README.md
- README 格式參考現有解決方案
- 代碼應包含詳細註釋和文檔字符串

## 📞 支持與反饋

- **問題報告**: 請在 GitHub Issues 提出
- **功能建議**: 歡迎提交 Pull Request
- **文檔問題**: 請參考主專案 [README](../README.md)

---

**最後更新**: 2025-01-19
**維護者**: Data Analysis with Chatbots Team
**授權**: MIT License
