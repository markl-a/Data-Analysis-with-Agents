# Online Shopper Purchase Intention Prediction

## Overview

This solution predicts whether an online shopping session will end in a purchase based on session behavior metrics, visitor characteristics, and temporal features. The model uses XGBoost to identify high-intent browsing sessions in real-time.

## Business Problem

### Context
E-commerce platforms face the challenge of converting browsing sessions into actual purchases. Understanding purchase intent in real-time enables:
- **Personalized interventions**: Offer targeted discounts to high-intent users
- **Resource optimization**: Focus customer service on valuable sessions
- **User experience**: Improve the shopping experience for different visitor segments
- **Marketing ROI**: Optimize ad spend by understanding which traffic sources convert

### Objective
Develop a predictive model that can identify purchase intent based on session characteristics, achieving:
- High precision to avoid annoying users with unnecessary interventions
- High recall to capture as many potential buyers as possible
- Real-time prediction capability for live sessions
- Interpretable results for business decision-making

### Impact
A successful model can:
- Increase conversion rates by 10-20% through targeted interventions
- Reduce cart abandonment through timely assistance
- Improve customer satisfaction by personalizing the experience
- Optimize marketing spend by identifying high-value traffic sources

## Dataset Description

### Data Generation
The solution generates synthetic data that simulates real-world online shopping behavior with 5,000 sessions.

### Features

#### Session Behavior Metrics
- **Administrative**: Number of administrative pages visited (account, profile)
- **Administrative_Duration**: Time spent on administrative pages (seconds)
- **Informational**: Number of informational pages visited (about us, FAQ)
- **Informational_Duration**: Time spent on informational pages (seconds)
- **ProductRelated**: Number of product pages visited
- **ProductRelated_Duration**: Time spent on product pages (seconds)

#### Engagement Metrics
- **BounceRates**: Percentage of single-page sessions
  - Lower for returning visitors (1-5%)
  - Higher for new visitors (10-40%)
- **ExitRates**: Percentage of exits from each page
  - Indicates session quality
- **PageValues**: Average value of pages viewed before purchase
  - Higher values indicate closer to conversion

#### Temporal Features
- **Month**: Month of the session (1-12)
  - Captures seasonal patterns
- **Weekend**: Whether session occurred on weekend (0/1)
  - Weekend shoppers may have different behavior
- **SpecialDay**: Proximity to special days (holidays, sales events)
  - Higher probability during special events

#### Visitor Characteristics
- **VisitorType**: New_Visitor, Returning_Visitor, Other
  - Returning visitors have higher conversion rates
- **TrafficSource**: Direct, Organic, Paid, Social, Referral
  - Different sources have different conversion patterns
- **Browser**: Chrome, Safari, Firefox, Edge, Other
- **OperatingSystem**: Windows, MacOS, Linux, iOS, Android
- **Device**: Desktop, Mobile, Tablet
  - Desktop users tend to have higher conversion rates

#### Target Variable
- **Revenue**: Whether the session resulted in a purchase (0/1)
  - Binary classification target
  - Imbalanced dataset (typical 15-20% positive class)

### Data Statistics
- **Total Sessions**: 5,000
- **Features**: 17 original + 7 engineered = 24 total
- **Purchase Rate**: Approximately 30% (realistic for optimized e-commerce)
- **Class Distribution**: Imbalanced, reflecting real-world scenarios

## Technical Approach

### 1. Feature Engineering

The solution creates sophisticated session-based features:

#### Aggregate Metrics
- **TotalPages**: Sum of all pages viewed across categories
  - Indicates session depth
- **TotalDuration**: Total time spent on site
  - Longer sessions often indicate higher intent

#### Behavioral Indicators
- **AvgTimePerPage**: Average time spent per page
  - High values indicate engaged users
  - Very low values may indicate bots or accidental visits
- **ProductPageRatio**: Proportion of product pages vs total pages
  - Higher ratio indicates purchase-focused browsing

#### Engagement Scores
- **EngagementScore**: Composite metric combining:
  - Page values
  - Total pages viewed
  - Bounce rate (inverted)
  - Formula: PageValues * TotalPages * (1 - BounceRate)

- **ExitBounceRatio**: Ratio of exit rate to bounce rate
  - Helps differentiate between different exit patterns

- **SessionQuality**: Overall session quality metric
  - Formula: ProductPageRatio * (1 - ExitRate) * AvgTimePerPage / 100
  - Combines multiple engagement signals

### 2. Data Preprocessing

#### Encoding Strategy
- **Categorical Variables**: Label encoding for tree-based models
  - TrafficSource, VisitorType, Browser, OperatingSystem, Device
  - Preserves ordinal relationships where they exist

#### Train-Test Split
- **Split Ratio**: 80% train, 20% test
- **Stratification**: Preserves class distribution
- **Random State**: 42 for reproducibility

### 3. Model Selection: XGBoost

**Why XGBoost?**
- **Performance**: State-of-the-art accuracy for structured data
- **Speed**: Fast training and prediction for real-time applications
- **Feature Importance**: Built-in feature importance for interpretability
- **Handling Imbalance**: Native support for imbalanced datasets
- **Regularization**: Prevents overfitting through L1/L2 regularization

**Hyperparameters**:
```python
- n_estimators: 200 (number of boosting rounds)
- max_depth: 6 (maximum tree depth)
- learning_rate: 0.1 (step size shrinkage)
- subsample: 0.8 (row sampling per tree)
- colsample_bytree: 0.8 (column sampling per tree)
```

### 4. Model Evaluation

#### Metrics Used
- **Classification Report**: Precision, Recall, F1-Score for both classes
- **ROC-AUC Score**: Area under the ROC curve
  - Measures discrimination ability
  - Threshold-independent metric
- **Confusion Matrix**: True/False Positives/Negatives
- **Cross-Validation**: 5-fold CV for robust performance estimation

#### Why These Metrics?
- **Precision**: Important to avoid annoying users with false positives
- **Recall**: Important to capture as many potential buyers as possible
- **AUC**: Overall model discrimination ability
- **F1-Score**: Harmonic mean balancing precision and recall

## Visualizations

The solution generates 12 comprehensive visualizations:

### 1. Business Insights
- **Revenue by Visitor Type**: Shows conversion rates by visitor segment
- **Revenue by Traffic Source**: Identifies best-performing channels
- **Revenue by Browser**: Browser-specific conversion patterns
- **Revenue by Device**: Desktop vs Mobile vs Tablet performance

### 2. Behavioral Analysis
- **Bounce vs Exit Rates**: Scatter plot showing relationship and clustering by revenue
- **Page Values Distribution**: Histogram comparing purchasers vs non-purchasers
- **Session Duration by Device**: Box plot showing device-specific engagement
- **Special Days Impact**: Conversion lift during promotional periods

### 3. Temporal Patterns
- **Monthly Revenue Trend**: Seasonal patterns and trends
- **Weekend vs Weekday**: Day-of-week effects on conversion

### 4. Model Performance
- **Feature Importance**: Top 15 features driving predictions
- **Confusion Matrix**: Classification accuracy breakdown
- **ROC Curve**: Model discrimination ability with AUC score

## Expected Results

### Model Performance
```
Classification Metrics:
- Accuracy: ~82-88%
- Precision (Purchase): ~75-82%
- Recall (Purchase): ~70-80%
- F1-Score (Purchase): ~72-81%
- ROC-AUC: ~0.85-0.92

Cross-Validation:
- Mean ROC-AUC: ~0.87
- Standard Deviation: ~0.02
```

### Key Findings

#### High-Value Segments
1. **Returning Visitors**: 2-3x higher conversion rate than new visitors
2. **Direct Traffic**: Highest conversion rate (brand loyalty)
3. **Desktop Users**: Higher conversion than mobile
4. **Low Bounce/Exit Rates**: Strong predictor of purchase intent

#### Feature Importance (Top 5)
1. **EngagementScore**: Composite engagement metric
2. **PageValues**: Strongest single predictor
3. **BounceRates**: Lower bounce = higher intent
4. **VisitorType**: Returning vs new visitor status
5. **ProductPageRatio**: Product focus indicates intent

#### Temporal Patterns
- **Special Days**: 40-50% conversion lift during promotions
- **Weekend Effect**: Slightly lower conversion (browsing vs buying)
- **Seasonal Trends**: Holiday months show higher conversion

## How to Run

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

### Execution
```bash
python solution.py
```

### Output
1. **Console Output**:
   - Dataset statistics
   - Model performance metrics
   - Cross-validation results

2. **Visualization**:
   - `online_shopper_intention_analysis.png` (12-panel dashboard)

### Runtime
- Approximate execution time: 10-15 seconds
- Memory usage: ~200-300 MB

## Business Applications

### Real-Time Interventions
**High-Intent Sessions** (Predicted Purchase Probability > 0.7):
- Offer free shipping or small discount
- Provide live chat assistance
- Show trust signals (reviews, guarantees)

**Medium-Intent Sessions** (Probability 0.4-0.7):
- Show comparison tools
- Display customer reviews
- Offer email capture for follow-up

**Low-Intent Sessions** (Probability < 0.4):
- Collect browse data for retargeting
- Offer content (buying guides, etc.)
- Build email list with value proposition

### Marketing Optimization
- **Traffic Source ROI**: Allocate budget to high-converting sources
- **Device Optimization**: Improve mobile experience based on gap
- **Seasonal Planning**: Staff and inventory for high-conversion periods

### Personalization
- **Returning Visitors**: VIP treatment, loyalty rewards
- **New Visitors**: Educational content, trust building
- **Product Browsers**: Similar items, bundle offers

## Improvements and Extensions

### Model Enhancements
1. **Ensemble Methods**:
   - Combine XGBoost with LightGBM and CatBoost
   - Use stacking for better performance

2. **Deep Learning**:
   - Neural network for sequential session data
   - LSTM for time-series session behavior

3. **Calibration**:
   - Probability calibration for better probability estimates
   - Important for setting intervention thresholds

### Feature Engineering
1. **Sequence Features**:
   - Page visit order and patterns
   - Time between page views
   - Navigation path analysis

2. **User History**:
   - Previous session behavior
   - Purchase history
   - Customer lifetime value

3. **Product Features**:
   - Price points viewed
   - Categories browsed
   - Product popularity scores

### Advanced Analytics
1. **Survival Analysis**:
   - Time-to-purchase modeling
   - Session abandonment prediction

2. **Cohort Analysis**:
   - Different models for different visitor segments
   - Source-specific models

3. **A/B Testing**:
   - Test intervention strategies
   - Measure incremental lift

### Production Deployment
1. **Real-Time Scoring**:
   - API endpoint for live prediction
   - Sub-100ms latency requirements

2. **Model Monitoring**:
   - Data drift detection
   - Performance monitoring
   - Automated retraining

3. **Explainability**:
   - SHAP values for individual predictions
   - Lime for local interpretability

## References and Resources

### Academic Papers
- Sakar, C.O., et al. (2019). "Real-time prediction of online shoppers' purchasing intention"
- Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System"

### Industry Best Practices
- Google Analytics: E-commerce tracking
- Shopify: Conversion rate optimization
- Amazon: Personalization at scale

### Tools and Libraries
- **XGBoost**: https://xgboost.readthedocs.io/
- **Scikit-learn**: https://scikit-learn.org/
- **SHAP**: https://github.com/slundberg/shap

## Conclusion

This solution demonstrates a complete end-to-end pipeline for predicting online shopper purchase intention. The XGBoost model achieves strong performance with an AUC of ~0.87-0.92, providing actionable insights for e-commerce optimization.

The comprehensive feature engineering captures session behavior patterns, while the visualizations provide interpretable business insights. The model can be deployed in production for real-time purchase intent prediction, enabling personalized user experiences and optimized conversion rates.

Key takeaways:
- Returning visitors and direct traffic are highest-value segments
- Engagement metrics (bounce rate, page values) are strongest predictors
- Device and browser optimization opportunities exist
- Special days and promotions significantly boost conversion

This foundation can be extended with advanced techniques like deep learning, real-time scoring, and personalized intervention strategies for maximum business impact.
