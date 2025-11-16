# 15. Legal Document Classification

## Overview
Classify legal documents into categories (Contract, Patent, Court Filing, Agreement, Regulation) using advanced NLP techniques and ensemble learning methods.

**Difficulty**: ⭐⭐⭐⭐ Advanced
**Domain**: Legal Tech, Document Management, Compliance
**Techniques**: TF-IDF, Ensemble Learning, Legal Text Analysis

## Problem Statement
Given a legal document, automatically classify it into one of five categories:
- **Contract** - Commercial contracts and service agreements
- **Patent** - Patent applications and specifications
- **Court Filing** - Legal motions, complaints, and court documents
- **Agreement** - Mutual agreements and memorandums of understanding
- **Regulation** - Regulatory documents and compliance requirements

## Dataset
- **Size**: 1,000 synthetic legal documents
- **Features**: Legal terminology, section references, party mentions
- **Categories**: 5 legal document types
- **Distribution**: Balanced across categories

## Methodology

### 1. Data Generation
- Domain-specific legal vocabularies for each category
- Realistic legal phrases and terminology
- Section references (Articles, Claims, Counts)
- Party designations appropriate to each document type
- Multi-paragraph document structure

### 2. Feature Engineering
- **TF-IDF vectorization** (1-3 grams, 5000 features, sublinear_tf)
- Document and word count statistics
- Average word length
- Section count (Article, Clause, Claim references)
- Party mention frequency
- Legal term density (pursuant, hereby, thereof)
- Citation pattern detection

### 3. Model
- **Ensemble Voting Classifier**
  - Logistic Regression (C=1.0)
  - Linear SVM (C=1.0)
  - Multinomial Naive Bayes (alpha=0.1)
  - Hard voting strategy
  - 5-fold cross-validation

### 4. Evaluation Metrics
- Classification accuracy
- Per-category precision, recall, F1-score
- Confusion matrix
- Cross-validation scores

## Key Features
- Ensemble learning for robust classification
- Legal domain-specific feature extraction
- Multiple n-gram analysis (1-3 grams)
- Citation and reference pattern detection
- Comprehensive legal terminology coverage

## Results
- **Expected Accuracy**: ~88-93%
- **Model Type**: Voting Classifier (Ensemble)
- **Insights**: Patents have distinct claim language; Court Filings have unique procedural terminology

## Visualizations
1. **Category distribution** - Balance across document types
2. **Average word count** - Document length by category
3. **Legal terms density** - Formal legal language usage
4. **Section count** - Document structure complexity
5. **Confusion matrix** - Classification performance

## Use Cases
- Automated document routing in law firms
- Legal document management systems
- Contract lifecycle management
- Patent portfolio organization
- Compliance document classification
- E-discovery and document review
- Legal research automation

## Running the Code
```bash
python solution.py
```

## Output Files
- `legal_analysis.png` - Data distribution visualizations
- `legal_confusion_matrix.png` - Model performance

## Key Learnings
1. Legal documents have highly specialized vocabularies
2. Ensemble methods improve classification robustness
3. Section references are strong category indicators
4. Trigrams capture legal phrases better than unigrams
5. Party designations vary significantly by document type
6. Formal legal language patterns are highly predictive

## Extensions
- Extract specific clauses and terms
- Identify key parties and dates
- Summarize legal documents
- Detect document anomalies
- Extract obligation and rights
- Implement clause similarity matching
- Add temporal analysis (document dates)
- Detect document versions and amendments
- Multi-label classification for hybrid documents
- Extract and validate citations
