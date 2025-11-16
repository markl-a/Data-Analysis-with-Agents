# 13. Resume Parsing - Skill Extraction and Classification

## Overview
Extract and classify technical skills and experience from resumes using NLP techniques. This project demonstrates how to parse unstructured resume text and categorize candidates into different technical roles.

**Difficulty**: ⭐⭐⭐ Intermediate
**Domain**: Human Resources, Recruitment
**Techniques**: TF-IDF, Gradient Boosting, Text Feature Engineering

## Problem Statement
Given resume text, automatically:
1. Extract technical skills and keywords
2. Classify the resume into job categories (Data Science, Software Engineering, DevOps, Web Development, Mobile Development)
3. Analyze experience level indicators

## Dataset
- **Size**: 1,000 synthetic resumes
- **Features**: Resume text with skills, experience, and qualifications
- **Categories**: 5 job categories
- **Distribution**: Balanced across categories

## Methodology

### 1. Data Generation
- Domain-specific skill vocabularies for each category
- Experience-level templates (junior, mid, senior)
- Realistic resume structure with multiple sections

### 2. Feature Engineering
- **TF-IDF vectorization** (1-3 grams, 2000 features)
- Text length and word count
- Average word length
- Technical term frequency
- Experience indicator counts

### 3. Model
- **Gradient Boosting Classifier**
  - 100 estimators
  - Learning rate: 0.1
  - Max depth: 5
  - 5-fold cross-validation

### 4. Evaluation Metrics
- Classification accuracy
- Per-category precision, recall, F1-score
- Confusion matrix
- Cross-validation scores

## Key Features
- Comprehensive skill extraction for 5 technical domains
- Multi-level experience indicators
- Feature engineering from unstructured text
- Visualization of resume distributions and patterns
- High accuracy classification

## Results
- **Expected Accuracy**: ~85-95%
- **Cross-validation**: Robust performance across folds
- **Insights**: Different categories have distinct skill patterns

## Visualizations
1. **Category distribution** - Balance across job types
2. **Word count distribution** - Resume length patterns
3. **Average word length by category** - Technical terminology complexity
4. **Experience indicators** - Leadership and expertise signals
5. **Confusion matrix** - Classification performance

## Use Cases
- Automated resume screening
- Candidate-job matching
- Skill gap analysis
- Recruitment automation
- Talent pool categorization

## Running the Code
```bash
python solution.py
```

## Output Files
- `resume_analysis.png` - Data distribution visualizations
- `resume_confusion_matrix.png` - Model performance

## Key Learnings
1. Domain-specific vocabularies are crucial for resume classification
2. Experience indicators (action verbs) help identify seniority
3. TF-IDF with n-grams captures technical skill combinations
4. Gradient Boosting handles multi-class text classification well
5. Feature engineering beyond text is valuable (length, complexity metrics)

## Extensions
- Extract years of experience using regex
- Identify education level and institutions
- Detect programming language proficiency
- Match resumes to job descriptions
- Extract contact information and certifications
- Implement named entity recognition (NER) for skills
