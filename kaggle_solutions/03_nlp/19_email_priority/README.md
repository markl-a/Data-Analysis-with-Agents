# 19. Email Priority Classification

## Overview
Automatically classify emails by priority level (High, Medium, Low) based on content, subject line, and sender information. This helps in email triage and inbox management.

**Difficulty**: ⭐⭐⭐ Intermediate
**Domain**: Productivity, Email Management, Business Communication
**Techniques**: TF-IDF, Gradient Boosting, Multi-feature Classification

## Problem Statement
Given an email (including subject, sender, and body), classify it into priority levels:
- **High Priority** - Urgent, time-sensitive, requires immediate action
- **Medium Priority** - Important but not urgent, needs attention soon
- **Low Priority** - Informational, no immediate action required

## Dataset
- **Size**: 1,000 synthetic emails
- **Features**: Email content with subject, sender, and body
- **Categories**: 3 priority levels
- **Distribution**: Realistic (20% High, 45% Medium, 35% Low)

## Methodology

### 1. Data Generation
- Priority-specific vocabularies:
  - **High**: Urgent keywords, executive senders, critical subjects
  - **Medium**: Follow-up language, collaborative tone, project updates
  - **Low**: Informational content, newsletters, general announcements
- Realistic email structure:
  - From field
  - Subject line
  - Body content
  - Signature
- Context-appropriate language for each priority

### 2. Feature Engineering
- **TF-IDF vectorization** (1-2 grams, 2500 features)
- Email length and word count
- Line count
- Urgent keyword frequency
- Subject line analysis:
  - Contains urgent words
  - All caps detection
  - Subject length
- Sender analysis:
  - Executive/VIP sender detection
- Punctuation patterns:
  - Exclamation marks
  - Question marks

### 3. Model
- **Gradient Boosting Classifier**
  - 100 estimators
  - Learning rate: 0.1
  - Max depth: 5
  - 5-fold cross-validation

### 4. Evaluation Metrics
- Classification accuracy
- Per-priority precision, recall, F1-score
- Confusion matrix
- Cross-validation scores

## Key Features
- Multi-source feature extraction (subject, sender, body)
- Urgent keyword detection
- Sender importance analysis
- Subject line pattern recognition
- Comprehensive email structure modeling

## Results
- **Expected Accuracy**: ~85-92%
- **Model**: Gradient Boosting
- **Insights**: Subject line and urgent keywords are strong predictors; executive senders indicate high priority

## Visualizations
1. **Priority distribution** - Email priority balance
2. **Word count by priority** - Email length patterns
3. **Urgent words** - Keyword frequency by priority
4. **Subject length** - Subject line characteristics
5. **Executive senders** - VIP sender proportion
6. **ALL CAPS subjects** - Urgency indicator analysis

## Use Cases
- Automated email triage
- Inbox organization and filtering
- Smart notification systems
- Email client features
- Productivity applications
- Customer service routing
- Help desk ticket prioritization
- Alert systems

## Running the Code
```bash
python solution.py
```

## Output Files
- `email_analysis.png` - Feature distribution visualizations
- `email_confusion_matrix.png` - Model performance

## Key Learnings
1. Subject line is highly predictive of priority
2. All-caps subjects often indicate urgency
3. Executive senders correlate with high priority
4. Urgent keywords strongly predict priority level
5. Low priority emails tend to be longer and more informational
6. High priority emails are concise and action-oriented
7. Medium priority has most variation and is hardest to classify

## Priority Indicators

### High Priority Signals
- Urgent keywords (ASAP, critical, emergency)
- Executive/VIP senders
- All-caps subject lines
- Time-based language (today, now, immediately)
- Action-required phrases

### Medium Priority Signals
- Follow-up language
- Meeting requests
- Review requests
- Collaborative language
- Project updates

### Low Priority Signals
- FYI, informational language
- Newsletter format
- No action required
- General announcements
- "No response needed"

## Extensions
- Extract deadlines and dates
- Detect automated emails
- Sender relationship analysis
- Thread context consideration
- Time-of-day patterns
- Response urgency prediction
- Auto-categorization into folders
- Smart notification scheduling
- Email summarization
- Priority score (not just class)
- Multi-label priority (urgent + important matrix)
- Personalized priority learning
- Integration with calendar for deadline detection
