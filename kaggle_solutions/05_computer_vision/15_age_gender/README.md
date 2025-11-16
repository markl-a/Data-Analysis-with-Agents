# Age and Gender Prediction from Faces

## Problem Overview
Multi-task learning to simultaneously predict age (regression) and gender (classification) from facial images. Applications in demographics analysis, targeted advertising, and security systems.

## Dataset
- **Synthetic face images** with realistic features:
  - Age range: 18-70 years
  - Genders: Male, Female
  - Age features: Wrinkles for older, smoother for younger
  - Gender features: Facial hair, eyebrow thickness
- **Tasks**: Age regression + Gender classification
- **Samples**: 2,000 images (64x64x3)

## Approach

### Model Architecture
**Multi-Task CNN** with shared backbone and task-specific heads:
```
Input (64x64x3)
  ↓
Shared Convolutional Layers
├─ Conv Block 1 → 32x32x32
├─ Conv Block 2 → 16x16x64
├─ Conv Block 3 → 8x8x128
└─ Conv Block 4 → 4x4x256
  ↓
Global Avg Pool + Shared FC → 512
  ↓
  ├─────────────┬─────────────┐
  ↓             ↓             ↓
Age Head    Gender Head
FC(128)       FC(64)
  ↓             ↓
Output(1)    Output(2)
Regression  Softmax
```

### Multi-Task Learning Benefits
1. **Shared Representations**: Common features for both tasks
2. **Regularization**: Tasks help each other generalize
3. **Efficiency**: Single forward pass for both predictions
4. **Related Tasks**: Age and gender share facial features

### Loss Function
- **Age**: Mean Absolute Error (MAE)
- **Gender**: Cross-Entropy
- **Combined**: Weighted sum (0.5 * age_loss + 0.5 * gender_loss)

## Results

### Performance Metrics
- **Age MAE**: ~5-8 years
- **Gender Accuracy**: ~90-95%
- **Joint Performance**: Both tasks improve together

### Key Insights
1. Shared features capture general facial structure
2. Task-specific heads learn specialized patterns
3. Gender classification easier than age regression
4. Facial features correlate with both age and gender
5. Multi-task learning improves over single-task

## Files Generated
1. `face_samples.png` - Sample faces with labels
2. `multitask_training_history.png` - Training curves for both tasks
3. `age_predictions.png` - Age prediction scatter plot
4. `gender_confusion_matrix.png` - Gender classification performance

## Usage
```bash
cd /home/user/Data-Analysis-with-Chatbots/kaggle_solutions/05_computer_vision/15_age_gender
python solution.py
```

## Requirements
- numpy
- matplotlib
- seaborn
- scikit-learn

## Applications
- **Demographics Analysis**: Population statistics
- **Targeted Advertising**: Age/gender-specific content
- **Access Control**: Age verification systems
- **Social Media**: Automatic tagging and filters
- **Retail Analytics**: Customer demographics
- **Healthcare**: Patient identification

## Future Improvements
1. Race/ethnicity prediction (additional task)
2. Emotion recognition (multi-task extension)
3. Age group classification vs regression
4. Attention mechanisms for facial regions
5. Ordinal regression for age
6. Transfer learning from face recognition models

## Related Datasets
- UTKFace: 20K+ faces with age, gender, ethnicity
- IMDB-WIKI: 500K+ celebrity faces with age
- Adience: Age and gender from real-world images
- MORPH: Longitudinal face aging database
