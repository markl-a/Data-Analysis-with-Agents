# Food Image Classification

## Problem Overview
Automatically classify food items from images for applications in nutrition tracking, restaurant automation, and dietary monitoring. This solution uses a VGG-style CNN to recognize 5 common food categories.

## Dataset
- **Synthetic food images** with distinctive visual characteristics:
  - Pizza: Circular shape with yellow/red toppings
  - Burger: Layered rectangular structure
  - Sushi: Rice rolls with seaweed and fish
  - Salad: Green textures with varied vegetables
  - Pasta: Yellowish strands with sauce
- **Classes**: 5 food categories
- **Samples**: 3,000 images (64x64x3)
- **Features**: Color, texture, shape patterns

## Approach

### Model Architecture
**VGG-Style CNN** with multiple convolutional blocks:
```
Input (64x64x3)
  ↓
Conv Block 1 (2x Conv 64) → MaxPool → 32x32x64
  ↓
Conv Block 2 (2x Conv 128) → MaxPool → 16x16x128
  ↓
Conv Block 3 (2x Conv 256) → MaxPool → 8x8x256
  ↓
Flatten → 16,384
  ↓
FC (512) → Dropout → ReLU
  ↓
FC (256) → ReLU
  ↓
Output (5) → Softmax
```

### Key Features
1. **Deep Architecture**: Multiple conv layers extract hierarchical features
2. **Small Filters**: 3x3 convolutions like VGG
3. **Dropout**: Prevents overfitting on food patterns
4. **Color Features**: RGB channels capture distinctive food colors
5. **Texture Analysis**: Multiple layers capture food textures

### Food Recognition Challenges
- **Intra-class Variation**: Same food prepared differently
- **Inter-class Similarity**: Similar colors across foods
- **Presentation**: Different plating and angles
- **Portion Sizes**: Variable food amounts

## Results

### Performance Metrics
- **Overall Accuracy**: ~75-85%
- **Per-Class Performance**: Varies by food distinctiveness
- **Best Performance**: Pizza, Sushi (distinctive shapes/colors)
- **Challenges**: Pasta vs Salad (texture similarity)

### Insights
1. Color is a strong discriminative feature
2. Shape matters for structured foods (pizza, sushi)
3. Texture patterns help distinguish similar-colored items
4. Deep networks better capture complex food features

## Files Generated
1. `food_samples.png` - Sample images from each class
2. `food_training_history.png` - Training curves
3. `food_confusion_matrix.png` - Classification confusion matrix
4. `per_class_accuracy.png` - Individual class performance

## Usage
```bash
cd /home/user/Data-Analysis-with-Chatbots/kaggle_solutions/05_computer_vision/12_food_classification
python solution.py
```

## Requirements
- numpy
- matplotlib
- seaborn
- scikit-learn

## Applications
- **Nutrition Apps**: Automatic food logging
- **Restaurant Automation**: Order recognition
- **Dietary Monitoring**: Track food intake
- **Recipe Recommendation**: Suggest based on food type
- **Portion Control**: Combined with size estimation

## Future Improvements
1. Multi-label classification (ingredients)
2. Fine-grained recognition (cuisine types)
3. Portion size estimation
4. Calorie prediction
5. Ingredient detection
6. Transfer learning from Food-101 dataset

## Related Datasets
- Food-101: 101,000 images of 101 food categories
- Recipe1M: Food images with recipes
- UEC Food datasets: Japanese/Asian foods
