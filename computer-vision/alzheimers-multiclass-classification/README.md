# Alzheimer's Disease Multiclass Classification with Transfer Learning

[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue)](https://www.kaggle.com/datasets/aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented/data)

## 🧠 Project Overview

This project implements an efficient deep learning approach for classifying Alzheimer's disease severity using brain MRI scans. The model utilizes **transfer learning** and **progressive fine-tuning** to achieve high accuracy while significantly reducing computational costs.

### Key Features
- **High Accuracy**: 99.61% training accuracy, 98.98% test accuracy
- **Computationally Efficient**: Two-stage training approach reduces computational requirements
- **Transfer Learning**: Leverages ResNet-50 pre-trained on RadImageNet
- **Progressive Fine-tuning**: Strategic layer unfreezing for optimal performance

## 📊 Dataset

The dataset contains **MRI brain scans** categorized into four classes representing different stages of Alzheimer's disease:

| Class | Label | Description |
|-------|-------|-------------|
| NonDemented | 2 | Normal cognitive function |
| VeryMildDemented | 3 | Very mild cognitive decline |
| MildDemented | 0 | Mild cognitive impairment |
| ModerateDemented | 1 | Moderate dementia |

**Dataset Split:**
- Training: 35,200 images (80%)
- Validation: 4,400 images (10%)  
- Testing: 4,400 images (10%)

## 🏗️ Model Architecture

### Base Model: ResNet-50 + Custom Head
- Pre-trained on **RadImageNet** with **include_top=False**
- **Custom classification head**:
  - Global Average Pooling layer
  - Dense layer (256 units, ReLU)
  - Dense layer (128 units, ReLU) 
  - Output layer (4 units, Softmax)
- Total layers: ~180 (ResNet-50 base + custom layers)
- Input size: 256×256×3 RGB images

### Two-Stage Training Strategy

#### Stage 1: Full Network Training (10 epochs)
- **All layers trainable** with include_top=False
- Added Global Average Pooling layer
- Added 2 Dense layers + Softmax for classification
- Learning rate: 1e-4
- Focus: Initial adaptation of entire network to Alzheimer's classification

#### Stage 2: Selective Fine-tuning (20 epochs)
- **Freeze first 120 layers** (preserve low-level features)
- **Unfreeze last 60 layers** + custom classification head
- **67% computational cost reduction** in this stage
- Focus: Fine-tune high-level features and classification layers


## 📈 Results

| Metric | Training | Validation | Testing |
|--------|----------|------------|---------|
| **Accuracy** | 99.61% | 98.30% | 98.98% |
| **Loss** | 0.0107 | 0.0517 | 0.0297 |

### Model Performance Analysis
- **Excellent generalization**: Minimal overfitting between train/test
- **Consistent validation**: Stable performance across datasets  
- **Efficient training**: Achieved high accuracy with reduced computational cost

## 🚀 Key Innovations

### 1. Strategic Transfer Learning
- **RadImageNet weights**: Medical imaging pre-training provides domain-specific features
- **Progressive unfreezing**: Gradual adaptation prevents catastrophic forgetting

### 2. Computational Optimization  
- **Selective fine-tuning**: Only 33% of layers trainable in final stage
- **Two-stage approach**: Reduces total training time while maintaining accuracy
- **Smart layer selection**: Preserve low-level medical imaging features, adapt high-level classifiers

### 3. Robust Training Pipeline
- **Early stopping**: Prevents overfitting (patience=20)
- **Learning rate scheduling**: Adaptive reduction on plateau
- **Model checkpointing**: Save best performing weights


## 📚 References

- Dataset: [Alzheimer's Multiclass Dataset - Kaggle](https://www.kaggle.com/datasets/aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented/data)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve the model performance or add new features.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

