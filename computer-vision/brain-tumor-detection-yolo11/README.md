# Brain Tumor Detection using YOLOv11

A deep learning project for automated brain tumor detection in medical images using the state-of-the-art YOLOv11 object detection model.

## 🎯 Project Overview

This project implements a brain tumor detection system by fine-tuning YOLOv11-medium on medical imaging data. The model achieves high accuracy in identifying and localizing brain tumors, making it a valuable tool for medical diagnosis assistance.

## 📊 Performance Metrics

Our fine-tuned YOLOv11 model achieved exceptional performance:

- **Precision**: 97.93%
- **Recall**: 94.60%
- **mAP@0.5**: 97.66%
- **mAP@0.5:0.95**: 71.51%

These metrics demonstrate the model's reliability in accurately detecting brain tumors while minimizing false positives and false negatives.

## 🗂️ Dataset

The project uses a curated brain tumor dataset from Roboflow:
- **Source**: [Tumor Detection Dataset](https://universe.roboflow.com/selencakmak/tumor-dj2a1)
- **Format**: YOLOv11 compatible annotations


## 🚀 Usage

### 1. Data Preparation
```python
from roboflow import Roboflow

# Initialize Roboflow with your API key
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("ziad-nassif").project("tumor-vliim-k1cjm")
version = project.version(1)
dataset = version.download("yolov11")
```

### 2. Model Training
```python
from ultralytics import YOLO

# Load YOLOv11-medium pretrained model
model = YOLO("yolo11m.pt")

# Train the model
model.train(
    data="/path/to/your/Tumor-1/data.yaml",
    epochs=200,
    imgsz=512,
    batch=16,
    device=0,
    verbose=False
)
```

### 3. Model Inference
```python
# Load trained model
model = YOLO("runs/detect/train102/weights/best.pt")

# Run inference on new images
results = model("path/to/test/image.jpg")
results[0].show()  # Display results
```

## ⚙️ Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model | YOLOv11-medium | Base architecture |
| Epochs | 200 | Training iterations |
| Image Size | 512×512 | Input resolution |
| Batch Size | 16 | Training batch size |

## 📈 Results Analysis

The model demonstrates excellent performance across all metrics:

- **High Precision (97.93%)**: Minimal false positive detections
- **Strong Recall (94.60%)**: Effectively captures most tumor instances
- **Excellent mAP@0.5 (97.66%)**: Outstanding detection accuracy at IoU threshold 0.5
- **Good mAP@0.5:0.95 (71.51%)**: Solid performance across varying IoU thresholds

## 🏥 Medical Applications

This model can assist healthcare professionals in:
- Early brain tumor detection
- Medical image analysis automation
- Screening process enhancement
- Diagnostic decision support

## 🔬 Technical Details

### Model Architecture
- **Base Model**: YOLOv11-medium
- **Input Resolution**: 512×512 pixels
- **Output**: Bounding boxes with confidence scores for tumor locations

### Training Environment
- **Platform**: Kaggle Notebooks
- **GPU**: Tesla P100/V100 (recommended)
- **Training Time**: ~2-3 hours for 200 epochs

---

**Note**: This project is for educational and research purposes. Always consult with medical professionals for actual diagnosis and treatment decisions.
