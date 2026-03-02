# 📦 SKU Detection using YOLO11 Variants

This project evaluates multiple YOLO11 model variants (n, m, l, x) across different image sizes and confidence thresholds for Stock Keeping Unit (SKU) detection. The goal is to identify the optimal configuration for accurate and robust product detection in retail environments.

## 🧪 Experimental Setup
We conducted systematic experiments varying:

Model Variants: YOLO11n (nano), YOLO11m (medium), YOLO11l (large), YOLO11x (x-large)

Image Sizes: 640×640 and 960×960 pixels

Confidence Thresholds: 0.25 and 0.15

Additional Configuration: Augmented training for YOLO11x at 960px

## Evaluation Metrics
mAP50: Mean Average Precision at IoU=0.50

mAP50-95: Mean Average Precision across IoU thresholds 0.50 to 0.95

Precision: Accuracy of positive predictions

Recall: Ability to find all positive instances

F1 Score: Harmonic mean of Precision and Recall

## 📊 Model Evaluation Summary
| Model | Img Size | Conf | mAP50 (Acc) | Precision | Recall | F1 Score | mAP50-95 |
|-------|----------|------|-------------|-----------|--------|----------|----------|
| YOLO11n | 640 | 0.25 | 0.7343 | 0.6989 | 0.7414 | 0.7195 | 0.4868 |
| YOLO11n | 640 | 0.15 | 0.7333 | 0.6837 | 0.7414 | 0.7114 | 0.4858 |
| YOLO11m | 640 | 0.25 | 0.6970 | 0.6692 | 0.6657 | 0.6674 | 0.4653 |
| YOLO11m | 640 | 0.15 | 0.7741 | 0.7031 | 0.7662 | 0.7333 | 0.4929 |
| YOLO11m | 960 | 0.25 | 0.8195 | 0.7839 | 0.8464 | 0.8140 | 0.5364 |
| YOLO11m | 960 | 0.15 | 0.8399 | 0.7828 | 0.8572 | 0.8183 | 0.5505 |
| YOLO11l | 640 | 0.25 | 0.7284 | 0.6981 | 0.6971 | 0.6976 | 0.4738 |
| YOLO11l | 640 | 0.15 | 0.7424 | 0.6874 | 0.7342 | 0.7101 | 0.4804 |
| YOLO11l | 960 | 0.25 | 0.8217 | 0.7380 | 0.8132 | 0.7738 | 0.5445 |
| YOLO11l | 960 | 0.15 | 0.8245 | 0.7164 | 0.8156 | 0.7628 | 0.5462 |
| YOLO11x | 640 | 0.25 | 0.8262 | 0.8248 | 0.8050 | 0.8148 | 0.5035 |
| YOLO11x | 640 | 0.15 | 0.8801 | 0.7662 | 0.8671 | 0.8135 | 0.5198 |
| YOLO11x | 960 | 0.25 | 0.8202 | 0.7393 | 0.8313 | 0.7826 | 0.5298 |
| YOLO11x | 960 | 0.15 | 0.8512 | 0.7431 | 0.8659 | 0.7998 | 0.5371 |
| YOLO11x (Aug) | 960 | 0.25 | 0.8231 | 0.7397 | 0.8318 | 0.7831 | 0.5279 |
| YOLO11x (Aug) | 960 | 0.15 | 0.8278 | 0.7151 | 0.8351 | 0.7704 | 0.5232 |

## 🏆 Best Performing Configuration

Highest mAP50: YOLO11x (640, conf=0.15) → 0.8801

Highest mAP50-95: YOLO11m (960, conf=0.15) → 0.5505

Best Balanced Model (Precision + Recall): YOLO11m (960, conf=0.15)

## 🔍 Key Observations

Increasing image size from 640 → 960 significantly improves performance, especially for YOLO11m and YOLO11l.

Lower confidence threshold (0.15) generally increases recall and mAP.

YOLO11x achieves highest raw mAP50 at 640 resolution.

Augmentation did not significantly improve YOLO11x performance at 960.

YOLO11m (960, 0.15) provides the best overall trade-off between accuracy and generalization.

## ⚙️ Training Configuration Example
### Yolo model 
```python 
results = pretrained_model.train(
    data="/kaggle/input/datasets/munazermontasirakash/data-yaml-kaggle/data.yaml",
    epochs=100,
    # imgsz=640,
    imgsz=960,
    batch=16,
    device=0,
    patience=20,
    project="/kaggle/working/sku_detection",
    name="yolo11x-960"
)
```

### Augmented Yolo Model
```python
results = pretrained_model.train(
    data="/kaggle/input/datasets/munazermontasirakash/data-yaml-kaggle/data.yaml",
    epochs=100,
    imgsz=960,
    batch=8,

    lr0=0.003,
    lrf=0.1,
    
    mosaic=0.5,       
    mixup=0.1,
    fliplr=0.5,
    degrees=2.0,
    translate=0.05,
    scale=0.3,

    patience=30,
    project="/kaggle/working/sku_detection",
    name="yolo11x_aug_ft"
)
```
## 📌 Conclusion

For SKU detection:

If GPU memory allows → YOLO11m @ 960, conf=0.15 is recommended.

If prioritizing highest mAP50 → YOLO11x @ 640, conf=0.15

For production trade-off → YOLO11m (960)
