# 📦 SKU Detection using YOLO11 Variants

# Task 1 – Object Detection Model Evaluation - Model Optimization(recall, precision)

This project evaluates multiple YOLO11 model variants (n, m, l, x) across different image sizes and confidence thresholds for Stock Keeping Unit (SKU) detection. The goal is to identify the optimal configuration for accurate and robust product detection in retail environments.

## 🧪 Experimental Setup
We conducted systematic experiments varying:

* Model Variants: YOLO11n (nano), YOLO11m (medium), YOLO11l (large), YOLO11x (x-large)
* Image Sizes: 640×640 and 960×960 pixels
* Confidence Thresholds: 0.25 and 0.15
* Additional Configuration: Augmented training for YOLO11x at 960px

## Evaluation Metrics
* mAP50: Mean Average Precision at IoU=0.50
* mAP50-95: Mean Average Precision across IoU thresholds 0.50 to 0.95
* Precision: Accuracy of positive predictions
* Recall: Ability to find all positive instances
* F1 Score: Harmonic mean of Precision and Recall

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

* Highest mAP50: YOLO11x (640, conf=0.15) → 0.8801
* Highest mAP50-95: YOLO11m (960, conf=0.15) → 0.5505
* Best Balanced Model (Precision + Recall): YOLO11m (960, conf=0.15)

## 🔍 Key Observations

* Increasing image size from 640 → 960 significantly improves performance, especially for YOLO11m and YOLO11l.
* Lower confidence threshold (0.15) generally increases recall and mAP.
* YOLO11x achieves highest raw mAP50 at 640 resolution.
* Augmentation did not significantly improve YOLO11x performance at 960.
* YOLO11m (960, 0.15) provides the best overall trade-off between accuracy and generalization.

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
## 📌 Task 1 Observation

For SKU detection:

* If GPU memory allows → YOLO11m @ 960, conf=0.15 is recommended.
* If prioritizing highest mAP50 → YOLO11x @ 640, conf=0.15
* For production trade-off → YOLO11m (960)


# 🧮 Task 2: Share of Shelf (SoS) Analysis
## 🎯 Objective

The second task evaluates the Share of Shelf (SoS) for each detected SKU in retail shelf images.

Share of Shelf represents the percentage distribution of detected products relative to the total detected products on the shelf.

This metric is widely used in:

* Retail analytics
* Brand performance tracking
* Shelf optimization
* Market competition analysis

## 📐 Share of Shelf Formula

SoS(i) (%) = (Number of detections of a SKU(i) /
Total detections of all SKUs) × 100
	​
## ⚙️ Implementation

1. We use YOLO model predictions to:
2. Run inference on test images
3. Count detections per class
4. Compute percentage share
5. Sort results by dominance
6. Visualize distribution

Core function
```python
import os
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

def compute_share_of_shelf(model, test_path, conf=0.25, iou=0.6, visualize=True):

    # Run prediction
    results = model.predict(
        source=test_path,
        conf=conf,
        iou=iou,
        save=False,
        verbose=False
    )

    # Count detections per class
    class_counts = Counter()

    for r in results:
        if r.boxes is not None:
            for cls in r.boxes.cls:
                class_id = int(cls.item())
                class_counts[class_id] += 1

    # Total detections
    total_detections = sum(class_counts.values())

    if total_detections == 0:
        print("No detections found.")
        return None

    # Compute Share of Shelf
    data = []
    for class_id, count in class_counts.items():
        share = (count / total_detections) * 100
        data.append([class_id, count, share])

    df = pd.DataFrame(data, columns=["Class_ID", "Count", "Share_of_Shelf (%)"])
    df = df.sort_values(by="Share_of_Shelf (%)", ascending=False)

    # Visualization
    if visualize:
        plt.figure(figsize=(14,6))
        plt.bar(df["Class_ID"], df["Share_of_Shelf (%)"])
        plt.xlabel("SKU (Class ID)")
        plt.ylabel("Share of Shelf (%)")
        plt.title("Share of Shelf per SKU")
        plt.xticks(rotation=90)
        plt.show()

    return df
```

## 📊 Output Format

The function returns a sorted DataFrame:
| Class_ID | Count | Share_of_Shelf (%) |
| -------- | ----- | ------------------ |
| 12       | 134   | 18.52              |
| 3        | 120   | 16.58              |
| 7        | 110   | 15.20              |

## 📈 Visualization

The bar chart displays:

* X-axis → SKU Class ID
* Y-axis → Share of Shelf (%)
* Sorted in descending order

This provides a clear view of:

* Most dominant SKU
* Underperforming SKUs
* Shelf distribution imbalance

## 🔍 Business Insight

* Higher SoS indicates stronger shelf presence
* Helps identify over/under represented brands
* Supports merchandising decisions
* Useful for retail performance monitoring

## 🏁 Conclusion

The project successfully completes:

* Task 1 – Object Detection Model Evaluation
* Task 2 – Retail Share of Shelf Analysis
