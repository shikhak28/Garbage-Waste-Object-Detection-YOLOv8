# Garbage Waste Object Detection using YOLOv8

This project detects and classifies different types of garbage in images using the YOLOv8 object detection model.

## Classes

The model detects six types of waste:

* Biodegradable
* Cardboard
* Glass
* Metal
* Paper
* Plastic

## Project Overview

The project follows these steps:

1. Download the garbage dataset.
2. Convert COCO annotations to YOLO format.
3. Prepare the dataset configuration.
4. Load a pretrained YOLOv8 model.
5. Train the model.
6. Validate the model.
7. Generate sample predictions.
8. Save the best trained model.

## Dataset Structure

After converting the dataset to YOLO format:

```text
garbage_yolo/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

Each label file follows the YOLO format:

```text
class_id center_x center_y width height
```

The coordinates are normalized between `0` and `1`.

## Model

This project uses the pretrained `YOLOv8n` model from Ultralytics.

Training configuration:

```text
Model: YOLOv8n
Epochs: 25
Image Size: 416
Batch Size: 16
Patience: 8
Number of Classes: 6
```

The pretrained model is fine-tuned on the garbage dataset.

## Installation

Install the required packages:

```bash
pip install ultralytics huggingface_hub
```

## Training

The model can be trained using:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="garbage_yolo/data.yaml",
    epochs=25,
    imgsz=416,
    batch=16,
    patience=8
)
```

## Validation

The trained model is evaluated on the validation dataset using:

* mAP@50
* mAP@50-95
* Per-class mAP@50
* Confusion matrix

These metrics are used to evaluate object detection and classification performance.

## Prediction

The trained model can detect garbage objects in images and provide:

* Bounding boxes
* Predicted class
* Confidence score

Example:

```python
results = model.predict(
    "path/to/image.jpg",
    conf=0.25
)
```

## Output

The best trained model is saved as:

```text
garbage_yolov8n_best.pt
```

Training, validation, and prediction results are stored in the `runs` directory.

## How to Run

1. Open `garbage_detection.ipynb`.
2. Install the required packages.
3. Download and prepare the dataset.
4. Convert the COCO annotations to YOLO format.
5. Create `data.yaml`.
6. Train the YOLOv8 model.
7. Validate the model.
8. Run predictions on sample images.
9. Use `garbage_yolov8n_best.pt` for further inference.

## Project Goal

The goal of this project is to automatically detect and classify different types of garbage in images using object detection.
