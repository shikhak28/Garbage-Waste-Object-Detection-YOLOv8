# Garbage / Waste Object Detection — YOLOv8

Object detection (bounding boxes, not just single-label classification) for waste sorting: **biodegradable, cardboard, glass, metal, paper, plastic**.

## Dataset

10,464 images, COCO-annotated, sourced via Roboflow ("Garbage Classification 3"), hosted publicly on [Hugging Face](https://huggingface.co/datasets/keremberke/garbage-object-detection). No login or API key needed to download.

## What this covers

- Downloading the dataset via `huggingface_hub` (handles the storage backend correctly regardless of upload method)
- A custom COCO → YOLO label conversion script (tested against a synthetic dataset before use — no third-party converter dependency)
- Fine-tuning YOLOv8n (transfer learning from COCO-pretrained weights, not training from scratch)
- Evaluation with mAP50, mAP50-95, and **per-class** AP — correctly handling classes with zero validation instances (a common YOLO evaluation pitfall: `ap50` arrays only include classes present in the val set, so naively zipping them against the full class list silently mislabels results)
- Visualizing predictions on held-out validation images
- A confusion matrix to see which waste categories get confused with each other

## Results

*(Fill in after your full training run — these are placeholders from a short verification run, not the real result)*

| Metric | Value |
|---|---|
| mAP50 | ___ |
| mAP50-95 | ___ |

| Class | AP50 |
|---|---|
| biodegradable | ___ |
| cardboard | ___ |
| glass | ___ |
| metal | ___ |
| paper | ___ |
| plastic | ___ |

## How to run

Open `garbage_detection.ipynb` in Google Colab. **Set the runtime to GPU first**: `Runtime` → `Change runtime type` → `T4 GPU`. Run all cells top to bottom — the dataset downloads automatically, no manual steps needed.

```
pip install -r requirements.txt
jupyter notebook garbage_detection.ipynb
```

## Tech stack

Python, PyTorch, Ultralytics YOLOv8, huggingface_hub, matplotlib
