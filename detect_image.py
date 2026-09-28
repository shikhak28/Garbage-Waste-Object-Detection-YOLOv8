import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from yolo_postprocess import postprocess


MODEL_PATH = "runs/yolov8n_merged_100ep/weights/best.onnx"

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMAGE_SIZE = 640


def detect(image_path):
    # --------------------------------------------------
    # 1. Load ONNX model
    # --------------------------------------------------

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    # --------------------------------------------------
    # 2. Load image
    # --------------------------------------------------

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    original_height, original_width = image.shape[:2]

    # --------------------------------------------------
    # 3. Preprocess
    # --------------------------------------------------

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    resized = cv2.resize(
        image_rgb,
        (IMAGE_SIZE, IMAGE_SIZE),
    )

    input_tensor = resized.astype(
        np.float32
    ) / 255.0

    # HWC -> CHW
    input_tensor = np.transpose(
        input_tensor,
        (2, 0, 1),
    )

    # Add batch dimension
    input_tensor = np.expand_dims(
        input_tensor,
        axis=0,
    )

    # --------------------------------------------------
    # 4. ONNX inference
    # --------------------------------------------------

    outputs = session.run(
        None,
        {input_name: input_tensor},
    )

    raw_output = outputs[0]

    # --------------------------------------------------
    # 5. YOLO post-processing
    # --------------------------------------------------

    detections = postprocess(
        raw_output,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
    )

    # --------------------------------------------------
    # 6. Scale boxes to original image
    # --------------------------------------------------

    scale_x = original_width / IMAGE_SIZE
    scale_y = original_height / IMAGE_SIZE

    results = []

    for detection in detections:

        box = detection["box"]

        x1 = int(box[0] * scale_x)
        y1 = int(box[1] * scale_y)
        x2 = int(box[2] * scale_x)
        y2 = int(box[3] * scale_y)

        # Keep boxes inside image
        x1 = max(0, min(x1, original_width - 1))
        y1 = max(0, min(y1, original_height - 1))
        x2 = max(0, min(x2, original_width - 1))
        y2 = max(0, min(y2, original_height - 1))

        results.append({
            "class_name": detection["class_name"],
            "confidence": detection["confidence"],
            "box": [x1, y1, x2, y2],
        })

        # --------------------------------------------------
        # Draw bounding box
        # --------------------------------------------------

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        label = (
            f"{detection['class_name']} "
            f"{detection['confidence']:.2f}"
        )

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # --------------------------------------------------
    # 7. Save result
    # --------------------------------------------------

    input_path = Path(image_path)

    output_path = (
        input_path.parent /
        f"{input_path.stem}_detected.jpg"
    )

    cv2.imwrite(
        str(output_path),
        image,
    )

    return results, output_path


# ======================================================
# Command-line interface
# ======================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "  python detect_image.py <image_path>"
        )
        sys.exit(1)

    image_path = sys.argv[1]

    results, output_path = detect(image_path)

    print("\nDETECTION RESULTS")
    print("=" * 50)

    if not results:
        print("No objects detected.")

    else:
        for result in results:
            print(
                f"{result['class_name']:15s}"
                f" confidence={result['confidence']:.4f}"
                f" box={result['box']}"
            )

    print("\nSaved result:")
    print(output_path)
