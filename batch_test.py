from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

from yolo_postprocess import postprocess


# =========================
# CONFIG
# =========================

MODEL_PATH = "runs/yolov8n_merged_100ep/weights/best.onnx"
TEST_DIR = Path("data/garbage_raw/test")

IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

NUM_IMAGES = 20


# =========================
# LOAD MODEL ONCE
# =========================

print("Loading ONNX model...")

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

print("Model loaded successfully")
print("Input:", input_name)
print()


# =========================
# GET TEST IMAGES
# =========================

image_paths = sorted(
    list(TEST_DIR.glob("*.jpg")) +
    list(TEST_DIR.glob("*.jpeg")) +
    list(TEST_DIR.glob("*.png"))
)

image_paths = image_paths[:NUM_IMAGES]

print(f"Testing {len(image_paths)} images")
print("=" * 80)


# =========================
# DETECTION FUNCTION
# =========================

def detect(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        return None, None

    original_height, original_width = image.shape[:2]

    # BGR -> RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize
    resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))

    # Normalize
    input_image = resized.astype(np.float32) / 255.0

    # HWC -> CHW
    input_image = np.transpose(input_image, (2, 0, 1))

    # Add batch dimension
    input_tensor = np.expand_dims(input_image, axis=0)

    # ONNX inference
    output = session.run(None, {input_name: input_tensor})[0]

    # YOLO post-processing
    detections = postprocess(
        output,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD
    )

    # Scale boxes back to original image
    scale_x = original_width / IMAGE_SIZE
    scale_y = original_height / IMAGE_SIZE

    for det in detections:

        x1, y1, x2, y2 = det["box"]

        det["box"] = [
            int(x1 * scale_x),
            int(y1 * scale_y),
            int(x2 * scale_x),
            int(y2 * scale_y)
        ]

    return image, detections


# =========================
# RUN TEST
# =========================

for image_path in image_paths:

    image, detections = detect(image_path)

    if image is None:
        print(f"{image_path.name:<45} ERROR")
        continue

    print(f"{image_path.name:<45}", end=" ")

    if not detections:
        print("NO DETECTION")
        continue

    results = []

    for det in detections:
        results.append(
            f'{det["class_name"]} {det["confidence"]:.2f}'
        )

    print(", ".join(results))


print("=" * 80)
print("Batch testing completed.")
