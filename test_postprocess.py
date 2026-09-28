import cv2
import numpy as np
import onnxruntime as ort

from yolo_postprocess import postprocess


MODEL_PATH = "runs/yolov8n_merged_100ep/weights/best.onnx"

IMAGE_PATH = (
    "data/garbage_raw/test/"
    "plastic355_jpg.rf.ec6dc9f4b8904416df9c82e5646f0855.jpg"
)


# --------------------------------------------------
# 1. Load ONNX model
# --------------------------------------------------

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name

print("ONNX model loaded")
print("Input:", input_name)


# --------------------------------------------------
# 2. Load image
# --------------------------------------------------

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(IMAGE_PATH)

original_height, original_width = image.shape[:2]

print("Original image size:")
print(f"  width  = {original_width}")
print(f"  height = {original_height}")


# --------------------------------------------------
# 3. Preprocess
# --------------------------------------------------

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB,
)

image_resized = cv2.resize(
    image_rgb,
    (640, 640),
)

image_input = image_resized.astype(
    np.float32
) / 255.0

# HWC -> CHW
image_input = np.transpose(
    image_input,
    (2, 0, 1),
)

# Add batch dimension
image_input = np.expand_dims(
    image_input,
    axis=0,
)

print("Model input shape:", image_input.shape)


# --------------------------------------------------
# 4. Run ONNX
# --------------------------------------------------

outputs = session.run(
    None,
    {input_name: image_input},
)

raw_output = outputs[0]

print("Raw output shape:", raw_output.shape)


# --------------------------------------------------
# 5. YOLO post-processing
# --------------------------------------------------

detections = postprocess(
    raw_output,
    confidence_threshold=0.25,
    iou_threshold=0.45,
)


# --------------------------------------------------
# 6. Scale boxes back to original image
# --------------------------------------------------

scale_x = original_width / 640
scale_y = original_height / 640


print("\nFINAL DETECTIONS")
print("=" * 50)

for detection in detections:

    box = detection["box"]

    x1 = box[0] * scale_x
    y1 = box[1] * scale_y
    x2 = box[2] * scale_x
    y2 = box[3] * scale_y

    print(
        f"{detection['class_name']:15s}"
        f" confidence={detection['confidence']:.4f}"
        f" box=["
        f"{x1:.2f}, "
        f"{y1:.2f}, "
        f"{x2:.2f}, "
        f"{y2:.2f}"
        f"]"
    )
