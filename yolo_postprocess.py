import numpy as np

CLASS_NAMES = [
    "biodegradable",
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
]


def xywh_to_xyxy(boxes):
    """
    Convert boxes from:
        center_x, center_y, width, height

    to:
        x1, y1, x2, y2
    """
    result = np.zeros_like(boxes)

    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    return result


def calculate_iou(box, boxes):
    """
    Calculate IoU between one box and multiple boxes.
    """

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection_width = np.maximum(0, x2 - x1)
    intersection_height = np.maximum(0, y2 - y1)

    intersection = intersection_width * intersection_height

    box_area = (
        (box[2] - box[0]) *
        (box[3] - box[1])
    )

    boxes_area = (
        (boxes[:, 2] - boxes[:, 0]) *
        (boxes[:, 3] - boxes[:, 1])
    )

    union = box_area + boxes_area - intersection

    return intersection / (union + 1e-6)


def nms(boxes, scores, iou_threshold=0.45):
    """
    Non-Maximum Suppression.

    Keeps the highest-confidence box and removes
    overlapping boxes.
    """

    order = scores.argsort()[::-1]

    keep = []

    while len(order) > 0:
        current = order[0]
        keep.append(current)

        if len(order) == 1:
            break

        ious = calculate_iou(
            boxes[current],
            boxes[order[1:]]
        )

        remaining = np.where(
            ious < iou_threshold
        )[0]

        order = order[remaining + 1]

    return np.array(keep, dtype=np.int64)


def postprocess(
    output,
    confidence_threshold=0.25,
    iou_threshold=0.45,
):
    """
    Convert YOLO raw output:

        [1, 10, 8400]

    into final detections.

    10 =
        4 box values
        +
        6 class scores
    """

    # Remove batch dimension
    predictions = output[0]

    # [10, 8400] -> [8400, 10]
    predictions = predictions.T

    # First 4 values are bounding boxes
    boxes = predictions[:, :4]

    # Remaining 6 values are class scores
    class_scores = predictions[:, 4:]

    # Best class for each candidate
    class_ids = np.argmax(class_scores, axis=1)
    scores = np.max(class_scores, axis=1)

    # Confidence filtering
    mask = scores >= confidence_threshold

    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return []

    # Convert xywh -> xyxy
    boxes = xywh_to_xyxy(boxes)

    # NMS separately for each class
    final_detections = []

    for class_id in np.unique(class_ids):

        class_mask = class_ids == class_id

        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]

        keep = nms(
            class_boxes,
            class_scores,
            iou_threshold,
        )

        original_indices = np.where(class_mask)[0]

        for idx in keep:

            original_idx = original_indices[idx]

            final_detections.append({
                "class_id": int(class_id),
                "class_name": CLASS_NAMES[class_id],
                "confidence": float(scores[original_idx]),
                "box": boxes[original_idx].tolist(),
            })

    # Highest confidence first
    final_detections.sort(
        key=lambda x: x["confidence"],
        reverse=True,
    )

    return final_detections
