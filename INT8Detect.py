import os
import cv2

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction




MODEL_PATH = r"PCB\best (1)_int8.onnx"
IMAGE_PATH = r"04_short_14_jpg.rf.05430cdfea750328c3b1c306ba758212.jpg"

CONF = 0.35

SLICE_SIZE = 512
OVERLAP = 0.20

OUTPUT_DIR = "sahi_output"



image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not read: {IMAGE_PATH}"
    )

H, W = image.shape[:2]

print(f"Image size: {W} x {H}")


# ============================================================
# LOAD INT8 YOLO THROUGH SAHI
# ============================================================

print("Loading INT8 YOLO model...")

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=MODEL_PATH,
    confidence_threshold=CONF,
    device="cpu"
)

print("Model loaded.")


# ============================================================
# SAHI INFERENCE
# ============================================================

print("\nRunning INT8 + SAHI...\n")

result = get_sliced_prediction(
    IMAGE_PATH,

    detection_model,

    # --------------------------------------------------------
    # Slice size
    # --------------------------------------------------------
    slice_height=SLICE_SIZE,
    slice_width=SLICE_SIZE,

    # --------------------------------------------------------
    # Overlap
    # --------------------------------------------------------
    overlap_height_ratio=OVERLAP,
    overlap_width_ratio=OVERLAP,

    # --------------------------------------------------------
    # Also run normal YOLO on the complete image
    # --------------------------------------------------------
    perform_standard_pred=True,

    # --------------------------------------------------------
    # Merge overlapping detections
    # --------------------------------------------------------
    postprocess_type="GREEDYNMM",

    postprocess_match_metric="IOS",

    postprocess_match_threshold=0.5,

    verbose=1
)


# ============================================================
# GET DETECTIONS
# ============================================================

detections = result.object_prediction_list


print("\n")
print("=" * 60)
print("INT8 + SAHI FINAL DETECTIONS")
print("=" * 60)

print(
    f"Total detections: {len(detections)}"
)


# ============================================================
# PRINT DETECTIONS
# ============================================================

for i, prediction in enumerate(detections):

    class_name = prediction.category.name

    confidence = prediction.score.value

    x1, y1, x2, y2 = (
        prediction.bbox.to_xyxy()
    )

    print(
        f"{i + 1}. "
        f"{class_name} | "
        f"{confidence:.3f} | "
        f"({x1:.1f}, {y1:.1f}) -> "
        f"({x2:.1f}, {y2:.1f})"
    )


# ============================================================
# SAVE SAHI VISUALIZATION
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

result.export_visuals(
    export_dir=OUTPUT_DIR,
    file_name="int8_sahi_result"
)


# ============================================================
# DISPLAY
# ============================================================

output_path = os.path.join(
    OUTPUT_DIR,
    "int8_sahi_result.png"
)

output = cv2.imread(output_path)

if output is not None:

    cv2.imshow(
        "INT8 YOLO + SAHI",
        output
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


print("\nResult saved to:")
print(output_path)