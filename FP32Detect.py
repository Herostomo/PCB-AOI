import os
import cv2

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction



MODEL_PATH = r"PCB\best (1).pt"
IMAGE_PATH = r"04_short_14_jpg.rf.05430cdfea750328c3b1c306ba758212.jpg"

OUTPUT_DIR = "sahi_output"

CONFIDENCE = 0.40
SLICE_SIZE = 512
OVERLAP = 0.20

DEVICE = "cpu"



image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Cannot read image: {IMAGE_PATH}"
    )

H, W = image.shape[:2]

print(f"Input image: {W} x {H}")



model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=MODEL_PATH,
    confidence_threshold=CONFIDENCE,
    device=DEVICE
)



result = get_sliced_prediction(

    IMAGE_PATH,

    model,

    # These are SLICE dimensions,
    # NOT original image dimensions.
    slice_height=SLICE_SIZE,
    slice_width=SLICE_SIZE,

    overlap_height_ratio=OVERLAP,
    overlap_width_ratio=OVERLAP,

    # Run normal YOLO on the complete image too
    perform_standard_pred=True,

    # Merge overlapping detections
    postprocess_type="GREEDYNMM",
    postprocess_match_metric="IOS",
    postprocess_match_threshold=0.5,

    verbose=1
)


# ============================================================
# RESULTS
# ============================================================

detections = result.object_prediction_list

print("\n" + "=" * 60)
print("FINAL DETECTIONS")
print("=" * 60)

print(f"Total detections: {len(detections)}")


for i, pred in enumerate(detections):

    name = pred.category.name
    confidence = pred.score.value

    x1, y1, x2, y2 = pred.bbox.to_xyxy()

    print(
        f"{i + 1}. "
        f"{name} | "
        f"{confidence:.3f} | "
        f"({x1:.1f}, {y1:.1f}, "
        f"{x2:.1f}, {y2:.1f})"
    )


# ============================================================
# SAVE RESULT
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

result.export_visuals(
    export_dir=OUTPUT_DIR,
    file_name="pcb_result"
)


# ============================================================
# DISPLAY
# ============================================================

visualized = cv2.imread(
    os.path.join(
        OUTPUT_DIR,
        "pcb_result.png"
    )
)

if visualized is not None:

    cv2.imshow(
        "PCB YOLO + SAHI",
        visualized
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


print("\nResult saved in:", OUTPUT_DIR)