from ultralytics import YOLO
import os


# ============================================================
# 1. PATHS
# ============================================================

# Your trained YOLOv8m model
MODEL_PATH = r"PCB\best (1).pt"

# EXACT dataset used for training
DATA_YAML = r"content\pcb-2\data.yaml"

    
# ============================================================
# 2. SETTINGS
# ============================================================

IMAGE_SIZE = 640

# Use 100% of validation images for INT8 calibration
CALIBRATION_FRACTION = 1.0


# ============================================================
# 3. CHECK FILES
# ============================================================

print("=" * 60)
print("CHECKING FILES")
print("=" * 60)

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )

if not os.path.isfile(DATA_YAML):
    raise FileNotFoundError(
        f"\ndata.yaml not found:\n{DATA_YAML}"
    )

print("Model found:")
print(MODEL_PATH)

print("\ndata.yaml found:")
print(DATA_YAML)


# ============================================================
# 4. LOAD MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING YOLO MODEL")
print("=" * 60)

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully.")


# ============================================================
# 5. INT8 ONNX EXPORT
# ============================================================

print("\n" + "=" * 60)
print("STARTING INT8 ONNX EXPORT")
print("=" * 60)

print(f"Image size           : {IMAGE_SIZE}")
print(f"Calibration fraction : {CALIBRATION_FRACTION}")
print("Calibration split    : validation")
print("Quantization         : INT8")
print("Dynamic input        : False")
print("Device               : CPU")

export_path = model.export(
    format="onnx",

    # Input resolution
    imgsz=IMAGE_SIZE,

    # INT8 quantization
    quantize=8,

    # Dataset used for calibration
    data=DATA_YAML,
    split="val",
    fraction=CALIBRATION_FRACTION,

    # Fixed input size
    dynamic=False,

    # Simplify ONNX graph
    simplify=True,

    # CPU is sufficient for export
    device="cpu"
)


# ============================================================
# 6. EXPORT COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("INT8 ONNX EXPORT COMPLETE")
print("=" * 60)

print("Exported model:")
print(export_path)


# ============================================================
# 7. CHECK FILE SIZE
# ============================================================

if os.path.isfile(export_path):

    size_mb = os.path.getsize(export_path) / (1024 * 1024)

    print(f"\nINT8 ONNX size: {size_mb:.2f} MB")

else:

    print("\nWARNING: Export path returned but file was not found.")


print("\nDone!")