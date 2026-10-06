# ============================================================
# YOLOv8m QAT -> INT8 ONNX EXPORT
# Windows-safe deployment script
# ============================================================

from ultralytics import YOLO
import os
import sys


# ============================================================
# 1. PATHS
# ============================================================

QAT_MODEL_PATH = r"QAT\yolov8m_qat\weights\best.pt"

# Change this to an actual PCB test image
TEST_IMAGE = r"04_short_14_jpg.rf.05430cdfea750328c3b1c306ba758212.jpg"

# Dataset YAML
DATA_YAML = r"content\pcb-2\data.yaml"

# Image size used during QAT
IMG_SIZE = 640


# ============================================================
# 2. CHECK FILES
# ============================================================

print("=" * 70)
print("YOLOv8m QAT -> INT8 ONNX EXPORT")
print("=" * 70)

print("\nChecking QAT model...")

if not os.path.isfile(QAT_MODEL_PATH):
    raise FileNotFoundError(
        f"\nQAT model not found:\n{QAT_MODEL_PATH}\n"
    )

print("QAT model found:")
print(QAT_MODEL_PATH)

model_size_mb = os.path.getsize(QAT_MODEL_PATH) / (1024 * 1024)

print(f"QAT best.pt size: {model_size_mb:.2f} MB")


# ============================================================
# 3. LOAD QAT MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING QAT MODEL")
print("=" * 70)

model = YOLO(QAT_MODEL_PATH)

print("QAT model loaded successfully.")


# ============================================================
# 4. EXPORT TO INT8 ONNX
# ============================================================

print("\n" + "=" * 70)
print("EXPORTING QAT MODEL TO INT8 ONNX")
print("=" * 70)

print(f"Image size: {IMG_SIZE}")
print("Quantization: INT8")
print("Format: ONNX")

try:

    exported_path = model.export(
        format="onnx",
        imgsz=IMG_SIZE,
        quantize=8
    )

except Exception as e:

    print("\nERROR DURING ONNX EXPORT")
    print("----------------------------------------")
    print(str(e))

    raise


# ============================================================
# 5. FIND EXPORTED MODEL
# ============================================================

print("\n" + "=" * 70)
print("EXPORT COMPLETE")
print("=" * 70)

print("Returned export path:")
print(exported_path)

if isinstance(exported_path, str):

    ONNX_PATH = exported_path

else:

    # Fallback: derive ONNX filename from QAT checkpoint
    ONNX_PATH = os.path.splitext(QAT_MODEL_PATH)[0] + ".onnx"


# Check that ONNX actually exists

if not os.path.isfile(ONNX_PATH):

    raise FileNotFoundError(
        f"\nONNX file was not found:\n{ONNX_PATH}"
    )


# ============================================================
# 6. PRINT ONNX SIZE
# ============================================================

onnx_size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)

print("\nINT8 ONNX MODEL:")
print(ONNX_PATH)

print(f"\nINT8 ONNX size: {onnx_size_mb:.2f} MB")


# ============================================================
# 7. VERIFY ONNX MODEL
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING ONNX MODEL")
print("=" * 70)

try:

    import onnx

    onnx_model = onnx.load(ONNX_PATH)

    # Check graph validity
    onnx.checker.check_model(onnx_model)

    print("ONNX checker: PASS")

except ImportError:

    print(
        "onnx package is not installed."
    )

    print(
        "Install it with:\n"
        "pip install onnx"
    )

    sys.exit(1)

except Exception as e:

    print("ONNX checker: FAILED")

    print(str(e))

    raise


# ============================================================
# 8. CHECK QUANTIZATION NODES
# ============================================================

print("\n" + "=" * 70)
print("CHECKING QUANTIZATION NODES")
print("=" * 70)

nodes = onnx_model.graph.node

quantize_nodes = [
    node
    for node in nodes
    if node.op_type == "QuantizeLinear"
]

dequantize_nodes = [
    node
    for node in nodes
    if node.op_type == "DequantizeLinear"
]

print(f"Total ONNX nodes       : {len(nodes)}")
print(f"QuantizeLinear nodes   : {len(quantize_nodes)}")
print(f"DequantizeLinear nodes : {len(dequantize_nodes)}")


if len(dequantize_nodes) > 0:

    print("\nQ/DQ nodes detected.")
    print("The exported ONNX graph contains quantization operations.")

else:

    print("\nWARNING:")
    print("No DequantizeLinear nodes were detected.")


# ============================================================
# 9. LOAD INT8 ONNX WITH ULTRALYTICS
# ============================================================

print("\n" + "=" * 70)
print("LOADING INT8 ONNX")
print("=" * 70)

try:

    int8_model = YOLO(ONNX_PATH)

    print("INT8 ONNX loaded successfully.")

except Exception as e:

    print("Could not load exported ONNX with Ultralytics.")

    print(str(e))

    raise


# ============================================================
# 10. OPTIONAL TEST IMAGE INFERENCE
# ============================================================

print("\n" + "=" * 70)
print("TEST INFERENCE")
print("=" * 70)

if os.path.isfile(TEST_IMAGE):

    print("Test image:")
    print(TEST_IMAGE)

    results = int8_model.predict(
        source=TEST_IMAGE,
        imgsz=IMG_SIZE,
        conf=0.25,
        save=True,
        verbose=True
    )

    print("\nInference completed successfully.")

else:

    print("Test image not found.")

    print("Skipping inference.")

    print("\nIf you want to test an image, change:")
    print("TEST_IMAGE = r'path\\to\\your\\image.jpg'")


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print("\nQAT checkpoint:")
print(QAT_MODEL_PATH)

print(f"QAT checkpoint size: {model_size_mb:.2f} MB")

print("\nINT8 ONNX:")
print(ONNX_PATH)

print(f"INT8 ONNX size: {onnx_size_mb:.2f} MB")

print("\nQuantization nodes:")
print(f"QuantizeLinear   : {len(quantize_nodes)}")
print(f"DequantizeLinear : {len(dequantize_nodes)}")

print("\nSTATUS:")
print("QAT checkpoint -> INT8 ONNX export completed.")

print("\nNext step:")
print("Benchmark the INT8 ONNX model against your original FP32 model.")