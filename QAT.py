# ============================================================
# YOLOv8m — QAT FINE-TUNING
# Windows-safe version
# ============================================================

from ultralytics import YOLO
import torch
import os


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # 1. PATHS
    # ========================================================

    MODEL_PATH = r"PCB\best (1).pt"

    DATA_YAML = r"content\pcb-2\data.yaml"

    OUTPUT_DIR = r"C:\Users\kshit\PCB_AOI\QAT"


    # ========================================================
    # 2. GPU CHECK
    # ========================================================

    print("=" * 60)
    print("GPU CHECK")
    print("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available."
        )

    DEVICE = 0

    print("CUDA available :", True)
    print("GPU            :", torch.cuda.get_device_name(0))


    # ========================================================
    # 3. CHECK FILES
    # ========================================================

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not os.path.isfile(DATA_YAML):
        raise FileNotFoundError(
            f"Dataset YAML not found:\n{DATA_YAML}"
        )

    print("\nModel:")
    print(MODEL_PATH)

    print("\nDataset:")
    print(DATA_YAML)


    # ========================================================
    # 4. LOAD FP32 MODEL
    # ========================================================

    print("\n" + "=" * 60)
    print("LOADING FP32 MODEL")
    print("=" * 60)

    model = YOLO(MODEL_PATH)

    print("Model loaded successfully.")


    # ========================================================
    # 5. QAT TRAINING
    # ========================================================

    print("\n" + "=" * 60)
    print("STARTING QAT")
    print("=" * 60)

    results = model.train(

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------
        data=DATA_YAML,

        # ----------------------------------------------------
        # Quantization-aware training
        # ----------------------------------------------------
        quantize=8,

        # ----------------------------------------------------
        # Fine-tuning
        # ----------------------------------------------------
        epochs=10,
        imgsz=640,
        batch=8,

        device=DEVICE,

        # ----------------------------------------------------
        # IMPORTANT FOR WINDOWS
        # ----------------------------------------------------
        workers=0,

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------
        optimizer="AdamW",

        # Small LR because best.pt is already trained
        lr0=1e-5,
        lrf=0.1,

        weight_decay=0.0005,

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------
        patience=5,

        # ----------------------------------------------------
        # Conservative augmentation
        # ----------------------------------------------------
        degrees=0.0,
        translate=0.05,
        scale=0.5,
        shear=0.0,
        perspective=0.0,

        fliplr=0.5,
        flipud=0.0,

        mosaic=0.0,
        mixup=0.0,

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------
        val=True,

        # ----------------------------------------------------
        # Reproducibility
        # ----------------------------------------------------
        seed=42,
        deterministic=True,

        # ----------------------------------------------------
        # Saving
        # ----------------------------------------------------
        save=True,
        save_period=1,

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------
        project=OUTPUT_DIR,
        name="yolov8m_qat",
        exist_ok=False
    )


    # ========================================================
    # 6. OUTPUT PATHS
    # ========================================================

    QAT_BEST = os.path.join(
        OUTPUT_DIR,
        "yolov8m_qat",
        "weights",
        "best.pt"
    )

    QAT_LAST = os.path.join(
        OUTPUT_DIR,
        "yolov8m_qat",
        "weights",
        "last.pt"
    )


    # ========================================================
    # 7. COMPLETE
    # ========================================================

    print("\n" + "=" * 60)
    print("QAT TRAINING COMPLETE")
    print("=" * 60)

    print("QAT best:")
    print(QAT_BEST)

    print("\nQAT last:")
    print(QAT_LAST)

    if os.path.isfile(QAT_BEST):

        size_mb = (
            os.path.getsize(QAT_BEST)
            / (1024 * 1024)
        )

        print(
            f"\nQAT best.pt size: "
            f"{size_mb:.2f} MB"
        )

    else:

        print(
            "\nWARNING: QAT best.pt "
            "was not found."
        )


# ============================================================
# WINDOWS MULTIPROCESSING ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import multiprocessing

    multiprocessing.freeze_support()

    main()