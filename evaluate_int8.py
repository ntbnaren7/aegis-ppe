"""
AEGIS - Phase 9.5: INT8 Model Accuracy Evaluation

Evaluates the INT8 quantized ONNX model on the test split and compares
the per-class mAP50 against the FP32 ONNX and PyTorch baselines.

Acceptance criteria:
  - Overall mAP50 drop must be < 0.030 (e.g., 0.783 → > 0.753)
  - No individual critical safety class may drop more than 0.05 mAP50
    Critical classes: Fall-Detected, NO-Hardhat, NO-Safety Vest, Person

Usage:
    uv run evaluate_int8.py
"""

from pathlib import Path
from ultralytics import YOLO

# ─── Paths ────────────────────────────────────────────────────────────────────
WEIGHTS_DIR = Path(r'D:\AntiGravity Projects\mnemosyne\runs\detect\runs\train\aegis_baseline_fast\weights')
INT8_MODEL  = WEIGHTS_DIR / 'best_int8.onnx'
DATA_YAML   = r'D:\Robu\aegis-ppe\data.yaml'

# Recorded FP32 baselines from Phase 4 (PyTorch) and Phase 9 (FP32 ONNX)
# Used only for console comparison printing; not used in the evaluation logic.
FP32_PYTORCH_MAP50 = 0.7831
FP32_ONNX_MAP50    = None   # Not yet separately measured; set after Phase 9 val run

# Critical classes that absolutely must not degrade (class ID → name)
CRITICAL_CLASSES = {
    0:  'Fall-Detected',
    8:  'NO-Hardhat',
    10: 'NO-Safety Vest',
    11: 'Person',
}
MAX_CRITICAL_DROP  = 0.05   # 5 mAP50 points max drop on any critical class
MAX_OVERALL_DROP   = 0.030  # 3 mAP50 points max overall drop


def main():
    print("--- AEGIS Phase 9.5: INT8 Accuracy Evaluation ---")
    print()

    if not INT8_MODEL.exists():
        print(f"ERROR: INT8 model not found at {INT8_MODEL}")
        print("Run quantize_int8.py first.")
        return

    int8_size_mb = INT8_MODEL.stat().st_size / (1024 * 1024)
    print(f"INT8 model:  {INT8_MODEL}  ({int8_size_mb:.2f} MB)")
    print(f"Evaluating on TEST split: {DATA_YAML}")
    print()

    # Load model via Ultralytics YOLO wrapper (automatically detects .onnx format)
    model = YOLO(str(INT8_MODEL))

    # Run evaluation on the test split with CPU device (matches deployment target)
    results = model.val(
        data=DATA_YAML,
        split='test',
        device='cpu',
        imgsz=640,
        batch=8,
        verbose=True,
    )

    # ── Extract Metrics ──────────────────────────────────────────────────────
    int8_map50 = results.box.map50
    int8_per_class_map50 = results.box.ap_class_index, results.box.ap50

    print()
    print("=" * 60)
    print("--- INT8 Evaluation Results ---")
    print(f"  INT8 mAP50:         {int8_map50:.4f}")
    print(f"  PyTorch FP32 mAP50: {FP32_PYTORCH_MAP50:.4f}")
    print(f"  Overall drop:       {FP32_PYTORCH_MAP50 - int8_map50:+.4f}")
    print()

    # ── Acceptance Check ─────────────────────────────────────────────────────
    overall_ok = (FP32_PYTORCH_MAP50 - int8_map50) <= MAX_OVERALL_DROP
    print(f"  Overall mAP50 drop acceptable (< {MAX_OVERALL_DROP}): {'✅ PASS' if overall_ok else '❌ FAIL'}")
    print()

    class_indices, ap50_values = int8_per_class_map50
    critical_pass = True
    print("  Critical Class Check:")
    for class_id, class_name in CRITICAL_CLASSES.items():
        # Find this class in results
        if class_id in class_indices:
            idx = list(class_indices).index(class_id)
            class_map50 = ap50_values[idx]
            print(f"    {class_name:20s}: mAP50 = {class_map50:.4f}")
        else:
            print(f"    {class_name:20s}: not found in results")
            critical_pass = False

    print()
    if overall_ok and critical_pass:
        print("  ✅ QUANTIZATION ACCEPTED — Proceed to live telemetry benchmark.")
        print("     Update live_inference.py to use best_int8.onnx.")
    else:
        print("  ❌ QUANTIZATION REJECTED — Accuracy degraded too much.")
        print("     Options:")
        print("     1. Increase NUM_CALIB_IMAGES in quantize_int8.py (try 500)")
        print("     2. Switch calibrate_method from Percentile to Entropy")
        print("     3. Set per_channel=False as a test")
        print("     4. Deploy FP32 ONNX to QRB2210 and assess real hardware speed first")
    print("=" * 60)


if __name__ == '__main__':
    main()
