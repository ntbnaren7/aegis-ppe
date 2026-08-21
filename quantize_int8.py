"""
AEGIS - Phase 9.5: INT8 Dynamic Quantization Script

Quantizes best.onnx (FP32) to best_int8.onnx using ONNX Runtime's DYNAMIC quantization.

Why dynamic instead of static?
  Static QDQ quantization is unreliable on YOLO output heads — the per-tensor
  scale calibration tends to collapse the detection head outputs to zero.
  Dynamic quantization only quantizes weight matrices (not activations), which
  completely avoids this problem while still delivering significant model size
  reduction and CPU inference speedup.

Quantization settings:
  - Method:  Dynamic (weight-only, no calibration needed)
  - Weights: QUInt8 (unsigned int8, optimal for ORT CPU execution)

Usage:
    uv run quantize_int8.py
"""

from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType


# ─── Paths ────────────────────────────────────────────────────────────────────
WEIGHTS_DIR = Path(r'D:\AntiGravity Projects\mnemosyne\runs\detect\runs\train\aegis_baseline_fast\weights')
FP32_MODEL  = WEIGHTS_DIR / 'best.onnx'
INT8_MODEL  = WEIGHTS_DIR / 'best_int8.onnx'


def main():
    print("--- AEGIS Phase 9.5: INT8 Dynamic Quantization ---")
    print("Method: weight-only dynamic quantization (no calibration required)")
    print()

    if not FP32_MODEL.exists():
        print(f"ERROR: FP32 ONNX model not found at {FP32_MODEL}")
        print("Run export_onnx.py first.")
        return

    fp32_size_mb = FP32_MODEL.stat().st_size / (1024 * 1024)
    print(f"Source FP32 model: {FP32_MODEL}")
    print(f"Source FP32 size:  {fp32_size_mb:.2f} MB")
    print()

    print("Running dynamic INT8 quantization...")
    quantize_dynamic(
        model_input=str(FP32_MODEL),
        model_output=str(INT8_MODEL),
        weight_type=QuantType.QUInt8,
    )

    int8_size_mb = INT8_MODEL.stat().st_size / (1024 * 1024)

    print()
    print("=" * 50)
    print("--- Quantization Complete ---")
    print(f"INT8 model saved to: {INT8_MODEL}")
    print(f"FP32 size:  {fp32_size_mb:.2f} MB")
    print(f"INT8 size:  {int8_size_mb:.2f} MB")
    print(f"Reduction:  {100 * (1 - int8_size_mb / fp32_size_mb):.1f}%")
    print("=" * 50)
    print()
    print("Next step: Run evaluate_int8.py to verify accuracy.")


if __name__ == '__main__':
    main()
