"""
AEGIS - Phase 9: ONNX Export Script

Exports the trained PyTorch baseline (best.pt) to a clean, standard FP32 ONNX model
for deployment with ONNX Runtime on the Arduino UNO Q's Qualcomm QRB2210 Linux MPU.

No quantization is applied at this stage. We validate ONNX compatibility first,
and only optimize further once runtime performance is measured on real hardware.

Usage:
    uv run export_onnx.py
"""

import os
from pathlib import Path
from ultralytics import YOLO

def main():
    print("--- AEGIS Phase 9: ONNX Export ---")
    print("Target deployment: Qualcomm QRB2210 MPU (Debian Linux) via ONNX Runtime")
    print()

    model_path = r'D:\AntiGravity Projects\mnemosyne\runs\detect\runs\train\aegis_baseline_fast\weights\best.pt'

    print(f"Loading baseline model: {model_path}")
    model = YOLO(model_path)

    pt_size_mb = Path(model_path).stat().st_size / (1024 * 1024)
    print(f"Source model size: {pt_size_mb:.2f} MB")
    print()

    print("Exporting to ONNX (FP32, no quantization)...")
    print("NOTE: No half/int8 flags — clean baseline export only.")

    # opset=13 enables per_channel=True in quantize_int8.py (axis attr added in opset 13)
    # opset=13 is supported by all modern ONNX Runtime versions including 1.29.0
    export_path = model.export(
        format='onnx',
        imgsz=640,
        opset=13,
        simplify=True,  # Graph simplification for better runtime compatibility
    )

    export_path = Path(export_path)
    onnx_size_mb = export_path.stat().st_size / (1024 * 1024)

    print()
    print("--- Export Complete ---")
    print(f"ONNX model saved to: {export_path}")
    print(f"ONNX model size:     {onnx_size_mb:.2f} MB")
    print(f"Source .pt size:     {pt_size_mb:.2f} MB")
    print(f"Size delta:          {onnx_size_mb - pt_size_mb:+.2f} MB")
    print()
    print("Next step: Copy the .onnx file to the Arduino UNO Q and test with ONNX Runtime.")
    print("           Or run live_inference.py to benchmark ONNX vs PyTorch on this machine first.")

if __name__ == '__main__':
    main()
