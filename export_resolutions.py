import os
from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType

PT_MODEL = r"D:\AntiGravity Projects\mnemosyne\runs\detect\runs\train\aegis_baseline_fast\weights\best.pt"
OUTPUT_DIR = r"D:\AntiGravity Projects\mnemosyne\runs\detect\runs\train\aegis_baseline_fast\weights"

resolutions = [512, 416, 320]

def main():
    print("PHASE 10.1D: Re-exporting models at lower resolutions...", flush=True)
    
    for res in resolutions:
        print(f"\n--- Exporting at {res}x{res} (this may take a few minutes) ---", flush=True)
        
        # We copy best.pt to a temp file so Ultralytics export doesn't overwrite our main best.onnx
        temp_pt = os.path.join(OUTPUT_DIR, f"best_{res}.pt")
        if not os.path.exists(temp_pt):
            import shutil
            shutil.copy(PT_MODEL, temp_pt)
            
        model = YOLO(temp_pt)
        
        # 1. Export FP32 ONNX
        fp32_path = os.path.join(OUTPUT_DIR, f"best_{res}.onnx")
        if not os.path.exists(fp32_path):
            model.export(format="onnx", imgsz=res, dynamic=False, simplify=True)
            
        # 2. Dynamic INT8 Quantization
        int8_path = os.path.join(OUTPUT_DIR, f"best_int8_{res}.onnx")
        if not os.path.exists(int8_path):
            print(f"Quantizing to {int8_path} (this takes ~3-5 mins)...", flush=True)
            quantize_dynamic(
                model_input=fp32_path,
                model_output=int8_path,
                weight_type=QuantType.QUInt8
            )
        print(f"Successfully generated: {int8_path}")

if __name__ == "__main__":
    main()
