"""
AEGIS - Phase 10.1C: Resolution Accuracy Validation

Evaluates the exported INT8 ONNX models at different resolutions against the held-out test dataset.

Usage:
    uv run evaluate_resolutions.py
"""

import json
from pathlib import Path
from ultralytics import YOLO

# ─── Paths ────────────────────────────────────────────────────────────────────
WEIGHTS_DIR = Path(r'D:\AntiGravity Projects\mnemosyne\runs\detect\runs\train\aegis_baseline_fast\weights')
DATA_YAML   = r'D:\Robu\aegis-ppe\data.yaml'

RESOLUTIONS = [640, 512, 416, 320]

CRITICAL_CLASSES = {
    0:  'Fall-Detected',
    8:  'NO-Hardhat',
    10: 'NO-Safety Vest',
    11: 'Person',
}

def main():
    print("--- AEGIS Phase 10.1C: Resolution Accuracy Validation ---")
    
    results_out = {}
    
    for res in RESOLUTIONS:
        model_name = 'best_int8.onnx' if res == 640 else f'best_int8_{res}.onnx'
        model_path = WEIGHTS_DIR / model_name
        
        if not model_path.exists():
            print(f"Skipping {res}x{res}, model not found: {model_path}")
            continue
            
        print(f"\n==========================================")
        print(f"Evaluating {res}x{res} model: {model_name}")
        print(f"==========================================")
        
        model = YOLO(str(model_path))
        
        res_data = model.val(
            data=DATA_YAML,
            split='test',
            device='cpu',
            imgsz=res,
            batch=8,
            verbose=False,
        )
        
        metrics = res_data.box
        
        results_out[res] = {
            'precision': float(metrics.mp),
            'recall': float(metrics.mr),
            'map50': float(metrics.map50),
            'map50_95': float(metrics.map),
            'critical_classes': {}
        }
        
        class_indices = list(metrics.ap_class_index)
        ap50_values = metrics.ap50
        ap_values = metrics.ap
        
        for class_id, class_name in CRITICAL_CLASSES.items():
            if class_id in class_indices:
                idx = class_indices.index(class_id)
                results_out[res]['critical_classes'][class_name] = {
                    'map50': float(ap50_values[idx]),
                    'map50_95': float(ap_values[idx])
                }
            else:
                results_out[res]['critical_classes'][class_name] = None
                
    with open('aegis_resolution_accuracy.json', 'w') as f:
        json.dump(results_out, f, indent=4)
        
    print("\n\n--- SUMMARY ---")
    for res, data in results_out.items():
        print(f"\n{res}x{res}:")
        print(f"  Precision: {data['precision']:.4f} | Recall: {data['recall']:.4f}")
        print(f"  Overall mAP50: {data['map50']:.4f} | mAP50-95: {data['map50_95']:.4f}")
        print("  Critical Classes mAP50:")
        for cls, cls_data in data['critical_classes'].items():
            if cls_data:
                print(f"    {cls}: {cls_data['map50']:.4f}")
            else:
                print(f"    {cls}: Not found")
                
    print("\nDetailed results saved to aegis_resolution_accuracy.json")

if __name__ == '__main__':
    main()
