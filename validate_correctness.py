import os
import cv2
import numpy as np
import onnxruntime as ort
import json
import glob
import traceback
from pathlib import Path

# --- Configuration ---
VALID_IMAGES_DIR = "D:/Robu/aegis-ppe/valid/images"
VALID_LABELS_DIR = "D:/Robu/aegis-ppe/valid/labels"
OUTPUT_DIR = "D:/Robu/aegis-ppe/validation_artifacts"

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = {
    "fp32_640": "D:/AntiGravity Projects/mnemosyne/runs/detect/runs/train/aegis_baseline_fast/weights/best.onnx",
    "int8_640": "D:/AntiGravity Projects/mnemosyne/runs/detect/runs/train/aegis_baseline_fast/weights/best_int8.onnx"
}

CLASSES = ['Fall-Detected', 'Gloves', 'Goggles', 'Hardhat', 'Ladder', 'Mask', 'NO-Gloves', 'NO-Goggles', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest']
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def preprocess(frame: np.ndarray, size: int) -> np.ndarray:
    """Exact identical preprocessing from benchmark_uno_q.py"""
    img = cv2.resize(frame, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def decode_predictions(pred: np.ndarray, conf_thres: float = 0.2, iou_thres: float = 0.45):
    """
    Decodes YOLO11 ONNX (1, 18, 8400) to boxes, scores, classes using NMS.
    Returns: xyxy_boxes, scores, class_ids, pre_nms_count
    """
    pred = pred[0].T # (8400, 18)
    boxes = pred[:, :4] # cx, cy, w, h
    scores = pred[:, 4:] # class scores
    
    max_scores = scores.max(axis=1)
    class_ids = scores.argmax(axis=1)
    
    mask = max_scores >= conf_thres
    valid_boxes = boxes[mask]
    valid_scores = max_scores[mask]
    valid_class_ids = class_ids[mask]
    pre_nms_count = np.sum(mask)
    
    if pre_nms_count == 0:
        return np.array([]), np.array([]), np.array([]), 0
        
    # cx,cy,w,h to topleft_x, topleft_y, w, h for OpenCV NMS
    x1 = valid_boxes[:, 0] - valid_boxes[:, 2] / 2
    y1 = valid_boxes[:, 1] - valid_boxes[:, 3] / 2
    w = valid_boxes[:, 2]
    h = valid_boxes[:, 3]
    
    xywh_topleft = np.stack([x1, y1, w, h], axis=1).tolist()
    valid_scores_list = valid_scores.tolist()
    
    indices = cv2.dnn.NMSBoxes(xywh_topleft, valid_scores_list, conf_thres, iou_thres)
    
    if len(indices) > 0:
        indices = indices.flatten()
        # convert to xyxy for drawing
        x2 = x1 + w
        y2 = y1 + h
        xyxy = np.stack([x1, y1, x2, y2], axis=1)
        return xyxy[indices], valid_scores[indices], valid_class_ids[indices], pre_nms_count
    else:
        return np.array([]), np.array([]), np.array([]), pre_nms_count

def select_diverse_images(num_images=10):
    images = glob.glob(os.path.join(VALID_IMAGES_DIR, "*.jpg"))
    selected = []
    seen_classes = set()
    
    for img_path in images:
        label_path = os.path.join(VALID_LABELS_DIR, Path(img_path).stem + ".txt")
        if not os.path.exists(label_path): continue
            
        with open(label_path, 'r') as f:
            lines = f.readlines()
            if not lines: continue
            
            img_classes = {int(line.split()[0]) for line in lines if line.strip()}
            
        # Give priority to images that have classes we haven't seen yet, or have >1 class
        if not img_classes.issubset(seen_classes) or len(img_classes) > 1:
            selected.append((img_path, img_classes, len(lines)))
            seen_classes.update(img_classes)
            if len(selected) >= num_images:
                break
                
    # If we didn't find enough diverse images, pad with remaining
    if len(selected) < num_images:
        for img_path in images:
            if img_path not in [s[0] for s in selected]:
                selected.append((img_path, set(), 0))
            if len(selected) >= num_images:
                break
                
    return selected

def draw_boxes(image, boxes, scores, class_ids):
    img = image.copy()
    if len(boxes) == 0: return img
    
    # Scale boxes to image size
    h_ratio, w_ratio = img.shape[0] / 640.0, img.shape[1] / 640.0
    
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        x1, x2 = int(x1 * w_ratio), int(x2 * w_ratio)
        y1, y2 = int(y1 * h_ratio), int(y2 * h_ratio)
        
        color = COLORS[cls_id]
        label = f"{CLASSES[cls_id]}: {score:.2f}"
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

def main():
    print("=== PHASE 10.1B: Correctness Validation ===")
    
    test_images_info = select_diverse_images(10)
    print(f"Selected {len(test_images_info)} test images.")
    for img_path, cls_set, count in test_images_info:
        print(f"  {Path(img_path).name} | GT Objs: {count} | Classes: {[CLASSES[c] for c in cls_set]}")
        
    # --- CHECKPOINT 2: MODEL INSPECTION ---
    sessions = {}
    for name, path in MODELS.items():
        sessions[name] = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        
    print("\n--- RUNNING CHECKPOINTS ---")
    
    results_report = {}
    
    for idx, (img_path, gt_classes, gt_count) in enumerate(test_images_info):
        img_name = Path(img_path).stem
        print(f"\n[{idx+1}/10] Testing: {img_name}")
        
        orig_img = cv2.imread(img_path)
        img_report = {"gt_count": gt_count, "gt_classes": list(gt_classes)}
        
        # --- CHECKPOINT 1: INPUT CONSISTENCY ---
        tensor_fp32 = preprocess(orig_img, 640)
        tensor_int8 = preprocess(orig_img, 640)
        
        diff = np.abs(tensor_fp32 - tensor_int8).max()
        if diff > 0:
            print(f"  [!] CRITICAL FAIL: Tensors differ. Max diff: {diff}")
            return
            
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{img_name}_01_orig.jpg"), orig_img)
        
        # --- CHECKPOINT 3 & 4 & 6: RAW OUTPUT & NMS VALIDATION ---
        fp32_out = sessions["fp32_640"].run(None, {sessions["fp32_640"].get_inputs()[0].name: tensor_fp32})[0]
        int8_out = sessions["int8_640"].run(None, {sessions["int8_640"].get_inputs()[0].name: tensor_int8})[0]
        
        if np.isnan(fp32_out).any() or np.isnan(int8_out).any():
            print("  [!] CRITICAL FAIL: NaN in output!")
            return
            
        fp32_boxes, fp32_scores, fp32_cls, fp32_pre_nms = decode_predictions(fp32_out, 0.2, 0.45)
        int8_boxes, int8_scores, int8_cls, int8_pre_nms = decode_predictions(int8_out, 0.2, 0.45)
        
        img_report["fp32_raw_mean"] = float(fp32_out.mean())
        img_report["int8_raw_mean"] = float(int8_out.mean())
        img_report["fp32_pre_nms"] = int(fp32_pre_nms)
        img_report["int8_pre_nms"] = int(int8_pre_nms)
        img_report["fp32_post_nms"] = len(fp32_boxes)
        img_report["int8_post_nms"] = len(int8_boxes)
        
        print(f"  FP32: {fp32_pre_nms} pre-NMS -> {len(fp32_boxes)} post-NMS")
        print(f"  INT8: {int8_pre_nms} pre-NMS -> {len(int8_boxes)} post-NMS")
        
        fp32_vis = draw_boxes(orig_img, fp32_boxes, fp32_scores, fp32_cls)
        int8_vis = draw_boxes(orig_img, int8_boxes, int8_scores, int8_cls)
        
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{img_name}_03_fp32.jpg"), fp32_vis)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{img_name}_04_int8.jpg"), int8_vis)
        
        # --- CHECKPOINT 5: THRESHOLD SWEEP ---
        img_report["thresholds"] = {}
        for t in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]:
            f_box, _, _, _ = decode_predictions(fp32_out, t, 0.45)
            i_box, _, _, _ = decode_predictions(int8_out, t, 0.45)
            img_report["thresholds"][str(t)] = {"fp32": len(f_box), "int8": len(i_box)}
            
        results_report[img_name] = img_report

    with open(os.path.join(OUTPUT_DIR, "validation_report.json"), "w") as f:
        json.dump(results_report, f, indent=4)
        
    print(f"\nValidation complete. Artifacts saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
