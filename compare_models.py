import cv2
import numpy as np
import onnxruntime as ort
import sys
import yaml

MODEL_FP32 = r"D:\AntiGravity Projects\mnemosyne\runs\detect\runs\train\aegis_baseline_fast\weights\best.onnx"
MODEL_INT8 = r"D:\AntiGravity Projects\mnemosyne\runs\detect\runs\train\aegis_baseline_fast\weights\best_int8.onnx"
TEST_IMG = r"D:\Robu\aegis-ppe\valid\images\-1001-_png_jpg.rf.259b9e2b3fc199a9e3e9aefc8ac9520d.jpg"
CONF_THRESH = 0.2
INPUT_SIZE = 640

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    # Same standard YOLO letterbox as live_inference.py
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = np.mod(dw, 32) / 2, np.mod(dh, 32) / 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img

def preprocess(frame):
    img = letterbox(frame, new_shape=(INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[2] == 3 else img
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def analyze_model(model_path, label, img_tensor):
    print(f"\n{'='*60}")
    print(f"Analyzing {label}")
    print(f"Path: {model_path}")
    
    # 1. Load Session
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # 2. Inspect Inputs
    inputs = session.get_inputs()
    input_name = inputs[0].name
    input_shape = inputs[0].shape
    input_type = inputs[0].type
    print(f"Input Name: {input_name} | Shape: {input_shape} | Type: {input_type}")
    
    # 3. Inspect Outputs
    outputs = session.get_outputs()
    output_name = outputs[0].name
    output_shape = outputs[0].shape
    print(f"Output Name: {output_name} | Shape: {output_shape}")
    
    # 4. Run Inference
    raw_output = session.run([output_name], {input_name: img_tensor})[0]
    
    # 5. Raw Output Statistics
    print("\n--- Raw Output Tensor Stats ---")
    print(f"Shape: {raw_output.shape}")
    print(f"Mean : {np.mean(raw_output):.4f}")
    print(f"Max  : {np.max(raw_output):.4f}")
    print(f"Min  : {np.min(raw_output):.4f}")
    print(f"NaNs : {np.isnan(raw_output).sum()}")
    print(f"Infs : {np.isinf(raw_output).sum()}")
    
    # 6. Decode & Filter
    # output shape is [1, 16, 8400] for 12 classes YOLOv11 (4 bbox + 12 cls = 16)
    preds = np.transpose(raw_output[0]) # shape [8400, 16]
    
    # Extract boxes and class scores
    boxes = preds[:, :4]
    scores = preds[:, 4:]
    
    max_scores = np.max(scores, axis=1)
    class_ids = np.argmax(scores, axis=1)
    
    valid_mask = max_scores > CONF_THRESH
    valid_boxes = boxes[valid_mask]
    valid_scores = max_scores[valid_mask]
    valid_class_ids = class_ids[valid_mask]
    
    print("\n--- Confidence Filtering (Thresh: 0.2) ---")
    print(f"Total proposals: {len(max_scores)}")
    print(f"Proposals > {CONF_THRESH}: {len(valid_scores)}")
    if len(valid_scores) > 0:
        print(f"Highest confidence: {np.max(valid_scores):.4f}")
    
    # 7. Mock NMS count (just counting how many pass threshold to see discrepancies)
    print("\n--- Detections Passing Threshold (Pre-NMS) ---")
    if len(valid_scores) == 0:
        print("0 detections.")
    else:
        for i in range(min(5, len(valid_scores))):
            print(f"  Box: {valid_boxes[i].astype(int)}, Score: {valid_scores[i]:.4f}, Class: {valid_class_ids[i]}")
        if len(valid_scores) > 5:
            print(f"  ... and {len(valid_scores)-5} more")
            
    return raw_output

def main():
    print("PHASE 10.1A: FP32 vs INT8 Output Comparison")
    
    # Load and preprocess image
    frame = cv2.imread(TEST_IMG)
    if frame is None:
        print(f"Error loading {TEST_IMG}")
        return
        
    print(f"\nOriginal Image Shape: {frame.shape}")
    img_tensor = preprocess(frame)
    print(f"Preprocessed Tensor Shape: {img_tensor.shape}")
    print(f"Tensor Mean: {np.mean(img_tensor):.4f}, Max: {np.max(img_tensor):.4f}")
    
    out_fp32 = analyze_model(MODEL_FP32, "FP32 Model", img_tensor)
    out_int8 = analyze_model(MODEL_INT8, "INT8 Model", img_tensor)
    
    print(f"\n{'='*60}")
    print("Direct Tensor Comparison")
    diff = np.abs(out_fp32 - out_int8)
    print(f"Mean Absolute Difference: {np.mean(diff):.4f}")
    print(f"Max Absolute Difference:  {np.max(diff):.4f}")

if __name__ == "__main__":
    main()
