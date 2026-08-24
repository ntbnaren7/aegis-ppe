#!/usr/bin/env python3
"""
AEGIS Phase 10.1: Resolution Scaling Hardware Benchmark
Runs INT8 YOLOv11 ONNX inference natively on the Arduino UNO Q at multiple resolutions.
"""

import os
import sys
import time
import datetime
import traceback
import argparse
from pathlib import Path

import cv2
import numpy as np
import psutil
import onnxruntime as ort

# ─── Benchmark Configuration ──────────────────────────────────────────────────

CAMERA_INDEX    = 0
RESOLUTIONS     = [640, 512, 416, 320]
CONF_THRESH     = 0.2
WARMUP_FRAMES   = 30
BENCHMARK_SECS  = 30
NUM_CLASSES     = 14

DIR_NAME = os.path.dirname(os.path.abspath(__file__))


# ─── Utilities ───────────────────────────────────────────────────────────────

def read_thermal_celsius():
    """Read temperature from QRB2210 thermal zones."""
    try:
        zone_path = "/sys/class/thermal/thermal_zone0/temp"
        if os.path.exists(zone_path):
            with open(zone_path, "r") as f:
                return float(f.read().strip()) / 1000.0
    except:
        pass
    return None


def preprocess(frame: np.ndarray, size: int) -> np.ndarray:
    """Preprocess a BGR frame to ONNX model input (BCHW float32 [0,1])."""
    img = cv2.resize(frame, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def nms_numpy(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def postprocess_count(output: np.ndarray, conf_thres: float) -> dict:
    """
    Decodes YOLO11 ONNX (1, 18, 8400) to boxes using NMS and returns detailed metrics.
    Ensures exactly 14 class scores are interpreted.
    """
    pred = output[0].T # (8400, 18)
    boxes = pred[:, :4] # cx, cy, w, h
    scores = pred[:, 4:18] # Exactly 14 class scores
    
    max_scores = scores.max(axis=1)
    class_ids = scores.argmax(axis=1)
    
    mask = max_scores >= conf_thres
    pre_nms_count = int(np.sum(mask))
    
    if pre_nms_count == 0:
        return {
            'raw_predictions_above_threshold': 0,
            'final_detections': 0,
            'classes': [],
            'confidences': []
        }
        
    valid_boxes = boxes[mask]
    valid_scores = max_scores[mask]
    valid_class_ids = class_ids[mask]
    
    # cx,cy,w,h to x1, y1, x2, y2 for Numpy NMS
    x1 = valid_boxes[:, 0] - valid_boxes[:, 2] / 2
    y1 = valid_boxes[:, 1] - valid_boxes[:, 3] / 2
    x2 = valid_boxes[:, 0] + valid_boxes[:, 2] / 2
    y2 = valid_boxes[:, 1] + valid_boxes[:, 3] / 2
    
    xyxy = np.stack([x1, y1, x2, y2], axis=1)
    
    indices = nms_numpy(xyxy, valid_scores, 0.45)
    
    final_detections = len(indices)
    if final_detections > 0:
        final_classes = [int(valid_class_ids[i]) for i in indices]
        final_confs = [float(valid_scores[i]) for i in indices]
    else:
        final_classes = []
        final_confs = []
        
    return {
        'raw_predictions_above_threshold': pre_nms_count,
        'final_detections': final_detections,
        'classes': final_classes,
        'confidences': final_confs
    }


def load_model(model_path: Path):
    """Load an ONNX model and return the session."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    t0 = time.perf_counter()
    session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    load_time_ms = (time.perf_counter() - t0) * 1000
    active_provider = session.get_providers()[0]
    return session, active_provider, load_time_ms


def benchmark_model(label: str, model_path: Path, cap: cv2.VideoCapture, input_size: int) -> dict:
    """Run a full benchmark for a single model at a given resolution."""
    print(f"\n{'='*60}")
    print(f"  Benchmarking: {label}  ({model_path.name})")
    print(f"  ORT: {ort.__version__}")
    print("=" * 60)

    result = {
        'label':              label,
        'model_path':         str(model_path),
        'model_size_mb':      round(model_path.stat().st_size / (1024*1024), 2) if model_path.exists() else 0,
        'ort_version':        ort.__version__,
        'execution_provider': None,
        'model_load_time_ms': None,
        'warmup_latency_ms':  None,
        'sustained_fps':      None,
        'inf_latency_mean_ms':None,
        'inf_latency_p95_ms': None,
        'e2e_latency_mean_ms':None,
        'cpu_mean_pct':       None,
        'ram_mean_mb':        None,
        'thermal_mean_c':     None,
        'detections_per_frame': None,
        'zero_detection_frames': None,
        'runtime_errors':     []
    }

    print("  Loading model...")
    try:
        session, provider, load_ms = load_model(model_path)
        result['model_load_time_ms'] = round(load_ms, 1)
        result['execution_provider'] = provider
        input_name = session.get_inputs()[0].name
        print(f"  Provider: {provider}  |  Load: {load_ms:.0f} ms")
    except Exception as e:
        result['runtime_errors'].append(f"Load failed: {traceback.format_exc()}")
        print(f"  LOAD ERROR: {e}")
        return result

    process = psutil.Process(os.getpid())
    psutil.cpu_percent()

    # Warm-up
    print(f"  Warming up ({WARMUP_FRAMES} frames)...")
    warmup_latency_ms = None
    for i in range(WARMUP_FRAMES):
        ret, frame = cap.read()
        if not ret: break
        img = preprocess(frame, input_size)
        t0 = time.perf_counter()
        session.run(None, {input_name: img})
        elapsed = (time.perf_counter() - t0) * 1000
        if i == 0: warmup_latency_ms = elapsed
    
    result['warmup_latency_ms'] = round(warmup_latency_ms, 1) if warmup_latency_ms else None
    if warmup_latency_ms is not None:
        print(f"  Warm-up first frame: {warmup_latency_ms:.1f} ms")
    else:
        print("  Warm-up failed: no frames read")

    # Sustained benchmark
    print(f"  Running {BENCHMARK_SECS}s sustained benchmark...")
    
    inf_latencies, post_latencies, e2e_latencies = [], [], []
    cpus, rams, thermals, det_counts = [], [], [], []
    frames_done = 0
    t_bench_start = time.perf_counter()

    while (time.perf_counter() - t_bench_start) < BENCHMARK_SECS:
        t_e2e_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret: break
        
        img = preprocess(frame, input_size)
        t_inf_start = time.perf_counter()
        try:
            outputs = session.run(None, {input_name: img})
            inf_ms = (time.perf_counter() - t_inf_start) * 1000
            
            t_post_start = time.perf_counter()
            det_count = postprocess_count(outputs[0], CONF_THRESH)
            post_ms = (time.perf_counter() - t_post_start) * 1000
            
            e2e_ms = (time.perf_counter() - t_e2e_start) * 1000
            
            inf_latencies.append(inf_ms)
            post_latencies.append(post_ms)
            e2e_latencies.append(e2e_ms)
            cpus.append(psutil.cpu_percent())
            rams.append(process.memory_info().rss / (1024 * 1024))
            thermal = read_thermal_celsius()
            if thermal is not None: thermals.append(thermal)
            det_counts.append(det_count)
            frames_done += 1
            
            if frames_done % 50 == 0:
                elapsed = time.perf_counter() - t_bench_start
                print(f"    {frames_done} frames | {elapsed:.0f}s elapsed | {frames_done/elapsed:.1f} FPS")
        except Exception as e:
            result['runtime_errors'].append(f"Inference error: {e}")
            continue

    total_time = time.perf_counter() - t_bench_start
    if inf_latencies:
        result['sustained_fps']          = round(frames_done / total_time, 2)
        result['inf_latency_mean_ms']    = round(float(np.mean(inf_latencies)), 1)
        result['inf_latency_p95_ms']     = round(float(np.percentile(inf_latencies, 95)), 1)
        result['post_latency_mean_ms']   = round(float(np.mean(post_latencies)), 1)
        result['e2e_latency_mean_ms']    = round(float(np.mean(e2e_latencies)), 1)
        result['cpu_mean_pct']           = round(float(np.mean(cpus)), 1)
        result['ram_mean_mb']            = round(float(np.mean(rams)), 1)
        result['thermal_mean_c']         = round(float(np.mean(thermals)), 1) if thermals else None
        
        # Aggregate det_counts list of dicts
        avg_raw = np.mean([d['raw_predictions_above_threshold'] for d in det_counts])
        avg_final = np.mean([d['final_detections'] for d in det_counts])
        zero_frames = sum(1 for d in det_counts if d['final_detections'] == 0)
        
        all_classes = []
        all_confs = []
        for d in det_counts:
            all_classes.extend(d['classes'])
            all_confs.extend(d['confidences'])
            
        class_distribution = {c: all_classes.count(c) for c in set(all_classes)}
        mean_conf = float(np.mean(all_confs)) if all_confs else 0.0

        result['raw_predictions_above_threshold'] = round(avg_raw, 2)
        result['final_detections_post_nms'] = round(avg_final, 2)
        result['zero_final_detection_frames'] = int(zero_frames)
        result['class_distribution'] = class_distribution
        result['mean_final_confidence'] = round(mean_conf, 4)

        # Save a debug frame so the user can see what the camera was pointing at
        try:
            debug_img = frame.copy()
            if det_counts and det_counts[-1]['final_detections'] > 0:
                last_det = det_counts[-1]
                for cls_id, conf in zip(last_det['classes'], last_det['confidences']):
                    # Since we don't have the exact boxes stored in det_counts, we just add a text label indicating detections
                    cv2.putText(debug_img, f"Class {cls_id}: {conf:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(DIR_NAME, f"debug_scene_{label}.jpg"), debug_img)
        except Exception as e:
            print(f"    Warning: Could not save debug image: {e}")

    print(f"\n  Results for {label}:")
    print(f"    Sustained FPS:         {result.get('sustained_fps')}")
    print(f"    Inf latency (mean):    {result.get('inf_latency_mean_ms')} ms")
    print(f"    Post latency (mean):   {result.get('post_latency_mean_ms')} ms")
    print(f"    CPU utilization:       {result.get('cpu_mean_pct')} %")
    print(f"    Final Detections/frame:{result.get('final_detections_post_nms')}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='AEGIS Phase 10.1: Resolution Scaling Hardware Benchmark')
    parser.add_argument('--camera', type=int, default=CAMERA_INDEX)
    parser.add_argument('--duration', type=int, default=BENCHMARK_SECS)
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(DIR_NAME, f"aegis_benchmark_{timestamp}.json")

    print("=" * 60)
    print("AEGIS Phase 10.1: Resolution Scaling Hardware Benchmark")
    print(f"Output: {output_path}")
    print("=" * 60)

    # Open camera (scan 0-5, forcing V4L2)
    print("\nOpening camera (forcing V4L2 backend)...")
    cap = None
    actual_camera_idx = None
    for idx in range(6):
        cap_test = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap_test.isOpened():
            ret, _ = cap_test.read()
            if ret:
                cap = cap_test
                actual_camera_idx = idx
                print(f"  Successfully opened camera at index {idx}")
                break
            cap_test.release()

    if cap is None or not cap.isOpened():
        print("ERROR: Could not open any camera on indices 0-5. Exiting.")
        return

    print("Stabilizing camera (5 frames)...")
    for _ in range(5): cap.read()

    platform_info = {
        'python': sys.version,
        'ort_version': ort.__version__,
        'available_providers': ort.get_available_providers(),
        'cpu_count': psutil.cpu_count(),
        'total_ram_mb': round(psutil.virtual_memory().total / (1024*1024)),
    }

    results = {
        'timestamp': timestamp,
        'platform': platform_info,
        'config': {
            'requested_camera_index': args.camera,
            'actual_camera_index': actual_camera_idx,
            'resolutions_tested': RESOLUTIONS,
            'conf_threshold': CONF_THRESH,
            'warmup_frames': WARMUP_FRAMES,
            'benchmark_secs': args.duration,
        },
        'models': {},
        'decision': ''
    }

    # Define models to test
    models_to_test = {}
    for res in RESOLUTIONS:
        name = "best_int8.onnx" if res == 640 else f"best_int8_{res}.onnx"
        models_to_test[f"int8_{res}"] = Path(DIR_NAME) / name

    for label, path in models_to_test.items():
        res = int(label.split('_')[1])
        results['models'][label] = benchmark_model(label, path, cap, res)

    # Decision logic focusing on Phase 10.1 resolution scaling
    fps_320 = results['models'].get('int8_320', {}).get('sustained_fps', 0)
    
    if fps_320 is None:
        decision = "INCONCLUSIVE — 320x320 model failed to run."
    elif fps_320 >= 8.0:
        decision = "320x320 achieves target FPS. Must validate detection accuracy next."
    elif fps_320 >= 4.0:
        decision = "320x320 is faster but still misses 8 FPS target. Test even smaller resolution or consider QNN/HTP."
    else:
        decision = f"Even 320x320 is far too slow (<4 FPS, got {fps_320}). CPU inference is not viable."
        
    print(f"\nDECISION: {decision}")
    print("=" * 120)
    results['decision'] = decision

    import json
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\nAPP LAB COPY-PASTE ZONE START")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    print("=" * 60)
    print("APP LAB COPY-PASTE ZONE END\n")

if __name__ == '__main__':
    main()
