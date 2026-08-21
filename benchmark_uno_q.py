"""
AEGIS - Phase 10: Arduino UNO Q Hardware Benchmark

Benchmarks both best.onnx (FP32) and best_int8.onnx (INT8 Dynamic) under
IDENTICAL conditions on the target QRB2210 MPU hardware.

Rules:
  - Same camera index, resolution, confidence threshold, and preprocessing for both runs.
  - Results are written to a timestamped JSON file for reproducibility.
  - ONNX Runtime version and execution provider are recorded in every result.
  - NO performance predictions are made. This script MEASURES; the decision follows from data.

Usage (on the Arduino UNO Q):
    python3 benchmark_uno_q.py

    # Or benchmark a single model explicitly:
    python3 benchmark_uno_q.py --model best.onnx
    python3 benchmark_uno_q.py --model best_int8.onnx

Output:
    aegis_benchmark_<timestamp>.json  (written to the script's directory)

Metrics per model:
    - ort_version           : ONNX Runtime version string
    - execution_provider    : Active ORT execution provider (e.g. CPUExecutionProvider)
    - model_load_time_ms    : Time from InferenceSession() start to first ready state
    - warmup_latency_ms     : Inference latency of the very first frame (cold start)
    - sustained_fps         : Mean FPS over the sustained benchmark window
    - inf_latency_mean_ms   : Mean inference latency over sustained window
    - inf_latency_p95_ms    : 95th percentile inference latency (jitter indicator)
    - e2e_latency_mean_ms   : Mean end-to-end latency (cap.read → annotated frame ready)
    - cpu_mean_pct          : Mean CPU utilization (%)
    - ram_mean_mb           : Mean process RAM (MB)
    - thermal_mean_c        : Mean CPU thermal zone temperature (°C), if readable on device
    - detections_per_frame  : Mean number of bounding boxes per frame (stability indicator)
    - zero_detection_frames : Count of frames with zero detections (detection stability)
    - runtime_errors        : Any non-fatal errors encountered during the run
"""

import argparse
import json
import os
import time
import datetime
import traceback
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import psutil


# ─── Benchmark Configuration ──────────────────────────────────────────────────
# These MUST be identical for both model runs. Do not change between FP32 and INT8.

CAMERA_INDEX    = 0        # Camera device index — set to match your UNO Q camera
INPUT_SIZE      = 640      # Must match training resolution
CONF_THRESHOLD  = 0.2      # Confidence threshold used in live_inference.py
WARMUP_FRAMES   = 30       # Frames to discard before recording (let cache warm up)
BENCHMARK_SECS  = 30       # Duration of the sustained benchmark window (seconds)
NUM_CLASSES     = 14       # YOLO model output classes

# Thermal: Linux /sys/class/thermal — reads first available zone
THERMAL_PATH_GLOB = '/sys/class/thermal/thermal_zone*/temp'

# Model paths — set to match your UNO Q filesystem layout
SCRIPT_DIR  = Path(__file__).parent
MODEL_PATHS = {
    'fp32': SCRIPT_DIR / 'best.onnx',
    'int8': SCRIPT_DIR / 'best_int8.onnx',
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def read_thermal_celsius() -> float | None:
    """Read temperature from first available Linux thermal zone. Returns None if unavailable."""
    import glob
    paths = sorted(glob.glob(THERMAL_PATH_GLOB))
    for p in paths:
        try:
            val = int(Path(p).read_text().strip())
            return val / 1000.0   # millidegrees → degrees
        except Exception:
            continue
    return None


def preprocess(frame: np.ndarray, size: int) -> np.ndarray:
    """
    Preprocess a BGR frame to ONNX model input (BCHW float32 [0,1]).
    Must exactly match the preprocessing used in live_inference.py.
    """
    img = cv2.resize(frame, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))   # HWC → CHW
    img = np.expand_dims(img, axis=0)    # → BCHW
    return img


def postprocess_count(output: np.ndarray, conf: float) -> int:
    """
    Count detections above the confidence threshold.
    YOLO11 output shape: (1, num_classes+4, 8400) — columns are anchors.
    Confidence = max class score across all classes.
    """
    pred = output[0]                    # (num_classes+4, 8400)
    scores = pred[4:, :].max(axis=0)   # max class score per anchor
    return int((scores >= conf).sum())


def load_session(model_path: Path) -> tuple[ort.InferenceSession, str, float]:
    """Load an ONNX Runtime InferenceSession and return (session, provider, load_time_ms)."""
    t0 = time.perf_counter()
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=['CPUExecutionProvider'],
    )
    load_time_ms = (time.perf_counter() - t0) * 1000
    active_provider = session.get_providers()[0]
    return session, active_provider, load_time_ms


def benchmark_model(label: str, model_path: Path, cap: cv2.VideoCapture) -> dict:
    """
    Run a full benchmark for a single model. Returns a results dict.
    cap must already be opened and warmed up.
    """
    print(f"\n{'='*60}")
    print(f"  Benchmarking: {label}  ({model_path.name})")
    print(f"  ORT: {ort.__version__}")
    print(f"{'='*60}")

    result = {
        'label':              label,
        'model_path':         str(model_path),
        'model_size_mb':      round(model_path.stat().st_size / (1024*1024), 2),
        'ort_version':        ort.__version__,
        'execution_provider': None,
        'model_load_time_ms': None,
        'warmup_latency_ms':  None,
        'sustained_fps':      None,
        'inf_latency_mean_ms': None,
        'inf_latency_p95_ms': None,
        'e2e_latency_mean_ms': None,
        'cpu_mean_pct':       None,
        'ram_mean_mb':        None,
        'thermal_mean_c':     None,
        'detections_per_frame': None,
        'zero_detection_frames': None,
        'runtime_errors':     [],
    }

    if not model_path.exists():
        result['runtime_errors'].append(f"Model file not found: {model_path}")
        print(f"  ERROR: {result['runtime_errors'][-1]}")
        return result

    # ── Load ────────────────────────────────────────────────────────────────
    try:
        print("  Loading model...")
        session, provider, load_ms = load_session(model_path)
        result['model_load_time_ms'] = round(load_ms, 1)
        result['execution_provider'] = provider
        input_name = session.get_inputs()[0].name
        print(f"  Provider: {provider}  |  Load: {load_ms:.0f} ms")
    except Exception as e:
        result['runtime_errors'].append(f"Load failed: {traceback.format_exc()}")
        print(f"  LOAD ERROR: {e}")
        return result

    process = psutil.Process(os.getpid())
    psutil.cpu_percent()  # prime the counter

    # ── Warm-up ──────────────────────────────────────────────────────────────
    print(f"  Warming up ({WARMUP_FRAMES} frames)...")
    warmup_latency_ms = None
    for i in range(WARMUP_FRAMES):
        ret, frame = cap.read()
        if not ret:
            result['runtime_errors'].append("Camera read failed during warm-up")
            break
        img = preprocess(frame, INPUT_SIZE)
        t0 = time.perf_counter()
        session.run(None, {input_name: img})
        elapsed = (time.perf_counter() - t0) * 1000
        if i == 0:
            warmup_latency_ms = elapsed
    result['warmup_latency_ms'] = round(warmup_latency_ms, 1) if warmup_latency_ms else None
    if warmup_latency_ms is not None:
        print(f"  Warm-up first frame: {warmup_latency_ms:.1f} ms")
    else:
        print(f"  Warm-up failed: no frames read")

    # ── Sustained benchmark ──────────────────────────────────────────────────
    print(f"  Running {BENCHMARK_SECS}s sustained benchmark...")
    inf_latencies, e2e_latencies, cpus, rams, thermals, det_counts = [], [], [], [], [], []
    t_bench_start = time.perf_counter()
    frames_done = 0

    while (time.perf_counter() - t_bench_start) < BENCHMARK_SECS:
        t_e2e_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            result['runtime_errors'].append(f"Camera read failed at frame {frames_done}")
            break

        img = preprocess(frame, INPUT_SIZE)

        t_inf_start = time.perf_counter()
        try:
            outputs = session.run(None, {input_name: img})
        except Exception as e:
            result['runtime_errors'].append(f"Inference error frame {frames_done}: {e}")
            continue

        inf_ms = (time.perf_counter() - t_inf_start) * 1000
        e2e_ms = (time.perf_counter() - t_e2e_start) * 1000

        det_count = postprocess_count(outputs[0], CONF_THRESHOLD)
        cpu_pct   = psutil.cpu_percent()
        ram_mb    = process.memory_info().rss / (1024 * 1024)
        thermal   = read_thermal_celsius()

        inf_latencies.append(inf_ms)
        e2e_latencies.append(e2e_ms)
        cpus.append(cpu_pct)
        rams.append(ram_mb)
        if thermal is not None:
            thermals.append(thermal)
        det_counts.append(det_count)

        frames_done += 1

        if frames_done % 50 == 0:
            elapsed = time.perf_counter() - t_bench_start
            print(f"    {frames_done} frames | {elapsed:.0f}s elapsed | "
                  f"{frames_done/elapsed:.1f} FPS | "
                  f"Inf: {inf_ms:.1f}ms | RAM: {ram_mb:.0f}MB")

    total_time = time.perf_counter() - t_bench_start

    if inf_latencies:
        result['sustained_fps']          = round(frames_done / total_time, 2)
        result['inf_latency_mean_ms']    = round(float(np.mean(inf_latencies)), 1)
        result['inf_latency_p95_ms']     = round(float(np.percentile(inf_latencies, 95)), 1)
        result['e2e_latency_mean_ms']    = round(float(np.mean(e2e_latencies)), 1)
        result['cpu_mean_pct']           = round(float(np.mean(cpus)), 1)
        result['ram_mean_mb']            = round(float(np.mean(rams)), 1)
        result['thermal_mean_c']         = round(float(np.mean(thermals)), 1) if thermals else None
        result['detections_per_frame']   = round(float(np.mean(det_counts)), 2)
        result['zero_detection_frames']  = int(sum(1 for d in det_counts if d == 0))

    print(f"\n  Results for {label}:")
    print(f"    Sustained FPS:         {result['sustained_fps']}")
    print(f"    Inf latency (mean):    {result['inf_latency_mean_ms']} ms")
    print(f"    Inf latency (p95):     {result['inf_latency_p95_ms']} ms")
    print(f"    E2E latency (mean):    {result['e2e_latency_mean_ms']} ms")
    print(f"    CPU utilization:       {result['cpu_mean_pct']} %")
    print(f"    RAM usage:             {result['ram_mean_mb']} MB")
    print(f"    Thermal (mean):        {result['thermal_mean_c']} °C")
    print(f"    Detections/frame:      {result['detections_per_frame']}")
    print(f"    Zero-detection frames: {result['zero_detection_frames']} / {frames_done}")
    if result['runtime_errors']:
        print(f"    Runtime errors:        {len(result['runtime_errors'])}")

    return result


def apply_decision_matrix(fp32: dict, int8: dict) -> str:
    """Apply the Phase 10 decision matrix based on measured results."""
    if fp32.get('sustained_fps') is None or int8.get('sustained_fps') is None:
        return "INCONCLUSIVE — one or both models failed to run. Check runtime_errors."

    fps_fp32 = fp32['sustained_fps']
    fps_int8 = int8['sustained_fps']
    ram_fp32 = fp32.get('ram_mean_mb', float('inf'))
    ram_int8 = int8.get('ram_mean_mb', float('inf'))

    fps_delta_pct = (fps_int8 - fps_fp32) / fps_fp32 * 100
    SIGNIFICANCE_THRESHOLD = 10  # % difference considered meaningful

    if fps_int8 >= fps_fp32 * (1 - SIGNIFICANCE_THRESHOLD/100) and ram_int8 < ram_fp32:
        return f"PREFER INT8 — similar or better FPS ({fps_int8:.1f} vs {fps_fp32:.1f}), lower RAM ({ram_int8:.0f} vs {ram_fp32:.0f} MB)"
    elif fps_fp32 > fps_int8 * (1 + SIGNIFICANCE_THRESHOLD/100):
        return f"SELECT FP32 — significantly faster ({fps_fp32:.1f} vs {fps_int8:.1f} FPS, {fps_delta_pct:+.0f}%)"
    elif fps_fp32 < 8 and fps_int8 < 8:
        return "BOTH TOO SLOW — investigate resolution reduction, frame-skipping, or alternative runtimes (NNAPI/HTP delegate, QNN SDK)"
    else:
        return f"SIMILAR PERFORMANCE — prefer INT8 for storage/RAM benefit ({fps_delta_pct:+.0f}% FPS delta, within threshold)"


def main():
    parser = argparse.ArgumentParser(description='AEGIS Phase 10: UNO Q Hardware Benchmark')
    parser.add_argument('--model', choices=['fp32', 'int8', 'both'], default='both',
                        help='Which model(s) to benchmark (default: both)')
    parser.add_argument('--camera', type=int, default=CAMERA_INDEX,
                        help=f'Camera device index (default: {CAMERA_INDEX})')
    parser.add_argument('--duration', type=int, default=BENCHMARK_SECS,
                        help=f'Benchmark duration in seconds (default: {BENCHMARK_SECS})')
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = SCRIPT_DIR / f'aegis_benchmark_{timestamp}.json'

    print("=" * 60)
    print("  AEGIS Phase 10: Arduino UNO Q Hardware Benchmark")
    print(f"  ONNX Runtime: {ort.__version__}")
    print(f"  Available providers: {ort.get_available_providers()}")
    print(f"  Benchmark duration: {args.duration}s per model")
    print(f"  Camera index: {args.camera}")
    print(f"  Output: {output_path}")
    print("=" * 60)

    # Open camera: scan indices 0-5 if the default doesn't work
    print(f"\nOpening camera (forcing V4L2 backend)...")
    cap = None
    for idx in range(6):
        # Force V4L2 backend to bypass buggy GStreamer pipelines on Linux SBCs
        cap_test = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap_test.isOpened():
            ret, _ = cap_test.read()
            if ret:
                cap = cap_test
                print(f"  Successfully opened camera at index {idx}")
                args.camera = idx
                break
            else:
                cap_test.release()
        else:
            cap_test.release()

    if cap is None or not cap.isOpened():
        print("ERROR: Could not open any camera on indices 0-5. Exiting.")
        print("Make sure the USB camera is plugged in BEFORE clicking Deploy!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, INPUT_SIZE)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_SIZE)

    # Brief camera warm-up (hardware exposure stabilization)
    print("Stabilizing camera (5 frames)...")
    for _ in range(5):
        cap.read()

    results = {
        'timestamp': timestamp,
        'platform': {
            'python': __import__('sys').version,
            'ort_version': ort.__version__,
            'available_providers': ort.get_available_providers(),
            'cpu_count': psutil.cpu_count(),
            'total_ram_mb': round(psutil.virtual_memory().total / (1024*1024)),
        },
        'config': {
            'camera_index':    args.camera,
            'input_size':      INPUT_SIZE,
            'conf_threshold':  CONF_THRESHOLD,
            'warmup_frames':   WARMUP_FRAMES,
            'benchmark_secs':  args.duration,
        },
        'models': {},
        'decision': None,
    }

    models_to_run = {
        'fp32': MODEL_PATHS['fp32'],
        'int8': MODEL_PATHS['int8'],
    } if args.model == 'both' else {
        args.model: MODEL_PATHS[args.model]
    }

    for label, path in models_to_run.items():
        results['models'][label] = benchmark_model(label, path, cap)

    cap.release()

    # Apply decision matrix if both models ran
    if 'fp32' in results['models'] and 'int8' in results['models']:
        decision = apply_decision_matrix(results['models']['fp32'], results['models']['int8'])
        results['decision'] = decision
        print(f"\n{'='*60}")
        print(f"  DECISION: {decision}")
        print(f"{'='*60}")

    # Save results to file
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print results to console so user can copy-paste from Arduino App Lab
    print(f"\n{'='*60}")
    print("  APP LAB COPY-PASTE ZONE START")
    print(f"{'='*60}")
    print(json.dumps(results, indent=2))
    print(f"{'='*60}")
    print("  APP LAB COPY-PASTE ZONE END")
    print(f"{'='*60}\n")

    print(f"Results also written to file: {output_path}")

if __name__ == '__main__':
    main()
