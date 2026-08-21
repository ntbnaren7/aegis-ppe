import cv2
import time
import sys
import os
import psutil
from ultralytics import YOLO

# Add the project root to sys.path so we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from aegis_ppe.safety_interpreter import SafetyInterpreter

def main():
    print("--- AEGIS Phase 9.5: INT8 Runtime Inference ---")
    
    # Setup psutil process for RAM usage
    process = psutil.Process(os.getpid())
    # Prime the CPU percentage calculation
    psutil.cpu_percent()
    
    # INT8 dynamic quantized model — validated mAP50: 0.7688 (-1.4% vs FP32 baseline)
    model_path = r'D:\AntiGravity Projects\mnemosyne\runs\detect\runs\train\aegis_baseline_fast\weights\best_int8.onnx'
    
    try:
        print(f"Loading AEGIS Baseline Model from: {model_path}")
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize ArUco Detector
    print("Initializing ArUco Detector (DICT_6X6_250)...")
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # Initialize Safety Interpreter
    print("Initializing Safety Interpreter (Temporal Threshold: 5 frames)...")
    interpreter = SafetyInterpreter(temporal_threshold=5)

    # Open the webcam. 0 is the default camera, 1 is usually an external USB webcam (like ROG Eye S)
    # Try changing this to 1 if it opens your laptop's built-in webcam instead.
    cam_index = 0
    cap = cv2.VideoCapture(cam_index)
    
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {cam_index}")
        print("Tip: If you have multiple webcams, try changing cam_index to 1 or 2.")
        return
        
    print("\nWebcam opened successfully! Press 'q' to quit.")
    
    # FPS calculation variables
    prev_time = 0
    new_time = 0

    while True:
        loop_start_time = time.time()
        
        # Read a frame from the webcam
        cam_start_time = time.time()
        success, frame = cap.read()
        cam_latency_ms = (time.time() - cam_start_time) * 1000
        
        if not success:
            print("Failed to grab frame. Exiting...")
            break
            
        # 1. Run inference on the frame for Safety/PPE
        # Lowered conf to 0.2 to see if the model is detecting you but with lower confidence
        # due to differences between your webcam background/lighting and the dataset
        inf_start_time = time.time()
        results = model.predict(frame, conf=0.2, verbose=False, device='cpu')
        
        # Plot the predictions onto the frame
        annotated_frame = results[0].plot()
        
        # Extract detected class IDs from YOLO results
        # results[0].boxes.cls is a tensor of shape (N,) containing class IDs
        detected_class_ids = [int(cls.item()) for cls in results[0].boxes.cls] if len(results[0].boxes) > 0 else []
        
        # Process the detected classes through the Safety Interpreter
        active_violations = interpreter.process_frame(detected_class_ids)
        
        # Draw massive warning text if violations are active
        if active_violations:
            warning_text = f"VIOLATION: {', '.join(active_violations)}"
            cv2.putText(
                annotated_frame, 
                warning_text, 
                (10, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 255), # Red text
                3
            )
            
            # Optional: Add a subtle red overlay to the entire frame
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (annotated_frame.shape[1], annotated_frame.shape[0]), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.2, annotated_frame, 0.8, 0, annotated_frame)
        
        # 2. Run ArUco detection on the original frame
        corners, ids, rejected = aruco_detector.detectMarkers(frame)
        inf_latency_ms = (time.time() - inf_start_time) * 1000
        
        # If markers are found, draw them on the annotated frame
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(annotated_frame, corners, ids)
            # Optional: Print the IDs to console just to confirm it's tracking
            # print(f"Detected ArUco IDs: {ids.flatten()}")
        
        # Calculate Telemetry
        cpu_usage = psutil.cpu_percent()
        ram_usage_mb = process.memory_info().rss / (1024 * 1024)
        
        e2e_latency_ms = (time.time() - loop_start_time) * 1000
        fps = 1000.0 / e2e_latency_ms if e2e_latency_ms > 0 else 0
        
        # Draw Telemetry Panel on the frame
        # We will draw this in the bottom left corner to keep it out of the way of the warnings
        y_offset = annotated_frame.shape[0] - 120
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(annotated_frame, f"E2E Latency: {e2e_latency_ms:.1f}ms", (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(annotated_frame, f"Inf Latency: {inf_latency_ms:.1f}ms", (10, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(annotated_frame, f"Cam Latency: {cam_latency_ms:.1f}ms", (10, y_offset + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(annotated_frame, f"CPU: {cpu_usage:.1f}% | RAM: {ram_usage_mb:.0f}MB", (10, y_offset + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow("AEGIS Live Inference - ROG Eye S", annotated_frame)
        
        # Wait for 1 millisecond and check if the 'q' key was pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting live inference...")
            break
            
    # Clean up when done
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
