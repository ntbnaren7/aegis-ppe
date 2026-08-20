import cv2
import time
import sys
import os
from ultralytics import YOLO

# Add the project root to sys.path so we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.aegis_ppe.safety_interpreter import SafetyInterpreter

def main():
    print("--- AEGIS Phase 7: Safety Interpretation ---")
    
    # Path to your successfully trained baseline model
    model_path = r'D:\AntiGravity Projects\mnemosyne\runs\detect\runs\train\aegis_baseline_fast\weights\best.pt'
    
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
        # Read a frame from the webcam
        success, frame = cap.read()
        
        if not success:
            print("Failed to grab frame. Exiting...")
            break
            
        # 1. Run inference on the frame for Safety/PPE
        # Lowered conf to 0.2 to see if the model is detecting you but with lower confidence
        # due to differences between your webcam background/lighting and the dataset
        results = model.predict(frame, conf=0.2, verbose=False, device=0)
        
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
        
        # If markers are found, draw them on the annotated frame
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(annotated_frame, corners, ids)
            # Optional: Print the IDs to console just to confirm it's tracking
            # print(f"Detected ArUco IDs: {ids.flatten()}")
        
        # Calculate FPS
        new_time = time.time()
        fps = 1 / (new_time - prev_time) if prev_time != 0 else 0
        prev_time = new_time
        
        # Draw FPS on the frame
        cv2.putText(
            annotated_frame, 
            f"FPS: {int(fps)}", 
            (10, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), # Green text
            2
        )
        
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
