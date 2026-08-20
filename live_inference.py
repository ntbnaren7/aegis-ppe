import cv2
import time
from ultralytics import YOLO

def main():
    print("--- AEGIS Phase 5: Live Webcam Inference ---")
    
    # Path to your successfully trained baseline model
    model_path = r'D:\AntiGravity Projects\mnemosyne\runs\detect\runs\train\aegis_baseline_fast\weights\best.pt'
    
    try:
        print(f"Loading AEGIS Baseline Model from: {model_path}")
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

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
            
        # Run inference on the frame
        # Lowered conf to 0.2 to see if the model is detecting you but with lower confidence
        # due to differences between your webcam background/lighting and the dataset
        results = model.predict(frame, conf=0.2, verbose=False, device=0)
        
        # Plot the predictions onto the frame
        annotated_frame = results[0].plot()
        
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
