from ultralytics import YOLO

def main():
    print("--- AEGIS Baseline Evaluation ---")
    
    # We load the best weights produced from the Phase 3 training
    try:
        model = YOLO(r'D:\AntiGravity Projects\mnemosyne\runs\detect\runs\train\aegis_baseline_fast\weights\best.pt')
        print("Loaded aegis_baseline_fast model successfully.")
    except Exception as e:
        print(f"Could not load best.pt weights. Ensure training finished successfully. Error: {e}")
        return

    print("Evaluating on TEST dataset...")
    # Evaluate model performance on the test set
    metrics = model.val(data='data.yaml', split='test', device=0)
    
    print("\n--- Test Metrics ---")
    print(f"mAP50-95: {metrics.box.map}") 
    print(f"mAP50: {metrics.box.map50}")
    print(f"Precision: {metrics.box.mp}")
    print(f"Recall: {metrics.box.mr}")
    
    # Results are also saved in runs/val/
    print("\nEvaluation complete! Detailed metrics are saved in the Ultralytics runs directory.")

if __name__ == '__main__':
    main()
