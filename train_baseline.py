from ultralytics import YOLO

def main():
    print("--- AEGIS Baseline Training ---")
    print("Loading YOLO11n base model...")
    # Load a pretrained lightweight model
    model = YOLO('yolo11n.pt') 
    
    print("Starting training...")
    # Train the model on the custom dataset with optimizations
    results = model.train(
        data='data.yaml',
        epochs=50,
        imgsz=640,
        device=0,
        cache='disk',
        batch=-1,
        workers=4,
        patience=10,
        project='runs/train',
        name='aegis_baseline_fast'
    )
    
    print("Training complete!")

if __name__ == '__main__':
    # Fix for multiprocessing on Windows
    import multiprocessing
    multiprocessing.freeze_support()
    main()
