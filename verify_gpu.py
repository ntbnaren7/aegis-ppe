import torch
import ultralytics

def main():
    print("--- Hardware Verification ---")
    print(f"PyTorch Version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is NOT available. Training will run on CPU.")

    print(f"Ultralytics Version: {ultralytics.__version__}")

if __name__ == "__main__":
    main()
