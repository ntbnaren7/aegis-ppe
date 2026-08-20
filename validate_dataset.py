import os
import glob
import yaml

def validate_dataset(base_dir):
    data_yaml = os.path.join(base_dir, "data.yaml")
    
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
        
    print(f"Classes (nc): {data['nc']}")
    print(f"Class names: {data['names']}")
    
    expected_names = ['Fall-Detected', 'Gloves', 'Goggles', 'Hardhat', 'Ladder', 'Mask', 'NO-Gloves', 'NO-Goggles', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest']
    
    if data['names'] != expected_names:
        print("ERROR: Class names do not match expected classes.")
        return False
        
    if data['nc'] != 14:
        print("ERROR: Expected 14 classes.")
        return False

    splits = [('train', data['train']), ('val', data['val']), ('test', data.get('test', 'test/images'))]
    
    total_images = 0
    total_labels = 0
    errors = 0
    
    for split_name, img_dir in splits:
        img_path = os.path.join(base_dir, img_dir)
        lbl_dir = img_dir.replace('images', 'labels')
        lbl_path = os.path.join(base_dir, lbl_dir)
        
        if not os.path.exists(img_path):
            print(f"ERROR: Image directory not found for {split_name}: {img_path}")
            errors += 1
            continue
            
        images = glob.glob(os.path.join(img_path, "*.jpg")) + glob.glob(os.path.join(img_path, "*.png"))
        
        print(f"[{split_name}] Found {len(images)} images in {img_path}")
        
        split_errors = 0
        for img in images:
            basename = os.path.splitext(os.path.basename(img))[0]
            label_file = os.path.join(lbl_path, basename + ".txt")
            
            if not os.path.exists(label_file):
                print(f"ERROR: Missing label for image {img}")
                split_errors += 1
                errors += 1
                continue
                
            # Validate label content
            with open(label_file, 'r') as lf:
                lines = lf.readlines()
                for line_idx, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"ERROR: Invalid label format in {label_file} line {line_idx+1}")
                        split_errors += 1
                        errors += 1
                        continue
                    
                    class_id = int(parts[0])
                    if not (0 <= class_id < 14):
                        print(f"ERROR: Invalid class ID {class_id} in {label_file} line {line_idx+1}")
                        split_errors += 1
                        errors += 1
                        
            total_labels += 1
        
        if split_errors == 0:
            print(f"[{split_name}] OK. Labels match images and format is correct.")
        total_images += len(images)
            
    print(f"\n--- Validation Summary ---")
    print(f"Total Images: {total_images}")
    print(f"Total Labels: {total_labels}")
    print(f"Total Errors: {errors}")
    
    if errors == 0:
        print("Dataset is structurally valid.")
        return True
    else:
        print("Dataset validation FAILED.")
        return False

if __name__ == "__main__":
    import sys
    success = validate_dataset(".")
    sys.exit(0 if success else 1)
