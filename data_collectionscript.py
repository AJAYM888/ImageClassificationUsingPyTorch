# Simple Manufacturing Data Generator
import os
import numpy as np
import cv2
from pathlib import Path

def create_sample_data(num_samples=100):
    """Create synthetic manufacturing defect dataset"""
    
    # Create directories
    base_dir = Path("manufacturing_data")
    base_dir.mkdir(exist_ok=True)
    
    classes = ['good', 'defective', 'scratched', 'dented', 'discolored']
    for class_name in classes:
        (base_dir / class_name).mkdir(exist_ok=True)
    
    print(f"Creating sample dataset with {num_samples} images per class...")
    
    # Generate images for each class
    for class_name in classes:
        print(f"Generating {class_name} samples...")
        for i in range(num_samples):
            
            # Create base synthetic product image
            img = create_base_product()
            
            # Add class-specific defects
            if class_name == 'scratched':
                img = add_scratches(img)
            elif class_name == 'dented':
                img = add_dents(img)
            elif class_name == 'discolored':
                img = add_discoloration(img)
            elif class_name == 'defective':
                # Random defect type
                defect_type = np.random.choice(['scratch', 'dent', 'discolor'])
                if defect_type == 'scratch':
                    img = add_scratches(img)
                elif defect_type == 'dent':
                    img = add_dents(img)
                else:
                    img = add_discoloration(img)
            # 'good' class gets no defects
            
            # Save image
            filename = base_dir / class_name / f"{class_name}_{i:03d}.jpg"
            cv2.imwrite(str(filename), img)
    
    print("✅ Sample dataset created successfully!")
    print(f"📁 Location: {base_dir}")
    print(f"📊 Total images: {len(classes) * num_samples}")
    
    # Verify dataset
    for class_name in classes:
        count = len(list((base_dir / class_name).glob("*.jpg")))
        print(f"   {class_name}: {count} images")

def create_base_product(size=(224, 224)):
    """Create a base synthetic product image"""
    # Create metallic-looking base
    img = np.ones((size[0], size[1], 3), dtype=np.uint8) * 180
    
    # Add some texture/noise
    noise = np.random.normal(0, 10, (size[0], size[1]))
    for c in range(3):
        img[:, :, c] = np.clip(img[:, :, c] + noise, 0, 255)
    
    # Add geometric patterns (simulating manufactured parts)
    center = (size[1]//2, size[0]//2)
    cv2.circle(img, center, 80, (160, 160, 160), 2)
    cv2.rectangle(img, (50, 50), (size[1]-50, size[0]-50), (150, 150, 150), 1)
    
    return img.astype(np.uint8)

def add_scratches(img):
    """Add scratch defects to image"""
    img = img.copy()
    h, w = img.shape[:2]
    
    # Add 1-3 scratches
    num_scratches = np.random.randint(1, 4)
    for _ in range(num_scratches):
        start = (np.random.randint(0, w), np.random.randint(0, h))
        end = (np.random.randint(0, w), np.random.randint(0, h))
        thickness = np.random.randint(1, 4)
        cv2.line(img, start, end, (50, 50, 50), thickness)
    
    return img

def add_dents(img):
    """Add dent defects to image"""
    img = img.copy()
    h, w = img.shape[:2]
    
    # Add 1-2 dents
    num_dents = np.random.randint(1, 3)
    for _ in range(num_dents):
        center = (np.random.randint(30, w-30), np.random.randint(30, h-30))
        radius = np.random.randint(15, 30)
        # Create darker circular area
        cv2.circle(img, center, radius, (100, 100, 100), -1)
        cv2.circle(img, center, radius//2, (80, 80, 80), -1)
    
    return img

def add_discoloration(img):
    """Add discoloration defects to image"""
    img = img.copy()
    h, w = img.shape[:2]
    
    # Add 1-3 discolored areas
    num_spots = np.random.randint(1, 4)
    for _ in range(num_spots):
        center = (np.random.randint(0, w), np.random.randint(0, h))
        radius = np.random.randint(20, 40)
        
        # Random color shift
        color = (
            np.random.randint(100, 255),
            np.random.randint(50, 150), 
            np.random.randint(50, 150)
        )
        
        # Create overlay
        overlay = img.copy()
        cv2.circle(overlay, center, radius, color, -1)
        
        # Blend with original (70% original, 30% overlay)
        img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    
    return img

def main():
    """Main function"""
    print("Manufacturing Dataset Generator")
    print("=" * 40)
    
    try:
        num_samples = int(input("Enter number of samples per class (default 100): ") or "100")
    except ValueError:
        num_samples = 100
    
    create_sample_data(num_samples)
    
    print("\n🎉 Data generation complete!")
    print("Next step: Run 'python3 quality_control_system.py' to start training")

if __name__ == "__main__":
    main()