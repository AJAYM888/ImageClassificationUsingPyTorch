# Simple installation checker
import sys
print("Python version:", sys.version)

# Check PyTorch
try:
    import torch
    print("PyTorch: INSTALLED -", torch.__version__)
except ImportError:
    print("PyTorch: NOT INSTALLED")

# Check OpenCV
try:
    import cv2
    print("OpenCV: INSTALLED -", cv2.__version__)
except ImportError:
    print("OpenCV: NOT INSTALLED")

# Check other packages
packages = ['numpy', 'matplotlib', 'sklearn', 'tqdm']
for pkg in packages:
    try:
        module = __import__(pkg)
        print(pkg + ": INSTALLED")
    except ImportError:
        print(pkg + ": NOT INSTALLED")

print("\nTo install missing packages, run:")
print("pip install torch torchvision opencv-python numpy matplotlib scikit-learn tqdm")