# simple_api.py - Working version
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import io
import time
from pathlib import Path
import uvicorn

app = FastAPI(title="Manufacturing Quality Control API")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model_instance = None
class_mapping = {
    'good': 0,
    'defective': 1,
    'scratched': 2,
    'dented': 3,
    'discolored': 4
}
reverse_mapping = {v: k for k, v in class_mapping.items()}

# Model definition (same as your training)
class QualityControlModel(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(QualityControlModel, self).__init__()
        if pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.backbone = resnet50(weights=None)
        
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model():
    global model_instance
    
    # Look for model files
    model_files = list(Path('.').glob('*.pth'))
    
    if not model_files:
        print("❌ No .pth model files found")
        return False
    
    model_path = model_files[0]  # Use the first .pth file found
    print(f"📂 Found model file: {model_path}")
    
    try:
        # Create model
        model = QualityControlModel(num_classes=5)
        
        # Load weights
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        model_instance = {
            'model': model,
            'device': device
        }
        
        print(f"✅ Model loaded successfully on {device}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

@app.on_event("startup")
async def startup():
    load_model()

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
    <head><title>Quality Control API</title></head>
    <body>
        <h1>🏭 Manufacturing Quality Control API</h1>
        <h2>Endpoints:</h2>
        <ul>
            <li><a href="/health">GET /health</a> - Check API status</li>
            <li><a href="/docs">GET /docs</a> - API Documentation</li>
            <li>POST /predict - Upload image for analysis</li>
        </ul>
        
        <h3>Test with curl:</h3>
        <pre>
curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: multipart/form-data" \\
     -F "file=@your_image.jpg"
        </pre>
    </body>
    </html>
    """

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model_instance is not None,
        "device": model_instance['device'] if model_instance else "none",
        "supported_classes": list(class_mapping.keys())
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model_instance:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        start_time = time.time()
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Transform image
        input_tensor = transform(image).unsqueeze(0).to(model_instance['device'])
        
        # Predict
        with torch.no_grad():
            outputs = model_instance['model'](input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = reverse_mapping[predicted.item()]
        confidence_score = confidence.item()
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence_score,
            "all_probabilities": {
                reverse_mapping[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            },
            "processing_time_ms": round(processing_time, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("simple_api:app", host="0.0.0.0", port=8001, reload=True)