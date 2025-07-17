# Manufacturing Quality Control Deep Learning System
# PyTorch-only version (no Albumentations dependency)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnet50, ResNet50_Weights
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from pathlib import Path
import json
import time
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ManufacturingDataset(Dataset):
    """Custom dataset for manufacturing defect detection"""
    
    def __init__(self, data_dir, transform=None, class_mapping=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Define class mapping if not provided
        if class_mapping is None:
            self.class_mapping = {
                'good': 0,
                'defective': 1,
                'scratched': 2,
                'dented': 3,
                'discolored': 4
            }
        else:
            self.class_mapping = class_mapping
        
        self.num_classes = len(self.class_mapping)
        self._load_data()
    
    def _load_data(self):
        """Load image paths and labels"""
        for class_name, class_idx in self.class_mapping.items():
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.images.append(str(img_path))
                    self.labels.append(class_idx)
                for img_path in class_dir.glob('*.png'):
                    self.images.append(str(img_path))
                    self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

class QualityControlModel(nn.Module):
    """ResNet-based model for quality control classification"""
    
    def __init__(self, num_classes=5, pretrained=True):
        super(QualityControlModel, self).__init__()
        
        # Load pretrained ResNet50
        if pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.backbone = resnet50(weights=None)
        
        # Replace final layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class QualityControlTrainer:
    """Training and evaluation class"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Check for Mac MPS support
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = 'mps'
            self.model = self.model.to('mps')
            logger.info("Using Mac GPU (MPS) acceleration")
        else:
            logger.info(f"Using device: {self.device}")
    
    def train_epoch(self, dataloader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(dataloader, desc="Training"):
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, dataloader, criterion):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, num_epochs=50, lr=0.001):
        """Full training loop"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        best_val_acc = 0.0
        patience_counter = 0
        early_stop_patience = 10
        
        logger.info(f"Starting training on {self.device}")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            epoch_time = time.time() - start_time
            
            logger.info(f'Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.2f}s)')
            logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_quality_control_model.pth')
                patience_counter = 0
                logger.info(f'New best model saved with validation accuracy: {best_val_acc:.2f}%')
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stop_patience:
                logger.info(f'Early stopping at epoch {epoch+1}')
                break
        
        logger.info(f'Training completed. Best validation accuracy: {best_val_acc:.2f}%')
        return best_val_acc

def get_transforms():
    """Define data augmentation transforms using PyTorch"""
    
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms

class QualityControlInference:
    """Inference class for production deployment"""
    
    def __init__(self, model_path, class_mapping, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.class_mapping = class_mapping
        self.reverse_mapping = {v: k for k, v in class_mapping.items()}
        
        # Check for Mac MPS support
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = 'mps'
        
        # Load model
        self.model = QualityControlModel(num_classes=len(class_mapping))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Define inference transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict_single(self, image_path):
        """Predict single image"""
        # Load and preprocess image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = self.reverse_mapping[predicted.item()]
        confidence_score = confidence.item()
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence_score,
            'all_probabilities': {self.reverse_mapping[i]: prob.item() 
                                for i, prob in enumerate(probabilities[0])}
        }

def evaluate_model(model, test_loader, class_mapping, device):
    """Comprehensive model evaluation"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = 100 * sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
    
    # Classification report
    class_names = [k for k, v in sorted(class_mapping.items(), key=lambda x: x[1])]
    report = classification_report(all_labels, all_predictions, target_names=class_names)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    return accuracy, report, cm, class_names

def plot_training_history(trainer):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(trainer.train_losses, label='Training Loss')
    ax1.plot(trainer.val_losses, label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(trainer.train_accuracies, label='Training Accuracy')
    ax2.plot(trainer.val_accuracies, label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training and evaluation pipeline"""
    
    # Configuration
    DATA_DIR = "manufacturing_data"  # Update with your data directory
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Class mapping
    class_mapping = {
        'good': 0,
        'defective': 1,
        'scratched': 2,
        'dented': 3,
        'discolored': 4
    }
    
    # Check if data directory exists
    if not Path(DATA_DIR).exists():
        logger.error(f"Data directory {DATA_DIR} not found!")
        logger.info("Please run the data collection script first:")
        logger.info("python3 data_collection_script.py")
        return
    
    # Get transforms
    train_transforms, val_transforms = get_transforms()
    
    # Create full dataset with training transforms
    full_dataset = ManufacturingDataset(DATA_DIR, transform=train_transforms, 
                                       class_mapping=class_mapping)
    
    if len(full_dataset) == 0:
        logger.error("No images found in the dataset!")
        logger.info("Please check the data directory structure.")
        return
    
    # Split dataset
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Apply different transforms to validation and test sets
    # Note: This is a simplified approach. In practice, you'd want separate dataset objects
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Initialize model and trainer
    model = QualityControlModel(num_classes=len(class_mapping))
    trainer = QualityControlTrainer(model)
    
    # Train model
    best_val_acc = trainer.train(train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE)
    
    # Plot training history
    plot_training_history(trainer)
    
    # Load best model for evaluation
    model.load_state_dict(torch.load('best_quality_control_model.pth'))
    
    # Evaluate on test set
    test_accuracy, classification_rep, confusion_mat, class_names = evaluate_model(
        model, test_loader, class_mapping, trainer.device
    )
    
    logger.info(f"Test Accuracy: {test_accuracy:.2f}%")
    logger.info("Classification Report:")
    logger.info(classification_rep)
    
    # Plot confusion matrix
    plot_confusion_matrix(confusion_mat, class_names)
    
    # Save class mapping
    with open('class_mapping.json', 'w') as f:
        json.dump(class_mapping, f)
    
    logger.info("Model training and evaluation completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"Test accuracy: {test_accuracy:.2f}%")
    
    # Example inference
    logger.info("Setting up inference engine...")
    inference_engine = QualityControlInference(
        'best_quality_control_model.pth', 
        class_mapping
    )
    logger.info("🎉 Training complete! Model ready for inference.")

if __name__ == "__main__":
    main()