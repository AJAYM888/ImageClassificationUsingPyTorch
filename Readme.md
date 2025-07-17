# ImageClassificationUsingPyTorch

A comprehensive PyTorch-based project for image classification that demonstrates how to build, train, and deploy deep learning models for image classification tasks. This project includes a complete pipeline from data collection to web deployment.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Web Interface](#web-interface)
- [API](#api)
- [Results](#results)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This project provides a complete end-to-end solution for image classification using PyTorch. It includes data collection tools, model training scripts, evaluation metrics, and a web interface for easy interaction with the trained model. The project is designed to be modular and easily extensible for different datasets and classification tasks.

## Features

- **Deep Learning with PyTorch**: Utilizes PyTorch for building and training neural networks
- **Data Collection Tools**: Automated scripts for dataset preparation and augmentation
- **Configurable Architecture**: Flexible model architecture that can be easily modified
- **Training Pipeline**: Complete training and validation workflow with progress tracking
- **Results Visualization**: Comprehensive visualization of training metrics and confusion matrices
- **Web Interface**: User-friendly HTML interface for model interaction
- **REST API**: API endpoints for programmatic access to the model
- **Model Persistence**: Save and load trained models for future use
- **Performance Metrics**: Detailed evaluation metrics and confusion matrix generation

## Project Structure

```
ImageClassificationUsingPyTorch/
├── api.py                    # REST API for model inference
├── appf.html                 # Web interface for image classification
├── check_installation.py     # Installation verification script
├── class_mapping.json        # Class labels mapping
├── confusion_matrix.png      # Generated confusion matrix visualization
├── data_collectionscript.py  # Data collection and preprocessing
├── model.py                  # Model architecture definition
├── requirements.txt          # Python dependencies
├── training_history.png      # Training progress visualization
└── __pycache__/             # Python cache files
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AJAYM888/ImageClassificationUsingPyTorch.git
   cd ImageClassificationUsingPyTorch
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python check_installation.py
   ```

## Usage

### Quick Start

1. **Prepare your dataset:**
   ```bash
   python data_collectionscript.py --data_path /path/to/images --output_dir ./dataset
   ```

2. **Train the model:**
   ```bash
   python model.py --dataset ./dataset --epochs 50 --batch_size 32
   ```

3. **Start the web interface:**
   ```bash
   python api.py
   ```
   Then open `appf.html` in your browser to interact with the model.

### Advanced Usage

For custom configurations and advanced training options, modify the parameters in the respective scripts or use command-line arguments.

## Dataset

The project supports various image datasets and includes tools for:
- **Data Collection**: Automated downloading and organization of image data
- **Data Preprocessing**: Image resizing, normalization, and augmentation
- **Data Splitting**: Automatic train/validation/test splits
- **Class Mapping**: Automatic generation of class labels and mappings

To use your own dataset:
1. Organize images in folders by class
2. Run the data collection script to preprocess
3. Update `class_mapping.json` if needed

## Training

The training process includes:
- **Model Architecture**: Configurable CNN architecture optimized for image classification
- **Loss Function**: Cross-entropy loss for multi-class classification
- **Optimizer**: Adam optimizer with learning rate scheduling
- **Metrics**: Accuracy, precision, recall, and F1-score tracking
- **Visualization**: Real-time training progress plots

### Training Parameters

- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 32)
- `--learning_rate`: Learning rate for optimizer (default: 0.001)
- `--device`: Training device (cuda/cpu, auto-detected)

## Evaluation

The project provides comprehensive evaluation tools:
- **Confusion Matrix**: Visual representation of classification performance
- **Accuracy Metrics**: Overall and per-class accuracy
- **Training History**: Loss and accuracy plots over training epochs
- **Model Performance**: Detailed performance statistics

Results are automatically saved as:
- `confusion_matrix.png`: Confusion matrix visualization
- `training_history.png`: Training progress plots
- Model checkpoints for best performance

## Web Interface

The project includes a user-friendly web interface (`appf.html`) that allows:
- **Image Upload**: Drag-and-drop or file selection
- **Real-time Prediction**: Instant classification results
- **Confidence Scores**: Probability distributions for all classes
- **Visual Feedback**: Clear display of results and predictions

## API

The REST API (`api.py`) provides programmatic access with endpoints for:
- **POST /predict**: Image classification endpoint
- **GET /classes**: Available class labels
- **GET /model_info**: Model architecture and training information

### API Usage Example

```python
import requests

# Upload and classify an image
with open('image.jpg', 'rb') as f:
    response = requests.post('http://localhost:5000/predict', files={'image': f})
    result = response.json()
    print(f"Predicted class: {result['class']}")
    print(f"Confidence: {result['confidence']:.2f}")
```

## Results

The project generates comprehensive results including:
- **Training Accuracy**: Achieved training accuracy and loss curves
- **Validation Performance**: Validation accuracy and generalization metrics
- **Confusion Matrix**: Detailed classification performance per class
- **Sample Predictions**: Example predictions with confidence scores

Example performance metrics:
- Training Accuracy: 95%+
- Validation Accuracy: 90%+
- Inference Time: <100ms per image

## Requirements

- **Python**: 3.7+
- **PyTorch**: 1.9+
- **Key Dependencies**:
  - torch
  - torchvision
  - numpy
  - matplotlib
  - PIL (Pillow)
  - flask (for API)
  - scikit-learn (for metrics)

See `requirements.txt` for complete dependency list.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comments for complex functions
- Include unit tests for new features
- Update documentation as needed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **PyTorch Team**: For the excellent deep learning framework
- **OpenCV Community**: For image processing tools
- **Matplotlib**: For visualization capabilities
- **Flask**: For web framework support
- **Contributors**: Thanks to all contributors who helped improve this project

---

**Author**: AJAYM888  
**Last Updated**: July 2025  
**Version**: 1.0.0

For questions or support, please open an issue on GitHub or contact the maintainer.
