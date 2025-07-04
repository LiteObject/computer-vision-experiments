# Computer Vision Model Experiments

This project is designed to experiment with different computer vision models for object detection tasks. The primary focus includes YOLO architectures (YOLOv5, YOLOv8) and Facebook's Detectron2 framework, providing a comprehensive comparison of modern object detection approaches.

## Project Structure

- **src/**: Contains the source code for the models, data handling, training, inference, and utility functions.
  - **models/**: Implementation of computer vision models.
    - `yolo_v5.py`: YOLOv5 model implementation.
    - `yolo_v8.py`: YOLOv8 model implementation.
    - `detectron2_model.py`: Detectron2 model implementation.
    - `base_model.py`: Base class for computer vision models.
  - **data/**: Functions for dataset handling and preprocessing.
    - `dataset_loader.py`: Dataset loading and augmentation.
    - `preprocessing.py`: Image and annotation preprocessing.
  - **training/**: Training loop and configuration settings.
    - `trainer.py`: Training and validation logic.
    - `config.py`: Configuration settings for training.
  - **inference/**: Functions for running inference and visualizing results.
    - `detector.py`: Inference functions for predictions.
    - `visualizer.py`: Visualization functions for results.
  - **utils/**: Utility functions for metrics and logging.
    - `metrics.py`: Evaluation metrics calculations.
    - `helpers.py`: General utility functions.

- **notebooks/**: Jupyter notebooks for data exploration, model comparison, and results analysis.
  - `data_exploration.ipynb`: Explore and visualize the dataset.
  - `model_comparison.ipynb`: Compare YOLOv5, YOLOv8, and Detectron2 performance.
  - `results_analysis.ipynb`: Analyze and visualize experiment results.
  - `detectron2_training.ipynb`: Detectron2-specific training and evaluation.

- **tests/**: Unit tests for models and data handling.
  - `test_models.py`: Tests for model implementations.
  - `test_data.py`: Tests for data loading and preprocessing functions.

- **requirements.txt**: Lists the dependencies required for the project.

- **setup.py**: Metadata and dependencies for packaging the project.

## Installation

### Quick Setup

Run the automated installation script:

```bash
python install.py
```

This script will:
- Check Python version compatibility (3.8+ required)
- Install PyTorch with CUDA support (if available)
- Install Detectron2 and all dependencies
- Install YOLO models (YOLOv5, YOLOv8)
- Verify the installation

### Manual Installation

If you prefer manual installation:

1. **Clone the repository:**
```bash
git clone <repository-url>
cd computer-vision-experiments
```

2. **Install PyTorch:**
```bash
# For CUDA 11.8 (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision torchaudio
```

3. **Install basic requirements:**
```bash
pip install -r requirements.txt
```

4. **Install Detectron2:**
```bash
# Install dependencies
pip install fvcore iopath omegaconf hydra-core pycocotools

# Install Detectron2 (Windows)
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html

# Install Detectron2 (Linux/macOS)
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

5. **Install YOLO models:**
```bash
pip install ultralytics yolov5
```

### System Requirements

- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with CUDA 11.8+ (recommended, CPU also supported)
- **Memory**: 8GB+ RAM, 4GB+ VRAM for GPU training
- **Storage**: 2GB+ free space for models and dependencies

## Usage

### Training Your Models

1. **Prepare Dataset**: Convert your annotations to COCO format or use existing datasets
2. **Configure Training**: Set parameters in `src/training/config.py`
3. **Choose Architecture**: Select from available YOLO or Detectron2 models
4. **Start Training**: Use training scripts or notebooks
5. **Monitor Progress**: Track metrics with Weights & Biases integration

### Inference

Run inference on new images using any trained model:

```python
# Load trained model
model.load_model('path/to/trained/weights.pth')

# Run inference
results = model.predict(['image1.jpg', 'image2.jpg'])

# Visualize results
vis_image = model.visualize_prediction(image, results)
```

### Model Comparison

Use the comparison tools to evaluate different architectures:

```python
from src.utils.helpers import compare_model_outputs

# Compare multiple models on same image
predictions_dict = {
    'Faster R-CNN': faster_rcnn.predict(image),
    'YOLOv8': yolo_v8.predict(image),
    'RetinaNet': retinanet.predict(image)
}

compare_model_outputs(predictions_dict, 'test_image.jpg', 'comparison.png')
```

## Supported Models

This project supports multiple state-of-the-art object detection frameworks:

### YOLO Models
- **YOLOv5**: Fast and efficient single-stage detector with excellent balance of speed and accuracy
- **YOLOv8**: Latest YOLO version with improved architecture and better small object detection

### Detectron2 Models
- **Faster R-CNN**: Two-stage detector with high accuracy
- **RetinaNet**: Single-stage detector with focal loss for handling class imbalance
- **Mask R-CNN**: Instance segmentation capabilities in addition to object detection
- **FCOS**: Fully convolutional one-stage detector

### Key Features
- **Easy Model Switching**: Unified interface for training and inference across different architectures
- **Custom Dataset Support**: Train on your own data with minimal configuration
- **Performance Comparison**: Built-in tools to compare models on the same dataset
- **Visualization Tools**: Rich visualization for predictions and training metrics

## Quick Start

### 1. Basic Model Comparison
```python
from src.models.detectron2_model import Detectron2Model

# Initialize models
faster_rcnn = Detectron2Model('faster_rcnn')
retinanet = Detectron2Model('retinanet')

# Run inference
predictions = faster_rcnn.predict('path/to/image.jpg')
```

### 2. Training on Custom Dataset
```python
# Register your COCO format dataset
model = Detectron2Model('faster_rcnn', num_classes=20)
model.register_dataset("my_train", "train_annotations.json", "train_images/")
model.register_dataset("my_val", "val_annotations.json", "val_images/")

# Setup training
model.setup_training("my_train", "my_val", learning_rate=0.001, max_iter=1000)

# Train
trainer = model.train()
```

### 3. Running Notebooks
Start with the provided Jupyter notebooks:

```bash
jupyter notebook notebooks/
```

- **`detectron2_training.ipynb`**: Learn Detectron2 basics
- **`model_comparison.ipynb`**: Compare YOLO vs Detectron2 
- **`data_exploration.ipynb`**: Explore your dataset
- **`results_analysis.ipynb`**: Analyze training results

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or features.

## License

This project is licensed under the MIT License. See the LICENSE file for details.