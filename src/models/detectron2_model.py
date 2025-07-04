import os
import torch
import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from .base_model import BaseModel


class Detectron2Model(BaseModel):
    """
    Detectron2 model implementation supporting various architectures:
    - Faster R-CNN
    - RetinaNet
    - Mask R-CNN
    - FCOS
    """

    def __init__(self, model_name="faster_rcnn", num_classes=80, device="auto"):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        # Auto-detect device: use CUDA if available, otherwise CPU
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.cfg = None
        self.predictor = None
        self.trainer = None

        # Model configurations mapping
        self.model_configs = {
            "faster_rcnn": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
            "retinanet": "COCO-Detection/retinanet_R_50_FPN_3x.yaml",
            "mask_rcnn": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
            "fcos": "COCO-Detection/fcos_R_50_FPN_1x.yaml"
        }

        self._setup_config()

    def _setup_config(self):
        """Setup Detectron2 configuration"""
        self.cfg = get_cfg()

        # Load model configuration
        if self.model_name in self.model_configs:
            self.cfg.merge_from_file(model_zoo.get_config_file(
                self.model_configs[self.model_name]))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                self.model_configs[self.model_name])
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        # Basic configuration
        self.cfg.MODEL.DEVICE = self.device
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes

        if self.model_name == "retinanet":
            self.cfg.MODEL.RETINANET.NUM_CLASSES = self.num_classes
        elif self.model_name == "fcos":
            self.cfg.MODEL.FCOS.NUM_CLASSES = self.num_classes

    def setup_training(self, train_dataset_name, val_dataset_name, output_dir="./output",
                       learning_rate=0.00025, max_iter=1000, batch_size=2):
        """Setup training configuration"""
        self.cfg.DATASETS.TRAIN = (train_dataset_name,)
        self.cfg.DATASETS.TEST = (val_dataset_name,)
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.SOLVER.IMS_PER_BATCH = batch_size
        self.cfg.SOLVER.BASE_LR = learning_rate
        self.cfg.SOLVER.MAX_ITER = max_iter
        self.cfg.SOLVER.STEPS = []
        self.cfg.SOLVER.GAMMA = 0.1
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        self.cfg.TEST.EVAL_PERIOD = 500
        self.cfg.OUTPUT_DIR = output_dir

        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

    def register_dataset(self, dataset_name, json_file, image_root):
        """Register a COCO format dataset"""
        register_coco_instances(dataset_name, {}, json_file, image_root)

    def train(self, train_loader=None, val_loader=None, epochs=None):
        """Train the model"""
        if not self.cfg.DATASETS.TRAIN:
            raise ValueError(
                "Training dataset not configured. Use setup_training() first.")

        self.trainer = DefaultTrainer(self.cfg)
        self.trainer.resume_or_load(resume=False)
        self.trainer.train()

        return self.trainer

    def evaluate(self, val_loader=None, dataset_name=None):
        """Evaluate the model"""
        if dataset_name is None:
            dataset_name = self.cfg.DATASETS.TEST[0] if self.cfg.DATASETS.TEST else None

        if dataset_name is None:
            raise ValueError("No validation dataset specified")

        evaluator = COCOEvaluator(
            dataset_name, self.cfg, False, output_dir=self.cfg.OUTPUT_DIR)
        trainer = DefaultTrainer(self.cfg)
        trainer.test(self.cfg, trainer.model, evaluators=[evaluator])

    def predict(self, images):
        """Run inference on images"""
        if self.predictor is None:
            self.predictor = DefaultPredictor(self.cfg)

        if isinstance(images, str):
            # Single image path
            image = cv2.imread(images)
            predictions = self.predictor(image)
            return predictions
        elif isinstance(images, np.ndarray):
            # Single image array
            predictions = self.predictor(images)
            return predictions
        elif isinstance(images, list):
            # Multiple images
            results = []
            for img in images:
                if isinstance(img, str):
                    image = cv2.imread(img)
                else:
                    image = img
                predictions = self.predictor(image)
                results.append(predictions)
            return results
        else:
            raise ValueError(
                "Images must be a path, numpy array, or list of paths/arrays")

    def visualize_prediction(self, image, predictions, class_names=None):
        """Visualize predictions on image"""
        if isinstance(image, str):
            image = cv2.imread(image)

        metadata = MetadataCatalog.get(
            self.cfg.DATASETS.TRAIN[0] if self.cfg.DATASETS.TRAIN else "coco_2017_val")
        if class_names:
            metadata.thing_classes = class_names

        v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.2)
        out = v.draw_instance_predictions(predictions["instances"].to("cpu"))
        return out.get_image()[:, :, ::-1]

    def save_model(self, file_path):
        """Save model weights"""
        if self.trainer:
            torch.save(self.trainer.model.state_dict(), file_path)
        else:
            raise ValueError("No trained model to save")

    def load_model(self, file_path):
        """Load model weights"""
        self.cfg.MODEL.WEIGHTS = file_path
        self.predictor = DefaultPredictor(self.cfg)

    def get_model_info(self):
        """Get model information"""
        return {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "device": self.device,
            "config_file": self.model_configs.get(self.model_name, "Custom"),
            "input_format": "BGR",
            "architecture": "Two-stage" if self.model_name in ["faster_rcnn", "mask_rcnn"] else "Single-stage"
        }


class Detectron2Trainer(DefaultTrainer):
    """Custom trainer with additional evaluation capabilities"""

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
