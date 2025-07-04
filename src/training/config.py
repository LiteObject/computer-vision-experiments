# Configuration settings for training YOLO models

class Config:
    def __init__(self):
        # Model parameters
        self.model_name = "YOLOv5"  # Options: "YOLOv5", "YOLOv8"
        self.input_size = 640  # Size of the input images
        self.num_classes = 80  # Number of classes in the dataset

        # Training parameters
        self.batch_size = 16  # Number of samples per gradient update
        self.learning_rate = 0.001  # Learning rate for the optimizer
        self.num_epochs = 50  # Number of epochs to train the model
        self.weight_decay = 0.0005  # Weight decay for regularization

        # Paths
        self.train_data_path = "data/train"  # Path to training data
        self.val_data_path = "data/val"  # Path to validation data
        self.checkpoint_path = "checkpoints/"  # Path to save model checkpoints
        self.log_path = "logs/"  # Path to save training logs

        # Other settings
        self.use_cuda = True  # Use GPU if available
        self.seed = 42  # Random seed for reproducibility

    def display(self):
        print("Configuration Settings:")
        for key, value in vars(self).items():
            print(f"{key}: {value}")