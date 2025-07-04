from models.yolo_v5 import YOLOv5
from models.yolo_v8 import YOLOv8

class ObjectDetector:
    def __init__(self, model_type='yolov5', weights_path=None):
        if model_type == 'yolov5':
            self.model = YOLOv5(weights_path)
        elif model_type == 'yolov8':
            self.model = YOLOv8(weights_path)
        else:
            raise ValueError("Unsupported model type. Choose 'yolov5' or 'yolov8'.")

    def load_model(self):
        self.model.load_weights()

    def predict(self, image):
        return self.model.predict(image)

    def predict_from_path(self, image_path):
        image = self.load_image(image_path)
        return self.predict(image)

    def load_image(self, image_path):
        # Implement image loading logic here
        pass

    def visualize_predictions(self, predictions):
        # Implement visualization logic here
        pass