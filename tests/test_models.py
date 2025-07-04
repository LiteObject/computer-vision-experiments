import unittest
from src.models.yolo_v5 import YOLOv5
from src.models.yolo_v8 import YOLOv8

class TestYOLOModels(unittest.TestCase):

    def setUp(self):
        self.yolo_v5 = YOLOv5(weights='path/to/yolov5/weights.pt')
        self.yolo_v8 = YOLOv8(weights='path/to/yolov8/weights.pt')

    def test_yolo_v5_initialization(self):
        self.assertIsNotNone(self.yolo_v5)

    def test_yolo_v8_initialization(self):
        self.assertIsNotNone(self.yolo_v8)

    def test_yolo_v5_prediction(self):
        test_image = 'path/to/test/image.jpg'
        predictions = self.yolo_v5.predict(test_image)
        self.assertIsInstance(predictions, list)

    def test_yolo_v8_prediction(self):
        test_image = 'path/to/test/image.jpg'
        predictions = self.yolo_v8.predict(test_image)
        self.assertIsInstance(predictions, list)

if __name__ == '__main__':
    unittest.main()