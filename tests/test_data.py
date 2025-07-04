import unittest
from src.data.dataset_loader import load_dataset
from src.data.preprocessing import preprocess_image

class TestDataFunctions(unittest.TestCase):

    def test_load_dataset(self):
        dataset = load_dataset('path/to/dataset')
        self.assertIsNotNone(dataset)
        self.assertGreater(len(dataset), 0)

    def test_preprocess_image(self):
        image = 'path/to/image.jpg'
        processed_image = preprocess_image(image)
        self.assertIsNotNone(processed_image)
        self.assertEqual(processed_image.shape, (desired_height, desired_width, channels))

if __name__ == '__main__':
    unittest.main()