from torchvision import datasets, transforms
import os
import pandas as pd
import numpy as np
from PIL import Image

class DatasetLoader:
    def __init__(self, data_dir, img_size=(640, 640), batch_size=16, train_split=0.8):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.train_split = train_split
        self.train_data = []
        self.val_data = []

    def load_data(self):
        images = []
        annotations = []

        for img_file in os.listdir(os.path.join(self.data_dir, 'images')):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(self.data_dir, 'images', img_file)
                images.append(img_path)

                # Assuming annotations are in a corresponding 'labels' directory
                label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
                label_path = os.path.join(self.data_dir, 'labels', label_file)
                annotations.append(label_path)

        self._split_data(images, annotations)

    def _split_data(self, images, annotations):
        data = list(zip(images, annotations))
        np.random.shuffle(data)
        split_index = int(len(data) * self.train_split)
        train_data, val_data = data[:split_index], data[split_index:]

        self.train_data = train_data
        self.val_data = val_data

    def get_train_loader(self):
        return self._create_loader(self.train_data)

    def get_val_loader(self):
        return self._create_loader(self.val_data)

    def _create_loader(self, data):
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
        ])

        dataset = [(transform(Image.open(img_path)), open(label_path).read()) for img_path, label_path in data]
        return dataset

# Example usage:
# loader = DatasetLoader(data_dir='path/to/dataset')
# loader.load_data()
# train_loader = loader.get_train_loader()
# val_loader = loader.get_val_loader()