class BaseModel:
    def __init__(self):
        pass

    def train(self, train_loader, val_loader, epochs):
        raise NotImplementedError("Train method not implemented.")

    def evaluate(self, val_loader):
        raise NotImplementedError("Evaluate method not implemented.")

    def predict(self, images):
        raise NotImplementedError("Predict method not implemented.")

    def save_model(self, file_path):
        raise NotImplementedError("Save model method not implemented.")

    def load_model(self, file_path):
        raise NotImplementedError("Load model method not implemented.")