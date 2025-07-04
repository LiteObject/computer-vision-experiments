class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for images, targets in self.train_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(self.train_loader)
        print(f'Epoch [{epoch}], Loss: {average_loss:.4f}')

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        average_loss = total_loss / len(self.val_loader)
        print(f'Validation Loss: {average_loss:.4f}')

    def save_checkpoint(self, epoch, path):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f'Checkpoint saved at {path}')

    def train(self, num_epochs, checkpoint_path):
        for epoch in range(num_epochs):
            self.train_one_epoch(epoch)
            self.validate()
            self.save_checkpoint(epoch, checkpoint_path)