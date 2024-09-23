import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# Hyperparameters
num_epochs_list = [10, 20, 30]  # List of epochs to experiment with
validation_splits = [0.2, 0.3, 0.4]  # List of validation set proportions

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preprocessing and transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load entire training dataset
full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)
        # Activation functions and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.relu(self.conv1(x))  # [batch_size, 32, 28, 28]
        x = self.relu(self.conv2(x))  # [batch_size, 64, 28, 28]
        x = self.pool(x)             # [batch_size, 64, 14, 14]
        x = self.dropout(x)
        x = x.view(-1, 64 * 14 * 14)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Training and validation loop
for num_epochs in num_epochs_list:
    for validation_split in validation_splits:
        # Split training data into train and validation sets
        train_size = int((1 - validation_split) * len(full_train_dataset))
        valid_size = len(full_train_dataset) - train_size
        train_dataset, valid_dataset = random_split(full_train_dataset, [train_size, valid_size])

        # Dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        # Model, loss function, optimizer
        model = Net().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training and validation results
        train_losses = []
        train_accuracies = []
        valid_losses = []
        valid_accuracies = []

        for epoch in range(num_epochs):
            # Train the model
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            start_str = f'Epoch {epoch+1:2d}/{num_epochs} \033[34m' + 'Train ' + '\033[0m'
            with tqdm(total=len(train_loader), desc=startStr) as pbar:
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)

                    optimizer.zero_grad()
                    outputs = model(data)
                    loss = criterion(outputs, target)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * data.size(0)
                    _, predicted = torch.max(outputs, 1)
                    train_total += target.size(0)
                    train_correct += (predicted == target).sum().item()
                    pbar.update(1)

            train_loss = train_loss / train_total
            train_accuracy = 100 * train_correct / train_total
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            # Validate the model
            model.eval()
            valid_loss = 0.0
            valid_correct = 0
            valid_total = 0
            with torch.no_grad():
                for data, target in valid_loader:
                    data, target = data.to(device), target.to(device)
                    outputs = model(data)
                    loss = criterion(outputs, target)

                    valid_loss += loss.item() * data.size(0)
                    _, predicted = torch.max(outputs, 1)
                    valid_total += target.size(0)
                    valid_correct += (predicted == target).sum().item()

            valid_loss = valid_loss / valid_total
            valid_accuracy = 100 * valid_correct / valid_total
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_accuracy)

        # Plot results
        plt.figure()
        plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
        plt.plot(range(1, num_epochs+1), valid_losses, label='Valid Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Loss vs. Epochs (Epochs={num_epochs}, Validation Split={validation_split})')
        plt.legend()
        plt.grid(True)
        plt.show()

        # ... (additional plotting or analysis as needed)