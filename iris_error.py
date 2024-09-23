import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

num_epochs = 10
batch_size = 64

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preprocessing and transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load entire training dataset
full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Split training data (80% train, 20% validation)
train_size = int(0.8 * len(full_train_dataset))
valid_size = len(full_train_dataset) - train_size
train_dataset, valid_dataset = random_split(full_train_dataset, [train_size, valid_size])

# Load test dataset
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

# Model, loss function, optimizer
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and validation loss/accuracy lists
train_losses = []
train_accuracies = []
valid_losses = []
valid_accuracies = []

# Learning rate experiments
learning_rates = [0.001, 0.0001, 0.00001]

# Initialize training results per learning rate (dictionaries with loss & accuracy lists)
results = {lr: {'train_losses': [], 'train_accuracies': [], 'valid_losses': [], 'valid_accuracies': []} for lr in learning_rates}

def train(model, device, train_loader, optimizer, epoch, lr):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total