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
    train_total = 0
    startStr = f'Epoch {epoch+1:2d}/{num_epochs} \033[34m' + 'Train ' + '\033[0m'
    with tqdm(total=len(train_loader), desc=startStr) as pbar:
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer = optim.Adam(model.parameters(), lr=lr)  # Set learning rate here
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

        # 진행 바 끝날 때 메시지 수정
        str = f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.3f}%'
        pbar.set_postfix_str(str)

def valid_or_test(mode, model, device, dataloader):
    model.eval()
    loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        if mode == 'valid':
            startStr = f'Epoch {epoch+1:2d}/{num_epochs} \033[34m' + 'Valid ' + '\033[0m'
        elif mode == 'test':
            startStr = f'Epoch {epoch+1:2d}/{num_epochs} \033[34m' + 'Test ' + '\033[0m'

        with tqdm(total=len(dataloader), desc=startStr) as pbar:
        #print('\033[34m' + 'Validation ' + '\033[0m', end='')
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)

                loss += loss.item() * data.size(0)
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                pbar.update(1)

            loss = loss / total
            accuracy = 100 * correct / total
            if mode == 'valid':
                endStr = f'Valid Loss: {loss:.4f}, Valid Acc: {accuracy:.3f}%'
            elif mode == 'test':
                endStr = f'Test Loss: {loss:.4f}, Test Acc: {accuracy:.3f}%'
            pbar.set_postfix_str(endStr)

    if mode == 'valid':
        valid_losses.append(loss)
        valid_accuracies.append(accuracy)
        print('-'*110)

learning_rates = [0.001, 0.0001, 0.00001] 
# Initialize training loss and accuracy lists per learning rate
train_losses_lr = []
train_accuracies_lr = []
valid_losses_lr= []
valid_accuracies_lr = []

# 모델 학습 및 검증
for lr in learning_rates:
    # Initialize training loss and accuracy lists per learning rate
    train_losses_lr[lr] = []
    train_accuracies_lr[lr] = []
    valid_losses_lr[lr] = []
    valid_accuracies_lr[lr] = []
    
    for epoch in range(num_epochs):
        train(model, device, train_loader, optimizer, epoch, lr)
        valid_or_test('valid', model, device, valid_loader)

        # 검증 손실 확인하여 학습률 조정
        if valid_losses[-1] < best_valid_loss:
            best_valid_loss = valid_losses[-1]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print("Validation loss didn't improve for {} epochs. Reducing learning rate.".format(patience))
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5  # 학습률 0.5배 감소
                epochs_without_improvement = 0

valid_or_test('test', model, device, test_loader)

print(valid_losses)
# valid_losses의 device를 'cuda:0'에서 cpu로 변경
valid_losses = [loss.cpu().numpy() for loss in valid_losses]

train_losses, valid_losses

plt.figure()
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), valid_losses, label='Valid Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs. Epochs')
plt.legend()
plt.grid(True)
plt.show()

# 그래프 2: 손실 vs. 정확도
plt.figure()
plt.plot(range(1, num_epochs+1), train_accuracies, 'o-', label='Train')
plt.plot(range(1, num_epochs+1), valid_accuracies, 'o-', label='Valid')
plt.xlabel('Accuracy (%)')
plt.ylabel('Loss')
plt.title('Loss vs. Accuracy')
plt.legend()
plt.grid(True)
plt.show()