import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch import nn, optim
import sys
from PIL import Image
from pathlib import Path

# Data preprocessing
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Load MNIST data/home/m/mskmic017/Desktop/CSC3022F/mlAssignment1/MNIST_JPGS/testSample/testSample
train_set = MNIST(root='./data', train=True, download=True, transform=transform)
test_set = MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# Neural network definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Define fully connected layers
        self.fc1 = nn.Linear(28*28, 256)  # First fully connected layer, input size: 28*28, output size: 256
        self.fc2 = nn.Linear(256, 128)    # Second fully connected layer, input size: 256, output size: 128
        self.fc3 = nn.Linear(128, 64)     # Third fully connected layer, input size: 128, output size: 64
        self.fc4 = nn.Linear(64, 10)      # Fourth fully connected layer, input size: 64, output size: 10 (for classification)
        self.relu = nn.ReLU()             # Define activation function
        self.dropout = nn.Dropout(0.2)    # Dropout regularization with 20% probability to prevent overfitting

    def forward(self, x):
        x = x.view(-1, 28*28)                       # Flatten the input image tensor to a 1D tensor
        x = self.dropout(self.relu(self.fc1(x)))    # Apply ReLU and dropout to first fully connected layer
        x = self.dropout(self.relu(self.fc2(x)))    # Pass the output through the second fully connected layer, apply ReLU activation function, and dropout
        x = self.dropout(self.relu(self.fc3(x)))    # Pass the output through the third fully connected layer, apply ReLU activation function, and dropout
        x = self.fc4(x)                             # Pass the output through the fourth fully connected layer (output layer)
        return x

# Instantiate the neural network
model = Net()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.003)  # Adam optimizer with learning rate 0.003
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Learning rate scheduler

