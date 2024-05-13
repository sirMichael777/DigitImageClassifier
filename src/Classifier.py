import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch import nn, optim
import sys
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

# Data preprocessing
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load MNIST data
train_set = MNIST(root='data', train=True, download=False, transform=transform)
test_set = MNIST(root='data', train=False, download=False, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# Neural network definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define fully connected layers
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

# Instantiate the neural network
model = Net()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


def plot_metrics(train_losses, accuracies):
    epochs = len(train_losses)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Training loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(range(1, epochs + 1))  # Setting x-axis ticks to start from 1
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), accuracies, label='Training Accuracy')
    plt.title('Training Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.xticks(range(1, epochs + 1))  # Setting x-axis ticks to start from 1
    plt.legend()

    plt.show()


# Modify the training function to use this new plotting function
def train_model():
    epochs = 15
    train_losses = []
    accuracies = []

    for e in range(1, epochs + 1):  # Start from 1 to 15
        running_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        train_losses.append(train_loss)
        accuracies.append(accuracy)
        scheduler.step()
        print(f"Epoch {e} - Training loss: {train_loss}, Accuracy: {accuracy}%")

    plot_metrics(train_losses, accuracies)


# Function to predict digit from an image file
def predict(image_path):
    path = Path(image_path)
    if path.is_file():
        img = Image.open(image_path)
        img = transform(img).unsqueeze(0)
        with torch.no_grad():
            logps = model(img)
        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        print("Predicted Digit =", probab.index(max(probab)))
    else:
        print(f"The provided path '{image_path}' is not a file. Please provide a valid image file path.")

# Main function to handle user interaction
def main():
    train_model()
    path = input("Done!\nPlease enter a filepath to predict or enter 'exit' to quit:\n>")
    while True:
        if path.lower() == 'exit':
            print("Exiting...")
            break
        predict(path)
        path = input("Please enter a filepath to predict or enter 'exit' to quit:\n>")

if __name__ == "__main__":
    main()
