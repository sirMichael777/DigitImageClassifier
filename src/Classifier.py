import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch import nn, optim
import sys
from PIL import Image
from pathlib import Path

# Data preprocessing
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

# Load MNIST data/home/m/mskmic017/Desktop/CSC3022F/mlAssignment1/MNIST_JPGS/testSample/testSample
train_set = MNIST(root='data', train=True, download=False, transform=transform)
test_set = MNIST(root='data', train=False, download=False, transform=transform)

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

# Training loop
def train_model():
    epochs = 15  # Number of epochs for training
    for e in range(epochs):
        running_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            optimizer.zero_grad()  # Clear gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            running_loss += loss.item()  # Accumulate loss
            
            _, predicted = torch.max(outputs.data, 1)  # Get predictions from the maximum value
            total += labels.size(0)  # Total number of labels
            correct += (predicted == labels).sum().item()  # Total correct predictions
        
        accuracy = 100 * correct / total  # Calculate accuracy
        scheduler.step()  # Adjust learning rate
        print(f"Epoch {e+1} - Training loss: {running_loss/len(train_loader)}, Accuracy: {accuracy}%")


# Function to predict digit from an image file
def predict(image_path):
    path = Path(image_path)
    if path.is_file():  # Check if provided path is a file
        img = Image.open(image_path)  # Open image file
        img = transform(img).unsqueeze(0)  # Preprocess image
        with torch.no_grad():  # Disable gradient calculation
            logps = model(img)  # Get log probabilities
        ps = torch.exp(logps)  # Get probabilities
        probab = list(ps.numpy()[0])  # Convert tensor to list
        print("Predicted Digit =", probab.index(max(probab)))  # Print predicted digit
    else:
        print(f"The provided path '{image_path}' is not a file. Please provide a valid image file path.")

# Main function to handle user interaction
def main():
    train_model()  # Train the model
    print("Done!\nPlease enter a filepath to predict:")
    path = input(">")
    while True:
        if path.lower() == 'exit':
            print("Exiting...")
            break
        predict(path)  # Predict digit from provided image file path
        path = input("Enter another filepath to predict or enter 'exit' to quit\n> ")  # Get user input

if __name__ == "__main__":
    main()  # Run main function