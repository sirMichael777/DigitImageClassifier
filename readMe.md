# MNIST Digit Classifier

This project implements a neural network to classify handwritten digits from the MNIST dataset using PyTorch. The model is trained on the MNIST dataset and can be used to predict the digit in an input image file.

## Project Structure

- `Classifier.py`: Main script containing the implementation of the neural network, training procedure, and prediction functionality.
- `data/`: Directory where the MNIST dataset will be stored.
- `README.md`: This readme file.

## Requirements

To run this project, you need to have the following installed:

- Python 3.7 or higher
- PyTorch
- torchvision
- matplotlib
- PIL (Pillow)

You can install the required packages using `pip`:

```bash
pip install torch torchvision matplotlib pillow
```

## Project Components

### Data Preprocessing

The MNIST dataset is loaded and transformed using `torchvision.transforms`. The images are normalized to have a mean and standard deviation of 0.5.

### Neural Network

A simple feed-forward neural network is implemented using `torch.nn.Module`. The network consists of four fully connected layers with ReLU activation functions and dropout regularization.

### Training

The model is trained using the cross-entropy loss function and the Adam optimizer. The learning rate is scheduled to decrease every 5 epochs. Training progress, including loss and accuracy, is printed for each epoch.

### Prediction

The trained model can predict the digit in an input image file. The image is transformed and fed into the network, and the predicted digit is printed.

### Plotting Metrics

Training loss and accuracy are plotted over the epochs to visualize the model's performance.

## Usage

1. **Train the Model**: The script will train the model on the MNIST dataset.
2. **Predict Digits**: After training, you can provide an image file path to predict the digit. Enter 'exit' to quit the prediction loop.

### Running the Script

Run the main script to train the model and predict digits:

```bash
python Classifier.py
```

### Predicting a Digit

After training, you will be prompted to enter a file path to predict the digit in the image:

```plaintext
Done!
Please enter a filepath to predict or enter 'exit' to quit:
```

Provide the path to the image file, and the script will output the predicted digit. Enter 'exit' to quit.

### Example

```plaintext
Done!
Please enter a filepath to predict or enter 'exit' to quit:
> path/to/image.png
Predicted Digit = 7
```

## Notes

- Ensure that the MNIST dataset is available in the `data/` directory. If not, set the `download` parameter to `True` in the `MNIST` dataset loader.
- The input image for prediction should be a grayscale image of size 28x28 pixels. Ensure the image is properly formatted before prediction.

## License

This project is licensed under the MIT License. Feel free to use and modify the code.

## Contact

For any questions or suggestions, please contact sirMichael (michaelcmcmaseko@gmail.com).

Enjoy using the MNIST Digit Classifier!