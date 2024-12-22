Handwritten Digit Recognition using Convolutional Neural Network (CNN)

This project uses TensorFlow and Keras to build a Convolutional Neural Network (CNN) model that classifies handwritten digits from the MNIST dataset. The model is trained on the MNIST dataset and can make predictions on new images.

Project Overview
The goal of this project is to implement a CNN-based image classification model that recognizes handwritten digits (0-9). The model is trained using the MNIST dataset and is capable of making predictions on images.

Technologies Used
TensorFlow: For building and training the neural network.
Keras: For the high-level neural network API used in TensorFlow.
NumPy: For numerical computations.
Matplotlib: For visualizing the training history.
OpenCV: For image processing and prediction on custom images.

Setup
Install Dependencies
To run the project, you need Python installed on your machine. It is recommended to use a virtual environment. You can install the required dependencies using
pip install tensorflow opencv-python matplotlib numpy

Running the Project
Training the Model

Run the mnist_model.py script to train the model on the MNIST dataset.
python mnist_model.py


The script will:

Load and preprocess the MNIST dataset.
Build and compile the CNN model.
Train the model for 25 epochs, using a validation split of 20%.
Display training and validation loss and accuracy using Matplotlib.
Making Predictions

You can use the trained model to make predictions on new custom images. Ensure the image is resized to 28x28 pixels and in grayscale format before passing it into the model.

Run the predictions.py script to make predictions:
python predictions.py

The script will:

Load an image from the specified path.
Convert the image to grayscale and resize it to 28x28 pixels.
Normalize the image and reshape it for the model.
Predict the digit using the trained model and display the result.
Visualization

During training, the mnist_model.py script will generate plots for:

Training and validation loss
Training and validation accuracy
These plots will help visualize the model’s performance during the training process.

Example Custom Prediction
You can make predictions on your own images by placing them in the images/ directory. Ensure the image is in .jpg or .png format. The script predictions.py will load the image, process it, and predict the digit.

Accuracy
After training, the model achieves an accuracy of around 98.04% on the MNIST test dataset.

License
This project is open-source and available under the MIT License.

