# Fashion MNIST CNN From Scratch

## Project Overview
This project implements a Convolutional Neural Network (CNN) completely from scratch using NumPy for image classification on the Fashion MNIST dataset.

The main objective of this assignment is to understand internal working of CNNs by manually implementing:
* Convolution Layer
* Max Pooling Layer
* Fully Connected Layer
* Forward Propagation
* Backpropagation
* Weight Updates
* CNN Training Loop

The model is trained and evaluated on the Fashion MNIST dataset.

# Dataset Used

## Fashion MNIST
Fashion MNIST is a dataset of grayscale clothing images consisting of:
* 70,000 total images
* 10 classes
* Image size: 28 × 28 pixels

Classes include:
* T-shirt/top
* Trouser
* Pullover
* Dress
* Coat
* Sandal
* Shirt
* Sneaker
* Bag
* Ankle boot

Dataset source:
tensorflow.keras.datasets.fashion_mnist

# Technologies Used
* Python
* NumPy
* TensorFlow/Keras (only for dataset loading)

# CNN Architecture
The CNN model follows this architecture:

Input Image (28x28x1)
        ↓
Convolution Layer (8 Filters)
        ↓
Max Pooling Layer
        ↓
Flatten Layer
        ↓
Fully Connected Softmax Layer
        ↓
Output Prediction (10 Classes)

# Features Implemented

## Convolution Layer
* Manual 3×3 filter convolution
* Feature map generation
* Forward propagation
* Backpropagation

## Max Pooling Layer
* 2×2 max pooling
* Dimensionality reduction
* Backward propagation support

## Fully Connected Layer
* Softmax activation
* Weight initialization
* Forward propagation
* Backpropagation

## Training
* Cross entropy loss
* Gradient descent
* Weight updates
* Multi epoch training

# Training Details
* Epochs: 3
* Learning Rate: 0.005
* Training Samples Used: 1000
* Testing Samples Used: 100

# Output
The model successfully learns during training and achieves approximately:
Training Accuracy: ~80%
Testing Accuracy: ~68%
## Training Output Screenshot

![CNN Output](output.png)
# How To Run The Project
## Step 1:
Clone the repository:
git clone <repository_link>

## Step 2
Open the project folder in VS Code.

## Step 3
Activate virtual environment:

### Mac/Linux
source venv/bin/activate

### Windows
venv\Scripts\activate

## Step 4
Install required libraries:
pip install numpy matplotlib tensorflow

## Step 5
Run the program:
python cnn_from_scratch.py

# Project Structure
UE24CS645BC2_PES1PG25CS021_Fashion_MNIST_CNN/
│
├── cnn_from_scratch.py
├── README.md
├── requirements.txt
└── venv/

# Learning Outcomes
Through this project, the following concepts were understood:
* CNN architecture
* Feature extraction using convolution
* Pooling operations
* Forward propagation
* Backpropagation
* Weight optimization
* Image classification using deep learning

# Author
Deekshitha Jha
USN: PES1PG25CS021