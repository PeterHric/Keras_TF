# Keras_TF

##################################################################
# ImageClassification.py MNIST Digit Classifier (TensorFlow/Keras)
##################################################################

This project implements a Basic Neural Network to recognize handwritten digits. It is based on the official TensorFlow beginner tutorial.

## Model Architecture
* **Flatten Layer**: Reshapes the input data from a 28x28 matrix to a 784-element array.
* **Dense Layer (128 units)**: A fully connected hidden layer using the **ReLU** activation function.
* **Dropout (0.2)**: A regularization technique that prevents the model from memorizing the training data (overfitting).
* **Dense Layer (10 units)**: The output layer which generates a score (logit) for each digit from 0 to 9.

## Requirements
* Python 3.x
* TensorFlow 2.x
* NumPy
* Matplotlib (for visualization)

## Training Configuration
* **Optimizer**: Adam (Adaptive Moment Estimation)
* **Loss Function**: Sparse Categorical Crossentropy (from logits)
* **Metrics**: Accuracy
* **Epochs**: 10
