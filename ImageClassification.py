# Follow this tutorial:
# https://www.tensorflow.org/tutorials/quickstart/beginner

# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# convert data from INTs to FLOATs
(x_train, x_test) = x_train/255.0, x_test/255.0

# Create the model of NN  (you can tweak model parameters)
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # https://www.tensorflow.org/api_docs/python/tf/keras/activations
    tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dense(128, activation='sigmoid'),
    # tf.keras.layers.Dense(128, activation='linear'),
    # tf.keras.layers.Dense(128, activation='tanh'),
    # tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Dense(10)
    # tf.keras.layers.Dense(120),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

test = range(10)
select1 = test[:1]
x_train_01 = x_train[:1]

# Get the logits and log-ods
predictions = model(x_train[:1]).numpy()
# Turn them to probabilities for better interpretability
predictions_prob = tf.nn.softmax(predictions).numpy()

# Create loss function object, able to calculate loss from logits - for back_propagation
scce_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
scce_loss = scce_loss_fn(y_train[:1], predictions).numpy()

# Now that we have prepared all model parameters, compile the model
model.compile(optimizer='adam',
              loss=scce_loss_fn,
              metrics=['accuracy'])

# Adjust the models weights (params) to minimize the loss (note: we use the TRAINing set of course)
# This should train the model on several epochs
model.fit(x_train, y_train, epochs=10)

# Evaluate the trained model with the TESTing data set (How much is it generalized)
model.evaluate(x_test,  y_test, verbose=3)

scce_loss
