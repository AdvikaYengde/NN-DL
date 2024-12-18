#program to demonstrate the working of different activation function like sigmoid, tanh,relu & softmax to train neutral network using python  

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Generate synthetic data
def generate_data(samples=1000):
    # Generate random data
    X = np.random.rand(samples, 2)
    # Generate labels based on a simple rule
    Y = (X[:, 0] + X[:, 1] > 1).astype(int)
    return X, Y

# Function to create and compile a model with a specified activation function
def create_model(activation):
    model = keras.Sequential([
        layers.Dense(8, activation=activation, input_shape=(2,)),
        layers.Dense(8, activation=activation),
        layers.Dense(2, activation='softmax')  # Final layer with softmax for binary classification
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Generate the dataset
X, Y = generate_data()

# Split the dataset into training and testing sets
X_train, X_test = X[:800], X[800:]
Y_train, Y_test = Y[:800], Y[800:]

# Activation functions to demonstrate
activation_functions = ['sigmoid', 'tanh', 'relu', 'softmax']
results = {}

for activation in activation_functions:
    print(f"\nTraining model with {activation} activation:")
    model = create_model(activation)
    history = model.fit(X_train, Y_train, epochs=50, verbose=0, validation_split=0.2)
    results[activation] = history

# Evaluate models
for activation in activation_functions:
    model = create_model(activation)
    model.fit(X_train, Y_train, epochs=50, verbose=0)
    loss, accuracy = model.evaluate(X_test, Y_test)
    print(f"{activation.capitalize()} Activation - Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Plotting training history
plt.figure(figsize=(12, 6))
for activation in activation_functions:
    plt.plot(results[activation].history['accuracy'], label=f'{activation.capitalize()} Accuracy')
plt.title('Training Accuracy for Different Activation Functions')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
