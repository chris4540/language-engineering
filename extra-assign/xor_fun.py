import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from keras.models import Sequential

# define the input and output of the XOR function
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "float32")
outputs = np.array([[0], [1], [1], [0]], "float32")

# Build a simple two-layer feed-forward network
model = Sequential()
model.add(Dense(2, input_dim=2, use_bias=True, activation='relu'))
model.add(Dense(1, use_bias=False, activation='linear'))

# set the weightings and bias from the book
first_layer = model.layers[0]
first_layer.set_weights([np.ones((2, 2)), np.array([0, -1])])
second_layer = model.layers[1]
second_layer.set_weights([np.array([[1], [-2]])])

# check if they are equal
tf.debugging.assert_equal(model.predict(inputs), outputs)
print("Test pass!")
