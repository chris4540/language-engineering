import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from keras.models import Sequential

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "float32")
outputs = np.array([[0], [1], [1], [0]], "float32")

model = Sequential()
model.add(Dense(2, input_dim=2, use_bias=True, activation='relu'))
model.add(Dense(1, use_bias=False, activation='linear'))

# set values from the book
first_layer = model.layers[0]
first_layer.set_weights([np.ones((2, 2)), np.array([0, -1])])
second_layer = model.layers[1]
second_layer.set_weights([np.array([[1], [-2]])])

tf.debugging.assert_equal(model.predict(inputs), target_data)
