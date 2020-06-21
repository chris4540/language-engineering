import numpy as np
from tensorflow.keras.layers import Dense
from keras.models import Sequential

training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "float32")
target_data = np.array([[0], [1], [1], [0]], "float32")


model = Sequential()
model.add(Dense(2, input_dim=2, use_bias=True, activation='relu'))
model.add(Dense(1, use_bias=False, activation='linear'))

# set values from the book
first_layer = model.layers[0]
first_layer.set_weights([np.ones((2, 2)), np.array([0, -1])])
second_layer = model.layers[1]
second_layer.set_weights([np.array([[1], [-2]])])

print("predict:")
print(model.predict(training_data))
print("true:")
print(target_data)
