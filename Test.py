import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential(
    [
        tf.keras.Input(shape=(3,)),  # specify input size
        ### START CODE HERE ###
        Dense(1, activation='sigmoid'),

        ### END CODE HERE ###
    ], name="my_model"
)

model.summary()

x = np.array([[1,2,3], [4,5,6], [7,8,9]])
y = np.array([[1],[0],[0]])
print(x.shape)
print(y.shape)

[layer1] = model.layers
W1,b1 = layer1.get_weights()
print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
print(model.layers[0].weights)

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)

model.fit (
    x,y,
    epochs=10000
)
predictions = np.zeros(len(x))
for i in range(3):
    prediction = model.predict(x[i].reshape(1,3))
    if prediction >= 0.5:
        predictions[i] = 1
    else:
        predictions[i] = 0

print(predictions)