import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt

from keras.datasets import mnist

(mTrain_X, mTrain_y), (test_X, test_y) = mnist.load_data()

print('X_train: ' + str(mTrain_X.shape))
print('Y_train: ' + str(mTrain_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

mTrain_X = mTrain_X.reshape(60000, -1)
print(mTrain_X.shape)

train_X = np.array([[0 for x in range(784)] for y in range(5000)])
train_y = np.array([[0 for x in range(1)] for y in range(5000)])

for i in range(5000):
    train_X[i] = mTrain_X[i]
    train_y[i] = mTrain_y[i]

model = Sequential(
    [
        tf.keras.Input(shape=(784,)),

        Dense(25, activation='relu'),
        Dense(15, activation='relu'),
        Dense(10, activation='linear')
    ]
)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics = ['accuracy']
)

model.fit (
    train_X, train_y,
    epochs = 100
)
loss.values = model.estimator.loss_curve_
print(loss_values)

predictions = np.zeros(100)
for i in range(100):
    prediction = model.predict(test_X[i].reshape(1, 784))
    predictions[i] = np.argmax(prediction)

correct = 0
for i in range(100):
    print(predictions[i])
    print(str(test_y[i]) + "\n")
    if predictions[i] == test_y[i]:
        correct = correct+1

print(correct)


