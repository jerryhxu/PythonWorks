import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from matplotlib import pyplot

from keras.datasets import mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()

print(test_y[9])

#for i in range(1):
i = 8
pyplot.subplot(330 + 1 + i)
pyplot.imshow(test_X[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()
