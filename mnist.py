import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from dense import Dense
from convolutional import Convolutional
from reshape import Reshape
from activations import Sigmoid
from losses import mse, mse_prime
from network import train, predict

data = pd.read_csv("train.csv")
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_train = data[:int(m/2)].T
x_train = data_train[1:n].T
# need to transpose so that first row matches with first element of y_train
x_train= x_train.astype("float32") / 255
y_train = data_train[0]
# this should just be an array

data_test = data[int(m/2):m].T
x_test = data_test[1:n].T
x_test = x_test.astype("float32") / 255
y_test = data_test[0]

x_train = x_train.reshape(len(x_train), 1, 28, 28)
x_test = x_test.reshape(len(x_test), 1, 28, 28)

identity_matrix = np.eye(10)

y_train = identity_matrix[y_train]
y_test = identity_matrix[y_test]

y_train = y_train.reshape(len(y_train), 10, 1)
y_test = y_test.reshape(len(y_test), 10, 1)

# neural network
network = [
    Convolutional((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 10),
    Sigmoid()
]

# train
train(network, mse, mse_prime, x_train, y_train)

# test
for x, y, i in zip(x_test, y_test, range(10)):
    output = predict(network, x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")

    image = x.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(image, interpolation = "nearest")
    plt.show()