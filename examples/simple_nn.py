import numpy as np
from sklearn import datasets

from machine_learning.neural_networks.activation_functions import Tanh, Softmax
from machine_learning.neural_networks.layers import ForwardLayer, OutputLayer
from machine_learning.neural_networks.loss_functions import MSELoss, CategoricalCrossentropyLoss
from machine_learning.neural_networks.neural_network import NeuralNetwork


def run():
    # set random seed
    np.random.seed(42)

    # load mnist dataset
    mnist = datasets.load_digits()
    x = mnist['data'] / 255.0
    y = np.eye(10)[mnist['target']]

    # train test split
    n_train = round(len(x) * 0.9)
    x_train, y_train, x_test, y_test = x[:n_train], y[:n_train], x[n_train:], y[n_train:]

    # define neural network
    nn = NeuralNetwork([
        ForwardLayer(64, 32, Tanh),
        ForwardLayer(32, 16, Tanh),
        OutputLayer(16, 10, Softmax, CategoricalCrossentropyLoss),
    ])

    # train neural network
    nn.train(x, y, 100, 8, 0.1, True)

    # calculate accuracy
    y_pred_train = nn.forward(x_train)
    y_pred_test = nn.forward(x_test)
    correct_train = (y_pred_train.argmax(axis=1) == y_train.argmax(axis=1)).mean()
    correct_test = (y_pred_test.argmax(axis=1) == y_test.argmax(axis=1)).mean()
    print(f'{correct_train * 100:.2f}% correct in train')
    print(f'{correct_test * 100:.2f}% correct in test')


if __name__ == '__main__':
    run()
