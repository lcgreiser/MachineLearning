from typing import List, Optional

import numpy as np

from machine_learning.neural_networks.layers import Layer, OutputLayer


class NeuralNetwork:

    def __init__(self, layers: List[Layer]):
        self.layers: List[Layer] = layers
        self.output: Optional[np.ndarray] = None
        assert isinstance(layers[-1], OutputLayer), 'Last layer should be output layer.'
        if len(layers) > 1:
            for layer1, layer2 in zip(self.layers[:-1], self.layers[1:]):
                layer1.next_layer = layer2

    def forward(self, x: np.ndarray):
        for layer in self.layers:
            x = layer.forward(x)
        self.output = x
        return x

    def backward(self, t: np.ndarray):
        self.layers[-1].target = t
        for layer in self.layers[::-1]:
            layer.backward()
        return self.layers[-1].loss.eval(self.output, t)

    def adjust_weights(self, lr: float):
        for layer in self.layers:
            layer.adjust_weights(lr)

    def train(self, x_train, y_train, num_epochs, batch_size, lr, shuffle):
        num_train_samples = len(x_train)
        assert num_train_samples >= batch_size, "Batch size is smaller than training set."
        assert len(x_train.shape) == 2 and len(y_train.shape) == 2, "Training data has invalid shape."
        order = np.arange(num_train_samples)
        if shuffle:
            np.random.shuffle(order)
        epoch = 0
        idx = 0
        while epoch < num_epochs:
            if idx + batch_size <= num_train_samples:
                x_batch = x_train[order[idx: idx + batch_size]]
                y_batch = y_train[order[idx: idx + batch_size]]
            else:
                x_batch = x_train[np.append(order[idx:], order[:num_train_samples - idx])]
                y_batch = y_train[np.append(order[idx:], order[:num_train_samples - idx])]
            y_pred = self.forward(x_batch)
            self.backward(y_batch)
            self.adjust_weights(lr)
            idx += batch_size
            if idx >= num_train_samples:
                idx -= num_train_samples
                print(f"{epoch+1:4d} error:", np.mean((y_pred - y_batch)**2))  # todo: criterion
                epoch += 1
