import math
from typing import Type, Optional

import numpy as np

from .activation_functions import ActivationFunction
from .loss_functions import LossFunction


class Layer:

    def __init__(self, num_inp: int, num_out: int, activation_function: Type[ActivationFunction]):
        xavier_range = 1/math.sqrt(num_inp)

        self.weights: np.ndarray = np.random.uniform(low=-xavier_range, high=xavier_range, size=(num_inp+1, num_out))
        self.weights[-1, :] = 0  # set bias to zero
        self.activation_function: Type[ActivationFunction] = activation_function

        self.x_biased: Optional[np.ndarray] = None
        self.network_input: Optional[np.ndarray] = None
        self.activations: Optional[np.ndarray] = None
        self.grad_weights: Optional[np.ndarray] = None
        self.grad_netinp: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_biased = np.pad(x, ((0, 0), (0, 1)), 'constant', constant_values=(1,))  # (batch, num_inp+1)
        self.network_input = self.x_biased @ self.weights  # (batch, num_out)
        self.activations = self.activation_function.forward(self.network_input)  # (batch, num_out)
        return self.activations

    def backward(self):
        # todo: implement optimizer
        grad_activation = self.get_grad_activation()  # (batch, num_out)
        self.grad_netinp = np.einsum('ba,baz->bz', grad_activation, self.activation_function.derivative(self.network_input), optimize='greedy')  # (batch, num_out) E_j
        self.grad_weights = np.einsum('bj,bi->bij', self.grad_netinp, self.x_biased, optimize='greedy')  # (batch, num_inp+1, num_out) E_ij

    def adjust_weights(self, lr: float):
        self.weights -= lr * self.grad_weights.mean(axis=0)

    def get_grad_activation(self):
        raise NotImplementedError


class ForwardLayer(Layer):

    def __init__(self, num_inp: int, num_out: int, activation_function: Type[ActivationFunction]):
        self.next_layer: Optional[Layer] = None
        super().__init__(num_inp, num_out, activation_function)

    def get_grad_activation(self):
        return (self.next_layer.grad_netinp @ self.next_layer.weights.T)[:, :-1]  # (batch, num_out)


class OutputLayer(Layer):

    def __init__(self, num_inp: int, num_out: int, activation_function: Type[ActivationFunction], loss: Type[LossFunction], *args, **kwargs):
        self.loss: Type[LossFunction] = loss
        self.target: Optional[np.ndarray] = None
        super().__init__(num_inp, num_out, activation_function)

    def get_grad_activation(self):
        return self.loss.derivative(self.activations, self.target)  # (batch, num_out)
