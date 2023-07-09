import numpy as np


class ActivationFunction:

    @staticmethod
    def forward(x):
        raise NotImplementedError

    @staticmethod
    def derivative(x):
        # x: # (batch, num_out)
        # returns jacobian in shape of (batch, num_out_activation, num_out_netinp)
        raise NotImplementedError


class Linear(ActivationFunction):

    @staticmethod
    def forward(x):
        return x

    @staticmethod
    def derivative(x):
        return np.repeat(np.eye(x.shape[1])[np.newaxis, :, :], x.shape[0], axis=0)


class Sigmoid(ActivationFunction):

    @staticmethod
    def forward(x):
        return 1 / (1+np.exp(-x))

    @staticmethod
    def derivative(x):
        z = np.zeros((x.shape[0], x.shape[1], x.shape[1]), dtype=x.dtype)
        z[:, np.arange(x.shape[1]), np.arange(x.shape[1])] = Sigmoid.forward(x)*(1-Sigmoid.forward(x))
        return z


class Tanh(ActivationFunction):

    @staticmethod
    def forward(x):
        return np.tanh(x)

    @staticmethod
    def derivative(x):
        z = np.zeros((x.shape[0], x.shape[1], x.shape[1]), dtype=x.dtype)
        z[:, np.arange(x.shape[1]), np.arange(x.shape[1])] = 1 - np.tanh(x)**2
        return z


class Softmax(ActivationFunction):

    @staticmethod
    def forward(x):
        return np.exp(x) / np.exp(x).sum(axis=1)[:, np.newaxis]

    @staticmethod
    def derivative(x):
        fw = Softmax.forward(x)
        s_i = np.repeat(fw[:, :, np.newaxis], x.shape[1], axis=2)
        s_j = np.repeat(fw[:, np.newaxis, :], x.shape[1], axis=1)
        kronecker = np.repeat(np.eye(x.shape[1])[np.newaxis, :, :], x.shape[0], axis=0)
        return s_i * (kronecker-s_j)
