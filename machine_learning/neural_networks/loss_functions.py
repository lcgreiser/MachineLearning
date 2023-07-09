import numpy as np


class LossFunction:

    @staticmethod
    def eval(x, t):
        # x/t: (batch, num_out)
        raise NotImplementedError

    @staticmethod
    def derivative(x, t):
        raise NotImplementedError


class MSELoss(LossFunction):

    @staticmethod
    def eval(x, t):
        return ((x - t)**2).mean(axis=1)

    @staticmethod
    def derivative(x, t):
        return 2*(x - t)/x.shape[1]


class CategoricalCrossentropyLoss(LossFunction):

    @staticmethod
    def eval(x, t):
        return -(t * np.log(x)).sum(axis=1)

    @staticmethod
    def derivative(x, t):
        return -t/x
