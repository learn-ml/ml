import numpy as np

from func import AbstractFunction


class Sigmod(AbstractFunction):
    def apply(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return apply(x) * (1 - apply(x))


class Tanh(AbstractFunction):
    def apply(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def derivative(self, x):
        return 1 - np.power(apply(x), 2)


sigmod = Sigmod()
tanh = Tanh()
