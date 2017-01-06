import numpy as np

from func.activation import sigmoid
from ml.classifier import AbstractClassification
from ml.regression import AbstractRegression
from optimiz.optimizer import Optimizer


class LinearRegression(AbstractRegression):
    def __init__(self):
        pass

    def predict(self, x):
        pass

    def fit(self, x, y, optimizer):
        pass


class LogicRegression(AbstractClassification):
    def __init__(self, optimizer):
        if not optimizer or not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be the subclass of optimize.optimizer.Optimizer")
        self.__optimizer = optimizer
        self.__weights = None
        self.__bias = None
        self.__labels = None

    def __forward(self, x):
        if self.__weights is None or self.__bias is None:
            raise AssertionError("you should use this model after fitted it, call fit() first.")
        hx = sigmoid.apply(x.dot(self.__weights) + self.__bias)
        return 1 if hx[0] > 0.5 else 0

    def __backward(self, x, y, params):
        self.__bias = params[0]
        self.__weights = params[1:]
        yv = self.__labels["value"][y]
        hx = self.__forward(x)
        return np.r_[yv - hx, x * (yv - hx)].reshape(-1, 1)

    def fit(self, X, y):
        m, n = X.shape
        if m < 1 or n < 1:
            raise ValueError("invalid training set")

        self.__labels = {
            'class': {i: v for i, v in enumerate(np.unique(y))},
            'value': {v: i for i, v in enumerate(np.unique(y))}
        }

        self.__weights = np.random.rand(n, 1)
        self.__bias = np.random.rand(1, 1)
        self.__optimizer.optimize(self.__backward, X, y, np.r_[self.__bias, self.__weights])

        return self

    def test(self, X, y):
        hx = self.classify(X)
        count = 0
        for i in range(len(hx)):
            if hx[i] != y.iloc[i]:
                count += 1
        return count / float(len(X))

    def classify(self, X):
        return [self.__labels['class'][self.__forward(x)] for x in X]
