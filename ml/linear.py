import numpy as np

from func.activation import sigmoid
from ml.classifier import AbstractClassification
from ml.model import Model
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
    def __init__(self, optimizer, model=None):
        if not optimizer or not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be the subclass of optimize.optimizer.Optimizer")
        self.__optimizer = optimizer

        if model:
            self.__model = model
        else:
            self.__model = Model()

        self.__labels = None

    def __forward(self, x):
        if self.__model is None:
            raise AssertionError("you should use this model after fitted it, call fit() first.")

        w = self.__model.weights
        b = self.__model.bias

        hx = sigmoid.apply(x.dot(w) + b)
        return 1 if hx[0] > 0.5 else 0

    def __backward(self, x, y):
        yv = self.__labels["value"][y]
        hx = self.__forward(x)
        sigma = yv - hx
        return sigma, (sigma * x).reshape(-1, 1)

    def fit(self, X, y):
        m, n = X.shape
        if m < 1 or n < 1:
            raise ValueError("invalid training set")

        self.__labels = {
            'class': {i: v for i, v in enumerate(np.unique(y))},
            'value': {v: i for i, v in enumerate(np.unique(y))}
        }

        self.__model.weights = np.random.rand(n, 1)
        self.__model.bias = np.random.rand(1, 1)
        self.__optimizer.optimize(self.__backward, X, y, self.__model)

        return self

    def test(self, X, y):
        hx = self.classify(X)
        count = 0
        for i in range(len(hx)):
            if hx[i] != y.iloc[i]:
                count += 1
        return 1 - (count / float(len(X)))

    def classify(self, X):
        return [self.__labels['class'][self.__forward(x)] for x in X]
