import numpy as np
import pandas as pd
from scipy.io.arff import loadarff

from func.activation import sigmod
from ml.classifier import AbstractClassification
from optimiz.optimizer import Optimizer
from optimiz.sgd import StochasticGradientDescent


class LogicRegression(AbstractClassification):
    def __init__(self, optimizer):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be the subclass of optimize.optimizer.Optimizer")
        self.__optimizer = optimizer
        self.__theta = None
        self.__labels = None

    def __forward(self, x):
        if self.__theta is None:
            raise AssertionError("you should use this model after fitted it, call fit() first.")
        y = sigmod.apply(x.dot(self.__theta[1:]) + self.__theta[0])

        return [self.__labels['class'][1] if yi > 0.5 else self.__labels['class'][0] for yi in y]

    def __backward(self, x, y):
        yv = self.__labels['value'][y]
        hx = self.__labels['value'][self.__forward(x)[0]]
        bg = yv - hx
        tg = [(yv - hx) * x[j] for j in range(0, len(self.__theta) - 1)]
        tg.insert(0, bg)
        return np.array(tg).reshape(len(self.__theta), 1)

    """
    Fit the model according to the given training data.

    Parameters:
    -----------
    X:  {array-like, sparse matrix}, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and n_features is the number of features.
    y: array-like, shape (n_samples,)
        Target vector relative to X.

    Returns:
    --------
    self: object
        Return self
    """

    def fit(self, X, y):
        m, n = X.shape
        if m < 1 or n < 1:
            raise ValueError("invalid training set")
        # init params with bias self.__theta[0] is bias, self.__theta[1:] is weights
        self.__theta = np.random.rand(n + 1, 1)
        self.__labels = {
            'class': {i: v for i, v in enumerate(np.unique(y))},
            'value': {v: i for i, v in enumerate(np.unique(y))}
        }
        self.__optimizer.optimize(self.__backward, X, y, self.__theta)
        return self

    def test(self, X, y):
        hx = self.classify(X)
        count = 0
        for i in range(len(hx)):
            if hx[i] != [y.iloc[i]]:
                count += 1
        print count / float(len(X))

    def classify(self, X):
        return [self.__forward(x) for x in X]


if __name__ == '__main__':
    data, meta = loadarff("../dataset/electricity-normalized.arff")
    df = pd.DataFrame(data)
    x = np.array(df.iloc[:-100, :-1])
    test_x = np.array(df.iloc[-100:, :-1])
    y = df['class'][:-100]
    test_y = df['class'][-100:]

    optimizer = StochasticGradientDescent(iteration=100)
    lr = LogicRegression(optimizer)
    lr.fit(x, y)
    lr.test(test_x, test_y)
