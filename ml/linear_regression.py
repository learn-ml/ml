# -*-coding:utf-8-*-
import numpy as np

from ml.regression import AbstractRegression
from optimiz.sgd import StochasticGradientDescent


class LinearRegression(AbstractRegression):
    """
    - param_size must ge 2 and the first param is bias, the others are weights.
    """

    def __init__(self, params_size):
        if params_size < 2:
            raise ValueError("参数个数必须大于等于2")
        self.__theta = np.random.rand(params_size).reshape(params_size, 1)

    def __forward(self, x):
        return self.__theta.T.dot(np.r_[1, x])

    def __backward(self, x, y):
        ex = np.r_[1, x]
        return np.array([(y - self.__forward(x)) * ex[j] for j in range(len(self.__theta))])

    def fit(self, x, y, optimizer):
        optimizer.optimize(self.__backward, x, y, self.__theta)

    def predict(self, x):
        return self.__forward(x)


if __name__ == '__main__':
    regression = LinearRegression(3)
    sgd = StochasticGradientDescent(iteration=80000, alpha=0.01)
    x = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])
    y = x.dot(np.array([2, 3])) + 4
    regression.fit(x, y, sgd)
    print regression.predict(x[0])
