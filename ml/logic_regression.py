# -*-coding:utf-8-*-
from func.activation import sigmod
from ml.classifier import AbstractClassification
from optimiz.sgd import StochasticGradientDescent


class LogicRegression(AbstractClassification):
    """
    sample logic regression algorithm
    """

    def __init__(self):
        pass

    def __forward(self, x):
        return sigmod.apply(x)

    def __backward(self, x, y):
        pass

    def fit(self, x, y, optimizer):
        optimizer.optimize(self.__backward, x, y)
        pass

    def classify(self, x):
        pass


if __name__ == '__main__':
    lr = LogicRegression()
    optimizer = StochasticGradientDescent()
    lr.fit([], [], optimizer)
    lr.classify([])
