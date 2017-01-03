# -*-coding:utf-8-*-
import abc


class AbstractRegression:
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def predict(self, x):
        pass

    @abc.abstractmethod
    def fit(self, x, y, optimizer):
        pass
