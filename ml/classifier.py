# -*-coding:utf-8-*-
import abc


class AbstractClassification:
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def classify(self, x):
        pass

    @abc.abstractmethod
    def fit(self, x, y, optimizer):
        pass
