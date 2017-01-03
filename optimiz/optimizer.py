# -*-coding:utf-8-*-

import abc


class Optimizer:
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def optimize(self):
        pass
