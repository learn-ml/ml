import abc


class AbstractFunction:
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def apply(self, x):
        pass

    @abc.abstractmethod
    def derivative(self, x):
        pass
