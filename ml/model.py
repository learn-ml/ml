import abc


class Model:
    __metaclass__ = abc.ABCMeta

    def __init__(self, weights=None, bias=None):
        if weights:
            self.__weights = weights
        if bias:
            self.__bias = bias

    def __str__(self):
        return "weights:%s, bias:%s" % (self.weights, self.bias)

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, weights):
        self.__weights = weights

    @property
    def bias(self):
        return self.__bias

    @bias.setter
    def bias(self, bias):
        self.__bias = bias
