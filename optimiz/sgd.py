# -*-coding:utf-8-*-
import logging

from optimiz.optimizer import Optimizer


class StochasticGradientDescent(Optimizer):
    def __init__(self, iteration=100, alpha=0.01):
        self.__iteration = iteration
        self.__alpha = alpha

    def optimize(self, func, x, y, model):
        if len(x) != len(y):
            raise ValueError("len(x) must be equals to len(y).")
        for epoch in range(self.__iteration):
            bg = None
            wg = None
            for i in range(len(x)):
                bg, wg = func(x[i], y[i])
                model.weights += self.__alpha * wg
                model.bias += self.__alpha * bg
            if (epoch + 1) % 10 == 0:
                logging.info("epoch:%s", epoch)
                print "epoch:%s, params:%s, gradient: %s,%s" % (epoch, model, wg, bg)
