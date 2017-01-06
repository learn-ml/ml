# -*-coding:utf-8-*-
import logging

from optimiz.optimizer import Optimizer


class StochasticGradientDescent(Optimizer):
    def __init__(self, iteration=100, alpha=0.01):
        self.__iteration = iteration
        self.__alpha = alpha

    def optimize(self, func, x, y, theta):
        if len(x) != len(y):
            raise ValueError("len(x) must be equals to len(y).")
        for epoch in range(self.__iteration):
            for i in range(len(x)):
                gradient = func(x[i], y[i], theta)
                theta += self.__alpha * gradient
            if epoch % 20 == 0:
                logging.info("epoch:%s", epoch)
                print "epoch:%s, theta:%s" % (epoch, theta)
