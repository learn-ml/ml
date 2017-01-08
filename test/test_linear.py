import numpy as np
import pandas as pd
from scipy.io.arff import loadarff

from ml.linear import LogicRegression
from optimiz.sgd import StochasticGradientDescent

optimizer = StochasticGradientDescent(iteration=80)


def testLogicRegression():
    data, meta = loadarff("../dataset/electricity-normalized.arff")
    df = pd.DataFrame(data)
    x = np.array(df.iloc[:-100, :-1])
    test_x = np.array(df.iloc[-100:, :-1])
    y = df['class'][:-100]
    test_y = df['class'][-100:]

    lr = LogicRegression(optimizer)
    lr.fit(x, y)
    print lr.test(test_x, test_y)


if __name__ == '__main__':
    testLogicRegression()
