# -*-:coding:utf-8-*-
import numpy as np
import pandas as pd


class WineQualityWhite:
    def load(self):
        df = pd.read_csv("./winequality-white.csv")


if __name__ == '__main__':
    wine = WineQualityWhite()
    df = wine.load()

    print np.array_split(df, 3)
