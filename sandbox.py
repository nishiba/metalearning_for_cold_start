# coding: utf-8
import numpy as np
import chainer.links as links
import chainer.functions as functions
from chainer.variable import Variable


def main():
    n_item = 100
    n_factor = 2
    embed = links.EmbedID(n_item, n_factor)
    x0 = np.array([[1, 2, 0], [3, 0, 0], [0, 0, 0]])

    e0 = embed(x0)
    print(e0.data)
    r0 = functions.sum(e0, axis=1) / np.array([[10], [10], [10]])
    print(r0.data)

if __name__ == '__main__':
    main()
