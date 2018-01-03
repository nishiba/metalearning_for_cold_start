# coding: utf-8

import chainer
import chainer.functions as functions
import chainer.links as links
import pandas as pd
from chainer import training, iterators, optimizers
from chainer.training import extensions

from click_data_set import ClickDataSet


class LinearWeightAdaption(chainer.Chain):
    def __init__(self, n_user: int, n_item: int, n_factor=10):
        super(LinearWeightAdaption, self).__init__()

        # settings
        self.n_user = n_user
        self.n_item = n_item
        self.n_factor = n_factor

        with self.init_scope():
            # architecture
            self.embedId = links.EmbedID(self.n_item, self.n_factor)
            self.linear = links.Convolution2D(1, 1, (1, 2), nobias=True, pad=0)
            self.bias = links.Bias(shape=(1,))

    def __call__(self, x0, x1, x, t=None, train=True):
        # item embedding
        e0 = self.embedId(x0)
        e1 = self.embedId(x1)

        # user embedding
        r0 = functions.mean(e0, axis=1)
        r1 = functions.mean(e1, axis=1)

        # weight depending on users
        r = functions.dstack((r0, r1))
        r = functions.reshape(r, (-1, 2))
        r = functions.expand_dims(r, axis=0)
        r = functions.expand_dims(r, axis=0)
        w = self.linear(r)
        w = functions.reshape(w, (-1, self.n_factor))

        # output
        ei = functions.expand_dims(self.embedId(x), axis=1)
        w = functions.expand_dims(w, axis=1)
        v = functions.matmul(w, ei, transb=True)
        v = functions.reshape(v, shape=(-1, 1))
        v = self.bias(v)
        if train:
            loss = functions.sigmoid_cross_entropy(v, t)
            chainer.reporter.report({'loss': loss}, self)
            return loss
        else:
            return functions.sigmoid(v)


def make_data(k, passes, n_cold_start_item, cold_start_click_count):
    data = pd.read_csv('./sample_data/data.csv')
    train_data, cold_start_data = ClickDataSet.train_cold_start_split(data, n_cold_start_item,
                                                                      cold_start_click_count)
    train_data, test_data = ClickDataSet.train_test_split(train_data.data, test_size=0.1)

    n_user = len(data['user'].unique())
    n_item = len(data['item'].unique())
    train = list(train_data.values(k=k, passes=passes))
    test = list(test_data.values(k=k, passes=passes))
    cs_test = list(cold_start_data.values(k=k, passes=passes))
    results = (n_user, n_item, train, test, cs_test)
    return results


def main():
    max_history = 30
    n_epoch = 20
    batch_size = 128
    n_factor = 30
    n_cold_start_item = 100
    cold_start_click_count = 10
    passes = 10

    n_user, n_item, train, test, cs_test = make_data(max_history, passes, n_cold_start_item, cold_start_click_count)

    model = LinearWeightAdaption(n_user, n_item + 1, n_factor=n_factor)
    train_iter = iterators.SerialIterator(train, batch_size, shuffle=True)
    test_iter = iterators.SerialIterator(test, batch_size, repeat=False, shuffle=False)
    cs_test_iter = iterators.SerialIterator(cs_test, batch_size, repeat=False, shuffle=False)
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.003))

    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (n_epoch, 'epoch'), out='result')
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.Evaluator(test_iter, model), name='test')
    trainer.extend(extensions.Evaluator(cs_test_iter, model), name='cs_test')
    trainer.extend(
        extensions.PrintReport(entries=['epoch', 'main/loss', 'test/main/loss', 'cs_test/main/loss', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())
    trainer.run()


if __name__ == '__main__':
    main()
