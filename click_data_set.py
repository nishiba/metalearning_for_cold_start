# coding: utf-8
from collections import defaultdict
from typing import List, Tuple, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class ClickDataSet(object):
    def __init__(self, data: pd.DataFrame, test: pd.DataFrame = None) -> None:
        self.data = data
        self.test = test
        self.items_clicked = defaultdict(set)
        self.users_clicked = defaultdict(set)
        self.items = set(self.data.item.unique())
        self.users = set(self.data.user.unique())
        for u, i in zip(self.data.user.values, self.data.item.values):
            self.items_clicked[u].add(i)
            self.users_clicked[i].add(u)

    @staticmethod
    def train_test_split(data: pd.DataFrame, test_size: float) -> Tuple['ClickDataSet', 'ClickDataSet']:
        train_data, test_data = train_test_split(data, test_size=test_size, shuffle=True)
        return ClickDataSet(train_data), ClickDataSet(data, test_data)

    @staticmethod
    def train_cold_start_split(
            data: pd.DataFrame, test_size: int, item_click_count: int = 10) -> Tuple['ClickDataSet', 'ClickDataSet']:
        counts = data.groupby(by='item').count()
        cold_start_items = counts[counts['user'] > 2 * item_click_count].reset_index()['item'].values
        cold_start_items = set(shuffle(cold_start_items)[:test_size])

        train_data_list = []
        test_data_list = []
        data = shuffle(data)
        item_counts = defaultdict(int)
        for u, i in zip(data['user'].values, data['item'].values):
            item_counts[i] += 1
            if i in cold_start_items and item_counts[i] > item_click_count:
                test_data_list.append((u, i))
            else:
                train_data_list.append((u, i))

        train_data = pd.DataFrame(train_data_list, columns=['user', 'item'])
        test_data = pd.DataFrame(test_data_list, columns=['user', 'item'])
        return ClickDataSet(train_data), ClickDataSet(data, test_data)

    def values(self, k: int = 100, passes: int = 1) -> Iterable[Tuple[np.ndarray, np.ndarray, int, List[int]]]:
        # generate data
        if self.test is None:
            return self._train_values(k, passes)
        else:
            return self._test_values(k)

    def _train_values(self, k, passes):
        for u in np.random.choice(list(self.users), len(self.users) * passes):
            clicked_items = shuffle(list(self.items_clicked[u]))[:2 * k]
            not_clicked_items = shuffle(list(self.items - self.items_clicked[u]))[:2 * k]
            n = min(len(clicked_items), len(not_clicked_items)) // 2
            positive_feedbacks = self._pad(clicked_items[:n], k)
            negative_feedbacks = self._pad(not_clicked_items[:n], k)
            for i in clicked_items[n:]:
                yield (positive_feedbacks, negative_feedbacks, i, [1])
            for i in not_clicked_items[n:]:
                yield (positive_feedbacks, negative_feedbacks, i, [0])

    def _test_values(self, k):
        if self.test is None:
            return
        for i, u in shuffle(list(zip(self.test.item.values, self.test.user.values))):
            yield self._choice_history(u, i, [1], k)
            # choice user who did not item i
            nu = np.random.choice(list(self.users - self.users_clicked[i]))
            yield self._choice_history(nu, i, [0], k)

    def _choice_history(self, u: int, i: int, label: List[int], k: int):
        p = shuffle(list(self.items_clicked[u] - {i}))[:k]
        n = shuffle(list(self.items - self.items_clicked[u]))[:k]
        m = min(len(p), len(n))
        return self._pad(p[:m], k), self._pad(n[:m], k), i, label

    def _pad(self, xs, k):
        if len(xs) >= k:
            return np.array(xs[:k])
        return np.pad(xs, (0, k - len(xs)), 'constant', constant_values=len(self.items))

    def __train_values(self, k, passes):
        # choice item uniformly
        for i in np.random.choice(self.data.item.unique(), self.data.shape[0] * passes):
            # choice user who clicked item i
            pu = np.random.choice(list(self.users_clicked[i]))
            yield self._choice_history(pu, i, [1], k)
            # choice user who did not item i
            nu = np.random.choice(list(self.users - self.users_clicked[i]))
            yield self._choice_history(nu, i, [0], k)


def main():
    train_dataset = pd.read_csv('./sample_data/data.csv').head(10000)
    train, cs_test = ClickDataSet.train_cold_start_split(train_dataset, 20, 10)
    list(train.values())


if __name__ == '__main__':
    main()
