# coding: utf-8
from collections import defaultdict
from typing import List
from sklearn.utils import shuffle

import numpy as np

from data import ImplicitFeedback


class FeedbackSet(object):
    def __init__(self, feedbacks: List[ImplicitFeedback], max_item=100, passes=10) -> None:
        self.users = list(set([fb.user for fb in feedbacks]))
        self.items = list(set([fb.item for fb in feedbacks]))
        self.n_user = len(self.users)
        self.n_item = len(self.items)
        self.max_item = max_item
        self.passes = passes

        self.user_id_index_map = dict(zip(self.users, range(self.n_user)))
        self.item_id_index_map = dict(zip(self.items, range(self.n_item)))

        self.data = defaultdict(list)
        for fb in feedbacks:
            self.data[self.user_id_index_map[fb.user]].append(self.item_id_index_map[fb.item])
        for u in self.data.keys():
            self.data[u] = list(set(self.data[u]))

    def values(self):
        for p in range(self.passes):
            for u in range(self.n_user):
                clicked = self.data[u]
                not_clicked = list(set(range(self.n_item)) - set(clicked))
                x1 = self._choice(clicked, self.max_item)
                x0 = self._choice(not_clicked, self.max_item * 2)
                x0, t0 = self._split(x0)
                x1, t1 = self._split(x1)
                t1 = [t for t in t1 if t != self.n_item]
                t0 = [t for t in t0 if t != self.n_item][:len(t1)]
                for t in t1:
                    yield (x0, x1, t, [1])
                for t in t0:
                    yield (x0, x1, t, [0])

    def make_values(self, feedbacks: List[ImplicitFeedback]):
        for fb in feedbacks:
            user = self.user_id_index_map.get(fb.user, None)
            item = self.item_id_index_map.get(fb.item, None)
            if user is None or item is None:
                continue
            clicked = np.array(self.data[user])
            not_clicked = np.array(list(set(range(self.n_item)) - set(clicked) - {item}))
            x1 = self._choice(clicked, self.max_item)
            x0 = self._choice(not_clicked, self.max_item)
            yield (x0, x1, item, [1])

    def _split(self, xs):
        n = len(xs) // 2
        x0 = np.random.choice(xs, n)
        x1 = list(set(xs) - set(x0))
        return x0, x1

    def _choice(self, xs, n):
        if len(xs) < n:
            xs = np.pad(xs, (0, n - len(xs)), 'constant', constant_values=self.n_item)
        return shuffle(xs)[:n]
