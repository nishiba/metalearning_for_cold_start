# coding: utf-8


class User(int):
    pass


class Item(int):
    pass


class ImplicitFeedback(object):
    def __init__(self, user: User, item: Item) -> None:
        self.user = user
        self.item = item

    def __str__(self):
        return 'user: %d, item: %d' % (self.user, self.item)
