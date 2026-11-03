"""Bounded LRU cache keyed by the canonical resolved input + profile hashes + engine version (see api._cached_eval)."""

from collections import OrderedDict


class LRU:
    def __init__(self, size):
        self.size = size
        self.d = OrderedDict()

    def get(self, k):
        v = self.d.get(k)
        if v is not None:
            self.d.move_to_end(k)
        return v

    def put(self, k, v):
        self.d[k] = v
        self.d.move_to_end(k)
        while len(self.d) > self.size:
            self.d.popitem(last=False)
