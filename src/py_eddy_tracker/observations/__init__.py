# -*- coding: utf-8 -*-
"""
"""

class Store:
    def __init__(self):
        self.store = None

    def from_netcdf(self):
        raise Exception()

    def from_zarr(self):
        raise Exception()

    def to_netcdf(self):
        raise Exception()

    def to_zarr(self):
        raise Exception()

    def __getitem__(self, item):
        if isinstance(item, slice):
            raise Exception()
        else:
            return self.store[item]

    def __len__(self):
        return self.store.size

    def __eq__(self, other):
        raise Exception()

    def __iter__(self):
        raise Exception()

    def reset(self):
        raise Exception()

    def fields(self):
        raise Exception()

    def sort(self):
        raise Exception()
    
    def argsort(self):
        raise Exception()


class ZarrMemoryStore(Store):
    pass


class RecordArrayStore(Store):
    pass