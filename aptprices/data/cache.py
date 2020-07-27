"""Data loading and caching

TODO: Use clientz to download data. Requires loop since not all data can be
      retrieved in one request. Cache.

"""
import json
import os
import pickle

import attr
from clientz import paavo, statfin


CACHE = os.path.abspath(".aptprices-cache")


@attr.s(frozen=True)
class Source():

    download = attr.ib()
    load = attr.ib()
    save = attr.ib()

    def update(self, *args, **kwargs):
        return self.save(self.download(*args, **kwargs))


def PickleSource(filepath, download):

    def save(data):
        with open(filepath, "wb+") as f:
            pickle.dump(data, f)
        return data

    def load():
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return data

    return Source(
        download=download,
        load=load,
        save=save
    )


def JSONSource(filepath, download):

    def save(data):
        with open(filepath, "w+") as f:
            json.dump(data, f)
        return data

    def load():
        with open(filepath, "r") as f:
            data = json.load(f)
        return data

    return Source(
        download=download,
        load=load,
        save=save
    )


def YearlyMeta(filepath=os.path.join(CACHE, "yearly-meta.json")):
    """Metadata for yearly StatFin apartment prices

    """
    api = statfin.API()

    def download():
        return api.apartment_prices_yearly.get()

    return JSONSource(filepath, download)


def QuarterlyMeta(filepath=os.path.join(CACHE, "quarterly-meta.json")):
    """Metadata for quarterly StatFin apartment prices

    """
    api = statfin.API()

    def download():
        return api.apartment_prices_quarterly.get()

    return JSONSource(filepath, download)


def Yearly(filepath=os.path.join(CACHE, "yearly.p")):
    """Yearly StatFin apartment prices

    TODO: Download all

    """
    api = statfin.API()

    def download(*args, **kwargs):
        return api.apartment_prices_yearly.post(*args, **kwargs)

    return PickleSource(filepath, download)


def Quarterly(filepath=os.path.join(CACHE, "quarterly.p")):
    """Quarterly StatFin apartment prices

    """
    api = statfin.API()

    def download(*args, **kwargs):
        return api.apartment_prices_quarterly.post(*args, **kwargs)

    return PickleSource(filepath, download)
