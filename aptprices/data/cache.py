"""Data loading and caching

TODO: Postinumero column to string (leading zeros missing)
TODO: Function to update all relevant caches

"""
import json
import os
import pickle
from typing import Callable, Iterable

import attr
import pandas as pd

from clientz import paavo, statfin
from aptprices import utils
from aptprices.utils import compose


CACHE = os.path.abspath(".aptprices-cache")


@attr.s(frozen=True)
class Source():
    """Generic data source

    """

    # Download data from a remote location
    download = attr.ib()

    # Load data from disk cache
    load = attr.ib()

    # Download with saving
    update = attr.ib()


def Landfill(filepath: str, download, dump, load):

    # FIXME: api as input to download
    @utils.mkdir(filepath)
    def update():
        return dump(download(), filepath)

    return Source(
        download=download,
        load=lambda: load(filepath),
        update=update
    )


def Pickle(filepath, download):

    def dump(x, y):
        pickle.dump(x, open(y, "wb+"))
        return x

    def load(x):
        return pickle.load(open(x, "rb"))

    return Landfill(filepath, download, dump, load)


def JSON(filepath, download):

    def dump(x, y):
        json.dump(x, open(y, "w+"))
        return x

    def load(x):
        return json.load(open(x, "r"))

    return Landfill(filepath, download, dump, load)


def lift(f: Callable):
    """Lift a function

    Example
    -------

    ..code-block :: python

        triple = lift(lambda x: 3 * x)
        Tripled = triple(Source)

    """
    def lifted(*sources):

        return Source(
            download=lambda: f(
                *utils.tuplemap(lambda x: x.download())(sources)
            ),
            load=lambda: f(
                *utils.tuplemap(lambda x: x.load())(sources)
            ),
            update=lambda: f(
                *utils.tuplemap(lambda x: x.update())(sources)
            )
        )

    return lifted


def bind(f: Callable):
    """Bind a function which returns a data source

    """
    def bound(*sources):
        return Source(
            download=f(
                *utils.tuplemap(lambda x: x.download())(sources)
            ).download,
            load=f(
                *utils.tuplemap(lambda x: x.load())(sources)
            ).load,
            update=f(
                *utils.tuplemap(lambda x: x.update())(sources)
            ).update
        )

    return bound


def Concat(sources: Iterable, **kwargs):
    """Concatenate sources

    """

    def concat(*frames, **kwargs):
        return pd.concat(frames, **kwargs)

    return lift(concat)(*sources)


#
# Data sources for the end-user
#


def YearlyMeta(filepath=os.path.join(CACHE, "yearly-meta.json")):
    """Metadata for yearly StatFin apartment prices

    TODO: What is the difference between yearly and quarterly
          metadata?

    """
    api = statfin.API()

    def download():
        return api.apartment_prices_yearly.get()

    return JSON(filepath, download)


def QuarterlyMeta(filepath=os.path.join(CACHE, "quarterly-meta.json")):
    """Metadata for quarterly StatFin apartment prices

    """
    api = statfin.API()

    def download():
        return api.apartment_prices_quarterly.get()

    return JSON(filepath, download)


def YearlyZip(zip_code):
    """Yearly apartment prices for a zip code area

    """
    api = statfin.API()
    filepath = os.path.join(CACHE, zip_code, "yearly.p")

    def download(*args, **kwargs):
        return api.apartment_prices_yearly.post(
            query_code="Postinumero",
            query_selection_values=[zip_code]
        )

    return Pickle(filepath, download)


def QuarterlyZip(zip_code):
    """Quarterly apartment prices for a zip code area

    """
    api = statfin.API()
    filepath = os.path.join(CACHE, zip_code, "yearly.p")

    def download(*args, **kwargs):
        return api.apartment_prices_quarterly.post(
            query_code="Postinumero",
            query_selection_values=[zip_code]
        )

    return Pickle(filepath, download)


def Yearly():
    """All yearly data

    TODO: Use bind and metadata

    """
    return
