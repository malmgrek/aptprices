"""Data loading and caching

TODO: Postinumero column to string (leading zeros missing)
TODO: Function to update all relevant caches

"""
import json
import logging
import os
import pickle
from time import sleep
from typing import Callable, Iterable

import attr
import pandas as pd

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


def Landfill(
        filepath: str,
        download: Callable,
        dump: Callable,
        load: Callable
):
    """Dump load source

    """
    @utils.mkdir(filepath)
    def update(api):
        return dump(download(api), filepath)

    return Source(
        download=download,
        load=lambda: load(filepath),
        update=update
    )


def Pickle(filepath: str, download: Callable):
    """Landfill of pickle

    """
    def dump(data, path):
        pickle.dump(data, open(path, "wb+"))
        return data

    def load(path):
        return pickle.load(open(path, "rb"))

    return Landfill(filepath, download, dump, load)


def JSON(filepath: str, download: Callable):
    """Landfill of json

    """
    def dump(data, path):
        json.dump(data, open(path, "w+"))
        return data

    def load(path):
        return json.load(open(path, "r"))

    return Landfill(filepath, download, dump, load)


def lift(func: Callable):
    """Lift a function

    Example
    -------

    ..code-block :: python

        triple = lift(lambda x: 3 * x)
        Tripled = triple(Source)

    """
    def lifted(*sources: Source):

        return Source(
            download=lambda api: func(
                *utils.tuplemap(lambda x: x.download(api))(sources)
            ),
            load=lambda: func(
                *utils.tuplemap(lambda x: x.load())(sources)
            ),
            update=lambda api: func(
                *utils.tuplemap(lambda x: x.update(api))(sources)
            )
        )

    return lifted


def bind(func: Callable):
    """Bind a function which returns a data source

    """
    def bound(*sources: Source):
        return Source(
            download=lambda api: func(
                *utils.tuplemap(lambda x: x.download(api))(sources)
            ).download(api),
            load=lambda: func(
                *utils.tuplemap(lambda x: x.load())(sources)
            ).load(),
            update=lambda api: func(
                *utils.tuplemap(lambda x: x.update(api))(sources)
            ).update(api)
        )

    return bound


def Concat(sources: Iterable[Source], **kwargs):
    """Concatenate an iterable of sources

    """

    def concat(*frames: pd.DataFrame, **kwargs):
        return pd.concat(frames, **kwargs)

    return lift(concat)(*sources)


# =============================
# Data sources for the end-user
# =============================


def YearlyMeta(filepath=os.path.join(CACHE, "yearly-meta.json")):
    """Metadata for yearly StatFin apartment prices

    TODO: What is the difference between yearly and quarterly
          metadata?

    """
    def download(api):
        return api.apartment_prices_yearly.get()

    return JSON(filepath, download)


def QuarterlyMeta(filepath=os.path.join(CACHE, "quarterly-meta.json")):
    """Metadata for quarterly StatFin apartment prices

    """
    def download(api):
        return api.apartment_prices_quarterly.get()

    return JSON(filepath, download)


def YearlyZip(zip_code):
    """Yearly apartment prices for a zip code area

    """
    filepath = os.path.join(CACHE, zip_code, "yearly.p")

    def download(api):
        sleep(0.1)
        return api.apartment_prices_yearly.post(
            query_code="Postinumero",
            query_selection_values=[zip_code]
        )

    return Pickle(filepath, download)


def QuarterlyZip(zip_code):
    """Quarterly apartment prices for a zip code area

    """
    filepath = os.path.join(CACHE, zip_code, "yearly.p")

    def download(api):
        sleep(0.1)
        return api.apartment_prices_quarterly.post(
            query_code="Postinumero",
            query_selection_values=[zip_code]
        )

    return Pickle(filepath, download)


def ConstructionYear():

    @lift
    def construction_year(meta):
        return dict(zip(
            meta["Rakennusvuosi"]["values"],
            meta["Rakennusvuosi"]["valueTexts"]
        ))

    return construction_year(YearlyMeta())


def HouseTypes():

    @lift
    def house_types(meta):
        return dict(zip(
            meta["Talotyyppi"]["values"],
            meta["Talotyyppi"]["valueTexts"]
        ))

    return house_types(YearlyMeta())


def ZipCodes():

    @lift
    def zip_codes(meta):
        return dict(zip(
            meta["Postinumero"]["values"],
            meta["Postinumero"]["valueTexts"]
        ))

    return zip_codes(YearlyMeta())


def Yearly():
    """All yearly data

    """
    @bind
    def Create(zip_codes):
        return Concat(
            utils.tuplemap(YearlyZip)(zip_codes),
            axis=0
        )

    return Create(ZipCodes())


def Quarterly():
    """All quarterly data

    """
    @bind
    def Create(zip_codes):
        return Concat(
            utils.tuplemap(QuarterlyZip)(zip_codes),
            axis=0
        )

    return Create(ZipCodes())


# ===========
# Convenience
# ===========


def update_caches():
    """Update relevant caches

    """
    # TODO
    return
