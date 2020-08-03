"""Data caching

TODO: Postinumero column to string (leading zeros missing)
TODO: Function to update all relevant caches

"""
import os
from time import sleep

from clientz.common.caching import (JSON, Pickle, lift, bind, Concat)
from clientz.common.utils import tuplemap

# Cache location in local file system
CACHE = os.path.abspath(".aptprices-cache")


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
            tuplemap(YearlyZip)(zip_codes),
            axis=0
        )

    return Create(ZipCodes())


def Quarterly():
    """All quarterly data

    """
    @bind
    def Create(zip_codes):
        return Concat(
            tuplemap(QuarterlyZip)(zip_codes),
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
