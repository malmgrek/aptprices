import codecs
import json
import logging
import requests

import attr
import numpy as np
import pandas as pd

from aptprices import utils
from aptprices.utils import compose, pipe, identity


#
# StatFin: http://pxnet2.stat.fi/PXWeb/api/v1/fi/StatFin/
#
# Talotyyppi | Explanation
# ---------------------------
#          1 | One room
#          2 | Two rooms
#          3 | Three rooms
#          4 | All apartments
#          5 | All rowhouses
#          6 | All
#
# Rakennusvuosi | Explanation
# ---------------------------
#             0 |  All
#             1 | -1950
#             2 |  1950
#             3 |  1960
#             4 |  1970
#             5 |  1980
#             6 |  1990
#             7 |  2000
#             8 |  2010
#             9 |  2020
#



# Paavo: http://pxnet2.stat.fi/PXWeb/api/v1/fi/
#        Postinumeroalueittainen_avoin_tieto/
#


def add_params(url, **params):
    return url + (
        "" if not params else "&".join(
            "{0}={1}".format(k, v) for k, v in params.items() if v is not None
        )
    )


@attr.s(frozen=True)
class Endpoint():
    """Use to create endpoint nodes for REST API clients

    """

    url = attr.ib()
    session = attr.ib(factory=requests.Session)
    defaults = attr.ib(default={})
    headers = attr.ib(default={})
    tf_get_params = attr.ib(default=identity)
    tf_get_response = attr.ib(default=identity)
    tf_post_params = attr.ib(default=identity)
    tf_post_response = attr.ib(default=identity)
    timeout = attr.ib(default=10.05)

    def post(self, resource="", **params):
        # NOTE: In 2020 only POST was supported for retrieving table data
        url = self.url + resource
        logging.debug("POST {0}".format(url))
        r = self.session.post(
            url=url,
            json=self.tf_post_params(**params),
            headers=self.headers,
            timeout=self.timeout
        )
        r.raise_for_status()
        return self.tf_post_response(r)

    def get(self, resource="", **params):
        url = add_params(
            self.url + resource,
            **{**self.defaults, **self.tf_get_params(params)}
        )
        logging.debug("GET {0}".format(url))
        r = self.session.get(url, headers=self.headers, timeout=self.timeout)
        r.raise_for_status()
        return self.tf_get_response(r)


def tf_get_response(res):

    raw = res.json()

    @attr.s(frozen=True)
    class Variable():
        text = attr.ib()
        values = attr.ib()
        valueTexts = attr.ib()

    @attr.s(frozen=True)
    class Metadata():
        title = attr.ib()
        codes = attr.ib()

    metadata = Metadata(
        title=raw["title"],
        codes=[v["code"] for v in raw["variables"]]
    )
    for v in raw["variables"]:
        object.__setattr__(
            metadata,
            v["code"],
            Variable(
                text=v["text"],
                values=v["values"],
                valueTexts=v["valueTexts"]
            )
        )

    return metadata


def StatFin():

    session = requests.Session()

    def ApartmentPricesEndpoint(url, tf_get_response, tf_post_params,
                                tf_post_response):
        return Endpoint(
            url="http://pxnet2.stat.fi/PXWeb/api/v1/fi/StatFin/" + url,
            tf_get_response=tf_get_response,
            tf_post_params=tf_post_params,
            tf_post_response=tf_post_response,
            session=session
        )

    def tf_post_params(
        query_code="Postinumero",
        query_selection_filter="item",
        query_selection_values=["00920"],
        response_format="json",
        **kwargs
    ):
        return utils.update_dict(
            {
                "query": [{
                    "code": query_code,
                    "selection": {
                        "filter": query_selection_filter,
                        "values": query_selection_values
                    }
                }],
                "response": {"format": response_format}
            },
            kwargs
        )

    tf_post_response = compose(
        lambda x: x.apply(
            lambda s: (
                pd.to_datetime(s) if s.name.startswith("Vuosi")
                else pd.to_numeric(s, errors="coerce")
            )
        ),
        lambda x: pd.DataFrame(
            columns=[y["code"] for y in x["columns"]],
            data=[y["key"] + y["values"] for y in x["data"]]
        ),
        lambda res: json.loads(
            codecs.decode(res.content, "utf-8-sig")
        )
    )


    @attr.s(frozen=True)
    class Client():

        apartment_prices_quarterly = ApartmentPricesEndpoint(
            url="asu/ashi/nj/statfin_ashi_pxt_112p.px?",
            tf_get_response=tf_get_response,
            tf_post_params=tf_post_params,
            tf_post_response=tf_post_response
        )

        apartment_prices_yearly = ApartmentPricesEndpoint(
            url="asu/ashi/vv/statfin_ashi_pxt_112q.px?",
            tf_get_response=tf_get_response,
            tf_post_params=tf_post_params,
            tf_post_response=tf_post_response
        )

    return Client()


def Paavo():

    session = requests.Session()

    def PaavoEndpoint(url, tf_get_response, tf_post_params, tf_post_response):
        return Endpoint(
            url=(
                "http://pxnet2.stat.fi/PXWeb/api/v1/fi/"
                "Postinumeroalueittainen_avoin_tieto/" +
                url
            ),
            tf_get_response=tf_get_response,
            tf_post_params=tf_post_params,
            tf_post_response=tf_post_response,
            session=session
        )

    def tf_post_params(
        query_code="Postinumeroalue",
        query_selection_filter="item",
        query_selection_values=["00920"],
        response_format="json",
        **kwargs
    ):
        return utils.update_dict(
            {
                "query": [{
                    "code": query_code,
                    "selection": {
                        "filter": query_selection_filter,
                        "values": query_selection_values
                    }
                }],
                "response": {"format": response_format}
            },
            kwargs
        )

    tf_post_response = compose(
        lambda x: x.apply(pd.to_numeric, errors="coerce"),
        lambda x: x.set_index("Tiedot"),
        lambda x: pd.DataFrame(
            columns=[y["code"] for y in x["columns"]],
            data=[y["key"] + y["values"] for y in x["data"]]
        ),
        lambda res: json.loads(
            codecs.decode(res.content, "utf-8-sig")
        )
    )

    attr.s(frozen=True)
    class Client():

        # NOTE: There are variables 1, 2, ..., 9
        #       For more information, see http://www.stat.fi/org/avoindata/
        #       paikkatietoaineistot/paavo_en.html
        #
        all_variables_2018 = PaavoEndpoint(
            url="2018/paavo_9_koko_2018.px",
            tf_get_response=tf_get_response,
            tf_post_params=tf_post_params,
            tf_post_response=tf_post_response
        )

    return Client()
