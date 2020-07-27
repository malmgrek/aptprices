import codecs
import functools
import itertools
import json
import logging
import requests

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import sleep


logger = logging.getLogger(__file__)


#
# Function manipulation
#


def curryish(f):

    def g(*args, **kwargs):
        return functools.partial(f, *args, **kwargs)

    return g


def compose2(f, g):

    def h(*args, **kwargs):
        return f(g(*args, **kwargs))

    return h


def lift(func):
    # Could add func's *args, **kwargs here
    return lambda f: compose2(func, f)


def lift2(func):
    return (
        lambda f, g: (
            lambda *args, **kwargs: func(
                *[f(*args, **kwargs), g(*args, **kwargs)]
            )
        )
    )


def rlift(func):
    return lambda f: compose2(f, func)


def compose(*funcs):
    return functools.partial(functools.reduce, compose2)(funcs)


def pipe(arg, *funcs):
    return compose(*funcs[::-1])(arg)


listmap = curryish(compose(list, map))
tuplemap = curryish(compose(tuple, map))
listfilter = curryish(compose(list, filter))
tuplefilter = curryish(compose(tuple, filter))


def flatten(x):
    """Flatten a list of lists once

    """
    return functools.reduce(lambda cum, this: cum + this, x, [])


def update_dict(x, y):
    return {**x, **y}


MAP_AGE = {
    '0': 'all',
    '1': '-1950',
    '2': '1950',
    '3': '1960',
    '4': '1970',
    '5': '1980',
    '6': '1990',
    '7': '2000',
    '8': '2010'
}
MAP_HOUSE = {
    '1': 'apt.1',
    '2': 'apt.2',
    '3': 'apt.3',
    '4': 'apt.all',
    '5': 'row.all',
    '6': 'all'
}
STATFIN_QUARTERLY_PX = 'asu/ashi/nj/statfin_ashi_pxt_112p.px'
STATFIN_YEARLY_PX = 'asu/ashi/vv/statfin_ashi_pxt_112q.px'
PAAVO_2018_PX = '2018/paavo_9_koko_2018.px'


class HousePriceData(object):
    STATFIN_URL = 'http://pxnet2.stat.fi/PXWeb/api/v1/fi/StatFin/'
    PAAVO_URL = 'http://pxnet2.stat.fi/PXWeb/api/v1/fi/' \
                'Postinumeroalueittainen_avoin_tieto/'

    def __init__(self, statfin_px=None, paavo_px=None, statfin_time_code=None):
        self.statfin_px = statfin_px or STATFIN_YEARLY_PX
        self.paavo_px = paavo_px or PAAVO_2018_PX
        if statfin_time_code == 'Vuosi':
            self.statfin_px = STATFIN_YEARLY_PX
        elif statfin_time_code == 'Vuosineljännes':
            self.statfin_px = STATFIN_QUARTERLY_PX
        self.statfin_time_code = statfin_time_code
        self._metadata = None
        self._statfin_data = None

    @property
    def metadata(self):
        if self._metadata is None:
            metadata = {}
            # StatFin metadata
            res = requests.get(url=self.STATFIN_URL + self.statfin_px)
            raw = res.json()
            for x in raw['variables']:
                code = x['code']
                if code == 'Talotyyppi':
                    metadata[code] = [
                        (value_text, MAP_HOUSE[value])
                        for value_text, value in zip(x['valueTexts'],
                                                     x['values'])
                    ]
                elif code == 'Rakennusvuosi':
                    metadata[code] = [
                        (value_text, MAP_AGE[value])
                        for value_text, value in zip(x['valueTexts'],
                                                     x['values'])
                    ]
                else:
                    metadata[code] = [
                        (value_text, value)
                        for value_text, value in zip(x['valueTexts'],
                                                     x['values'])
                    ]
            # Paavo metadata; Duplicate zip code data ignored
            res = requests.get(url=self.PAAVO_URL + self.paavo_px)
            raw = res.json()
            metadata = {
                **metadata,
                **{x['code']: [(value_text, value)
                              for value_text, value in zip(x['valueTexts'],
                                                           x['values'])]
                   for x in raw['variables'] if x['code'] == 'Tiedot'}
            }
            self._metadata = metadata
        return self._metadata

    def _download_statfin_data(self, data, zip_code):
        res = requests.post(
            url=self.STATFIN_URL + self.statfin_px,
            json={
                'query': [{
                    'code': 'Postinumero',
                    'selection': {
                        'filter': 'item',
                        'values': [zip_code]
                    }
                }],
                'response': {'format': 'json'}
            }
        )
        raw = json.loads(codecs.decode(res.content, 'utf-8-sig'))
        # Any item in raw['data'] is of form
        # x = {'key': ['Postinumero', 'Talotyyppi',
        #              'Rakennusvuosi', 'Vuosineljännes],
        #      'values': ['keskihinta', 'lkm_julk']}
        for col in set(c[:-1] for c in data.columns):
            subset = list(
                filter(lambda x: (MAP_AGE[x['key'][2]],
                                  MAP_HOUSE[x['key'][1]]) == col, raw['data'])
            )
            if len(subset):
                locs = list(
                    map(lambda x: (x['key'][0], pd.to_datetime(x['key'][3])),
                        subset)
                )
                data[col + ('keskihinta',)].loc[locs] = \
                    pd.to_numeric([x['values'][0] for x in subset],
                                  errors='coerce')
                data[col + ('lkm_julk',)].loc[locs] = \
                    pd.to_numeric([x['values'][1] for x in subset],
                                  errors='coerce')
        return data

    def download_statfin_data(self, zip_codes=None):
        """Populate a Multi-index data frame with StatFin data"""
        zip_codes = zip_codes \
                       or [t[1] for t in self.metadata['Postinumero']]
        bar = get_progress_bar(len(zip_codes))
        bar.start()
        index = pd.MultiIndex.from_product(
            [list(map(lambda x: x[1], self.metadata['Postinumero'])),
             list(map(lambda x: pd.to_datetime(x[1]),
                      self.metadata[self.statfin_time_code]))],
            names=['Postinumero', self.statfin_time_code]
        )
        columns = pd.MultiIndex.from_product(
            [list(map(lambda x: x[1], self.metadata['Rakennusvuosi'])),
             list(map(lambda x: x[1], self.metadata['Talotyyppi'])),
             ['lkm_julk', 'keskihinta']],
            names=['Rakennusvuosi', 'Talotyyppi', 'Havainnot']
        )
        data = pd.DataFrame(index=index, columns=columns, dtype=float)
        for i, zip_code in enumerate(zip_codes):
            data = self._download_statfin_data(data, zip_code)
            bar.update(i + 1)
            sleep(.1)
        return data

    def download_paavo_data(self, zip_codes=None):
        zip_codes = zip_codes \
                       or [t[1] for t in self.metadata['Postinumero']]
        index = pd.Index(list(map(lambda x: x[1], self.metadata['Postinumero'])))
        columns = list(map(lambda x: x[1], self.metadata['Tiedot']))
        data = pd.DataFrame(index=index, columns=columns)
        res = requests.post(
            url=self.PAAVO_URL + self.paavo_px,
            json={
                'query': [{
                    'code': 'Postinumeroalue',
                    'selection': {
                        'filter': 'item',
                        'values': zip_codes
                    }
                }],
                'response': {'format': 'json'}
            }
        )
        raw = json.loads(codecs.decode(res.content, 'utf-8-sig'))
        # raw['comments'] ignored; might contain lots of important desrciptions
        # Any item in raw['data'] is of form
        # x = {'key': ['Postinumero', 'koodi'],
        #      'values': ['numeroarvo']}
        bar = get_progress_bar(len(raw.get('data')))
        bar.start()
        for i, x in enumerate(raw.get('data') or []):
            try:
                data[x['key'][1]].loc[x['key'][0]] = \
                    pd.to_numeric(x['values'][0])
            except Exception as error:
                logger.error('{}: Bad value in {}: {}'.format(x['key'][0],
                                                              x['key'][1],
                                                              error))
            finally:
                bar.update(i + 1)
        return data
