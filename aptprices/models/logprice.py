import itertools
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import pystan


logger = logging.getLogger(__file__)


def xsort(x):
    return sorted(list(set(x)))


def zip_to_zip_int(x):
    return int(x)


def zip_to_area1_int(x):
    return int(x[:2])


def zip_to_area2_int(x):
    return int(x[:3])


def log_population_density(population, area_m2):
    return -.1 * np.log(1.0e6 * population / area_m2)


def log_price(eur):
    return np.log(eur) - 6.


def coded_time(year):
    return (year - 2009.964) / 10.


class LogPriceModel(object):
    def __init__(self, model_src='/home/stastr/sandbox/ashi/ashi/model.stan',
                 post_mean=None, samples=None, zip_codes=None, **kwargs):
        self.model_src = model_src
        self.zip_codes = zip_codes
        self._samples = samples
        self._model = None
        self._post_mean = post_mean

        # hidden attributes to handle zip code mappings
        self._zip_ints = None
        self._area1_ints = None
        self._area2_ints = None
        training_data, stan_data = self.build_training_data(zip_codes=zip_codes,
                                                            **kwargs)
        self.training_data = training_data
        self.stan_data = stan_data

    def zip_to_zip_ind(self, zip_code):
        zip_int = zip_to_zip_int(zip_code)
        return self._zip_ints.index(zip_int) + 1

    def zip_to_area1_ind(self, zip_code):
        area1_int = zip_to_area1_int(zip_code)
        return self._area1_ints.index(area1_int) + 1

    def zip_to_area2_ind(self, zip_code):
        area2_int = zip_to_area2_int(zip_code)
        return self._area2_ints.index(area2_int) + 1

    def build_training_data(self, statfin_data=None, paavo_data=None,
                            zip_codes=None, key=('all', 'all')):
        # generic easily readable DataFrame for working with results
        data = statfin_data.loc[zip_codes, :]
        data = data.xs(key, axis=1)
        data.reset_index(inplace=True)
        data.rename(columns={'Postinumero': 'zip', 
                             'Vuosi': 'year',
                             'lkm_julk': 'count', 
                             'keskihinta': 'price'}, 
                    inplace=True)
        data = data[data['count'] >= 6]  # enforce at least 6 prices
        data['count'] = data['count'].apply(lambda x: int(x))
        data['log_price'] = log_price(data['price'])
        data['t'] = coded_time(pd.to_datetime(data['year'].values).year)
        data['d'] = (paavo_data.loc[data['zip']]).apply(
            lambda x: log_population_density(x['He_vakiy'], x['Pinta_ala']),
            axis=1
        ).values
        data.dropna(inplace=True)
        # must update zip related lists here as model parametrization
        # depends on training data
        self._zip_ints = sorted(set(map(zip_to_zip_int, data['zip'])))
        self._area1_ints = sorted(set(map(zip_to_area1_int, data['zip'])))
        self._area2_ints = sorted(set(map(zip_to_area2_int, data['zip'])))
        data['zip_ind'] = data['zip'].apply(self.zip_to_zip_ind)
        data['area1_ind'] = data['zip'].apply(self.zip_to_area1_ind)
        data['area2_ind'] = data['zip'].apply(self.zip_to_area2_ind)

        # Data for Stan
        # TODO change variable names to *_ind in the Stan code
        stan_data = {
            'N': len(data),
            'M': len(set(data['zip_ind'])),
            'M1': len(set(data['area1_ind'])),
            'M2': len(set(data['area2_ind'])),
        }
        for col in ['t', 'd', 'log_price', 'count',
                    'zip_ind', 'area1_ind', 'area2_ind']:
            stan_data[col] = list(data[col])
        return data, stan_data

    @property
    def model(self):
        if self._model is None:
            src = self.model_src
            if src.endswith('.stan'):
                model = pystan.StanModel(file=src)
            elif src.endswith('.pickle'):
                model = pickle.load(open(src), 'rb')
            else:
                raise(ValueError,
                      'Attribute `model_src` must be either Pickle or Stan')
            self._model = model
        return self._model

    @property
    def samples(self):
        if isinstance(self._samples, str):
            src = self._samples
            if src.endswith('.pickle'):
                with open(src, 'rb') as f:
                    samples = pickle.load(f)
                self._samples = samples
        return self._samples

    @property
    def post_mean(self):
        if self._post_mean is None:
            samples = self.samples
            post_mean = dict()
            post_mean['w'] = np.median(samples['w'], axis=0)
            post_mean['w1'] = np.median(samples['w1'], axis=0)
            post_mean['w2'] = np.median(samples['w2'], axis=0)
            post_mean['w_mean'] = np.median(samples['w_mean'], axis=0)
            self._post_mean = post_mean
        elif isinstance(self._post_mean, str):
            src = self._post_mean
            with open(src, 'rb') as f:
                post_mean = pickle.load(f)
            self._post_mean = post_mean
        return self._post_mean

    @staticmethod
    def design_matrix(training_data):
        ret = np.zeros((len(training_data), 6))
        ret[:, 0] = 1
        ret[:, 1] = training_data['t'].values
        ret[:, 2] = training_data['t'].values ** 2
        ret[:, 3] = training_data['d'].values
        ret[:, 4] = training_data['t'].values * training_data['d'].values
        ret[:, 5] = training_data['t'].values ** 2 * training_data['d'].values
        return ret

    def predict(self, x):
        """Predict from Stan-compatible model inputs"""
        # TODO think about other extreme cases needing extrapolation
        x = x.copy()
        log_price_pred = []
        X = self.design_matrix(x)
        for i, (row, val) in enumerate(x.iterrows()):
            X_row = X[i, :]
            area1_ind = int(val['area1_ind'])
            area2_ind = int(val['area2_ind'])
            # note below the -1 for Pythonic indexing
            if val['zip_ind'] is not None:
                zip_ind = int(val['zip_ind'])
                result = np.dot(X_row,
                                self.post_mean['w_mean'] +
                                self.post_mean['w1'][area1_ind-1, :] +
                                self.post_mean['w2'][area2_ind-1, :]) + \
                         np.dot(X_row[:3], self.post_mean['w'][zip_ind-1, :])
                log_price_pred.append(result)
            else:
                result = np.dot(X_row,
                                self.post_mean['w_mean'] +
                                self.post_mean['w1'][area1_ind - 1, :] +
                                self.post_mean['w2'][area2_ind - 1, :])
                log_price_pred.append(result)
        x['log_price_pred'] = log_price_pred
        x['price_pred'] = np.exp(x['log_price_pred'] + 6)  # hard-coded
        return x


def plot(statfin_data, paavo_data, zip_codes, metadata, log_model=None,
         fig=None, ages=None, houses=None, cmap='rainbow',
         ylim=None):
    nrows = int(np.ceil(np.sqrt(len(zip_codes))))
    fig = fig or plt.figure(figsize=(nrows * 4, nrows * 3))
    ages = ages or ('all',)
    houses = houses or ('apt.1', 'apt.2', 'apt.3', 'row.all')
    colors = getattr(mpl.cm, cmap)(np.linspace(0, 1,
                                               len(ages) * len(houses)))
    axs = [fig.add_subplot(nrows, nrows, i)
           for i in range(1, len(zip_codes))]
    # colors leged
    legend_elements = []
    labels = []
    for j, (age, house) in enumerate(itertools.product(ages, houses)):
        legend_elements.append(
            mpl.patches.Patch(
                facecolor=mpl.colors.rgb2hex(colors[j]),
                edgecolor='w', alpha=.5,
            ))
        labels.append('({}, {})'.format(age, house))
    # circle sizes another legend
    ax_helper = axs[0].twinx()
    ax_helper.set_yticks([])
    for count in [10, 50, 100, 250, 500, 1000]:
        ax_helper.scatter([], [], c='k', alpha=.5, s=count,
                          edgecolors='none',
                          label='{} sales'.format(count))
    # get min, max y-limits
    ylim = ylim or \
           (statfin_data.xs(('all', 'all', 'keskihinta'), axis=1).min() - 200.,
            statfin_data.xs(('all', 'all', 'keskihinta'), axis=1).max() + 2000)
    # plot to each Axis object
    for i, ax in enumerate(axs):
        # TODO more general checks
        # TODO sub optimal performance
        # try plot predictions
        xs_train = statfin_data.loc[zip_codes[i], :].xs(('all', 'all'),
                                                        axis=1)
        # build prediction inputs -- ugly way
        paavo_slice = \
            paavo_data[['He_vakiy', 'Pinta_ala']].loc[zip_codes[i]]
        index = xs_train.index.append(
            pd.DatetimeIndex(['2018', '2019']))
        d = [log_population_density(
            paavo_slice['He_vakiy'],
            paavo_slice['Pinta_ala'])] * len(index)
        t = coded_time(index.year)
        ax_twinx = ax.twinx()
        ax_twinx.axhline(np.exp(d[0] * -10.), c='k', lw=5, ls='--', alpha=.4)
        ax_twinx.set_ylim([0., 21000.])
        ax_twinx.set_ylabel('Pop / m2')
        if log_model is not None:
            try:
                zip_ind = [log_model.zip_to_zip_ind(zip_codes[i])] * \
                          len(index)
            except Exception as error:
                logger.info('{}: zip code unknown my model, '
                            'reverting to upper hierarchy for prediction. '
                            'Traceback: {}'.format(zip_codes[i], error))
            finally:
                zip_ind = [None] * len(index)
            area1_ind = [log_model.zip_to_area1_ind(zip_codes[i])] * \
                        len(index)
            area2_ind = [log_model.zip_to_area2_ind(zip_codes[i])] * \
                        len(index)
            inputs = {
                't': t,
                'zip_ind': zip_ind,
                'area1_ind': area1_ind,
                'area2_ind': area2_ind,
                'd': d
            }
            x = pd.DataFrame(
                index=index,
                data=inputs
            )
            y = log_model.predict(x)
            y['price_pred'].plot(ax=ax, c='k', lw=3)
        for j, (age, house) in enumerate(itertools.product(ages, houses)):
            xs = statfin_data.loc[zip_codes[i], :].xs((age, house), axis=1)
            xs.dropna(inplace=True)
            # plot observation data
            try:
                ax.scatter(
                    xs.index,
                    xs['keskihinta'],
                    c=[colors[j]] * len(xs),
                    s=xs['lkm_julk'],
                    edgecolors='none',
                    alpha=.8
                )
            except Exception as error:
                logger.error(
                    '{}: No data for ({}, {}): {}'.format(
                        zip_codes[i], age, house, error))
            finally:
                for tick in ax.get_xticklabels():
                    tick.set_rotation(45)
        ax.set_title(next(x[0] for x in metadata['Postinumero']
                          if x[1] == zip_codes[i]))
        ax.set_ylabel('Eur / m2')
        ax.set_ylim(ylim)
        ax.grid(True)
    axs[0].legend(
        handles=legend_elements,
        labels=labels,
        ncol=len(labels),
        loc=(0., 1.4),
        frameon=False,
        fontsize=16)
    ax_helper.legend(scatterpoints=1, frameon=False,
                     ncol=6, fontsize=12, loc=(0., 1.2))
    fig.tight_layout()
    return fig