

import numpy as np
from pandas import Series
import pandas as pd
from transform import Return, MaxRescale, BollingerBand
from sklearn.svm import SVR
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostRegressor
from stockAPI import QuandlAPI
import datetime
import pdb
import os

"""
For data, I need:
- Get initial raw data
- Get index S&P500 and probably an index that follows the sector
- Get the return over X periods which is the label to predict.
  This needs a forward buffer
- Augment data with bollinger band (rolling mean and std).
  This needs a back buffer
- Adjust Open, High and Low like Close was adjusted with Adjusted Close
- Normalize by the max value. Needs to be fitted so later data can be
  normalized the same way

Plan
- Get the buffer needed before and after
- Get full data
- Calculate return
- Get indexes
- Append indexes
- Adjust
- Rescale between the original real two dates, without the buffer
"""


class SVRPredictor(object):
    """docstring for SVRPredictor"""
    def __init__(self, tickers, periods):
        super(SVRPredictor, self).__init__()
        self.tickers = set(tickers)
        self.periods = periods
        self.stocks = dict()
        self.models = dict()
        self.pca = PCA()

    def fit(self, start_date, end_date):

        for ticker in self.tickers:
            self.stocks[ticker] = Stock(ticker)

        params_svr = [{
            'kernel': ['rbf', 'sigmoid', 'linear'],
            'C': [0.01, 0.1, 1, 10, 100],
            'epsilon': [0.0000001, 0.000001, 0.00001]
            }]
        params = ParameterGrid(params_svr)

        # Find the split for training and CV
        mid_date = train_test_split(start_date, end_date)
        for ticker, stock in self.stocks.items():

            X_train, y_train = stock.get_data(start_date, mid_date, fit=True)
            # X_train = self.pca.fit_transform(X_train.values)
            X_train = X_train.values
            # pdb.set_trace()
            X_cv, y_cv = stock.get_data(mid_date, end_date)
            # X_cv = self.pca.transform(X_cv.values)
            X_cv = X_cv.values

            lowest_mse = np.inf
            for i, param in enumerate(params):
                svr = SVR(**param)
                # ada = AdaBoostRegressor(svr)
                svr.fit(X_train, y_train.values)
                mse = mean_squared_error(
                    y_cv, svr.predict(X_cv))
                if mse <= lowest_mse:
                    self.models[ticker] = svr

        return self

    def transform(self, tickers, start_date, end_date):
        assert self.tickers.issubset(set(tickers))

        predictions = {}
        for ticker in tickers:
            # pdb.set_trace()
            stock = self.stocks[ticker]
            data, label = stock.get_data(start_date, end_date)
            # data = self.pca.transform(data.values)
            data = data.values
            model = self.models[ticker]
            y_pred = Series(
                model.predict(data),
                index=label.index)
            predictions[ticker] = [y_pred, label]

        return predictions


class AdaBoostPredictor(object):
    """docstring for AdaBoostPredictor"""
    def __init__(self, tickers, periods):
        super(AdaBoostPredictor, self).__init__()
        self.tickers = set(tickers)
        self.periods = periods
        self.stocks = dict()
        self.models = dict()
        self.pca = PCA()

    def fit(self, start_date, end_date):

        for ticker in self.tickers:
            self.stocks[ticker] = Stock(ticker)

        params_ada = [{
            'n_estimators': [25, 50, 100],
            'learning_rate': [0.01, 0.1, 1, 10],
            'loss': ['linear', 'square', 'exponential']
            }]

        params = ParameterGrid(params_ada)

        # Find the split for training and CV
        mid_date = train_test_split(start_date, end_date)
        for ticker, stock in self.stocks.items():

            X_train, y_train = stock.get_data(start_date, mid_date, fit=True)
            # X_train = self.pca.fit_transform(X_train.values)
            X_train = X_train.values
            # pdb.set_trace()
            X_cv, y_cv = stock.get_data(mid_date, end_date)
            # X_cv = self.pca.transform(X_cv.values)
            X_cv = X_cv.values

            lowest_mse = np.inf
            for i, param in enumerate(params):
                ada = AdaBoostRegressor(**param)
                ada.fit(X_train, y_train.values)
                mse = mean_squared_error(
                    y_cv, ada.predict(X_cv))
                if mse <= lowest_mse:
                    self.models[ticker] = ada

        return self

    def transform(self, tickers, start_date, end_date):
        assert self.tickers.issubset(set(tickers))

        predictions = {}
        for ticker in tickers:
            # pdb.set_trace()
            stock = self.stocks[ticker]
            data, label = stock.get_data(start_date, end_date)
            # data = self.pca.transform(data.values)
            data = data.values
            model = self.models[ticker]
            y_pred = Series(
                model.predict(data),
                index=label.index)
            predictions[ticker] = [y_pred, label]

        return predictions


class KNNPredictor(object):
    """docstring for KNNPredictor"""
    def __init__(self, tickers, periods):
        super(KNNPredictor, self).__init__()
        self.tickers = set(tickers)
        self.periods = periods
        self.stocks = dict()
        self.models = dict()

    def fit(self, start_date, end_date):

        for ticker in self.tickers:
            self.stocks[ticker] = Stock(ticker)

        params_svr = [{
            'n_neighbors': [2, 5, 10, 15]}]
        params = ParameterGrid(params_svr)

        # Find the split for training and CV
        mid_date = train_test_split(start_date, end_date)
        for ticker, stock in self.stocks.items():

            # pdb.set_trace()
            X_train, y_train = stock.get_data(start_date, mid_date, fit=True)
            X_cv, y_cv = stock.get_data(mid_date, end_date)

            lowest_mse = np.inf
            for i, param in enumerate(params):
                knn = KNeighborsRegressor(**param)
                # ada = AdaBoostRegressor(knn)
                knn.fit(X_train.values, y_train.values)
                mse = mean_squared_error(y_cv, knn.predict(X_cv.values))
                if mse <= lowest_mse:
                    self.models[ticker] = knn

        return self

    def transform(self, tickers, start_date, end_date):
        assert self.tickers.issubset(set(tickers))

        predictions = {}
        for ticker in tickers:
            # pdb.set_trace()
            stock = self.stocks[ticker]
            data, label = stock.get_data(start_date, end_date)
            model = self.models[ticker]
            y_pred = Series(
                model.predict(data.values),
                index=label.index)
            predictions[ticker] = [y_pred, label]

        return predictions


class Stock(object):
    """docstring for Stock"""
    def __init__(self, ticker, over=7):
        super(Stock, self).__init__()
        self.ticker = ticker
        self.over = over

        self.rescale = MaxRescale(
            ['Adjusted Close', 'Volume', 'Open', 'Low', 'High', 'SP Close'])
        self.bollinger10 = BollingerBand(window=10)
        self.bollinger5 = BollingerBand(window=5)

    def get_data(self, start_date, end_date, fit=False, cache=True):

        d_file_name = "d{}_{}_{}.csv".format(start_date, end_date, self.ticker)
        l_file_name = "l{}_{}_{}.csv".format(start_date, end_date, self.ticker)

        if cache and os.path.exists(d_file_name) and os.path.exists(l_file_name):
            data = pd.read_csv(d_file_name, index_col=0, parse_dates='Date')
            label = pd.read_csv(
                l_file_name,
                index_col=0,
                parse_dates=True,
                squeeze=True,
                header=None)
        else:
            data = QuandlAPI.get_data(self.ticker, start_date, end_date, 160, 15)
            label = Return(self.over).transform(data)

            #######################
            # Augment with index  #
            #######################
            sp500 = QuandlAPI.get_data(
                "INDEX_GSPC", start_date, end_date, 75, 15)
            data = data.join(sp500['Adj Close'], how='left')
            data.rename(columns={'Adj Close': 'SP Close'}, inplace=True)

            # Deal with outliers
            desc = data.describe()
            iqr = desc.ix['75%'] - desc.ix['25%']
            data = data[data <= desc.ix['50%'] + 1.5*iqr]
            data = data[data >= desc.ix['50%'] - 1.5*iqr]

            # Cache the raw data
            if cache:
                data.to_csv(d_file_name, index=True)
                label.to_csv(l_file_name, index=True)

        ##########
        # Adjust #
        ##########
        # TODO There may be a problem when subsequent call are made
        # The adjusted is always according to the last date
        # So two subsequent call with two different period will be adjusted
        # differently. Not sure if it matters or not
        ratio = data['Adjusted Close'] / data['Close']
        data['High'] = data['High']*ratio
        data['Low'] = data['Low']*ratio
        data['Open'] = data['Open']*ratio

        ##################
        # Rescale by max #
        ##################
        if fit:
            self.rescale.fit(data, start_date, end_date)
        data = self.rescale.transform(data)

        ######################
        # Average True Range #
        ######################
        # pdb.set_trace()
        data['ATR'] = ATR(data, 14)

        ##################
        # Bollinger band #
        ##################
        data = self.bollinger10(data)
        # data = self.bollinger5(data)

        #################
        # Rolling stats #
        #################
        data = high(22, data)  # Chandelier
        data = low(22, data)  # Chandelier

        # Ichimoku Clouds
        # http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ichimoku_cloud
        data['tenkansen9'] = sen(data, 9)
        data['kijunsen26'] = sen(data, 26)
        data['senkouA'] = (data['tenkansen9'] + data['kijunsen26']) / 2. # Not sure if redondant. Affine transformation of two other features
        data['senkouB52'] = sen(data, 52)
        data['chikou26'] = data['Adjusted Close'].shift(26)

        # Moving Average
        # http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_averages
        data['ema10'] = ema(data, 10)
        data['ema10-100'] = ema(data, 10) / ema(data, 100)
        data['ema25-100'] = ema(data, 25) / ema(data, 100)
        data['ema50-100'] = ema(data, 50) / ema(data, 100)
        data['kama'] = kama(data)
        data['ret_over2'] = return_over(data, 2)

        # RSI
        # http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:relative_strength_index_rsi
        data['rsi'] = rsi(data)

        # Price Volume Oscillator
        # http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:percentage_volume_oscillator_pvo
        ema_vol10 = ema(data, 10, 'Volume')
        ema_vol26 = ema(data, 26, 'Volume')
        data['pvo'] = (ema_vol10-ema_vol26)/ema_vol10

        # This is to make sure that both index are aligned.
        # If the start_date and end_date don't play well with the features
        # I'm making sure that it does not screw up the rest
        data.dropna(inplace=True)
        label.dropna(inplace=True)
        data, label = data[start_date:end_date], label[start_date:end_date]

        data_index, label_index = data.index, label.index
        max_start = max(data_index[0], label_index[0])
        min_end = min(data_index[-1], label_index[-1])
        data, label = data[max_start: min_end], label[max_start: min_end]

        del data['Close']
        # del data['Volume']
        # pdb.set_trace()
        return data, label


def train_test_split(start_date, end_date, numerator=4, denominator=5):
    date_format = "%Y-%m-%d"
    start = datetime.datetime.strptime(start_date, date_format)
    end = datetime.datetime.strptime(end_date, date_format)

    # TODO Not really pretty. Timedelta cannot be multiplied by a float
    # Have to find a better way than to provide numerator and denominator
    mid = start + (end - start) * numerator / denominator

    return datetime.datetime.strftime(mid, date_format)


def ATR(data, periods=14):
    TR1 = data['High'] - data['Low']
    TR2 = (data['High'] - data['Adjusted Close'].shift(1)).apply(np.abs)
    TR3 = (data['Low'] - data['Adjusted Close'].shift(1)).apply(np.abs)
    TR = pd.concat(
        [TR1.apply(np.abs), TR2.apply(np.abs), TR3.apply(np.abs)],
        axis=1).max(1)
    _atr = pd.rolling_mean(TR, window=periods)
    atr = (TR + _atr.shift(1)*(periods-1))/periods
    df = pd.concat([atr, _atr], axis=1)
    atr = df.apply(lambda x: x[0] if not np.isnan(x[0]) else x[1], axis=1)
    return atr


def high(window, data):
    data['r_high{}'.format(window)] = pd.rolling_max(
        data['Adjusted Close'], window=window)
    return data


def low(window, data):
    data['r_low{}'.format(window)] = pd.rolling_min(
        data['Adjusted Close'], window=window)
    return data


def sen(data, periods=9):
    t = pd.rolling_max(
        data['Adjusted Close'], window=periods)
    t += pd.rolling_min(
        data['Adjusted Close'], window=periods)
    t /= 2.
    return t


def ema(data, periods=10, column='Adjusted Close'):
    # pdb.set_trace()
    mult = 2./(periods+1)
    roll_mean = pd.rolling_mean(data[column], periods)
    _ema = roll_mean.shift(1)
    ema = (data[column] - _ema) * mult + _ema
    ema.ix[periods-1] = roll_mean.ix[periods-1]
    return ema


def return_over(data, over=2):
    return data['Adjusted Close'] - data['Adjusted Close'].shift(over)


def kama(data):
    # pdb.set_trace()
    er = efficiency_ratio(data)
    sc = (er * (2./(2+1) - 2./(30+1)) + 2./(30+1))**2
    rm = pd.rolling_mean(
            data['Adjusted Close'], window=10).shift(1)

    # KAMA = Prior KAMA + SC * (Price - Prior KAMA)
    return sc + rm * (data['Adjusted Close'] - rm)


def efficiency_ratio(data):
    change = return_over(data, 10).apply(np.abs)
    volatility = pd.rolling_sum(return_over(data, 1).apply(np.abs), window=10)
    return change/volatility

def rsi(data, periods=14):
    # pdb.set_trace()
    change = data['Adjusted Close'] - data['Adjusted Close'].shift(1)
    gains = change.apply(lambda x: 0 if x < 0 else x)
    losses = change.apply(lambda x: 0 if x >= 0 else -x)
    fag = pd.rolling_sum(gains, periods)
    fal = pd.rolling_sum(losses, periods)
    ag = (fag.shift(1) * (periods-1) + gains)/periods
    al = (fal.shift(1) * (periods-1) + losses)/periods
    rs = ag / al

    return 1 - 1/(1+rs)


# def buffer(start, end, before=15, after=15):
#     """Creating a buffer before and after the start and end date.
#     The rolling mean requires 10 open business days in the past and the return
#     is calculated over 7 business days.

#     Args:
#         start (string): Description
#         end (string): Description

#     Returns:
#         TYPE: Description
#     """
#     date_format = "%Y-%m-%d"
#     if start is None:
#         start = '1900-01-01'  # Default timestamp
#     else:
#         sdate = datetime.datetime.strptime(
#             start,
#             date_format)
#         sdate += datetime.timedelta(days=-before)
#         start = datetime.datetime.strftime(sdate, date_format)

#     if end is None:
#         end = datetime.datetime.strftime(
#             datetime.date.today(),
#             date_format)
#     else:
#         edate = datetime.datetime.strptime(
#             end,
#             date_format)
#         edate += datetime.timedelta(days=after)
#         end = datetime.datetime.strftime(edate, date_format)
#     return start, end

