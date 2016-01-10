#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: alexis
# @Date:   2016-01-03 14:04:55
# @Last Modified by:   alexis
# @Last Modified time: 2016-01-10 08:51:30

import pandas as pd
import pdb


class SPIndex(object):
    """docstring for SPIndex"""
    def __init__(self, stock, stard_date, end_date):
        super(SPIndex, self).__init__()
        self.arg = arg



class MaxRescale(object):
    """Normalize the provided columns by their max values. Values will then
    be between 0 and 1."""

    def __init__(self, columns):
        super(MaxRescale, self).__init__()
        self.columns = columns

    def fit(self, X):
        # pdb.set_trace()
        self.max = X[self.columns].max()
        return self

    def transform(self, X):

        # pdb.set_trace()
        _X = X.copy()
        _X[self.columns] = _X[self.columns] / self.max
        return _X


class BollingerBand(object):
    """Add the rolling mean and standard deviation for the BollingerBand"""
    def __init__(self, window):
        super(BollingerBand, self).__init__()
        self.window = window

    def __call__(self, X):
        X['r_mean'] = pd.rolling_mean(X['Adjusted Close'], window=self.window)
        X['r_std'] = pd.rolling_std(X['Adjusted Close'], window=self.window)
        # Back fill the NaN we got by doing the rolling window
        X.fillna(method='bfill', inplace=True)
        return X


class Return(object):
    """Calculate the Return over a period"""
    def __init__(self, period):
        super(Return, self).__init__()
        self.period = period

    def transform(self, X):
        shifted = X['Adjusted Close'].shift(-self.period) #  Shift by number of days to predict
        return shifted/X['Adjusted Close'] - 1
