#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: alexis
# @Date:   2016-01-03 14:10:15
# @Last Modified by:   alexis
# @Last Modified time: 2016-01-03 14:24:37

import datetime


class QuandlAPI(object):
    """Using the QuandlAPI to get the raw data. Without any transformation"""
    def __init__(self, stock, start = None, end = None, interval='d'):
        super(QuandlAPI, self).__init__()
        self.date_format = "%Y-%m-%d"
        self.stock = stock
        self.start = start
        if start is not None:
            self.start = start
        else:
            self.start = '1900-01-01'  # Earliest timestamp

        if end is not None:
            self.end = end
        else:
            self.end = datetime.datetime.strftime(
                datetime.date.today(),
                self.date_format)

        if interval == 'd':
            self.interval = 'daily'
        elif interval == 'w':
            self.interval = 'weekly'
        else:
            raise RuntimeError("Unsupported interval. Need to be d or w.")

        data = Quandl.get(
            'YAHOO/{}'.format(stock),
            trim_start = self.start,
            trim_end = self.end,
            collapse='daily')

        sp500 = Quandl.get(
            'YAHOO/INDEX_GSPC',
            trim_start = self.start,
            trim_end = self.end,
            collapse='daily')
        data = data.join(sp500['Adj Close'], how='left')
        data.rename(columns={'Adj Close': 'SP Close'}, inplace=True)
        self.data = data
