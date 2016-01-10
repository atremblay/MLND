import requests
import pandas as pd
from pandas import DataFrame
import datetime
import Quandl

class YahooAPI(object):
    """docstring for YahooAPI"""
    base_url = "http://ichart.yahoo.com/table.csv"

    @staticmethod
    def get_data(stock, start = None, end = None, interval='d'):
        params = dict(s=stock)
        format = "%Y-%m-%d"
        if start is not None:
            date = datetime.datetime.strptime(start, format)
            params['a'] = date.month - 1
            params['b'] = date.day
            params['c'] = date.year

        if end is not None:
            date = datetime.datetime.strptime(end, format)
            params['d'] = date.month - 1
            params['e'] = date.day
            params['f'] = date.year


        params['g'] = interval

        response = requests.get(YahooAPI.base_url, params=params)
        content = response.content.split('\n')
        headers = content[0].split(',')
        lines = [line.split(',') for line in content[1:-1]]  # last line empty
        import pdb
        pdb.set_trace()
        df = DataFrame(lines, columns=headers)
        df['Date'] = pd.to_datetime(df['Date'], format=format)
        df.set_index('Date', inplace = True)
        return df

class QuandlAPI(object):
    """Using the QuandlAPI to get the raw data. Without any transformation.
    It provides the corresponding S&P500 index"""
    # f_qzoEo5pn1_k353x29x
    @staticmethod
    def get_data(
        stock, start = None, end = None, bbuffer=0, fbuffer=0):
        """Summary

        Args:
            stock (string): Ticker (e.g. AAPL, GOOGL)
            start (string, optional): Start date. Format YYYY-MM-DD
            end (string, optional): End date. Format YYYY-MM-DD
            bbuffer (int, optional): Back buffer in days. No-op if start is none
            fbuffer (int, optional): Forward buffer in days. No-op if end is none

        Returns:
            Dataframe: A dataframe containing all the data provided by the
            Quandl API between the provided dates plus buffer if asked.
        """
        date_format = "%Y-%m-%d"

        if start is None:
            start = '1900-01-01'  # Earliest timestamp
        else:
            sdate = datetime.datetime.strptime(
                start,
                date_format)
            sdate += datetime.timedelta(days=-bbuffer)
            start = datetime.datetime.strftime(sdate, date_format)

        if end is None:
            end = datetime.datetime.strftime(
                datetime.date.today(),
                date_format)
        else:
            edate = datetime.datetime.strptime(
                end,
                date_format)
            edate += datetime.timedelta(days=fbuffer)
            end = datetime.datetime.strftime(edate, date_format)


        data = Quandl.get(
            'YAHOO/{}'.format(stock),
            trim_start = start,
            trim_end = end,
            collapse='daily',
            authtoken='f_qzoEo5pn1_k353x29x')

        return data