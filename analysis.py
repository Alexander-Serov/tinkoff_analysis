"""
Support functions for the main interface.

Candle limits:
// Получение свечей(ордеров)
// Внимание! Действуют ограничения на промежуток и доступный размер свечей за него
// Интервал свечи и допустимый промежуток запроса:
// - 1min [1 minute, 1 day]
// - 2min [2 minutes, 1 day]
// - 3min [3 minutes, 1 day]
// - 5min [5 minutes, 1 day]
// - 10min [10 minutes, 1 day]
// - 15min [15 minutes, 1 day]
// - 30min [30 minutes, 1 day]
// - hour [1 hour, 7 days]
// - day [1 day, 1 year]
// - week [7 days, 2 years]
// - month [1 month, 10 years]
"""

from openapi_client import openapi
import datetime as dt
from pytz import timezone
import os
from openapi_client.openapi_streaming import run_stream_consumer
from openapi_client.openapi_streaming import print_event
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import lru_cache
import math
import pytz
import warnings
import copy
import numpy as np

# Load the token
token_file = './token_sandbox'
with open(token_file, 'r') as f:
    token = f.read().strip()

LOCAL_TIMEZONE = dt.datetime.now(dt.timezone.utc).astimezone().tzinfo
MOSCOW_TIMEZONE = pytz.timezone('Europe/Moscow')
EARLIEST_DATE = dt.datetime.fromisoformat('2013-01-01').replace(tzinfo=MOSCOW_TIMEZONE)
# TODO: In the recieved results, Moscow timezone sometimes appears as +2:30 and sometimes as +3:00. To fix

# Initialize the openapi
client = openapi.sandbox_api_client(token)
market = client.market


def get_all_etfs(reload=False):
    if get_all_etfs.etfs is None or reload:
        get_all_etfs.etfs = market.market_etfs_get().payload.instruments
    return get_all_etfs.etfs


get_all_etfs.etfs = None


def get_figi_history(figi, start, end, interval):
    """
    Get history for a given figi identifier
    :param figi:
    :param start:
    :param end:
    :param interval:
    :return:
    """
    df = None
    try:
        # print('C2', start.isoformat(), end.isoformat())
        hist = market.market_candles_get(figi=figi, _from=start.isoformat(), to=end.isoformat(), interval=interval)
        candles = hist.payload.candles
        candles_dict = [candles[i].to_dict() for i in range(len(candles))]
        df = pd.DataFrame.from_dict(candles_dict)
    except Exception as e:
        print(f'Unable to load history for figi={figi}')
        print(e)
    return df


def get_figi_for_ticker(ticker):
    return market.market_search_by_ticker_get(ticker).payload.instruments[0].figi


def get_ticker_history(ticker, start, end, interval):
    """
    Get history for the given ticker.
    Just gets the figi and calls the appropriate figi history function.
    :param ticker:
    :param start:
    :param end:
    :param interval:
    :return:
    """
    figi = get_figi_for_ticker(ticker)
    return get_figi_history(figi, start, end, interval)


def get_etfs_history(end=dt.datetime.now(dt.timezone.utc),
                     start=dt.datetime.now(dt.timezone.utc) - dt.timedelta(weeks=52),
                     interval='month'):
    """
    Get history _data with a given interval for all available ETFs.
    :param interval:
    :return:
    """
    # print('C1', start, end)
    tickers = []
    etfs = get_all_etfs()
    all_etfs_history = pd.DataFrame()
    for i, etf in enumerate(etfs):
        figi = etf.figi
        ticker = etf.ticker
        tickers.append(ticker)
        one_etf_history = get_figi_history(figi=figi, start=start, end=end, interval=interval)
        # print(f'Got {len(one_etf_history)} elements from {start} till {end}')
        if one_etf_history.empty:
            continue

        one_etf_history.drop(columns=['interval'], inplace=True)
        # Rename columns to combine all ETFs in the same table
        # Also replace negative values by nans
        rename_dict = {}

        for column in set(one_etf_history.columns) - set(['time', 'interval']):
            if not isinstance(column, str):
                negative_price = one_etf_history[column] < 0
                if len(negative_price):
                    one_etf_history.loc[negative_price, column] = np.nan
                    warnings.warn('Some price values were non-positive and were replaced with nans.', RuntimeWarning)
            rename_dict[column] = f'{ticker}_{column}'
        one_etf_history.rename(columns=rename_dict, inplace=True)

        # Merge into the large table
        if all_etfs_history.empty:
            all_etfs_history = one_etf_history
        else:
            all_etfs_history = all_etfs_history.merge(one_etf_history, how='outer', on='time')

    # Convert to Moscow time zone (received time is in UTC)
    # print(all_etfs_history.time.dt.tz_convert(MOSCOW_TIMEZONE))
    # all_etfs_history['time'] = all_etfs_history.time.dt.tz_localize(MOSCOW_TIMEZONE)

    return all_etfs_history, tickers


def get_etfs_daily_history(end=dt.datetime.now(MOSCOW_TIMEZONE) - dt.timedelta(days=1),
                           start=dt.datetime.now(MOSCOW_TIMEZONE) - dt.timedelta(weeks=52, days=1)):
    """
    Due to API restrictions, an interval longer than a year must be divided into years when fetched
    """
    interval = 'day'
    interval_dt = dt.timedelta(days=1)

    # For daily forecasts drop hours
    # end = end.replace(hour=0, minute=0, second=0,microsecond=0)
    # start = start.replace(hour=0, minute=0, second=0, microsecond=0)

    # print('A2', start, end)
    # print(start.tzinfo, end.tzinfo)
    length = end.astimezone(MOSCOW_TIMEZONE) - start.astimezone(MOSCOW_TIMEZONE)
    # The server bugs if the time period is smaller than 1 interval
    if length < interval_dt:
        start = end - interval_dt
        length = interval_dt

    one_period = dt.timedelta(weeks=52)
    need_divide = length > one_period
    n_periods = int(math.ceil(length / one_period))
    periods = [[end - one_period * (i + 1), end - one_period * i] for i in range(n_periods)]
    periods[-1][0] = start
    periods = periods[::-1]

    dfs = []
    for period in tqdm(periods, desc=f'Getting forecast with interval={interval}'):
        df, tickers = get_etfs_history(start=period[0], end=period[1], interval=interval)
        if df.empty:
            print(f'Server returned an empty reply for the following period: {period}!')
            continue
        dfs.append(df)
    out = pd.concat(dfs, axis=0)
    # if len(out) < (end-start) / dt.timedelta(days=1):
    #     warnings.warn(f'Server returned fewer days than expected: {len(out)} v. {(end-start) / dt.timedelta(days=1)}',
    #                   RuntimeWarning)

    # Drop all _data points after the end date (because the API does not return exactly what was requested)
    # inds = out[out.time > end]
    # if len(inds) > 0:
    #     out.drop(inds.index, inplace=True)

    # # Drop nans in time
    # print(out[pd.isna(out.time)])
    # print(out.tail(1))
    # out.drop(pd.isna(out.time).index, inplace=True)

    return out, tickers


class History:
    """
    A set of functions to download, update and locally store the history of all provided ETFs.
    By default, only the _data up to now is stored. Today's _data is always incomplete, so the last day in the recorded
    data is always updated.

    Usage:
    History.update() # to update the local database of prices
    History.data and History.trackers to access a dataframe containing price history and a trackers list
    """

    def __init__(self, interval='day'):
        self.interval = interval
        self.data_file = f'ETFs_history_i={interval}.csv'
        self.tickers_file = f'tickers_i={interval}.dat'
        self._data = pd.DataFrame()
        self._tickers = []
        self._load_data()

        if not self._data.empty:
            self.start_date = self._data.time.min()
            self.end_date = self._data.time.max()
        else:
            self.start_date = None
            self.end_date = None

    @property
    def data(self):
        return copy.deepcopy(self._data)

    @property
    def tickers(self):
        return copy.deepcopy(self._tickers)

    def _load_data(self):
        loaded = False
        # print('Loading local history _data...')
        try:
            self._data = pd.read_csv(self.data_file)
            self._data.time = pd.to_datetime(self._data.time)
            with open(self.tickers_file, 'r') as f:
                self._tickers = f.read().split()
            loaded = True
        except FileNotFoundError:
            print('No saved ETF history found locally.')
        if loaded:
            print('Local history data loaded successfully')
        return loaded

    def _save_data(self):
        tmp_filename = self.data_file + '.tmp'
        success = False
        # Save to another file
        try:
            self._data.to_csv(tmp_filename, index=False)
            with open(self.tickers_file, 'w') as f:
                for ticker in self._tickers:
                    f.write(ticker + '\n')
            success = True
        except Exception as e:
            print('Unable to save ETFs history')
            raise e

        # Replace the  original file
        if success:
            try:
                os.unlink(self.data_file)
            except FileNotFoundError:
                pass

            try:
                os.rename(tmp_filename, self.data_file)
            except Exception as e:
                success = False
                raise e

        return success

    def update(self, reload=False):
        """
        Fetch the latest _data from server up to yesterday.
        :return:
        """
        today = dt.datetime.now(LOCAL_TIMEZONE)
        # If _data were loaded, only fetch _data for the missing period
        if self._data.empty or reload:
            start_date = EARLIEST_DATE
        else:
            start_date = self.end_date
        # print(start_date, today)
        # if start_date >= today:
        #     print('Data already up to date')
        #     return

        df, self._tickers = get_etfs_daily_history(start=start_date, end=today)
        # print('A3', df.time.tail(1))

        if self._data.empty or reload:
            self._data = df
        else:
            # try:
            # print(f'Range old: [{self._data.time.min()}, {self._data.time.max()}] and new: [{df.time.min()}, {df.time.max()}]')
            # print(self._data)
            # print(df)
            # Drop all data from the original table that repeats in the new one
            merged = copy.deepcopy(self._data)
            # print('A3', df.tail(), df.time.min())
            # print('A5:', df.index)
            # print('A8', df.time.tail(), merged.time.tail())
            # print('B1', merged.time >= df.time.min())
            # print('A6', np.any(merged.time >= df.time.min()), len(merged.loc[merged.time >= df.time.min()]))
            # print('A7')
            # print('A4', merged.loc[merged.time >= df.time.min()].index)
            merged.drop(merged[merged.time >= df.time.min()].index, inplace=True)
            # print(merged.tail())

            merged = merged.append(df, verify_integrity=True, ignore_index=True)
            # print(merged.tail())
            merged.time = pd.to_datetime(
                self._data.time)  # the time column converts to object althouth there are no nans
            # print(merged)
            self._data = merged
            # print('A4', self._data.time.dtype)
            # print(self._data)
            print('History data updated successfully!')
            # except Exception as e:
            #     print('Unable to update the _data table')

        self._data.sort_values(by='time', inplace=True)
        self._save_data()
