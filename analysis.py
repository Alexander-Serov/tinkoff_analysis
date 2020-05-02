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
import inspect
import time

from openapi_client import openapi
from openapi_genclient.exceptions import ApiException
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
import sys
import colorama
from colorama import Fore, Style
import pathlib
from pathlib import Path

colorama.init()

# Load the token
main_folder = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
token_file = main_folder / './token_sandbox'
with open(token_file, 'r') as f:
    token = f.read().strip()

LOCAL_TIMEZONE = dt.datetime.now(dt.timezone.utc).astimezone().tzinfo
MOSCOW_TIMEZONE = pytz.timezone('Europe/Moscow')
EARLIEST_DATE = dt.datetime.fromisoformat('2013-01-01').replace(
    tzinfo=MOSCOW_TIMEZONE)
# TODO: In the recieved results, Moscow timezone sometimes appears as +2:30 and sometimes as +3:00. To fix
obsolete_tickers = {'FXJP': 'BBG005HM5979', 'FXAU': 'BBG005HM6BL7', 'FXUK': 'BBG005HLK5V5'}

logfolder = Path('logs')
os.makedirs(logfolder, exist_ok=True)
logfile = logfolder / f'{str(dt.date.today())}.log'

# Initialize the openapi
client = openapi.sandbox_api_client(token)
market = client.market


def get_all_etfs(reload=False):
    if get_all_etfs.etfs is None or reload:
        get_all_etfs.etfs = market.market_etfs_get().payload.instruments
    return get_all_etfs.etfs


get_all_etfs.etfs = None


def log_to_file(*args):
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)
    caller = calframe[1][3]
    try:
        with open(logfile, 'a') as f:
            f.write(str(dt.datetime.now()) + f' function: {caller}\n')
            for arg in args:
                f.write(str(arg))
            f.write('\n\n')
    except Exception as e:
        print('Unable to save the following thing to log:')
        print(args)
        # raise e


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
        hist = market.market_candles_get(figi=figi, _from=start.isoformat(),
                                         to=end.isoformat(), interval=interval)
        # print('1', start.isoformat(), end.isoformat(),
        #       market.market_candles_get(figi=figi,
        #                                       _from=start.isoformat(),
        #                                  to=end.isoformat(), interval=interval))
        candles = hist.payload.candles
        candles_dict = [candles[i].to_dict() for i in range(len(candles))]
        df = pd.DataFrame.from_dict(candles_dict)
    except Exception as e:
        if figi not in obsolete_tickers.values():
            log_to_file(f'Unable to load history for figi={figi}')
            log_to_file(e)
    return df


@lru_cache(maxsize=1000)
def get_figi_for_ticker(ticker):
    res = market.market_search_by_ticker_get(ticker).payload.instruments
    if res:
        figi = res[0].figi
    else:
        figi = None
    return figi


@lru_cache(maxsize=1000)
def get_ticker_for_figi(figi):
    if figi in obsolete_tickers.values():
        return None
    try:
        return market.market_search_by_figi_get(figi).payload.ticker
    except ApiException as e:
        log_to_file(f'Unable to get ticker for figi={figi}.')
        log_to_file(str(e))
        return None


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
                     start=dt.datetime.now(dt.timezone.utc) - dt.timedelta(
                         weeks=52),
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
        one_etf_history = get_figi_history(figi=figi, start=start, end=end,
                                           interval=interval)

        if one_etf_history.empty:
            continue

        if not one_etf_history.time.is_unique and interval in ['day']:
            print(ticker, one_etf_history)
            raise ValueError(f'Received time stamps are not unique for '
                             f'ticker={ticker} and period=[{start}, {end}]')

        one_etf_history.drop(columns=['interval'], inplace=True)
        if 'ticker' not in one_etf_history.columns:
            one_etf_history['ticker'] = ticker

        # Merge into the large table
        if all_etfs_history.empty:
            all_etfs_history = one_etf_history
        else:
            # all_etfs_history = all_etfs_history.merge(one_etf_history, how='outer', on=['time', 'figi'])
            all_etfs_history = all_etfs_history.append(one_etf_history,
                                                       ignore_index=True)

    return all_etfs_history, tickers


def get_etfs_daily_history(
        end=dt.datetime.now(MOSCOW_TIMEZONE) - dt.timedelta(days=1),
        start=dt.datetime.now(MOSCOW_TIMEZONE) - dt.timedelta(weeks=52,
                                                              days=1)):
    """
    Get daily market history (1 point per day) in exactly the requested interval.

    Note:
    Due to API restrictions, an interval longer than a year must be divided into years when fetched
    """
    interval = 'day'
    interval_dt = dt.timedelta(days=1)
    print(f'Requesting ETF history from {start} till {end} with an interval={interval}')

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
    periods = [[end - one_period * (i + 1), end - one_period * i] for i in
               range(n_periods)]
    periods[-1][0] = start
    periods = periods[::-1]

    dfs = []
    for period in tqdm(periods,
                       desc=f'Getting forecast with interval={interval}'):
        df, tickers = get_etfs_history(start=period[0], end=period[1],
                                       interval=interval)
        if df.empty:
            print(
                f'Server returned an empty reply for the following period: {period}!')
            continue
        dfs.append(df)
    if dfs:
        out = pd.concat(dfs, axis=0, ignore_index=True)
    else:
        out, tickers = None, None
    # if len(out) < (end-start) / dt.timedelta(days=1):
    #     warnings.warn(f'Server returned fewer days than expected: {len(out)} v. {(end-start) / dt.timedelta(days=1)}',
    #                   RuntimeWarning)

    # Drop all _data points after the end date (because the API does not return exactly what was requested)
    # inds = out[out.time > end]
    # if len(inds) > 0:
    #     out.drop(inds.index, inplace=True)

    # Drop nans in time
    # print(out[pd.isna(out.time)])
    # print(out.tail(1))

    l1 = len(out)
    out.drop(out[pd.isna(out.time)].index, inplace=True)
    if len(out) < l1:
        log_to_file(f'{l1 - len(out)} NaN time stamps dropped.')

    return out, tickers


def get_current_price(figi: str = None, ticker: str = None):
    """
    If the market is open, return the lowest `ask` price for the given figi.
    Otherwise, return the close price of the last trading day.

    Note: close price should be used because it correctly represents the last transaction.
    See: https://www.quora.com/What-is-the-difference-between-last-traded-price-LTP-and-closing-price
    This explains the difference with Tinkoff Investment app where the `last_price` is shown instead
    Close price when the market is close was verified.
    #todo verify current price when the market is open

    Parameters
    ----------
    figi
    ticker

    Returns
    -------

    """
    if figi is None:
        if ticker is None:
            raise ValueError('Either ticker or figi should be provided.')
        else:
            figi = get_figi_for_ticker(ticker)
    elif ticker is not None and get_figi_for_ticker(ticker) != figi:
        raise ValueError(
            f'Ticker and figi point to different products: {figi} {get_figi_for_ticker(ticker)}')

    if figi in obsolete_tickers.values() or figi is None:
        return np.nan

    try:
        ans = market.market_orderbook_get(figi=figi, depth=1)
    except ApiException as e:
        log_to_file(f'Unable to get current price for figi={figi}.')
        log_to_file(str(e))
        return np.nan
    payload = ans.payload
    if payload.trade_status == 'NotAvailableForTrading':
        current_price = payload.close_price
    else:
        order_response = payload.asks[0]
        # print('debug-1', get_ticker_for_figi(figi), order_response)
        current_price = order_response.price

        # TODO not tested

    return current_price


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
        self.data_file = main_folder / f'ETFs_history_i={interval}.csv'
        self.tickers_file = main_folder / f'tickers_i={interval}.dat'
        self._data = pd.DataFrame()
        self._tickers = []
        self._load_data()

    @property
    def last_date(self):
        if not self._data.empty:
            last_date = self._data.time.max()
        else:
            last_date = None

        return last_date

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

            # Drop nans in time stamps
            l1 = len(self._data)
            self._data.drop(self._data[pd.isna(self._data.time)].index, inplace=True)
            if len(self._data) < l1:
                log_to_file(f'{l1 - len(self._data)} NaN time stamps dropped.')

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
        # Do not save nan time
        data_to_save = copy.deepcopy(self._data)
        data_to_save.drop(index = data_to_save[data_to_save['time'].isna()].index, inplace=True)


        tmp_data_file = self.data_file.with_suffix('.tmp')
        tmp_tickers_file = self.tickers_file.with_suffix('.tmp')
        success = False
        # Save to another file
        try:
            data_to_save.to_csv(tmp_data_file, index=False)
            with open(tmp_tickers_file, 'w') as f:
                for ticker in self._tickers:
                    f.write(ticker + '\n')
            success = True
        except Exception as e:
            print('Unable to save ETFs history')
            raise e

        # Replace the original files
        if success:
            for tmp_file, file in [(tmp_data_file, self.data_file),
                                   (tmp_tickers_file, self.tickers_file)]:
                try:
                    os.unlink(file)
                except FileNotFoundError:
                    pass

                try:
                    os.rename(tmp_file, file)
                except Exception as e:
                    success = False
                    raise e

        return success

    def update(self, reload=False, verbose=False):
        """
        Fetch the latest _data from server starting from 2 last known days.
        :return:
        """
        today = dt.datetime.now(MOSCOW_TIMEZONE)
        # If _data have been loaded, only fetch data starting with the last
        # date in the database
        if self._data.empty or reload:
            start_date = EARLIEST_DATE
        else:
            start_date = self.last_date - dt.timedelta(days=1)

        if verbose:
            print(f'Updating historical data starting from {start_date}')

        new_data, tickers = get_etfs_daily_history(start=start_date, end=today)
        if new_data is None or new_data.empty:
            raise RuntimeError(
                f'The received data frame cannot be empty because one of the '
                f'dates was already present in the history data. Requested dates: {start_date} -- {today}.')

        self._tickers = tickers
        if self._data.empty or reload:
            self._data = new_data
        else:
            """Drop all data from the original table that repeats in the new 
            one """
            old_data = copy.deepcopy(self._data)
            old_data.drop(old_data[old_data.time >= new_data.time.min()].index,
                          inplace=True)

            merged = old_data.append(new_data, verify_integrity=True,
                                     ignore_index=True)
            # Convert the time column to time because after merge it changes
            # to object
            merged.time = pd.to_datetime(self._data.time)
            self._data = merged
            print(f'History data updated successfully since {start_date}!')

        self._data.sort_values(by='time', inplace=True)
        self._save_data()

    def calculate_statistics(self, position='c'):
        """
        Print basic calculate_statistics such as increase and decrease from 52-week
        extrema.

        Note: to better exclude the extreme values for years and determine the true range,
        the max and min are replaced with close quantiles. For weeks, the true max and min
        are still kept. In theory, a similar procedure can be implemented, but higher resolution
        data are required.

        :param position what time of day (o, c, h, l) to use to
        assign a value to a day :return:
        #todo for correct week max and min, need to get at least hourly data for the latest week at least
        """
        figis = self._data.figi.unique()
        max_quantile = 0.97  # to make extrema calculations more robust to outliers,
        min_quantile = 1 - max_quantile  # calculate as close quantiles instead of real extremum

        filter_52w = dt.datetime.now(LOCAL_TIMEZONE) - dt.timedelta(weeks=52)
        filter_1w = dt.datetime.now(LOCAL_TIMEZONE) - dt.timedelta(weeks=1)
        filter_1d = dt.datetime.now(LOCAL_TIMEZONE) - dt.timedelta(days=1)

        statistics = {'last_price':
                          {figi: get_current_price(figi) for figi
                           in figis},
                      'max_52w': self._data.loc[self._data.time >=
                                                filter_52w, ['figi', 'h']].groupby(
                          by='figi').quantile(max_quantile)['h'].to_dict(),
                      'max_52w-10%': (self._data.loc[self._data.time >=
                                                     filter_52w, ['figi', 'h']].groupby(
                          by='figi').quantile(max_quantile)['h'] * 0.9).to_dict(),
                      'min_52w': self._data.loc[self._data.time >=
                                                filter_52w, ['figi', 'l']].groupby(
                          by='figi').quantile(min_quantile)['l'].to_dict(),
                      'max_1w':
                          self._data.loc[self._data.time >= filter_1w, ['figi', 'h']].groupby(
                              by='figi').quantile(1)['h'].to_dict(),
                      'min_1w':
                          self._data.loc[self._data.time >= filter_1w, ['figi', 'l']].groupby(
                              by='figi').quantile(0)['l'].to_dict(),
                      '1w':
                          self._data.loc[self._data.time >= filter_1w, ['figi', position]].groupby(
                              by='figi').first()[position].to_dict(),
                      '1d':
                          self._data.loc[self._data.time <= filter_1d, ['figi', position]].groupby(
                              by='figi').last()[position].to_dict()
                      }
        # print(self._data[self._data.time >= filter_1w].groupby(
        #                       by='figi'))
        # print(self._data[self._data.ticker == 'FXUS'].time.unique())

        # print(max_52w)

        # current_prices_df = pd.DataFrame.from_dict(current_prices)
        # print(current_prices_df)

        # Combine the calculate_statistics together
        statistics_df = pd.DataFrame(index=figis)
        statistics_df.index.name = 'figi'
        statistics_df['ticker'] = statistics_df.index.map(get_ticker_for_figi)
        for key, value in statistics.items():
            if (set(figis) - set(value.keys()) - set(obsolete_tickers.values())):
                warnings.warn(f'Not all figis were found for the column `'
                              f'{key}`. The analysis may be incorrect.')
            statistics_df[key] = statistics_df.index.map(value)
            if key != 'last_price':
                statistics_df[key + '_chg'] = (statistics_df['last_price'] -
                                               statistics_df[key]) / \
                                              statistics_df[key]

                statistics_df[key + '_chg_percent'] = (statistics_df[
                                                           'last_price'] -
                                                       statistics_df[key]) / \
                                                      statistics_df[key] * 100

        # print(statistics_df)
        # statistics_df.to_csv('tmp.csv')

        return statistics_df

    def recommend_simple(self, update: bool = True, _print=False, reload=False):
        if update:
            self.update(reload=reload)
        statistics_df = self.calculate_statistics()

        if _print:
            print()
            print(statistics_df.loc[:, ['ticker', 'max_52w', 'max_52w_chg_percent', 'max_1w',
                                        'last_price', 'max_52w-10%']])
            print(
                '* Note: max and min for weeks and years are calculated differently, and may seem '
                'inconsistent,\nbut should be OK to use. See docstring for details.')

        # Print recommendation according to current strategy
        # TODO formalize strategies into a separate class or functions
        statistics_sorted = statistics_df.sort_values('max_52w_chg_percent')
        THRESHOLD_MAX_52W_CHG_PERCENT = -10
        THRESHOLD_1W_CHG = -1e-8
        msg = [f'\n==\nGood opportunities to buy (criteria: i. 52w change <= '
               f'{THRESHOLD_MAX_52W_CHG_PERCENT} %, ii. 1w change <= '
               f'{THRESHOLD_1W_CHG:.2f}):']
        for figi in statistics_sorted.index:
            # filt = statistics_sorted.index == figi
            if (statistics_sorted.loc[figi, 'max_52w_chg_percent'] <=
                    THRESHOLD_MAX_52W_CHG_PERCENT
                    and statistics_sorted.loc[figi, '1w_chg'] <=
                    THRESHOLD_1W_CHG):
                msg.append(
                    f'{statistics_sorted.loc[figi, "ticker"]:s}\t52w max '
                    f'change: {Fore.RED}{statistics_sorted.loc[figi, "max_52w_chg_percent"]:.2f} %{Fore.RESET} '
                    f'\tlast week change: {Fore.RED}{statistics_sorted.loc[figi, "1w_chg_percent"]:.2f} %'
                    f'{Fore.RESET}'
                    f'\tlast_price: {statistics_sorted.loc[figi, "last_price"]:.2f}'
                )
        if len(msg) > 1:
            msg.append('\n==\n')
            print(*msg, sep='\n')
        else:
            print('Currently no good opportunities to buy :(')
        # warnings.warn('The output values have not been verified', UserWarning)

    def recommend_other(self, update: bool = True, _print=False, reload=False):
        if update:
            self.update(reload=reload)
        statistics_df = self.calculate_statistics()

        if _print:
            print(statistics_df)  # .loc[:, ['ticker', 'max_52w', 'max_1w', 'current']])

        # Print recommendation according to current strategy
        # TODO formalize strategies into a separate class or functions
        statistics_sorted = statistics_df.sort_values('max_52w_chg_percent')
        THRESHOLD_MAX_52W_CHG_PERCENT = -10
        THRESHOLD_1D_CHG = -1e-8
        msg = [f'\n==\nGood opportunities to buy (criteria: i. 52w change <= '
               f'{THRESHOLD_MAX_52W_CHG_PERCENT} %, ii. previous day change <= '
               f'{THRESHOLD_1D_CHG:.2f}):']
        for figi in statistics_sorted.index:
            # filt = statistics_sorted.index == figi
            if (statistics_sorted.loc[figi, 'max_52w_chg_percent'] <=
                    THRESHOLD_MAX_52W_CHG_PERCENT
                    and statistics_sorted.loc[figi, '1d_chg'] <=
                    THRESHOLD_1D_CHG):
                msg.append(
                    f'{statistics_sorted.loc[figi, "ticker"]:s}\t52w max '
                    f'change: {Fore.RED}{statistics_sorted.loc[figi, "max_52w_chg_percent"]:.2f} %{Fore.RESET} '
                    f'\tprevious day change: {Fore.RED}'
                    f'{statistics_sorted.loc[figi, "1d_chg_percent"]:.2f} %'
                    f'{Fore.RESET}'
                    f'\tlast_price: {statistics_sorted.loc[figi, "last_price"]:.3f}'
                )
        if len(msg) > 1:
            msg.append('\n==\n')
            print(*msg, sep='\n')
        else:
            print('Currently no good opportunities to buy :(')

        # warnings.warn('The output values have not been verified', UserWarning)
