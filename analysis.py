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

# Load the token
token_file = './token_sandbox'
with open(token_file, 'r') as f:
    token = f.read().strip()

LOCAL_TIMEZONE = dt.datetime.now(dt.timezone.utc).astimezone().tzinfo

# Initialize the openapi
client = openapi.sandbox_api_client(token)
market = client.market

@lru_cache
def get_all_etfs():
    etfs = market.market_etfs_get().payload.instruments
    return etfs


def get_figi_history(figi, start, end, interval):
    """
    Get history for a given figi identifier
    :param figi:
    :param start:
    :param end:
    :param interval:
    :return:
    """
    hist = market.market_candles_get(figi=figi, _from=start.isoformat(), to=end.isoformat(), interval=interval)
    candles = hist.payload.candles
    candles_dict = [candles[i].to_dict() for i in range(len(candles))]
    df = pd.DataFrame.from_dict(candles_dict)
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


def get_etfs_history(end = dt.datetime.now(dt.timezone.utc),
                    start = dt.datetime.now(dt.timezone.utc) - dt.timedelta(weeks=52),
                    interval='month'):
    """
    Get history data with a given interval for all available ETFs.
    :param interval:
    :return:
    """
    tickers = []
    etfs = get_all_etfs()
    for i, etf in enumerate(tqdm(etfs)):
        figi = etf.figi
        ticker = etf.ticker
        tickers.append(ticker)
        one_etf_history = get_figi_history(figi=figi, start=start, end=end, interval=interval)
        one_etf_history.drop(columns=['interval'])
        # Rename columns to combine all ETFs in the same table
        rename_dict = {}

        for column in set(one_etf_history.columns) - set(['time', 'interval']):
            rename_dict[column] = f'{ticker}_{column}'
        one_etf_history.rename(columns=rename_dict, inplace=True)

        # Merge into the large table
        if not i:
            all_etfs_history = one_etf_history
        else:
            all_etfs_history = all_etfs_history.merge(one_etf_history, how='outer', on='time')

    return all_etfs_history, tickers




