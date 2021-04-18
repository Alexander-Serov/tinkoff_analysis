import datetime as dt

import numpy as np
from openapi_client.openapi import SandboxOpenApi
from openapi_genclient import MarketApi

from market import Market
from utils import MOSCOW_TIMEZONE, OBSOLETE_TICKERS

tickers_for_tests = {
    "FXUS": "BBG005HLSZ23",
    "TBIO": "TCS00A102EK1",
    "FXIT": "BBG005HLTYH9",
}

market = Market()


def test_get_figi_for_ticker():
    for ticker, figi in tickers_for_tests.items():
        assert market.get_figi_for_ticker(ticker) == figi

    # nonexistent
    assert market.get_figi_for_ticker("PUFF") is None


def get_ticker_for_figi():
    for ticker, figi in tickers_for_tests.items():
        assert market.get_ticker_for_figi(figi) == ticker
    for figi in OBSOLETE_TICKERS.values():
        assert market.get_ticker_for_figi(figi) is None

    # nonexistent
    assert market.get_ticker_for_figi("PUFF12345678") is None


def test_properties():
    assert isinstance(market.client, SandboxOpenApi)
    assert isinstance(market.market, MarketApi)


def test_get_current_price():
    """Minimal test for getting the market price.
    Should be extended to test for price when markets opened and closed.
    """
    for ticker, figi in tickers_for_tests.items():
        figi_price = market.get_current_price(figi=figi)
        assert 0 < figi_price < np.inf
        assert np.isclose(market.get_current_price(ticker=ticker), figi_price)

    # obsolete
    ticker, figi = next(iter(OBSOLETE_TICKERS.items()))
    assert np.isnan(market.get_current_price(ticker=ticker))
    assert np.isnan(market.get_current_price(figi=figi))


def test_get_all_etfs():
    etfs = market.get_all_etfs()

    for etf in etfs:
        assert etf.currency in ["USD", "RUB", "EUR"]
        assert etf.type == "Etf"
        assert etf.ticker not in OBSOLETE_TICKERS
        assert 0 < etf.min_price_increment < np.inf
        assert isinstance(etf.lot, int) and 0 < etf.lot < np.inf


def test_get_candles():
    date_1 = MOSCOW_TIMEZONE.localize(dt.datetime(2020, 9, 9))
    date_2 = MOSCOW_TIMEZONE.localize(dt.datetime(2017, 12, 8))

    start_end_interval = [
        # interval = day
        {
            "start": dt.datetime.now(MOSCOW_TIMEZONE) - dt.timedelta(weeks=52, days=1),
            "end": dt.datetime.now(MOSCOW_TIMEZONE),
            "interval": "day",
        },
        {"start": date_1 - dt.timedelta(weeks=26), "end": date_1, "interval": "day"},
        {"start": date_2 - dt.timedelta(weeks=10), "end": date_2, "interval": "day"},
        # interval = hour
        {
            "start": dt.datetime.now(MOSCOW_TIMEZONE) - dt.timedelta(weeks=1),
            "end": dt.datetime.now(MOSCOW_TIMEZONE),
            "interval": "hour",
        },
        {"start": date_1 - dt.timedelta(weeks=1), "end": date_1, "interval": "hour"},
        {"start": date_2 - dt.timedelta(days=5), "end": date_2, "interval": "hour"},
        # interval = 5min
        {
            "start": dt.datetime.now(MOSCOW_TIMEZONE) - dt.timedelta(days=1),
            "end": dt.datetime.now(MOSCOW_TIMEZONE),
            "interval": "5min",
        },
        {"start": date_1 - dt.timedelta(days=1), "end": date_1, "interval": "5min"},
        {"start": date_2 - dt.timedelta(hours=8), "end": date_2, "interval": "5min"},
    ]

    for figi in tickers_for_tests.values():
        for st_end_int in start_end_interval:
            candles = market.get_candles(
                figi,
                _from=st_end_int["start"].isoformat(),
                to=st_end_int["end"].isoformat(),
                interval=st_end_int["interval"],
            ).payload.candles

            assert isinstance(candles, list)
            if candles:
                assert candles[0].interval == st_end_int["interval"]
                for candle in candles:
                    for atr in ["c", "h", "l", "o", "v"]:
                        assert 0 < getattr(candle, atr) < np.inf
