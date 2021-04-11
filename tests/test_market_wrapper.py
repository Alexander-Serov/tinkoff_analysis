import numpy as np
from openapi_client.openapi import SandboxOpenApi
from openapi_genclient import MarketApi

from market import Market
from utils import OBSOLETE_TICKERS

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
