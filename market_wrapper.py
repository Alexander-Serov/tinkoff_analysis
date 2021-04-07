from functools import lru_cache

import numpy as np
from openapi_client import openapi
from openapi_genclient.exceptions import ApiException

import utils


class MarketWrapper:
    def __init__(self, token=utils.TOKEN):
        self._token = token
        self._client = openapi.sandbox_api_client(self._token)  # Initialize the openapi
        self._market = self.client.market

    @lru_cache(maxsize=1000)
    def get_figi_for_ticker(self, ticker):
        res = self.market.market_search_by_ticker_get(ticker).payload.instruments
        return res[0].figi if res else None

    @lru_cache(maxsize=1000)
    def get_ticker_for_figi(self, figi):
        if figi in utils.OBSOLETE_TICKERS.values():
            return None
        try:
            return self.market.market_search_by_figi_get(figi).payload.ticker
        except ApiException as e:
            utils.log_to_file(f"Unable to get ticker for figi={figi}.")
            utils.log_to_file(str(e))
            return None

    def get_current_price(self, figi: str = None, ticker: str = None):
        """If the market is open, return the lowest `ask` price for the given figi.
        Otherwise, return the close price of the last trading day.

        Note: close price should be used because it correctly represents the last
        transaction.
        See: https://www.quora.com/What-is-the-difference-between-last-traded-price-LTP-and-closing-price  # noqa: E501
        This explains the difference with Tinkoff Investment app where the `last_price` is
        shown instead
        Close price when the market is closed was verified.
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
                raise ValueError("Either ticker or figi should be provided.")
            figi = self.get_figi_for_ticker(ticker)
        elif ticker is not None and self.get_figi_for_ticker(ticker) != figi:
            raise ValueError(
                f"Ticker and figi point to different products: {figi} "
                f"{self.get_figi_for_ticker(ticker)}"
            )

        if figi in utils.OBSOLETE_TICKERS.values() or figi is None:
            return np.nan

        try:
            ans = self.market.market_orderbook_get(figi=figi, depth=1)
        except ApiException as e:
            utils.log_to_file(f"Unable to get current price for figi={figi}.")
            utils.log_to_file(str(e))
            return np.nan
        payload = ans.payload
        if payload.trade_status == "NotAvailableForTrading":
            current_price = payload.close_price
        else:
            order_response = payload.asks[0]
            current_price = order_response.price

            # TODO needs tests

        return current_price

    @property
    def client(self):
        return self._client

    @property
    def market(self):
        return self._market