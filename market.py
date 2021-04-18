import time

import numpy as np
from openapi_client import openapi
from openapi_genclient.exceptions import ApiException

import utils
from utils import SLEEP_TIME, SLEEP_TRIES


class Market:
    def __init__(self, token=utils.TOKEN):
        self._token = token
        self._client = openapi.sandbox_api_client(self._token)  # Initialize the openapi
        self._market = self.client.market

    def get_figi_for_ticker(self, ticker):
        res = None
        count = 0

        while not res and count < SLEEP_TRIES:
            count += 1
            try:
                ticker_search = self.market.market_search_by_ticker_get(ticker)
                res = ticker_search.payload.instruments
            except Exception as e:
                utils.log_to_file(f"Unable to get figi for ticker={ticker}.")
                utils.log_to_file(str(e))
                utils.log_to_file(f"Sleep {SLEEP_TIME} seconds")
                time.sleep(SLEEP_TIME)

        return res[0].figi if res else None

    def get_ticker_for_figi(self, figi):
        if figi in utils.OBSOLETE_TICKERS.values():
            return None

        ticker = None
        try_ = 0

        while not ticker and try_ < SLEEP_TRIES:
            try_ += 1
            try:
                ticker = self.market.market_search_by_figi_get(figi).payload.ticker
            except ApiException as e:
                utils.log_to_file(f"Unable to get ticker for figi={figi}.")
                utils.log_to_file(str(e))
                utils.log_to_file(f"Sleep {SLEEP_TIME} seconds")
                time.sleep(SLEEP_TIME)

        return ticker if ticker else None

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

        ans = None
        count = 0

        while not ans and count < SLEEP_TRIES:
            count += 1
            try:
                ans = self.market.market_orderbook_get(figi=figi, depth=1)
            except ApiException as e:
                utils.log_to_file(f"Unable to get current price for figi={figi}.")
                utils.log_to_file(str(e))

                utils.log_to_file(f"Unable to get ticker for figi={figi}.")
                utils.log_to_file(str(e))
                utils.log_to_file(f"Sleep {SLEEP_TIME} seconds")
                time.sleep(SLEEP_TIME)

        if not ans and count >= SLEEP_TRIES:
            return np.nan

        payload = ans.payload
        if payload.trade_status == "NotAvailableForTrading" or not payload.asks:
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

    def get_all_etfs(self):
        etfs = self._market.market_etfs_get().payload.instruments
        for etf in etfs:
            if etf.ticker in utils.OBSOLETE_TICKERS.keys():
                etfs.remove(etf)

        return etfs

    def get_candles(self, figi, _from, to, interval, **kwargs):
        """

        Notes
        -----
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
        return self._market.market_candles_get(figi, _from, to, interval, **kwargs)
