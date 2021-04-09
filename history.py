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

import copy
import datetime as dt
import math
import os
import time
import warnings
from typing import List, Tuple

import pandas as pd
from colorama import Fore
from pytz import UTC
from tqdm import tqdm

import utils
from market_wrapper import MarketWrapper
from utils import (
    EARLIEST_DATE,
    LOCAL_TIMEZONE,
    MOSCOW_TIMEZONE,
    SLEEP_COUNT,
    SLEEP_TIME,
)


class History:
    """
    A set of functions to download, update and locally store the history of all provided
    ETFs.
    By default, only the _data up to now is stored. Today's _data is always incomplete,
    so the last day in the recorded
    data is always updated.

    Usage:
    History().update() # to update the local database of prices
    History.data and History.trackers to access a dataframe containing price history
    and a trackers list
    """

    def __init__(self, interval="day", market_wrapper=MarketWrapper(), verbose=False):
        self._data = pd.DataFrame()
        self._tickers = []
        self._etfs = None

        self.market_wrapper = market_wrapper
        self.market = self.market_wrapper.market
        self.interval = interval
        self.data_file = utils.MAIN_FOLDER / f"ETFs_history_i={interval}.csv"
        self.tickers_file = utils.MAIN_FOLDER / f"tickers_i={interval}.dat"
        self.verbose = verbose

        # Update
        self._load_data()

    def _load_data(self):
        loaded = False
        # print('Loading local history _data...')
        try:
            self._data = pd.read_csv(self.data_file)

            # Drop nans in time stamps
            l1 = len(self._data)
            self._data.drop(self._data[pd.isna(self._data.time)].index, inplace=True)
            if len(self._data) < l1:
                utils.log_to_file(f"{l1 - len(self._data)} NaN time stamps dropped.")

            self._data.time = pd.to_datetime(self._data.time)
            with open(self.tickers_file, "r") as f:
                self._tickers = f.read().split()
            loaded = True
        except FileNotFoundError:
            print("No saved ETF history found locally.")
        if loaded:
            print("Local history data loaded successfully")
        return loaded

    def _save_data(self):
        # Do not save nan time
        data_to_save = copy.deepcopy(self._data)
        data_to_save.drop(
            index=data_to_save[data_to_save["time"].isna()].index, inplace=True
        )

        tmp_data_file = self.data_file.with_suffix(".tmp")
        tmp_tickers_file = self.tickers_file.with_suffix(".tmp")
        # Save to another file
        try:
            data_to_save.to_csv(tmp_data_file, index=False)
            with open(tmp_tickers_file, "w") as f:
                for ticker in self._tickers:
                    f.write(ticker + "\n")
            success = True
        except Exception as e:
            print("Unable to save ETFs history")
            raise e

        # Replace the original files
        if success:
            for tmp_file, file in [
                (tmp_data_file, self.data_file),
                (tmp_tickers_file, self.tickers_file),
            ]:
                try:
                    os.unlink(file)
                except FileNotFoundError:
                    pass

                try:
                    os.rename(tmp_file, file)
                except Exception as e:
                    raise e

        return success

    def _get_all_etfs(self, reload=False):
        if self._etfs is None or reload:
            downloaded_etfs = self.market.market_etfs_get().payload.instruments

            for etf in downloaded_etfs:
                if etf.ticker in utils.OBSOLETE_TICKERS.keys():
                    downloaded_etfs.remove(etf)

            self._etfs = downloaded_etfs

        return self._etfs

    def get_ticker_history(self, ticker, start, end, interval):
        """
        Get history for the given ticker.
        Just gets the figi and calls the appropriate figi history function.
        :param ticker:
        :param start:
        :param end:
        :param interval:
        :return:
        """

        figi = self.market_wrapper.get_figi_for_ticker(ticker)
        return self.get_figi_history(figi, start, end, interval)

    def get_figi_history(
        self, figi: str, start: dt.datetime, end: dt.datetime, interval: str
    ):
        """
        Get history for a given figi identifier
        :param figi:
        :param start:
        :param end:
        :param interval:
        :return:
        """
        hist = None
        count = 0

        while not hist and count < SLEEP_COUNT:
            count += 1
            try:
                hist = self.market.market_candles_get(
                    figi=figi,
                    _from=start.isoformat(),
                    to=end.isoformat(),
                    interval=interval,
                    # _request_timeout=1000,
                )

            except Exception as e:
                utils.log_to_file(e)
                utils.log_to_file(f"Sleep {SLEEP_TIME} seconds")
                time.sleep(SLEEP_TIME)

        if self.verbose:
            print("Received market response:", hist.payload.candles)
        candles = hist.payload.candles
        candles_dicts = [candles[i].to_dict() for i in range(len(candles))]
        df = pd.DataFrame(candles_dicts)
        return df

    def get_etfs_history(
        self,
        end=dt.datetime.now(dt.timezone.utc),
        start=dt.datetime.now(dt.timezone.utc) - dt.timedelta(weeks=52),
        freq="month",
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Get history _data with a given interval for all available ETFs.

        Parameters
        ----------
        end
            End of the time frame for which to get history.
        start
            Start of the time string for which to return history.
        freq
            Frequency of the returned values.

            Returns
            -------
            pd.DataFrame
                A dataframe with historical values of all ETFs.
            list
                List of ETF tickers.
        """
        tickers = []
        etfs = self._get_all_etfs()
        all_etfs_history = pd.DataFrame()
        for i, etf in enumerate(etfs):
            figi = etf.figi
            ticker = etf.ticker
            tickers.append(ticker)

            if self.verbose:
                print(
                    f"Getting ETF history for figi={figi} from {start} till {end},"
                    f" interval={freq}."
                )
            one_etf_history = self.get_figi_history(
                figi=figi, start=start, end=end, interval=freq
            )

            # If could not receive the data after some tries.
            # The server did not respond or there was no trading
            if one_etf_history.empty:
                continue

            if not one_etf_history.time.is_unique and freq in ["day"]:
                print(ticker, one_etf_history)
                raise ValueError(
                    f"Received time stamps are not unique for "
                    f"ticker={ticker} and period=[{start}, {end}]"
                )

            one_etf_history.drop(columns=["interval"], inplace=True)
            if "ticker" not in one_etf_history.columns:
                one_etf_history["ticker"] = ticker

            # Append to the large table
            if all_etfs_history.empty:
                all_etfs_history = one_etf_history
            else:
                all_etfs_history = all_etfs_history.append(
                    one_etf_history, ignore_index=True
                )

        return all_etfs_history, tickers

    def get_etfs_daily_history(
        self,
        end=dt.datetime.now(MOSCOW_TIMEZONE) - dt.timedelta(days=1),
        start=dt.datetime.now(MOSCOW_TIMEZONE) - dt.timedelta(weeks=52, days=1),
    ):
        """
        Get daily market history (1 point per day) in exactly the requested interval.

        Note:
        Due to API restrictions, an interval longer than a year must be divided into years
        when fetched
        """

        interval = "day"
        interval_dt = dt.timedelta(days=1)
        print(
            f"Requesting ETF history from {start} till {end} with an interval={interval}"
        )

        length = end.astimezone(MOSCOW_TIMEZONE) - start.astimezone(MOSCOW_TIMEZONE)
        # The server bugs if the time period is smaller than 1 interval
        if length < interval_dt:
            start = end - interval_dt
            length = interval_dt

        one_period = dt.timedelta(weeks=52)
        n_periods = int(math.ceil(length / one_period))
        periods = [
            [end - one_period * (i + 1), end - one_period * i] for i in range(n_periods)
        ]
        periods[-1][0] = start
        periods = periods[::-1]

        dfs = []
        out, tickers = pd.DataFrame(), []
        for period in tqdm(periods, desc=f"Getting forecast with interval={interval}"):
            if self.verbose:
                print(f"Requesting history from {period[0]} till {period[1]}.")
            df, tickers = self.get_etfs_history(
                start=period[0], end=period[1], freq=interval
            )
            if df.empty:
                print(
                    f"Server returned an empty reply for the following period: {period}!"
                )
                continue
            dfs.append(df)
        if dfs:
            out = pd.concat(dfs, axis=0, ignore_index=True)

        l1 = len(out)
        out.drop(out[pd.isna(out.time)].index, inplace=True)
        if len(out) < l1:
            utils.log_to_file(f"{l1 - len(out)} NaN time stamps dropped.")

        return out, tickers

    def update(self, reload=False):
        """Fetch the latest price data from server from the last cached date till now.

        Parameters
        ----------
        reload
            If False, will use the stored price cache. Otherwise, will reload all
            from the server.
        """
        today = dt.datetime.now(MOSCOW_TIMEZONE)
        # If _data have been loaded, only fetch data starting with the last
        # date in the database
        if self._data.empty or reload:
            start_date = EARLIEST_DATE
            self._etfs = None  # Drop etfs list to reload
        else:
            start_date = (self.last_date - dt.timedelta(days=1)).astimezone(
                MOSCOW_TIMEZONE
            )

        if self.verbose:
            print(f"Updating historical data from {start_date} till {today}")

        new_data, tickers = self.get_etfs_daily_history(start=start_date, end=today)

        if self.verbose:
            print("Update function received the following new data: ", new_data)
        if new_data is None or new_data.empty:
            raise RuntimeError(
                "The received data frame cannot be empty because one of the "
                "dates was already present in the history data. "
                f"Requested dates: {start_date} -- {today}."
            )

        self._tickers = tickers
        if self._data.empty or reload:
            new_data["time"] = new_data["time"].dt.tz_convert(UTC)
            self._data = new_data
        else:
            """Drop all data from the original table that repeats in the new
            one"""
            old_data = copy.deepcopy(self._data)
            old_data.drop(
                old_data[old_data.time >= new_data.time.min()].index, inplace=True
            )

            new_data["time"] = new_data["time"].dt.tz_convert(UTC)
            merged = old_data.append(new_data, verify_integrity=True, ignore_index=True)
            # Convert the time column to time because after merge it changes
            # to object
            merged.time = pd.to_datetime(merged.time)
            self._data = merged
            print(f"History data updated successfully since {start_date}!")

        self._data.sort_values(by="time", inplace=True)
        self._save_data()

    def calculate_statistics(self, position="c"):
        """
        Print basic calculate_statistics such as increase and decrease from 52-week
        extrema.

        Note: to better exclude the extreme values for years and determine the true
        range,
        the max and min are replaced with close quantiles. For weeks, the true max and
        min
        are still kept. In theory, a similar procedure can be implemented, but higher
        resolution
        data are required.

        :param position
        What time of day (o, c, h, l: open, close, high, low) to use to
        assign a value to a day

        :return:
        #todo for correct week max and min, need to get at least hourly data for the
        latest week at least
        """
        figis = self._data.figi.unique()
        max_quantile = 0.99  # to make extrema calculations more robust to outliers,
        min_quantile = 1 - max_quantile  # calculate as close quantiles instead of
        # real extremum

        filter_52w = dt.datetime.now(LOCAL_TIMEZONE) - dt.timedelta(weeks=52)
        filter_1w = dt.datetime.now(LOCAL_TIMEZONE) - dt.timedelta(weeks=1)
        filter_1d = dt.datetime.now(LOCAL_TIMEZONE) - dt.timedelta(days=1)

        statistics = {
            "last_price": {
                figi: self.market_wrapper.get_current_price(figi) for figi in figis
            },
            "max_52w": self._data.loc[self._data.time >= filter_52w, ["figi", "h"]]
            .groupby(by="figi")
            .quantile(max_quantile)["h"]
            .to_dict(),
            "max_52w-10%": (
                self._data.loc[self._data.time >= filter_52w, ["figi", "h"]]
                .groupby(by="figi")
                .quantile(max_quantile)["h"]
                * 0.9
            ).to_dict(),
            "min_52w": self._data.loc[self._data.time >= filter_52w, ["figi", "l"]]
            .groupby(by="figi")
            .quantile(min_quantile)["l"]
            .to_dict(),
            "max_1w": self._data.loc[self._data.time >= filter_1w, ["figi", "h"]]
            .groupby(by="figi")
            .quantile(1)["h"]
            .to_dict(),
            "min_1w": self._data.loc[self._data.time >= filter_1w, ["figi", "l"]]
            .groupby(by="figi")
            .quantile(0)["l"]
            .to_dict(),
            "1w": self._data.loc[self._data.time >= filter_1w, ["figi", position]]
            .groupby(by="figi")
            .first()[position]
            .to_dict(),
            "1d": self._data.loc[self._data.time <= filter_1d, ["figi", position]]
            .groupby(by="figi")
            .last()[position]
            .to_dict(),
        }

        # Combine the calculate_statistics together
        statistics_df = pd.DataFrame(index=figis)
        statistics_df.index.name = "figi"
        statistics_df["ticker"] = statistics_df.index.map(
            self.market_wrapper.get_ticker_for_figi
        )

        for key, value in statistics.items():
            if set(figis) - set(value.keys()):
                warnings.warn(f"Not all figis were found for the column '{key}'.")
            statistics_df[key] = statistics_df.index.map(value)
            if key != "last_price":
                statistics_df[key + "_chg"] = (
                    statistics_df["last_price"] - statistics_df[key]
                ) / statistics_df[key]

                statistics_df[key + "_chg_percent"] = (
                    (statistics_df["last_price"] - statistics_df[key])
                    / statistics_df[key]
                    * 100
                )

        return statistics_df

    def recommend_simple(self, update: bool = True, _print=False, reload=False):
        if update:
            self.update(reload=reload)
        statistics_df = self.calculate_statistics()

        if _print:
            print()
            statistics_df_sorted = statistics_df.sort_values("max_52w_chg_percent")
            with pd.option_context("precision", 2):
                print(
                    statistics_df_sorted.loc[
                        :,
                        [
                            "ticker",
                            "max_52w",
                            "max_52w_chg_percent",
                            "1w_chg_percent",
                            "max_1w",
                            "last_price",
                            "max_52w-10%",
                        ],
                    ]
                )
            print(
                "* Note: max and min for weeks and years are calculated differently, "
                "and may seem "
                "inconsistent,\nbut should be OK to use. See docstring for details."
            )

        # Print recommendation according to current strategy
        # TODO formalize strategies into a separate class or functions
        statistics_sorted = statistics_df.sort_values("max_52w_chg_percent")
        THRESHOLD_MAX_52W_CHG_PERCENT = -10
        THRESHOLD_1W_CHG = -1e-8
        msg = [
            f"\n==\nGood opportunities to buy (criteria: i. 52w change <= "
            f"{THRESHOLD_MAX_52W_CHG_PERCENT} %, ii. 1w change <= "
            f"{THRESHOLD_1W_CHG:.2f} %):"
        ]
        for figi in statistics_sorted.index:
            if (
                statistics_sorted.loc[figi, "max_52w_chg_percent"]
                <= THRESHOLD_MAX_52W_CHG_PERCENT
                and statistics_sorted.loc[figi, "1w_chg"] <= THRESHOLD_1W_CHG
            ):
                msg.append(
                    f'{statistics_sorted.loc[figi, "ticker"]:s}\t52w max '
                    f"change: {Fore.RED}"
                    f'{statistics_sorted.loc[figi, "max_52w_chg_percent"]:.2f}'
                    f" %{Fore.RESET} "
                    f"\tlast week change: "
                    f'{Fore.RED}{statistics_sorted.loc[figi, "1w_chg_percent"]:.2f} %'
                    f"{Fore.RESET}"
                    f'\tlast_price: {statistics_sorted.loc[figi, "last_price"]:.2f}'
                )
        if len(msg) > 1:
            msg.append("\n==\n")
            print(*msg, sep="\n")
        else:
            print("Currently no good opportunities to buy :(")
        # warnings.warn('The output values have not been verified', UserWarning)

    def recommend_other(self, update: bool = True, _print=False, reload=False):
        if update:
            self.update(reload=reload)
        statistics_df = self.calculate_statistics()

        if _print:
            with pd.option_context("precision", 2):
                print(statistics_df)

        # Print recommendation according to current strategy
        # TODO formalize strategies into a separate class or functions
        statistics_sorted = statistics_df.sort_values("max_52w_chg_percent")
        THRESHOLD_MAX_52W_CHG_PERCENT = -10
        THRESHOLD_1D_CHG = -1e-8
        msg = [
            f"\n==\nGood opportunities to buy (criteria: i. 52w change <= "
            f"{THRESHOLD_MAX_52W_CHG_PERCENT} %, ii. previous day change <= "
            f"{THRESHOLD_1D_CHG:.2f}):"
        ]
        for figi in statistics_sorted.index:
            if (
                statistics_sorted.loc[figi, "max_52w_chg_percent"]
                <= THRESHOLD_MAX_52W_CHG_PERCENT
                and statistics_sorted.loc[figi, "1d_chg"] <= THRESHOLD_1D_CHG
            ):
                msg.append(
                    f'{statistics_sorted.loc[figi, "ticker"]:s}\t52w max '
                    f"change: {Fore.RED}"
                    f'{statistics_sorted.loc[figi, "max_52w_chg_percent"]:.2f} '
                    f"%{Fore.RESET} "
                    f"\tprevious day change: {Fore.RED}"
                    f'{statistics_sorted.loc[figi, "1d_chg_percent"]:.2f} %'
                    f"{Fore.RESET}"
                    f'\tlast_price: {statistics_sorted.loc[figi, "last_price"]:.3f}'
                )
        if len(msg) > 1:
            msg.append("\n==\n")
            print(*msg, sep="\n")
        else:
            print("Currently no good opportunities to buy :(")

        # warnings.warn('The output values have not been verified', UserWarning)

    @property
    def last_date(self):
        return self._data.time.max() if not self._data.empty else None

    @property
    def data(self):
        return copy.deepcopy(self._data)

    @property
    def tickers(self):
        return copy.deepcopy(self._tickers)

    @property
    def etfs(self):
        return copy.deepcopy(self._etfs)
