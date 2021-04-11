import copy
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from utils import MAIN_FOLDER, log_to_file


class Strategy:
    def __init__(self, strategy: [Dict, pd.DataFrame] = None, data_file=None):
        """
        Contain a float value of ratio in portfolio for each ticker

        :param strategy: either a pd.DataFrame with columns "ticker" and "ratio" or
            a dict with keys "ticker" and "ratio" (for dict)
        :param data_file: a path to a json file where to load or to save the strategy

        If both parameters are not None then the file is used to save the strategy
        from the strategy parameter. If data_file is None then the default file
        Strategy.json is used
        """
        self._strategy = pd.DataFrame(
            {
                "ticker": pd.Series([], dtype="str"),
                "ratio": pd.Series([], dtype="float64"),
            }
        )
        self.data_file = data_file if data_file else MAIN_FOLDER / "Strategy.json"

        # Set _strategy field using the provided arguments
        assert (
            strategy is None
            or isinstance(strategy, pd.DataFrame)
            or isinstance(strategy, Dict)
        ), (
            "Strategy param is not None but is neither pd.DataFrame nor Dict."
            " Strategy is not loaded"
        )

        if strategy is not None:
            if isinstance(strategy, pd.DataFrame):
                self._strategy["ticker"] = strategy["ticker"].astype("str")
                self._strategy["ratio"] = strategy["ratio"].astype("float64")
            else:
                self._strategy["ticker"] = pd.Series(strategy["ticker"]).astype("str")
                self._strategy["ratio"] = pd.Series(strategy["ratio"]).astype("float64")
        else:  # load from file
            self.load_data()

        assert np.isclose(self._strategy["ratio"].sum(), 1, atol=1e-6)

    def load_data(self):
        try:
            self._strategy = (
                pd.read_json(self.data_file)
                .reset_index()
                .rename({"index": "ticker"}, axis=1)
            )
            print("Strategy data loaded successfully")
        except Exception as e:
            print(f"An error occurred during strategy json-file loading: {e}")
            log_to_file(f"An error occurred during strategy json-file loading: {e}")
            raise e

    def save_data(self, data_file: Path = None):
        """
        :param data_file: (optional) file to save the strategy to
        :return:
        """
        file_to_save = data_file if data_file else self.data_file

        # Set index to easily change the ratios in the file
        data_to_save = copy.deepcopy(self._strategy).set_index("ticker")

        tmp_data_file = file_to_save.with_suffix(".tmp")
        # Save to the tmp file
        try:
            data_to_save.to_json(tmp_data_file)
        except Exception as e:
            print("Unable to save strategy")
            raise e

        # Replace the original files
        try:
            os.unlink(file_to_save)
        except FileNotFoundError:
            pass

        try:
            os.rename(tmp_data_file, file_to_save)
        except Exception as e:
            raise e

    @property
    def strategy(self):
        return copy.deepcopy(self._strategy)
