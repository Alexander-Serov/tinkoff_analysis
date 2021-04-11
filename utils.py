import datetime as dt
import inspect
import os
from pathlib import Path

import colorama
import pytz

LOCAL_TIMEZONE = dt.datetime.now(dt.timezone.utc).astimezone().tzinfo
MOSCOW_TIMEZONE = pytz.timezone("Europe/Moscow")
EARLIEST_DATE = dt.datetime.fromisoformat("2013-01-01").replace(tzinfo=MOSCOW_TIMEZONE)

OBSOLETE_TICKERS = {
    "FXJP": "BBG005HM5979",
    "FXAU": "BBG005HM6BL7",
    "FXUK": "BBG005HLK5V5",
}

SLEEP_TIME = 5  # in seconds
SLEEP_TRIES = 10

logfolder = Path("logs")
os.makedirs(logfolder, exist_ok=True)
logfile = logfolder / f"{str(dt.date.today())}.log"

colorama.init()

# Load the token
MAIN_FOLDER = Path(__file__).parent
token_file = MAIN_FOLDER / "token_sandbox"
with open(token_file, "r") as f:
    TOKEN = f.read().strip()


def log_to_file(*args):
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)
    caller = calframe[1][3]
    try:
        with open(logfile, "a") as f:
            f.write(str(dt.datetime.now()) + f" function: {caller}\n")
            for arg in args:
                f.write(str(arg))
            f.write("\n\n")
    except Exception:
        print("Unable to save the following thing to the log:")
        print(args)
