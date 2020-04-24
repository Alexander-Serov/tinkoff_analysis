"""
File allowing to access investment recommendation from the terminal
"""

from analysis import History

hist_daily = History(interval='day')
hist_daily.recommend_simple(update=True, _print=True)
hist_daily.recommend_other(update=False)
