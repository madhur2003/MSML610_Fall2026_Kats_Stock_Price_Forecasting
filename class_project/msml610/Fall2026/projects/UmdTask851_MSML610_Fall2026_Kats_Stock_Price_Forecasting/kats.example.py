# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Kats example: forecasting Tesla's closing price
#
# - This notebook forecasts the next 30 trading days of Tesla's closing price with
#   Kats, and checks how far the result can be trusted
# - The data is the daily Tesla price history from 2010-07-01 to 2021-10-14,
#   shipped in `data/TSLA.csv`
# - The pedagogical arc:
#   - Load the prices, put them on a regular business-day grid, and hand them to
#     Kats
#   - Define the problem: a chronological split, a baseline, and the metrics
#   - Forecast with a naive baseline, ARIMA, and Prophet, and plot the forecasts
#     with their prediction intervals
#   - Score one test window, then repeat it on eight windows to see what holds up
#   - Bonus: a trading-volume feature with `MLARModel`, other time intervals, and
#     other stocks
# - The Kats calls live in `kats_utils.py` and are introduced in `kats.API.ipynb`.
#   This notebook orchestrates them

# %%
# %load_ext autoreload
# %autoreload 2

import logging
import os

# Kats warns at import time about optional packages (torch, ax-platform) that
# this notebook does not use. Raise the log level before importing it.
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("prophet.plot").setLevel(logging.CRITICAL)

import numpy as np
import pandas as pd

# %%
import helpers.hintrospection as hintros
import helpers.hnotebook as hnotebook

import kats_utils as utils

# Initialize notebook configuration and logging.
hnotebook.config_notebook()
_LOG = logging.getLogger(__name__)
utils.init_loggers(_LOG)

# Convert `display` into `print()` when running outside IPython.
try:
    from IPython.display import display
except ImportError:
    display = print  # type: ignore

# %% [markdown]
# # Part 1: From a price file to a Kats series

# %% [markdown]
# ## Cell 1.1: Load the prices
#
# - The file is a daily price history with `Date`, `Open`, `High`, `Low`, `Close`,
#   `Adj Close`, and `Volume`
# - The loader sorts the rows by date and checks that each day appears once

# %%
# Load the Tesla prices that ship with the tutorial.
prices = utils.load_prices(utils.get_default_data_path())
display(prices.head(3))
# Outcome: three rows from July 2010, when the price was a few dollars.

# %% [markdown]
# ## Cell 1.2: Put the closing price on a business-day grid
#
# - The market is closed on holidays, so the file skips some business days
# - ARIMA needs a regular grid, so each missing business day gets the last close
#   carried forward: the price did not change while the market was closed

# %%
# Build the series of closing prices, indexed by business day.
close = utils.to_business_day_series(prices)
print("rows in the file =", len(prices))
print("business days =", len(close))
print("days filled =", len(close) - len(prices))
# Outcome: 2,843 trading days become 2,946 business days, with 103 holidays filled.

# %% [markdown]
# ## Cell 1.3: Plot the price history
#
# **Goal**
# - See the shape of the series that the models have to forecast

# %% [markdown]
# **Description**
# - Panels
#   - `Tesla closing price`: the closing price in dollars over the full history

# %%
# Plot the closing price.
utils.plot_series(close, title="Tesla closing price", figsize=(12, 3.5))

# %% [markdown]
# **Guided usage**
# - Compare the level in 2010 with the level in 2021
#   - Observe a rise from about $3 to a peak of about $883 in January 2021
# - Look at the size of the daily swings early and late in the history
#   - Observe that the swings in dollars grow with the price
#   - This matters for the prediction intervals in Part 6

# %% [markdown]
# **Implementation** `plot_series(series, *, title="", figsize=(12, 3.5))`
# - Draws the values against the index as one line
# - Adds the title, a label for the y axis, and a light grid

# %%
# Link the function to its source and list its interface.
hintros.print_obj_info(utils.plot_series)

# %% [markdown]
# ## Cell 1.4: Hand the series to Kats
#
# - Every Kats model takes a `TimeSeriesData`, with a `time` and a `value` column
# - The wrappers in `kats_utils.py` do this conversion, so it happens inside each
#   forecast call

# %%
# Convert the series and inspect the Kats object.
close_ts = utils.to_timeseries_data(close)
print("type =", type(close_ts).__name__)
print("points =", len(close_ts))
# Outcome: a `TimeSeriesData` with the same 2,946 points.

# %% [markdown]
# # Part 2: Define the problem

# %% [markdown]
# ## Cell 2.1: The forecasting task
#
# - **Task**: forecast the closing price of the next 30 business days, about six
#   weeks
# - **Metric**: MAE and RMSE in dollars, plus the share of true prices that fall
#   inside the 95% prediction interval
# - **Baseline**: the naive forecast, the last price carried forward
#   - Prices are close to a random walk, so a model that cannot beat this adds
#     nothing
#   - The column `MAE / naive MAE` below 1 means the model beats the baseline
# - **Split**: chronological, with the last 30 business days as the test set
#   - A random split would let the model train on days that come after the ones
#     it is scored on

# %%
# Hold out the last 30 business days.
horizon = 30
train, test = utils.split_train_test(close, horizon)
# Outcome: the test window starts on 2021-09-03 and ends on 2021-10-14.

# %% [markdown]
# ## Cell 2.2: Is the series stationary?
#
# - ARIMA needs a stationary series, and the differencing order `d` is the number
#   of differences needed to get there
# - The augmented Dickey-Fuller test has the null hypothesis that the series has a
#   unit root, so a p-value below 0.05 means stationary

# %%
# Test the level and the first difference of the training set.
display(utils.check_stationarity(train))
# Outcome: the level is not stationary (p close to 1) and the first difference
# is (p near 0), so use `d=1`.

# %% [markdown]
# ## Cell 2.3: Choose the ARIMA order on the training set
#
# - With `d=1` fixed, try every `(p, q)` up to 2 and keep the one with the lowest
#   AIC
# - Only the training set is used, so the choice cannot see the test window

# %%
# Search the order by AIC.
arima_order = utils.select_arima_order(train)
print("arima_order =", arima_order)
# Outcome: the order (2, 1, 2).

# %% [markdown]
# # Part 3: Forecast

# %% [markdown]
# ## Cell 3.1: Forecast with the three models
#
# - `forecast_all()` fits the naive baseline, Kats ARIMA, and Kats Prophet on the
#   training set and forecasts the 30 test days
# - It sets the interval of every model to 95%
#   - The Kats defaults differ: ARIMA's band is 95% and Prophet's is 80%, see
#     `kats.API.ipynb`

# %%
# Forecast the test window with the three models.
forecasts = utils.forecast_all(train, horizon, arima_order=arima_order)
display(forecasts[utils.PROPHET].head(3))
# Outcome: a frame with `time`, `fcst`, `fcst_lower`, and `fcst_upper`, the same
# format for all three models.

# %% [markdown]
# ## Cell 3.2: Plot the forecasts with their intervals
#
# **Goal**
# - See each forecast and its 95% interval next to what really happened

# %% [markdown]
# **Description**
# - Panels
#   - `Naive`, `ARIMA`, `Prophet`: one panel per model, with the last 120 training
#     days, the test prices, the forecast, and its interval

# %%
# Plot the three forecasts on a shared y axis.
utils.plot_forecasts(train, test, forecasts, num_history=120)

# %% [markdown]
# **Guided usage**
# - Compare the forecast line with the black line of actual prices in each panel
#   - Observe that the naive and ARIMA forecasts are nearly flat
#   - Observe that Prophet continues the upward trend
# - Compare the shape of the three intervals
#   - Observe that the naive and ARIMA intervals fan out with the horizon
#   - Observe that Prophet's interval is wide from the first day
# - Check whether the actual prices stay inside each interval
#   - Observe that the naive and ARIMA intervals miss the last days, when the
#     price climbs, and that Prophet's covers the whole window

# %% [markdown]
# **Implementation** `plot_forecasts(train, test, forecasts, *, num_history=120,
# ylabel="Close price ($)")`
# - Draws the last `num_history` training points in each panel
# - Draws the test values and each model's forecast with its interval
# - Shares the y axis so the panels are comparable

# %%
# Link the function to its source and list its interface.
hintros.print_obj_info(utils.plot_forecasts)

# %% [markdown]
# ## Cell 3.3: Compare the interval widths
#
# - Intervals are comparable only at the same level, so check the widths after
#   the alignment

# %%
# Compute the mean width of each 95% interval in dollars.
widths = pd.Series(
    {
        name: (f["fcst_upper"] - f["fcst_lower"]).mean()
        for name, f in forecasts.items()
    },
    name="mean interval width ($)",
)
display(widths.round(1).to_frame())
# Outcome: ARIMA ($107) and the naive baseline ($110) are close, since both widen
# like a random walk, and Prophet's interval is wider ($184).

# %% [markdown]
# # Part 4: Evaluate on the test window

# %% [markdown]
# ## Cell 4.1: Score the forecasts

# %%
# Score each forecast on the 30 held-out days.
scores = utils.evaluate_forecasts(test, forecasts)
display(scores.round(3))
# Outcome: Prophet has the lowest MAE ($19.6, about half of the naive MAE), while
# ARIMA is no better than the baseline (0.98). The Prophet interval covers all 30
# days, the other two cover 93% to 97%.

# %% [markdown]
# ## Cell 4.2: What the score does not say
#
# - Prophet won this window, but one window is one draw
# - Prophet extrapolates a trend, so it wins whenever the price keeps rising

# %%
# Compute the price change over the test window.
change = 100 * (test.iloc[-1] / train.iloc[-1] - 1)
print("price change over the test window = %.1f%%" % change)
# Outcome: the price rose 11.7%, a window that rewards a model that extrapolates
# the trend. Part 5 checks whether the win repeats.

# %% [markdown]
# # Part 5: Does the result hold? A rolling backtest

# %% [markdown]
# ## Cell 5.1: Walk back through eight windows
#
# - The backtest cuts the last 240 business days into eight consecutive windows
#   of 30 days
# - At each origin, the models are refitted on the data before the window and
#   scored on the window
# - The ARIMA order is chosen on the data before the first window, so no window
#   leaks into the choice

# %%
# Choose the order before the first window and run the backtest.
num_origins = 8
backtest_order = utils.select_arima_order(close.iloc[: -horizon * num_origins])
results = utils.rolling_origin_evaluation(
    close, horizon, num_origins=num_origins, arima_order=backtest_order
)
print("rows =", len(results))
# Outcome: 24 rows, one for each of the 8 origins and the 3 models.

# %% [markdown]
# ## Cell 5.2: The ratio to the baseline at each origin

# %%
# Show the MAE ratio of each model, with the price change of the window.
ratio = results.pivot(index="origin", columns="model", values="MAE / naive MAE")
ratio["price change (%)"] = results.groupby("origin")["price change (%)"].first()
display(ratio.round(2))
# Outcome: ARIMA stays at about 1 in every window. Prophet loses to the baseline
# in the first four windows (ratios from 1.75 to 3.08) and wins in the last four.

# %% [markdown]
# ## Cell 5.3: Plot the ratio at each origin
#
# **Goal**
# - See at a glance in which windows each model beats the baseline

# %% [markdown]
# **Description**
# - Panels
#   - `MAE / naive MAE`: one line for ARIMA and one for Prophet over the forecast
#     origins, with the baseline at 1

# %%
# Plot the ratio of each model at each origin.
utils.plot_rolling_evaluation(results)

# %% [markdown]
# **Guided usage**
# - Look at where each line crosses the dashed baseline
#   - Observe that ARIMA hugs the baseline in every window
#   - Observe that Prophet sits above the baseline in the first four windows
#     and below it in the last four
# - Compare the size of the wins and the losses
#   - Observe that the worst Prophet loss (about 3 times the baseline error) is
#     larger than its best win (about half)

# %% [markdown]
# **Implementation** `plot_rolling_evaluation(results, *, figsize=(14, 4.5))`
# - Plots the `MAE / naive MAE` of ARIMA and Prophet at each origin
# - Draws the baseline as a dashed line at 1
# - Uses a log scale, so a ratio of 2 and a ratio of 0.5 are equally far from 1

# %%
# Link the function to its source and list its interface.
hintros.print_obj_info(utils.plot_rolling_evaluation)

# %% [markdown]
# ## Cell 5.4: Summarize the backtest

# %%
# Summarize each model over the eight windows.
summary = utils.summarize_rolling_evaluation(results)
display(summary.round(2))
# Outcome: no model beats the baseline on average (ARIMA 1.01, Prophet 1.58).
# Each of the two beats it in 4 of 8 windows, and the intervals cover the true
# price in only 42% to 55% of the days, not 95%.

# %% [markdown]
# # Part 6: Why the intervals miss, and the log fix

# %% [markdown]
# ## Cell 6.1: A constant spread in dollars
#
# - The naive and ARIMA intervals assume a fixed spread of the daily change in
#   dollars, estimated on the whole training history
# - Tesla's swings in dollars grew with its price, so a spread estimated on the
#   cheap early years is too narrow for the recent windows
# - The log of the price turns a swing proportional to the price into a constant
#   spread, so the intervals can be modeled in logs and converted back

# %%
# Rerun the backtest on the log of the prices, scored in dollars.
log_results = utils.rolling_origin_evaluation(
    close,
    horizon,
    num_origins=num_origins,
    arima_order=backtest_order,
    log_scale=True,
)
log_summary = utils.summarize_rolling_evaluation(log_results)
display(log_summary.round(2))
# Outcome: the coverage of the naive and ARIMA intervals rises from 0.44 and 0.42
# to 0.86, close to the 95% target, while the MAE ratio is unchanged. Prophet
# gets worse (ratio 6.26): its log trend extrapolates the growth of the whole
# history.

# %% [markdown]
# ## Cell 6.2: Compare the coverage before and after

# %%
# Put the coverage of the two runs side by side.
coverage = pd.DataFrame(
    {
        "dollars": summary["mean interval coverage"],
        "log prices": log_summary["mean interval coverage"],
    }
)
display(coverage.round(2))
# Outcome: the log fixes the intervals of the two simple models and not the
# Prophet point forecast, so the right scale depends on the model.

# %% [markdown]
# # Part 7: Bonus, a trading-volume feature with `MLARModel`

# %% [markdown]
# ## Cell 7.1: Build the frame with the volume
#
# - `MLARModel` is a gradient-boosted regressor over lagged values, and it can
#   also read the lags of a covariate
# - The volume varies by orders of magnitude, so use its log
# - On a holiday the volume is the one of the previous trading day, as for the
#   price

# %%
# Put the closing price and the log volume on the same grid.
log_volume = np.log(utils.to_business_day_series(prices, column="Volume"))
frame = pd.DataFrame({"Close": close, "log_volume": log_volume})
display(frame.tail(3))
# Outcome: two columns that share the business-day index.

# %% [markdown]
# ## Cell 7.2: Backtest `MLARModel` with and without the volume
#
# - The windows are the same eight as in Part 5, so the ratios are comparable
# - `MLARModel` gives no prediction interval, so only the MAE is scored

# %%
# Run both variants on the eight windows.
mlar_results = utils.rolling_origin_mlar(
    frame, horizon, num_origins=num_origins, covariate="log_volume"
)
mlar_ratio = mlar_results.pivot(
    index="origin", columns="model", values="MAE / naive MAE"
)
display(mlar_ratio.round(2))
# Outcome: the volume lowers the ratio in 7 of the 8 windows, most of all in the
# last two (from 1.96 and 1.83 to 0.75 and 0.80).

# %% [markdown]
# ## Cell 7.3: All the models in one table

# %%
# Collect the mean MAE ratio of every model over the eight windows.
mlar_mean = mlar_results.groupby("model")["MAE / naive MAE"].mean()
all_models = pd.concat([summary["mean MAE / naive MAE"], mlar_mean])
display(all_models.sort_values().round(2).to_frame("mean MAE / naive MAE"))
# Outcome: the volume brings `MLARModel` from 1.57 to 1.14, which is the best of
# the learned models, and still above the baseline (1.00).

# %% [markdown]
# # Part 8: Bonus, other intervals and other stocks

# %% [markdown]
# ## Cell 8.1: Weekly and monthly prices
#
# - `resample_series()` keeps the last price of each week or month, and drops the
#   last period if the data stops before it ends
# - The horizons cover a similar span: 30 days, 6 weeks, and 2 months
# - The same functions run on any interval through the `freq` argument

# %%
# Backtest the three models on daily, weekly, and monthly prices.
intervals = {
    "daily": (utils.BUSINESS_DAY, 30),
    "weekly": (utils.WEEKLY, 6),
    "monthly": (utils.MONTHLY, 2),
}
interval_summaries = {}
for name, (freq, interval_horizon) in intervals.items():
    series = (
        close
        if freq == utils.BUSINESS_DAY
        else utils.resample_series(close, freq)
    )
    interval_summaries[name] = utils.backtest_prices(
        series, horizon=interval_horizon, num_origins=num_origins, freq=freq
    )
display(pd.concat(interval_summaries, names=["interval"]).round(2))
# Outcome: ARIMA is slightly better than the baseline on weekly prices (0.93) and
# worse on monthly ones (1.12). Prophet is worse than the baseline at every
# interval, and its intervals cover none of the monthly windows.

# %% [markdown]
# ## Cell 8.2: Other stocks
#
# - The loop runs on every CSV in `data/`, and the ticker is the file name
# - To add a stock, drop a file with the same columns in `data/`, for example
#   `AAPL.csv` from Yahoo Finance, and rerun the cell
# - This copy of the tutorial ships with Tesla only

# %%
# Backtest every stock found in the data directory.
data_dir = os.path.dirname(utils.get_default_data_path())
stock_summaries = {}
for ticker, path in utils.list_price_files(data_dir).items():
    stock_close = utils.to_business_day_series(utils.load_prices(path))
    stock_summaries[ticker] = utils.backtest_prices(
        stock_close, horizon=horizon, num_origins=num_origins
    )
display(pd.concat(stock_summaries, names=["ticker"]).round(2))
# Outcome: one block of three rows per stock. With Tesla only, it repeats the
# daily result of Cell 5.4.

# %% [markdown]
# ## Summary: What the example shows
#
# - Kats gives the three forecasters the same inputs and outputs, so the whole
#   comparison is one loop over a dictionary
# - On the last 30 days, Prophet wins by a wide margin, and the rolling backtest
#   shows that this is not a result to trust: no model beats the naive baseline
#   on average over eight windows
# - A single test window is one draw. Score several windows before choosing a
#   model, and compare with the naive baseline, not only with each other
# - Check the interval coverage, not only the point error. The 95% intervals
#   covered 42% to 55% of the prices, because a spread in dollars is too narrow
#   for a price that grew almost 200-fold. Modeling log prices restores most of the
#   coverage for the simple models
# - A volume feature helps `MLARModel` in 7 of 8 windows, which shows that extra
#   information can matter, even when the model stays above the baseline
# - Kats is unmaintained and needs the pinned environment of the Dockerfile, so
#   treat it as a teaching tool for the common forecasting pattern
