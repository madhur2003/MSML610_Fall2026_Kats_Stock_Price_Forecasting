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
# # Kats API: forecasting a toy series with ARIMA, Prophet, and MLAR
#
# - This notebook explores the Kats forecasting API on one small synthetic series
#   (a trend, a repeating pattern, and noise), so every result can be checked by
#   eye
# - The pedagogical arc:
#   - `TimeSeriesData`: the one input type every Kats model accepts
#   - Params and Model objects: configure, `fit()`, `predict()`
#   - The same pattern for ARIMA and Prophet, and where the two differ
#   - A minimal end-to-end workflow with the wrappers in `kats_utils.py`
#   - `MLARModel`: a model that does not follow the pattern

# %%
# %load_ext autoreload
# %autoreload 2

import logging

# Kats warns at import time about optional packages (torch, ax-platform) that
# this notebook does not use. Raise the log level before importing it.
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("prophet.plot").setLevel(logging.CRITICAL)

import pandas as pd
from kats.consts import TimeSeriesData
from kats.models.arima import ARIMAModel, ARIMAParams
from kats.models.ml_ar import MLARModel, MLARParams
from kats.models.prophet import ProphetModel, ProphetParams

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
# ## Library Overview
#
# - **What problem it solves**: Kats (Kit to Analyze Time Series, from Meta) puts
#   several forecasting models behind one small set of classes, so trying a second
#   model costs a few lines instead of a new library
# - **Key abstraction**: `TimeSeriesData` wraps a time column and a value column,
#   and every model takes one as input
# - **Mental model**: a `Params` object holds the settings, a `Model` object holds
#   the data and the params, `fit()` trains it in place, and `predict()` returns a
#   `DataFrame`
#
# | Object | Description | Comments |
# |--------|-------------|----------|
# | `TimeSeriesData(df)` | Time series container | Columns `time` and `value` |
# | `ARIMAParams(p, d, q)` | Settings of an ARIMA model | Plain attributes |
# | `ARIMAModel(data, params)` | ARIMA forecaster | Wraps a `statsmodels` model |
# | `ProphetParams(...)` | Settings of a Prophet model | Many optional arguments |
# | `ProphetModel(data, params)` | Prophet forecaster | Wraps `prophet.Prophet` |
# | `model.fit()` | Train in place | Returns `None` |
# | `model.predict(steps, freq)` | Forecast ahead | Frame with `time`, `fcst`, `fcst_lower`, `fcst_upper` |
# | `MLARModel(params)` | Gradient-boosted autoregression | Uses `train()`, not `fit()` |
#
# - **Key classes**:
#   - `TimeSeriesData`: the input container
#   - `ARIMAModel`: the classical statistical baseline
#   - `ProphetModel`: a trend plus seasonality model
#   - `MLARModel`: a LightGBM model that can use extra features

# %% [markdown]
# # Part 1: `TimeSeriesData`, the input type

# %% [markdown]
# ## Cell 1.1: Build a toy series
#
# - The series has 80 business days with an upward trend, a repeating pattern, and
#   noise
# - Business days (`"B"`, Monday to Friday) are the frequency used for stock
#   prices in the example notebook

# %%
# Build a toy series indexed by business day.
series = utils.make_synthetic_series(num_periods=80, freq="B")
display(series.head(3))
# Outcome: values near 50 that start to climb, indexed from 2020-01-01.

# %% [markdown]
# ## Cell 1.2: Plot the toy series
#
# **Goal**
# - See the trend and the repeating pattern that the models have to capture

# %% [markdown]
# **Description**
# - Panels
#   - `Toy series`: the 80 values as a line over time

# %%
# Plot the series.
utils.plot_series(series, title="Toy series: trend plus repeating pattern")

# %% [markdown]
# **Guided usage**
# - Compare the line at the start and at the end
#   - Observe the level rising by about 8 over the 80 days, from the trend
# - Look at a few neighbouring days
#   - Observe the small up and down swings, from the repeating pattern and noise

# %% [markdown]
# **Implementation** `plot_series(series, *, title="", figsize=(12, 3.5))`
# - Draws the values against the index as one line
# - Adds the title, a label for the y axis, and a light grid

# %%
# Link the function to its source and list its interface.
hintros.print_obj_info(utils.plot_series)

# %% [markdown]
# ## Cell 1.3: Minimal construction
#
# - `TimeSeriesData` takes a `DataFrame` with a `time` column and a `value` column

# %%
# Put the series in the layout that Kats expects.
df = pd.DataFrame({"time": series.index, "value": series.to_numpy()})
# Wrap the frame in a Kats container.
ts = TimeSeriesData(df)
print("type(ts)=", type(ts))
# Outcome: a `kats.consts.TimeSeriesData` object.

# %% [markdown]
# ## Cell 1.4: Inspect the object

# %%
# Count the observations.
print("len(ts)=", len(ts))
# Outcome: 80, one per business day.

# %%
# The time and the values are two separate pandas Series.
print("type(ts.time)=", type(ts.time), "dtype:", ts.time.dtype)
print("type(ts.value)=", type(ts.value), "dtype:", ts.value.dtype)
# Outcome: `time` is `datetime64[ns]` and `value` is `float64`.

# %%
# Get the data back as a DataFrame.
display(ts.to_dataframe().head(3))
# Outcome: the same two columns that went in, `time` and `value`.

# %% [markdown]
# ## Cell 1.5: Construct from separate time and value
#
# - The other way to build it is with `time=` and `value=` arguments
# - The `value` argument can be a named Series, and the name becomes the column
#   name

# %%
# Build the same container from the index and the Series, with no DataFrame.
ts_kw = TimeSeriesData(time=series.index, value=series)
print("ts_kw.value.name=", ts_kw.value.name)
print("len(ts_kw)=", len(ts_kw))
# Outcome: the name `value` comes from `series.name`, and the length is 80.

# %% [markdown]
# ## Cell 1.6: Public interface

# %%
# Link the class to its source on GitHub, and list its public interface.
utils.print_kats_obj_info(TimeSeriesData)
# Outcome: methods such as `to_dataframe`, `interpolate`, `plot`, `extend`.

# %% [markdown]
# ## Cell 1.7: Slice into a training set and a test set
#
# - Slicing a `TimeSeriesData` returns another `TimeSeriesData`
# - Forecasting always splits by time, never at random, so the model never sees the
#   future

# %%
# Keep the first 70 days for training and the last 10 days for testing.
train_ts = ts[:70]
test_ts = ts[70:]
print("type(train_ts)=", type(train_ts))
print("len(train_ts)=", len(train_ts), "len(test_ts)=", len(test_ts))
print("last train day=", train_ts.time.iloc[-1].date())
print("first test day=", test_ts.time.iloc[0].date())
# Outcome: 70 and 10 points, and the test set starts the day after the training set.

# %% [markdown]
# # Part 2: ARIMA, the params, model, `fit()`, `predict()` pattern

# %% [markdown]
# ## Cell 2.1: Params hold the settings
#
# - `ARIMAParams(p, d, q)` only stores three integers, nothing is trained yet
#   - `p`: number of past values used (autoregressive part)
#   - `d`: number of times the series is differenced to remove the trend
#   - `q`: number of past errors used (moving-average part)

# %%
# Choose a small model: one lag, one difference, no moving-average term.
arima_params = ARIMAParams(p=1, d=1, q=0)
print("type(arima_params)=", type(arima_params))
print("p, d, q=", arima_params.p, arima_params.d, arima_params.q)
# Outcome: the object just echoes back the three numbers.

# %% [markdown]
# ## Cell 2.2: A model holds the data and the params

# %%
# Combine the training data and the params in a model.
arima_model = ARIMAModel(train_ts, arima_params)
print("type(arima_model)=", type(arima_model))
print("arima_model.model=", arima_model.model)
# Outcome: the `model` attribute is `None`, because nothing has been trained yet.

# %% [markdown]
# ## Cell 2.3: Public interface

# %%
# Link the class to its source on GitHub, and list its public interface.
utils.print_kats_obj_info(ARIMAModel)
# Outcome: `fit`, `predict`, `plot`, and `get_parameter_search_space`.

# %% [markdown]
# ## Cell 2.4: `fit()` trains in place
#
# - `fit()` returns `None` and stores the trained `statsmodels` model in
#   `.model`
# - `utils.quiet_model_logs()` hides the warnings that `statsmodels` prints while
#   fitting, which would otherwise repeat on every call

# %%
# Fit the model on the training data.
with utils.quiet_model_logs():
    fit_result = arima_model.fit()
print("fit_result=", fit_result)
print("type(arima_model.model)=", type(arima_model.model))
# Outcome: `None` is returned, and `.model` is now a statsmodels results object.

# %% [markdown]
# ## Cell 2.5: `predict()` returns a DataFrame
#
# - `steps` is the number of points to forecast
# - `freq` is the spacing of the forecast dates, `"B"` for business days

# %%
# Forecast the next 10 business days.
arima_forecast = arima_model.predict(steps=10, freq="B")
display(arima_forecast)
# Outcome: 10 rows with a point forecast `fcst` and a band `fcst_lower`,
# `fcst_upper`.

# %% [markdown]
# ## Cell 2.6: Check the forecast dates

# %%
# The forecast starts on the first day after the training data.
print("last train day=", train_ts.time.iloc[-1].date())
print("first forecast day=", arima_forecast["time"].iloc[0].date())
print("last forecast day=", arima_forecast["time"].iloc[-1].date())
# Outcome: the same 10 days as the test set, skipping weekends.

# %% [markdown]
# ## Cell 2.7: The `alpha` argument is ignored
#
# - `predict()` accepts `alpha`, which should set the width of the band
#   - `alpha=0.05` is a 95% band, and `alpha=0.2` is an 80% band
# - Check whether the band changes

# %%
# Ask for an 80% band and for a 95% band.
forecast_80 = arima_model.predict(steps=10, freq="B", alpha=0.2)
forecast_95 = arima_model.predict(steps=10, freq="B", alpha=0.05)
# Compare the width of the two bands.
widths = pd.DataFrame(
    {
        "alpha=0.2": forecast_80["fcst_upper"] - forecast_80["fcst_lower"],
        "alpha=0.05": forecast_95["fcst_upper"] - forecast_95["fcst_lower"],
    }
)
display(widths.head(3))
# Outcome: the two columns are identical, so this Kats version ignores `alpha`.

# %% [markdown]
# ## Cell 2.8: Set the band width through the wrapper
#
# - `utils.forecast_arima()` recomputes the band from the trained `statsmodels`
#   model, so `interval_level` works as expected
# - The wrapper takes a `pandas` Series, so split the series first

# %%
# Split the pandas series the same way as the Kats container.
train, test = utils.split_train_test(series, 10)
# Ask for an 80% band and for a 95% band.
wrapped_80 = utils.forecast_arima(train, 10, order=(1, 1, 0), interval_level=0.8)
wrapped_95 = utils.forecast_arima(
    train, 10, order=(1, 1, 0), interval_level=0.95
)
# Compare the average width of the two bands.
mean_widths = pd.Series(
    {
        "80% band": (wrapped_80["fcst_upper"] - wrapped_80["fcst_lower"]).mean(),
        "95% band": (wrapped_95["fcst_upper"] - wrapped_95["fcst_lower"]).mean(),
    },
    name="mean width",
)
display(mean_widths.to_frame())
# Outcome: the 80% band is narrower than the 95% band, as it should be.

# %% [markdown]
# # Part 3: Prophet, the same pattern with different settings

# %% [markdown]
# ## Cell 3.1: Params hold the settings
#
# - `ProphetParams` has many optional arguments, and the defaults are often good
# - The defaults that matter most for stock prices are the seasonalities and the
#   width of the band

# %%
# Switch off the yearly pattern, since 70 days cannot show a yearly cycle.
prophet_params = ProphetParams(weekly_seasonality=True, yearly_seasonality=False)
# Read back the default width of the band.
print("interval_width=", prophet_params.interval_width)
print("seasonality_mode=", prophet_params.seasonality_mode)
# Outcome: the band is 80% by default, while ARIMA's is 95%.

# %% [markdown]
# ## Cell 3.2: A model holds the data and the params

# %%
# Combine the training data and the params in a model.
prophet_model = ProphetModel(train_ts, prophet_params)
print("type(prophet_model)=", type(prophet_model))
print("prophet_model.model=", prophet_model.model)
# Outcome: as with ARIMA, `.model` is `None` before `fit()`.

# %% [markdown]
# ## Cell 3.3: Public interface

# %%
# Link the class to its source on GitHub, and list its public interface.
utils.print_kats_obj_info(ProphetModel)
# Outcome: the same `fit` and `predict` as `ARIMAModel`.

# %% [markdown]
# ## Cell 3.4: `fit()` trains in place
#
# - Prophet runs a Stan program and logs a lot, so `quiet_model_logs()` keeps the
#   notebook readable

# %%
# Fit the model on the training data.
with utils.quiet_model_logs():
    prophet_fit_result = prophet_model.fit()
print("prophet_fit_result=", prophet_fit_result)
print("type(prophet_model.model)=", type(prophet_model.model))
# Outcome: `None` is returned, and `.model` is now a `prophet.Prophet` object.

# %% [markdown]
# ## Cell 3.5: `predict()` has the same output format

# %%
# Forecast the next 10 business days.
prophet_forecast = prophet_model.predict(steps=10, freq="B")
display(prophet_forecast)
# Outcome: the same four columns as ARIMA, so downstream code works for both.

# %% [markdown]
# ## Cell 3.6: The default band is narrower than ARIMA's

# %%
# Compare the average band width of Prophet's default and ARIMA's default.
band_widths = pd.DataFrame(
    {
        "Prophet (80%)": [
            (
                prophet_forecast["fcst_upper"] - prophet_forecast["fcst_lower"]
            ).mean()
        ],
        "ARIMA (95%)": [
            (arima_forecast["fcst_upper"] - arima_forecast["fcst_lower"]).mean()
        ],
    },
    index=["mean width"],
)
display(band_widths)
# Outcome: Prophet's band is several times narrower than ARIMA's.

# %% [markdown]
# ## Cell 3.7: Align the band with `interval_width`
#
# - Set `interval_width=0.95` so the two models report the same kind of band

# %%
# Refit Prophet with a 95% band.
prophet_params_95 = ProphetParams(
    weekly_seasonality=True, yearly_seasonality=False, interval_width=0.95
)
prophet_model_95 = ProphetModel(train_ts, prophet_params_95)
with utils.quiet_model_logs():
    prophet_model_95.fit()
prophet_forecast_95 = prophet_model_95.predict(steps=10, freq="B")
# Measure the average width of the new band.
width_95 = (
    prophet_forecast_95["fcst_upper"] - prophet_forecast_95["fcst_lower"]
).mean()
print("mean width at 95%=", round(width_95, 2))
# Outcome: wider than the 80% band, but still much narrower than ARIMA's, so the
# level explains only part of the gap.

# %% [markdown]
# # Part 4: Composition, two models on one split

# %% [markdown]
# ## Cell 4.1: What the wrapper layer adds
#
# - The raw calls above work, but a backtest repeats them many times
# - `kats_utils.py` wraps them so that every model has the same signature and the
#   same output
#   - `forecast_naive()`: last value carried forward, the baseline to beat
#   - `forecast_arima()`: Kats ARIMA, with the band width fixed
#   - `forecast_prophet()`: Kats Prophet, with the band width and logs handled
#   - `forecast_all()`: the three at once, in a dictionary, used in the example
#     notebook

# %% [markdown]
# ## Cell 4.2: Forecast with the three models
#
# - Prophet fits a yearly cycle by default, and 70 days of data cannot show one
#   - With the default, the forecast on this toy series is far off, so switch it
#     off
# - The example notebook has 11 years of prices, where the default is fine

# %%
# Forecast the 10 test days with the baseline, ARIMA, and Prophet.
forecasts = {
    utils.NAIVE: utils.forecast_naive(train, 10),
    utils.ARIMA: utils.forecast_arima(train, 10, order=(1, 1, 0)),
    utils.PROPHET: utils.forecast_prophet(train, 10, yearly_seasonality=False),
}
print("models=", list(forecasts.keys()))
# Outcome: three forecast frames, one per model, all with the same columns.

# %% [markdown]
# ## Cell 4.3: Compare them on the test set

# %%
# Score each forecast on the 10 held-out days.
scores = utils.evaluate_forecasts(test, forecasts)
display(scores)
# Outcome: one row per model, sorted by MAE; `MAE / naive MAE` below 1 means the
# model beats the baseline. Prophet does, while ARIMA(1, 1, 0) does not, since it
# forecasts a flat line while the series keeps climbing.

# %% [markdown]
# ## Cell 4.4: Plot the forecasts
#
# **Goal**
# - See each forecast and its band next to what really happened

# %% [markdown]
# **Description**
# - Panels
#   - `Naive`, `ARIMA`, `Prophet`: one panel per model, with the recent history,
#     the test values, the forecast, and its band

# %%
# Plot the three forecasts on a shared y axis.
utils.plot_forecasts(train, test, forecasts, num_history=40, ylabel="value")

# %% [markdown]
# **Guided usage**
# - Compare the band width across the three panels
#   - Observe ARIMA's band widen quickly, since it differences the series
# - Compare where the test values fall relative to each band
#   - Observe whether the true values stay inside the 95% band

# %% [markdown]
# **Implementation** `plot_forecasts(train, test, forecasts, *, num_history=120,
# ylabel="Close price ($)")`
# - Draws the last `num_history` training points in each panel
# - Draws the test values and each model's forecast with its band
# - Shares the y axis so the panels are comparable

# %%
# Link the function to its source and list its interface.
hintros.print_obj_info(utils.plot_forecasts)

# %% [markdown]
# # Part 5: Patterns beyond `fit()` and `predict()`

# %% [markdown]
# ## Cell 5.1: Search spaces for tuning
#
# - Every model class has a static `get_parameter_search_space()`
# - It lists the settings worth tuning, in a format that Kats' tuning tools read

# %%
# Read the first two entries of the ARIMA search space.
arima_space = ARIMAModel.get_parameter_search_space()
display(pd.DataFrame(arima_space[:2])[["name", "values"]])
# Outcome: `p` and `d` with the integer values to try.

# %% [markdown]
# ## Cell 5.2: `MLARModel` takes only params
#
# - `MLARModel` is a LightGBM regressor over lagged values, with calendar and
#   Fourier features
# - It breaks the pattern in three ways
#   - The model is built from params alone, and the data goes to `train()`
#   - The method is `train()`, not `fit()`
#   - `predict()` returns a dictionary of frames, one per series

# %%
# Predict 10 steps ahead from the last 10 values of the series.
mlar_params = MLARParams(
    target_variable="value", horizon=10, input_window=10, freq="B"
)
mlar_model = MLARModel(mlar_params)
print("type(mlar_model)=", type(mlar_model))
# Outcome: a model that holds the settings and no data.

# %% [markdown]
# ## Cell 5.3: `train()` takes a list of series

# %%
# Train on the training set, passed as a list of one series.
with utils.quiet_model_logs():
    mlar_model.train([train_ts])
print("trained on", len(train_ts), "points")
# Outcome: no output, the model keeps its state inside.

# %% [markdown]
# ## Cell 5.4: `predict()` returns a dictionary

# %%
# Forecast the horizon that was set in the params.
with utils.quiet_model_logs():
    mlar_output = mlar_model.predict()
print("type(mlar_output)=", type(mlar_output))
print("keys=", list(mlar_output.keys()))
# Outcome: a dictionary with one key, since there is a single series.

# %%
# Look at the frame of the only series.
mlar_forecast = mlar_output[0]
display(mlar_forecast.head(3))
# Outcome: the forecast sits in the `forecast` column, with no band.

# %% [markdown]
# ## Cell 5.5: Bring it into the common format

# %%
# Rename to the common column names so it can be compared with the others.
mlar_common = mlar_forecast.rename(columns={"forecast": "fcst"})[
    ["time", "fcst"]
]
mlar_mae = utils.compute_mae(test.to_numpy(), mlar_common["fcst"].to_numpy())
print("MLAR MAE=", round(mlar_mae, 3))
# Outcome: an error in the same units as the series, comparable to Cell 4.3.

# %% [markdown]
# # Part 6: Interactive exploration

# %% [markdown]
# ## Cell 6.1: What happens to ARIMA when the order changes?
#
# - Change `order` and rerun the cell
# - Try `(0, 1, 0)`, `(1, 1, 1)`, and `(2, 1, 0)`
# - Ask which order has the lowest MAE, and whether it is also the one with the
#   lowest AIC from `utils.select_arima_order()`

# %%
# Edit the order and rerun.
order = (1, 1, 0)
experiment_forecast = utils.forecast_arima(train, 10, order=order)
experiment_mae = utils.compute_mae(
    test.to_numpy(), experiment_forecast["fcst"].to_numpy()
)
print("order=", order, "MAE=", round(experiment_mae, 3))
# Outcome: one number per order, to compare across reruns.

# %% [markdown]
# ## Cell 6.2: What happens to Prophet when the seasonality changes?
#
# - Set `weekly` to `False` and rerun
# - Ask what the weekly pattern does on a series with no weekend rows

# %%
# Edit the flag and rerun.
weekly = True
experiment_prophet = utils.forecast_prophet(
    train, 10, yearly_seasonality=False, weekly_seasonality=weekly
)
experiment_prophet_mae = utils.compute_mae(
    test.to_numpy(), experiment_prophet["fcst"].to_numpy()
)
print("weekly=", weekly, "MAE=", round(experiment_prophet_mae, 3))
# Outcome: one number per setting, to compare across reruns.

# %% [markdown]
# ## Summary: The Mental Model
#
# - `TimeSeriesData` is the only input type: a `time` column and a `value` column,
#   sliced like a list and converted back with `to_dataframe()`
# - Every forecaster is built from a `Params` object and a `Model` object, trained
#   in place with `fit()`, and asked for a `DataFrame` with `predict(steps, freq)`
# - The forecast has the columns `time`, `fcst`, `fcst_lower`, and `fcst_upper`,
#   so ARIMA and Prophet can be compared with the same code
# - The defaults are not aligned: ARIMA's band is 95% and ignores `alpha`, and
#   Prophet's is 80%, so set `interval_width` or use the wrappers before comparing
# - `MLARModel` follows a different pattern (`train()` and a dictionary output),
#   so check the interface of each model before assuming `fit()` and `predict()`
