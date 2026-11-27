# Kats Tutorial Notes

- Three short notes that do not belong in the notebooks: the comparison with the
  Prophet tutorial, why Kats needs care, and how the notebook skills were used

## Kats vs `tutorials/Prophet`

- Both tutorials forecast with Prophet, but they teach different things

| Point       | `tutorials/Prophet`                                                                                  | This tutorial                                                                                                                                           |
| :---------- | :--------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Scope       | One model in depth: trend types, Fourier seasonality, holidays, component plots, external regressors | Kats ARIMA and Prophet behind one `Params` / `Model` / `fit()` / `predict()` pattern, plus a naive baseline and MLAR, which does not follow the pattern |
| Data        | Synthetic daily series with a known trend, weekly pattern, and holiday effect                        | Tesla closing prices, 2010 to 2021, where the true process is unknown                                                                                   |
| Backtest    | `cross_validation()` and `performance_metrics()`, for Prophet only                                   | `rolling_origin_evaluation()` in `kats_utils.py`, for any model, scored against a naive baseline                                                        |
| Intervals   | Set to 95% with `interval_width=0.95`                                                                | Kats ARIMA always gives 95% and ignores `alpha`, Kats Prophet gives 80% by default, so `forecast_*()` aligns them to 95%                                |
| Environment | Standard template image (Python 3.12, pandas 2.2.3)                                                  | Python 3.11, pandas 1.5.3, Kats pinned to a commit                                                                                                      |

- What the Tesla example adds from the forecasting point of view
  - On the last 30 days, Prophet has the lowest error (0.555 of the naive MAE)
  - Over 8 windows it beats the naive baseline in 4 and loses in 4, by up to about 3
    times the baseline error
  - A single window is one draw, so score several and compare with the naive baseline
- When to use which
  - Prophet directly: when the work needs a Prophet feature, such as holidays,
    external regressors, component plots, or its own cross-validation
  - Kats: when the goal is to put several models on the same series through one
    interface and compare them

## Why Kats Needs Care

- Kats is unmaintained, and the pieces below are the cost of running it today
  - The last PyPI release (0.2.0, 2022) pins `numpy<1.22`, `pandas<=1.3.5`, and
    `statsmodels==0.12.2`, which have no wheels for a current Python
    - The Dockerfile installs Kats from a pinned GitHub commit with `--no-deps`, and
      `requirements.txt` pins versions that work together
  - `ARIMAModel.predict` calls `Series.ravel()`, which pandas 2.0 removed
    - That forces pandas 1.5.3 and so Python 3.11, not the 3.12 of the template
  - `matplotlib==3.8.4` cannot draw in a Jupyter kernel with IPython 9, so
    `requirements.txt` keeps `ipython<9`
  - Importing Kats warns about optional packages and changes the root logger
    - `init_loggers()` in `kats_utils.py` cleans this up
  - Kats ARIMA ignores `alpha`, so `forecast_arima()` recomputes its intervals
- Treat Kats as a teaching tool for the common forecasting pattern, and check that it
  still installs before building on it

## How the Notebook Skills Were Used

- The notebooks were written to `.claude/skills/notebook.rules.md` first
  - The API notebook follows the structure of `notebook.create_api_intro`: a Library
    Overview with a mental-model table, a small synthetic series, one concept per
    cell, and a closing mental-model summary
  - The example notebook uses the same rules: Parts, numbered cells, and plot cells
    written as Goal, Description, Guided usage, and Implementation
- Then each skill was run on both notebooks, in the order of `tutorials/README.gp.md`

| Skill                          | What it checked                                                        | Result                                    |
| :----------------------------- | :--------------------------------------------------------------------- | :---------------------------------------- |
| `notebook.fix_logging`         | Setup cells against `notebook.template.py`, and `utils.init_loggers()` | No change                                 |
| `notebook.split_cells`         | One task per code cell, over 67 cells                                  | No change                                 |
| `notebook.split_header_cells`  | One header per markdown cell, over 90 cells                            | No change                                 |
| `notebook.format_md_cells`     | Nested bullets and width                                               | No change                                 |
| `notebook.refactor_to_utils`   | Functions or plotting code left in a notebook                          | No change, all code is in `kats_utils.py` |
| `notebook.delete_dead_code`    | Unused functions, imports, and commented-out code                      | No change                                 |
| `notebook.lint_numbered_cells` | Part and Cell numbers without gaps                                     | No change                                 |
| `notebook.fix_packages`        | Every import against `requirements.txt` and the Dockerfile             | No change                                 |

- Where the notebooks differ from the skills on purpose
  - No `Generated by` line in the title cell, since a skill did not generate the
    notebooks: they were written to the rules and checked afterwards
  - No `pip install` cell, since the packages ship in the image and a notebook-side
    install would break the Kats pins
  - No `cellN_*()` function names, since the functions in `kats_utils.py` are shared
    by both notebooks
- Not run: `notebook.format_interactive_cells` (no widgets) and the skills outside
  `notebook.*`
- The Docker image was not built where the skills ran, so the first
  `./docker_build.sh` is the check of the Dockerfile
