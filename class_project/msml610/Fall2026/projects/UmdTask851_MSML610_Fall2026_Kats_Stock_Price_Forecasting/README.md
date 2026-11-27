# Kats Tutorial

- Learn Kats in 60 minutes: several forecasting models behind one interface, with
  Tesla stock prices as the example

## Quick Start

- From the root of the repository, change directory to the tutorial folder:
  ```bash
  > cd class_project/msml610/Fall2026/projects/UmdTask851_MSML610_Fall2026_Kats_Stock_Price_Forecasting
  ```
- Build the Docker image (Python 3.11, Kats pinned to a GitHub commit):
  ```bash
  > ./docker_build.sh
  ```
- Launch Jupyter Lab inside the container:
  ```bash
  > ./docker_jupyter.sh
  ```
- Open the notebooks in this order:
  1. **`kats.API.ipynb`**: the Kats API on a small synthetic series, with
     `TimeSeriesData`, ARIMA and Prophet behind one `Params` / `Model` / `fit()` /
     `predict()` pattern, and MLAR as the model that breaks it
  2. **`kats.example.ipynb`**: forecast the next 30 trading days of Tesla's price,
     check the result with a rolling backtest, and try a volume feature and other
     intervals
- For more information on the Docker build system refer to the
  [Project template README](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/class_project/project_template/docker_scripts.README.md)
- For the comparison with the Prophet tutorial, why Kats is fragile, and how the
  notebook skills were used, see `kats.notes.md`
