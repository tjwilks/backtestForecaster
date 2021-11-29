import pandas as pd
from backtest_forecaster.models.primitive_models import AbstractPrimitiveModel
from typing import Dict
from collections import namedtuple


def get_model_forecasts_all_series(series_data: pd.DataFrame,
                                   models: Dict[str, AbstractPrimitiveModel],
                                   forecasting_function,
                                   max_horizon: int = 12,
                                   min_train_window_len: int = 12,
                                   max_windows=30):
    all_series_models_forecasts = []
    unique_series_ids = series_data["series_id"].unique()
    n_series_to_test = 1
    for series_id in unique_series_ids:
        series = series_data[series_data["series_id"] == series_id]
        windows = get_windows(
            time_series_len=len(series),
            max_horizon=max_horizon,
            min_train_window_len=min_train_window_len+1
        )
        if len(windows) == 0:
            continue
        series_models_forecasts = get_models_forecasts_all_windows(
            tseries=series,
            models=models,
            windows=windows,
            max_windows=max_windows,
            forecasting_function=forecasting_function
        )
        series_models_forecasts["series_id"] = series_id
        all_series_models_forecasts.append(series_models_forecasts)
        if n_series_to_test == 5:
            break
        n_series_to_test += 1
    all_series_models_forecasts = pd.concat(all_series_models_forecasts)
    all_series_models_forecasts = all_series_models_forecasts.sort_values("predict_from", ascending=False)
    return all_series_models_forecasts


def get_windows(time_series_len,
                max_horizon,
                min_train_window_len):

    final_index = time_series_len
    test_index_starts = reversed(range(min_train_window_len+1, final_index))
    get_horizon = lambda test_index_start: (
        max_horizon if (test_index_start + max_horizon < final_index)
        else final_index - test_index_start
    )
    window = namedtuple("Window", "train_index test_index")
    windows = [
        window(
            train_index=list(range(1, test_index_start)),
            test_index=list(range(test_index_start, test_index_start+get_horizon(test_index_start)))
        )
        for test_index_start in test_index_starts
    ]
    return windows


def get_models_forecasts_all_windows(tseries,
                                     models,
                                     windows,
                                     max_windows,
                                     forecasting_function):

    max_windows = max_windows if len(windows) > max_windows else len(windows)
    models_forecasts_backtest_all_windows = []
    for window in windows[:max_windows]:
        train_series, test_series = tseries.iloc[window.train_index, :], tseries.iloc[window.test_index, :]
        models_backtest_window = forecasting_function(
            models,
            train_series,
            test_series
        )
        models_forecasts_backtest_all_windows.append(models_backtest_window)
    models_forecasts_backtest_all_windows = pd.concat(models_forecasts_backtest_all_windows)
    return models_forecasts_backtest_all_windows
