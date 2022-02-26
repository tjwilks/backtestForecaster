from abc import abstractmethod, ABC
import pandas as pd
from backtest_forecaster.models.primitive_models import AbstractPrimitiveModel
from typing import Dict
from collections import namedtuple
import numpy as np
import re
from datetime import date
from dateutil.relativedelta import relativedelta



class BacktestForecaster:
    """

    """

    def get_windows(self):
        final_index = self.series_data["date_index"].max()
        first_index = self.series_data["date_index"].min()
        test_index_starts = pd.date_range(first_index, final_index, freq="MS", inclusive="right").tolist()[::-1]
        get_horizon = lambda test_index_start: (
            test_index_start + relativedelta(months=+self.max_horizon - 1)
            if test_index_start + relativedelta(months=+self.max_horizon) <= final_index
            else final_index
        )
        window = namedtuple("Window", "train_index test_index")
        windows = [
            window(
                train_index=pd.date_range(first_index, test_index_start, freq="MS", inclusive="left").tolist(),
                test_index=pd.date_range(test_index_start, get_horizon(test_index_start), freq="MS").tolist()
            )
            for test_index_start in test_index_starts
        ]
        windows = [
            window for window in windows if len(window.train_index) >= self.min_train_window_len
        ]
        return windows

    def get_backtest_models_and_forecasts(self):
        all_fit_models = {}
        backtest_forecasts = []
        unique_series_ids = self.series_data["series_id"].unique()
        for series_id in unique_series_ids:
            series = self.series_data[self.series_data["series_id"] == series_id]
            fit_models_all_windows, series_models_forecasts = self.get_model_forecasts_all_windows(
                tseries=series,
                windows=self.windows,
            )
            series_models_forecasts["series_id"] = series_id
            all_fit_models[series_id] = fit_models_all_windows
            backtest_forecasts.append(series_models_forecasts)
        backtest_forecasts = pd.concat(backtest_forecasts)
        backtest_forecasts = backtest_forecasts.sort_values(
            by=["series_id", "date_index"],
            ascending=False
        )
        return all_fit_models, backtest_forecasts

    def get_model_forecasts_all_windows(self, tseries, windows):
        max_windows = self.max_windows if len(windows) > self.max_windows else len(windows)
        models_forecasts_backtest_all_windows = []
        fit_models_all_windows = {}
        for window in windows[:max_windows]:
            train_series = tseries.loc[tseries["date_index"].isin(window.train_index), :]
            test_series = tseries.loc[tseries["date_index"].isin(window.test_index), :]
            fit_models = self.get_fit_models(
                train_series
            )
            models_backtest_window = self.get_forecasts_all_models(
                fit_models,
                test_series
            )
            fit_models_all_windows[test_series["date_index"].min()] = fit_models
            models_forecasts_backtest_all_windows.append(models_backtest_window)
        models_forecasts_backtest_all_windows = pd.concat(models_forecasts_backtest_all_windows)
        return fit_models_all_windows, models_forecasts_backtest_all_windows

    @abstractmethod
    def get_fit_models(self, train_series, test_series):
        pass

    @abstractmethod
    def get_forecasts_all_models(self, train_series, test_series):
        pass


class PrimitiveModelBacktestForecaster(BacktestForecaster):

    def __init__(
            self,
            series_data: pd.DataFrame,
            models,
            max_horizon: int = 12,
            min_train_window_len: int = 12,
            max_windows=30
    ):
        self.series_data = series_data
        self.max_horizon = max_horizon
        self.min_train_window_len = min_train_window_len
        self.max_windows = max_windows
        self.windows = self.get_windows()
        self.models = models

    def get_fit_models(self, train_series):
        y_train = np.array(train_series["actuals"], dtype=np.float64)
        fit_models = dict()
        for primitive_model_name, primitive_model in self.models.copy().items():
            primitive_model.fit(y=y_train)
            fit_models[primitive_model_name] = primitive_model
        return fit_models

    def get_forecasts_all_models(self, fit_models, test_series):
        """
        Fit models in model_list to trainset, predict on test set.

        :param train: training data
        :param test: test data
        :param seasonal_period: the seasonal period of the data
        ...
        :returns: predictions and actuals of backtest
        """
        y_test = np.array(test_series["actuals"], dtype=np.float64)
        primitive_model_forecasts = dict()
        primitive_model_forecasts["date_index"] = test_series["date_index"].max()
        primitive_model_forecasts["actuals"] = y_test
        for primitive_model_name, primitive_model in self.models.copy().items():
            forecast = primitive_model.predict(h=len(test_series))
            primitive_model_forecasts[primitive_model_name] = forecast
        primitive_model_forecasts = pd.DataFrame(primitive_model_forecasts, index=test_series.index)
        return primitive_model_forecasts


class CombinerBacktestForecaster(BacktestForecaster):

    def __init__(
            self,
            forecast_data: pd.DataFrame,
            models,
            max_horizon: int = 12,
            min_train_window_len: int = 12,
            max_windows: int = 30,
            horizon_length: int = 1
    ):

        self.series_data = self._handle_horizons(forecast_data, horizon_length)
        self.models = models
        self.max_horizon = max_horizon
        self.min_train_window_len = min_train_window_len
        self.windows = self.get_windows()
        self.max_windows = max_windows

    def get_fit_models(self, train_series):
        """
        Fit models in model_list to trainset, predict on test set.

        :param train: training data
        :param test: test data
        :param seasonal_period: the seasonal period of the data
        ...
        :returns: predictions and actuals of backtest
        """
        X_train = train_series.drop(
            columns=["predict_from", "actuals", "series_id"]).apply(pd.to_numeric)
        y_train = train_series["actuals"]
        fit_models = dict()
        for combiner_model_name, combiner_model in self.models.copy().items():
            combiner_model.fit(
                x=X_train,
                y=y_train,
                epochs=100
            )
            fit_models[combiner_model_name] = combiner_model
        return fit_models

    def get_forecasts_all_models(self, fit_models, test_series):
        """
        Fit models in model_list to trainset, predict on test set.

        :param train: training data
        :param test: test data
        :param seasonal_period: the seasonal period of the data
        ...
        :returns: predictions and actuals of backtest
        """
        # Check test set starts 1 point on from train set
        X_test = test_series.drop(
            columns=["predict_from", "actuals", "series_id"]).apply(pd.to_numeric)
        combiner_forecasts = dict()
        combiner_forecasts["predict_from"] = test_series["predict_from"].min()
        combiner_forecasts["date_index"] = test_series["date_index"].min()
        combiner_forecasts["actuals"] = test_series["actuals"]
        for combiner_model_name, combiner_model in fit_models.copy().items():
            forecast = combiner_model.predict(x=X_test)
            combiner_forecasts[combiner_model_name] = forecast
        combiner_forecasts = pd.DataFrame(combiner_forecasts, index=test_series.index)
        return combiner_forecasts

    @staticmethod
    def _handle_horizons(forecast_data, horizon_length):
        forecast_data["horizon"] = (
                (forecast_data['date_index'] - forecast_data['predict_from'])/np.timedelta64(1, 'M')
        ).astype(int) + 1
        forecast_data = forecast_data[forecast_data["horizon"] == horizon_length]
        forecast_data = forecast_data.drop(["date_index", "horizon"], axis=1)
        series_data = forecast_data.groupby(
            by=["series_id", "predict_from"],
            as_index=False).sum()
        series_data["date_index"] = series_data["predict_from"]
        return series_data
